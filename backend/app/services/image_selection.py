from dataclasses import dataclass, replace
from time import perf_counter

from backend.app.domain.documents import Evidence, EvidenceImage
from backend.app.domain.rag import ImageSelectionResult, RoutePlan, TaskDecision
from backend.app.providers.llm import LlmProvider


@dataclass(frozen=True)
class ImageSelectionContext:
    # 图片选择只消费最终已验证证据；route_plan 只用于判断选择策略，不替代证据约束。
    question: str
    answer: str
    conversation_context: str | None
    route_plan: RoutePlan | None
    completed_route_steps: list[dict[str, object]]
    task_decision: TaskDecision | None
    verified_evidence: list[Evidence]
    max_images: int = 3


class ImageSelectionService:
    def __init__(self, llm_provider: LlmProvider) -> None:
        self.llm_provider = llm_provider

    def select(self, context: ImageSelectionContext) -> ImageSelectionResult:
        started_at = perf_counter()
        candidate_count = _image_candidate_count(context.verified_evidence)
        if candidate_count == 0:
            return ImageSelectionResult(
                images={},
                strategy="none",
                candidate_count=0,
                selected_count=0,
                latency_ms=(perf_counter() - started_at) * 1000,
                reason="最终已验证证据中没有候选图片。",
            )

        figure_evidence = [item for item in context.verified_evidence if _is_figure_evidence(item)]
        if _has_figure_intent(context) and figure_evidence:
            selected = _select_first_figure_images(figure_evidence, context.max_images)
            return ImageSelectionResult(
                images=selected,
                strategy="figure_index",
                used_llm=False,
                fallback_used=False,
                candidate_count=candidate_count,
                selected_count=_selected_image_count(selected),
                latency_ms=(perf_counter() - started_at) * 1000,
                reason="route_plan、已完成步骤、任务类型或证据来源表明这是图片相关结果。",
            )

        result = self.llm_provider.select_images_for_answer(
            _build_contextual_question(context.question, context.conversation_context),
            context.answer,
            context.verified_evidence,
            context.max_images,
        )
        return replace(result, latency_ms=(perf_counter() - started_at) * 1000)


def _has_figure_intent(context: ImageSelectionContext) -> bool:
    if context.route_plan and any(step.route == "figure_index" for step in context.route_plan.steps):
        return True
    if any(step.get("route") == "figure_index" for step in context.completed_route_steps):
        return True
    if context.task_decision and context.task_decision.task_type == "figure_query":
        return True
    return any(_is_figure_evidence(item) for item in context.verified_evidence)


def _is_figure_evidence(item: Evidence) -> bool:
    section_title = (item.section_title or "").strip().lower()
    return bool(item.images) and (
        item.chunk_id.startswith("figure:") or section_title == "figure index" or section_title.startswith("figure ")
    )


def _select_first_figure_images(evidence: list[Evidence], max_images: int) -> dict[str, list[EvidenceImage]]:
    selected: dict[str, list[EvidenceImage]] = {}
    count = 0
    seen: set[str] = set()
    for item in evidence:
        if count >= max_images:
            break
        if not item.images:
            continue
        image = item.images[0]
        if image.image_id in seen:
            continue
        seen.add(image.image_id)
        selected[item.chunk_id] = [
            EvidenceImage(
                image_id=image.image_id,
                page=image.page,
                width=image.width,
                height=image.height,
                label=image.label,
                caption=image.caption,
                nearby_text=image.nearby_text,
                reason=image.caption or image.nearby_text or "Selected from the figure index.",
            )
        ]
        count += 1
    return selected


def _build_contextual_question(question: str, conversation_context: str | None) -> str:
    context = (conversation_context or "").strip()
    if not context:
        return question
    return (
        "对话上下文仅用于解析追问、指代和限定条件；回答仍必须依据本轮检索证据。\n"
        f"{context}\n\n"
        f"当前问题：{question}"
    )


def _image_candidate_count(evidence: list[Evidence]) -> int:
    return sum(len(item.images) for item in evidence)


def _selected_image_count(selected: dict[str, list[EvidenceImage]]) -> int:
    return sum(len(images) for images in selected.values())
