from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import json
import re

from backend.app.config import Settings
from backend.app.domain.documents import Evidence, EvidenceImage


class LlmProvider(ABC):
    @abstractmethod
    def answer_with_evidence(self, question: str, evidence: list[Evidence]) -> str:
        raise NotImplementedError

    def select_images_for_answer(
        self,
        question: str,
        answer: str,
        evidence: list[Evidence],
        max_images: int = 3,
    ) -> dict[str, list[EvidenceImage]]:
        return {}


class DeepSeekLlmProvider(LlmProvider):
    def __init__(self, settings: Settings) -> None:
        if not settings.deepseek_api_key:
            raise ValueError("DEEPSEEK_API_KEY is required for DeepSeekLlmProvider.")
        from openai import OpenAI

        self.model = settings.deepseek_model
        self.image_selection_enabled = settings.llm_image_selection_enabled
        self.timeout_seconds = settings.deepseek_timeout_seconds
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="deepseek")
        self.client = OpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            timeout=settings.deepseek_timeout_seconds,
        )

    def answer_with_evidence(self, question: str, evidence: list[Evidence]) -> str:
        if _is_figure_listing_question(question):
            quick_answer = _answer_figure_listing_question(evidence)
            if quick_answer:
                return quick_answer

        context = "\n\n".join(
            f"[{idx + 1}] page {item.page_start}-{item.page_end}, {item.section_title}\n{item.text}"
            for idx, item in enumerate(evidence)
        )
        messages = [
            {
                "role": "system",
                "content": "You answer chemistry/materials-science questions using only the provided evidence. Cite page numbers when possible.",
            },
            {"role": "user", "content": f"Question:\n{question}\n\nEvidence:\n{context}"},
        ]
        future = self.executor.submit(self.client.chat.completions.create, model=self.model, messages=messages)
        try:
            response = future.result(timeout=self.timeout_seconds)
            return response.choices[0].message.content or ""
        except FutureTimeoutError:
            return _fallback_answer_with_evidence(question, evidence)
        except Exception as exc:
            if "timeout" in exc.__class__.__name__.lower() or "timed out" in str(exc).lower():
                return _fallback_answer_with_evidence(question, evidence)
            raise

    def select_images_for_answer(
        self,
        question: str,
        answer: str,
        evidence: list[Evidence],
        max_images: int = 3,
    ) -> dict[str, list[EvidenceImage]]:
        if not getattr(self, "image_selection_enabled", True):
            return _fallback_selected_images(question, answer, evidence, max_images)

        candidates = []
        by_chunk: dict[str, dict[str, EvidenceImage]] = {}
        for item in evidence:
            by_chunk[item.chunk_id] = {image.image_id: image for image in item.images}
            for image in item.images:
                candidates.append(
                    {
                        "chunk_id": item.chunk_id,
                        "image_id": image.image_id,
                        "page": image.page,
                        "width": image.width,
                        "height": image.height,
                        "label": image.label,
                        "caption": image.caption,
                        "nearby_text": image.nearby_text[:800] if image.nearby_text else None,
                        "section_title": item.section_title,
                        "evidence_preview": item.text[:500],
                    }
                )
        if not candidates:
            return {}

        messages = [
            {
                "role": "system",
                "content": (
                    "Select images that are actually useful for the answer. "
                    "Only choose image_id values from the provided candidates. "
                    "Use caption, nearby_text, and evidence_preview to judge relevance. "
                    "If the image caption/nearby_text has no clear relationship to the answer, do not select it. "
                    f"Return strict JSON: {{\"images\":[{{\"chunk_id\":\"...\",\"image_id\":\"...\",\"reason\":\"...\"}}]}}. "
                    f"Select at most {max_images} images. Return an empty list if no image is clearly useful."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\nAnswer:\n{answer}\n\n"
                    f"Candidate images:\n{json.dumps(candidates, ensure_ascii=False)}"
                ),
            },
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            if "timeout" in exc.__class__.__name__.lower() or "timed out" in str(exc).lower():
                return _fallback_selected_images(question, answer, evidence, max_images)
            raise
        try:
            payload = json.loads(response.choices[0].message.content or "{}")
        except json.JSONDecodeError:
            return _fallback_selected_images(question, answer, evidence, max_images)
        selected = _sanitize_selected_images(payload.get("images", []), by_chunk, max_images)
        return selected or _fallback_selected_images(question, answer, evidence, max_images)


class FakeLlmProvider(LlmProvider):
    def answer_with_evidence(self, question: str, evidence: list[Evidence]) -> str:
        if not evidence:
            return f"I could not find enough evidence to answer: {question}"
        pages = ", ".join(str(item.page_start) for item in evidence)
        return f"Based on the retrieved evidence on page(s) {pages}, this is a grounded draft answer to: {question}"

    def select_images_for_answer(
        self,
        question: str,
        answer: str,
        evidence: list[Evidence],
        max_images: int = 3,
    ) -> dict[str, list[EvidenceImage]]:
        selected: dict[str, list[EvidenceImage]] = {}
        remaining = max_images
        for item in evidence:
            if remaining <= 0:
                break
            if not item.images:
                continue
            image = item.images[0]
            selected[item.chunk_id] = [
                EvidenceImage(
                    image_id=image.image_id,
                    page=image.page,
                    width=image.width,
                    height=image.height,
                    label=image.label,
                    caption=image.caption,
                    nearby_text=image.nearby_text,
                    reason="Selected by fake provider from candidate evidence images.",
                )
            ]
            remaining -= 1
        return selected


def _sanitize_selected_images(
    selected: list[object],
    by_chunk: dict[str, dict[str, EvidenceImage]],
    max_images: int,
) -> dict[str, list[EvidenceImage]]:
    result: dict[str, list[EvidenceImage]] = {}
    count = 0
    for item in selected:
        if count >= max_images or not isinstance(item, dict):
            break
        chunk_id = str(item.get("chunk_id") or "")
        image_id = str(item.get("image_id") or "")
        candidate = by_chunk.get(chunk_id, {}).get(image_id)
        if candidate is None:
            continue
        reason = str(item.get("reason") or "")[:240] or None
        result.setdefault(chunk_id, []).append(
            EvidenceImage(
                image_id=candidate.image_id,
                page=candidate.page,
                width=candidate.width,
                height=candidate.height,
                label=candidate.label,
                caption=candidate.caption,
                nearby_text=candidate.nearby_text,
                reason=reason,
            )
        )
        count += 1
    return result


def _fallback_selected_images(
    question: str,
    answer: str,
    evidence: list[Evidence],
    max_images: int,
) -> dict[str, list[EvidenceImage]]:
    scored: list[tuple[int, int, Evidence, EvidenceImage]] = []
    query_terms = _meaningful_terms(f"{question} {answer}")
    for evidence_rank, item in enumerate(evidence):
        has_figure_reference = bool(
            re.search(r"\b(?:fig(?:ure)?|scheme|table)\.?\s*\d+|[图表]\s*\d+", f"{answer}\n{item.text}", re.IGNORECASE)
        )
        for image in item.images:
            metadata_context = " ".join(value for value in [image.caption, image.nearby_text] if value)
            overlap = len(query_terms & _meaningful_terms(metadata_context))
            score = 0
            if metadata_context and overlap > 0:
                score += overlap + 1
            if has_figure_reference:
                score += 2
            if score > 0:
                scored.append((score, evidence_rank, item, image))

    scored.sort(key=lambda candidate: (-candidate[0], candidate[1], candidate[3].page, candidate[3].image_id))
    result: dict[str, list[EvidenceImage]] = {}
    seen: set[str] = set()
    for _score, _rank, item, image in scored:
        if len(seen) >= max_images:
            break
        if image.image_id in seen:
            continue
        reason = image.caption or image.nearby_text or "Selected from candidate images linked to figure-related evidence."
        result.setdefault(item.chunk_id, []).append(
            EvidenceImage(
                image_id=image.image_id,
                page=image.page,
                width=image.width,
                height=image.height,
                label=image.label,
                caption=image.caption,
                nearby_text=image.nearby_text,
                reason=reason[:240],
            )
        )
        seen.add(image.image_id)
    return result


def _meaningful_terms(text: str) -> set[str]:
    terms = {term.lower() for term in re.findall(r"[A-Za-z][A-Za-z0-9-]{3,}", text)}
    terms.update(re.findall(r"[\u4e00-\u9fff]{2,}", text))
    return terms


def _fallback_answer_with_evidence(question: str, evidence: list[Evidence]) -> str:
    if not evidence:
        return f"未检索到足够证据回答：{question}"
    lines = [
        "LLM 响应超时，先基于检索证据给出快速摘要：",
    ]
    for index, item in enumerate(evidence[:3], start=1):
        snippet = re.sub(r"\s+", " ", item.text).strip()[:260]
        lines.append(f"[{index}] p. {item.page_start}-{item.page_end}，{item.section_title}：{snippet}")
    return "\n".join(lines)


def _is_figure_listing_question(question: str) -> bool:
    normalized = question.lower()
    return any(
        marker in normalized
        for marker in [
            "有什么图",
            "哪些图",
            "性能图",
            "图表",
            "figure",
            "fig.",
            "plot",
            "chart",
        ]
    )


def _answer_figure_listing_question(evidence: list[Evidence]) -> str:
    rows: list[str] = []
    seen: set[str] = set()
    figure_pattern = re.compile(r"(?:Figure|Fig\.)\s*\d+(?:\.\d+)?|图\s*\d+(?:\.\d+)?", re.IGNORECASE)
    sentence_pattern = re.compile(r"[^。.!?\n]*(?:(?:Figure|Fig\.)\s*\d+(?:\.\d+)?|图\s*\d+(?:\.\d+)?)[^。.!?\n]*(?:[。.!?]|$)", re.IGNORECASE)
    for item in evidence:
        text = re.sub(r"\s+", " ", item.text).strip()
        for match in sentence_pattern.finditer(text):
            sentence = match.group(0).strip(" ;,，")
            figure = figure_pattern.search(sentence)
            key = f"{figure.group(0).lower() if figure else sentence[:20]}:{item.page_start}"
            if not sentence or key in seen:
                continue
            seen.add(key)
            rows.append(f"- p. {item.page_start}-{item.page_end}: {sentence[:220]}")
            if len(rows) >= 8:
                break
        if len(rows) >= 8:
            break
    if not rows:
        return ""
    return "根据检索到的原文片段，该文献涉及的图/性能图包括：\n" + "\n".join(rows)


def build_llm_provider(settings: Settings) -> LlmProvider:
    if settings.env == "test":
        return FakeLlmProvider()
    return DeepSeekLlmProvider(settings)
