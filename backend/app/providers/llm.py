from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import json
import re

from backend.app.config import Settings
from backend.app.domain.documents import Evidence, EvidenceImage
from backend.app.domain.rag import EvidenceAssessment, RoutePlan, RouteStep, TaskDecision


class LlmProvider(ABC):
    @abstractmethod
    def answer_with_evidence(self, question: str, evidence: list[Evidence]) -> str:
        raise NotImplementedError

    def uses_llm_for_answer(self) -> bool:
        return True

    def healthcheck(self) -> None:
        """验证当前 provider 是否能执行其配置的问答能力。"""
        return None

    def plan_route(
        self,
        question: str,
        *,
        conversation_context: str | None,
        scope: str,
        document_ids: list[str] | None,
    ) -> RoutePlan | None:
        return None

    def assess_evidence(self, question: str, evidence: list[Evidence]) -> EvidenceAssessment:
        valid_evidence = [item for item in evidence if item.text.strip()]
        return EvidenceAssessment(
            sufficient=bool(valid_evidence),
            covered_aspects=["已检索到非空候选证据"] if valid_evidence else [],
            missing_aspects=[] if valid_evidence else ["没有检索到可用于回答的证据"],
            supporting_chunk_ids=[item.chunk_id for item in valid_evidence],
            reason="当前 LLM provider 未实现结构化证据评估，已按非空证据进行回退。",
            fallback_used=True,
        )

    def assess_evidence_for_task(
        self,
        question: str,
        evidence: list[Evidence],
        *,
        task_decision: TaskDecision | None,
        requirements: list[str],
        retrieval_round: int,
        max_retrieval_rounds: int,
    ) -> EvidenceAssessment:
        return self.assess_evidence(question, evidence)

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

    def healthcheck(self) -> None:
        self.client.models.list()

    def plan_route(
        self,
        question: str,
        *,
        conversation_context: str | None,
        scope: str,
        document_ids: list[str] | None,
    ) -> RoutePlan | None:
        context = {
            "scope": scope,
            "document_ids": document_ids or [],
            "conversation_context": (conversation_context or "")[:3000],
            "allowed_routes": ["chunk_search", "figure_index", "document_profile"],
            "max_steps": 3,
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "你是化学与材料文献问答系统的中文路由规划器。"
                    "根据用户问题、会话上下文和文档范围，输出有界 route_plan。"
                    "不要回答问题，只决定检索/处理步骤。"
                    "route 只能取 allowed_routes 中的值；最多 3 步。"
                    "图片、图号、谱图、曲线定位走 figure_index；"
                    "整篇论文摘要、主题、关键词、跨文档找论文走 document_profile；"
                    "数值、实验条件、机制、结论支撑、高风险安全/合成问题走 chunk_search。"
                    "复合问题可拆成多步，例如先 document_profile 再 figure_index。"
                    "高风险问题必须包含 chunk_search。"
                    "严格输出 JSON："
                    '{"plan_type":"single_step或multi_step","rationale":"简短理由",'
                    '"steps":[{"route":"...","task_type":"...","query":"...","purpose":"...",'
                    '"max_rounds":1,"requires_assessment":true,"requires_answer_verification":false}]}。'
                ),
            },
            {
                "role": "user",
                "content": f"用户问题：\n{question}\n\n路由上下文：\n{json.dumps(context, ensure_ascii=False)}",
            },
        ]
        future = self.executor.submit(
            self.client.chat.completions.create,
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
        )
        try:
            response = future.result(timeout=min(self.timeout_seconds, 3))
            payload = json.loads(response.choices[0].message.content or "{}")
        except (FutureTimeoutError, json.JSONDecodeError):
            return None
        except Exception as exc:
            if _is_recoverable_llm_error(exc):
                return None
            raise
        return _sanitize_route_plan(payload, question)

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
            if _is_recoverable_llm_error(exc):
                return _fallback_answer_with_evidence(question, evidence)
            raise

    def assess_evidence(self, question: str, evidence: list[Evidence]) -> EvidenceAssessment:
        return self._assess_evidence(question, evidence, task_context=None)

    def assess_evidence_for_task(
        self,
        question: str,
        evidence: list[Evidence],
        *,
        task_decision: TaskDecision | None,
        requirements: list[str],
        retrieval_round: int,
        max_retrieval_rounds: int,
    ) -> EvidenceAssessment:
        context = {
            "task_type": task_decision.task_type if task_decision else "unknown",
            "risk_level": task_decision.risk_level if task_decision else "unknown",
            "assessment_requirements": requirements,
            "retrieval_round": retrieval_round,
            "max_retrieval_rounds": max_retrieval_rounds,
        }
        return self._assess_evidence(question, evidence, task_context=json.dumps(context, ensure_ascii=False))

    def _assess_evidence(
        self,
        question: str,
        evidence: list[Evidence],
        task_context: str | None,
    ) -> EvidenceAssessment:
        valid_evidence = [item for item in evidence if item.text.strip()]
        if not valid_evidence:
            return EvidenceAssessment(
                sufficient=False,
                missing_aspects=["没有检索到可用于回答的证据"],
                next_search_goal="补充与问题直接相关的文献证据",
                reason="没有可用于评估的非空检索证据。",
            )

        candidates = [
            {
                "chunk_id": item.chunk_id,
                "document_id": item.document_id,
                "pages": f"{item.page_start}-{item.page_end}",
                "section_title": item.section_title,
                "text": item.text[:2400],
            }
            for item in valid_evidence
        ]
        messages = [
            {
                "role": "system",
                "content": (
                    "你是化学与材料文献问答系统的证据充分性评估器。"
                    "判断给定检索证据是否已覆盖用户问题的关键方面。"
                    "严格依据任务判定上下文中的覆盖标准，不得自行扩大用户要求的范围。"
                    "只有证据明确支撑所有必要方面时 sufficient 才能为 true；"
                    "仅主题相关、信息残缺或需要猜测时必须为 false。"
                    "supporting_chunk_ids 只能包含提供的 chunk_id，并只列出回答可以引用的证据。"
                    "missing_aspects 列出尚缺证据的方面，conflicts 列出相互冲突的信息，"
                    "next_search_goal 在资料不足时说明下一轮应补查的目标。"
                    "confidence 为 0 到 1 的估计值，仅用于观察。"
                    "严格输出 JSON："
                    '{"sufficient":true或false,"covered_aspects":["..."],'
                    '"missing_aspects":["..."],"conflicts":["..."],'
                    '"supporting_chunk_ids":["..."],"next_search_goal":"..."或null,'
                    '"confidence":0.0,"reason":"简短理由"}。'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"用户问题：\n{question}\n\n"
                    f"任务判定上下文：\n{task_context or '未提供，按用户原问题的明确要求判断。'}\n\n"
                    f"检索证据：\n{json.dumps(candidates, ensure_ascii=False)}"
                ),
            },
        ]
        future = self.executor.submit(
            self.client.chat.completions.create,
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
        )
        try:
            response = future.result(timeout=self.timeout_seconds)
            payload = json.loads(response.choices[0].message.content or "{}")
        except (FutureTimeoutError, json.JSONDecodeError):
            return _fallback_evidence_assessment(valid_evidence, "证据评估响应超时或格式无效，已保留候选证据。")
        except Exception as exc:
            if _is_recoverable_llm_error(exc):
                return _fallback_evidence_assessment(valid_evidence, "证据评估服务暂时不可用，已保留候选证据。")
            raise
        return _sanitize_evidence_assessment(payload, valid_evidence)

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
            if _is_recoverable_llm_error(exc):
                return _fallback_selected_images(question, answer, evidence, max_images)
            raise
        try:
            payload = json.loads(response.choices[0].message.content or "{}")
        except json.JSONDecodeError:
            return _fallback_selected_images(question, answer, evidence, max_images)
        selected = _sanitize_selected_images(payload.get("images", []), by_chunk, max_images)
        return selected or _fallback_selected_images(question, answer, evidence, max_images)


class FakeLlmProvider(LlmProvider):
    def uses_llm_for_answer(self) -> bool:
        return False

    def assess_evidence(self, question: str, evidence: list[Evidence]) -> EvidenceAssessment:
        valid_evidence = [item for item in evidence if item.text.strip()]
        return EvidenceAssessment(
            sufficient=bool(valid_evidence),
            covered_aspects=["测试证据覆盖问题"] if valid_evidence else [],
            missing_aspects=[] if valid_evidence else ["没有检索到可用于回答的证据"],
            supporting_chunk_ids=[item.chunk_id for item in valid_evidence],
            confidence=1.0 if valid_evidence else 0.0,
            reason="测试 provider 将非空检索证据视为充分。",
        )

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


class ComposedLlmProvider(LlmProvider):
    """将回答与证据评估 provider 解耦，供离线评测按成本组合能力。"""

    def __init__(self, answer_provider: LlmProvider, assessment_provider: LlmProvider) -> None:
        self.answer_provider = answer_provider
        self.assessment_provider = assessment_provider

    def uses_llm_for_answer(self) -> bool:
        return self.answer_provider.uses_llm_for_answer()

    def healthcheck(self) -> None:
        self.answer_provider.healthcheck()
        if self.assessment_provider is not self.answer_provider:
            self.assessment_provider.healthcheck()

    def plan_route(
        self,
        question: str,
        *,
        conversation_context: str | None,
        scope: str,
        document_ids: list[str] | None,
    ) -> RoutePlan | None:
        return self.answer_provider.plan_route(
            question,
            conversation_context=conversation_context,
            scope=scope,
            document_ids=document_ids,
        )

    def answer_with_evidence(self, question: str, evidence: list[Evidence]) -> str:
        return self.answer_provider.answer_with_evidence(question, evidence)

    def assess_evidence(self, question: str, evidence: list[Evidence]) -> EvidenceAssessment:
        return self.assessment_provider.assess_evidence(question, evidence)

    def assess_evidence_for_task(
        self,
        question: str,
        evidence: list[Evidence],
        *,
        task_decision: TaskDecision | None,
        requirements: list[str],
        retrieval_round: int,
        max_retrieval_rounds: int,
    ) -> EvidenceAssessment:
        return self.assessment_provider.assess_evidence_for_task(
            question,
            evidence,
            task_decision=task_decision,
            requirements=requirements,
            retrieval_round=retrieval_round,
            max_retrieval_rounds=max_retrieval_rounds,
        )

    def select_images_for_answer(
        self,
        question: str,
        answer: str,
        evidence: list[Evidence],
        max_images: int = 3,
    ) -> dict[str, list[EvidenceImage]]:
        return self.answer_provider.select_images_for_answer(question, answer, evidence, max_images)


def _sanitize_route_plan(payload: object, question: str) -> RoutePlan | None:
    if not isinstance(payload, dict):
        return None
    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list):
        return None
    allowed_routes = {"chunk_search", "figure_index", "document_profile"}
    steps: list[RouteStep] = []
    for index, raw_step in enumerate(raw_steps[:3], start=1):
        if not isinstance(raw_step, dict):
            continue
        route = str(raw_step.get("route") or "").strip()
        if route not in allowed_routes:
            continue
        task_type = str(raw_step.get("task_type") or route).strip()[:80]
        query = str(raw_step.get("query") or question).strip()[:1000] or question
        purpose = str(raw_step.get("purpose") or "").strip()[:300]
        try:
            max_rounds = int(raw_step.get("max_rounds")) if raw_step.get("max_rounds") is not None else None
        except (TypeError, ValueError):
            max_rounds = None
        if max_rounds is not None:
            max_rounds = max(1, min(max_rounds, 3))
        steps.append(
            RouteStep(
                step_id=f"step-{index}",
                route=route,
                task_type=task_type or route,
                query=query,
                purpose=purpose,
                max_rounds=max_rounds,
                requires_assessment=bool(raw_step.get("requires_assessment", True)),
                requires_answer_verification=bool(raw_step.get("requires_answer_verification", False)),
            )
        )
    if not steps:
        return None
    plan_type = str(payload.get("plan_type") or ("multi_step" if len(steps) > 1 else "single_step")).strip()[:50]
    return RoutePlan(
        steps=steps,
        plan_type=plan_type,
        rationale=str(payload.get("rationale") or "").strip()[:500],
        used_llm=True,
    )


def _sanitize_evidence_assessment(payload: object, evidence: list[Evidence]) -> EvidenceAssessment:
    if not isinstance(payload, dict):
        return _fallback_evidence_assessment(evidence, "证据评估返回结构无效，已保留候选证据。")
    sufficient = bool(payload.get("sufficient", payload.get("answerable", False)))
    reason = str(payload.get("reason") or "")[:500]
    covered_aspects = _sanitize_string_list(payload.get("covered_aspects"))
    missing_aspects = _sanitize_string_list(payload.get("missing_aspects"))
    conflicts = _sanitize_string_list(payload.get("conflicts"))
    next_search_goal = str(payload.get("next_search_goal") or "").strip()[:500] or None
    confidence = _sanitize_confidence(payload.get("confidence"))
    available_ids = {item.chunk_id for item in evidence}
    raw_ids = payload.get("supporting_chunk_ids", [])
    supporting_ids = [
        str(chunk_id)
        for chunk_id in raw_ids
        if isinstance(chunk_id, str) and chunk_id in available_ids
    ] if isinstance(raw_ids, list) else []
    supporting_ids = list(dict.fromkeys(supporting_ids))
    if sufficient and not supporting_ids:
        return EvidenceAssessment(
            sufficient=False,
            covered_aspects=covered_aspects,
            missing_aspects=missing_aspects or ["没有返回可引用的有效支撑证据片段"],
            conflicts=conflicts,
            next_search_goal=next_search_goal or "检索能够直接支撑结论的证据片段",
            confidence=confidence,
            reason="模型判定证据充分，但没有返回有效的支撑证据片段。",
            used_llm=True,
        )
    return EvidenceAssessment(
        sufficient=sufficient,
        covered_aspects=covered_aspects,
        missing_aspects=missing_aspects,
        conflicts=conflicts,
        supporting_chunk_ids=supporting_ids,
        next_search_goal=next_search_goal,
        confidence=confidence,
        reason=reason or ("模型判断检索证据足以回答问题。" if sufficient else "模型判断检索证据不足以直接回答问题。"),
        used_llm=True,
    )


def _fallback_evidence_assessment(evidence: list[Evidence], reason: str) -> EvidenceAssessment:
    return EvidenceAssessment(
        sufficient=False,
        covered_aspects=["已保留非空候选证据"] if evidence else [],
        missing_aspects=["证据充分性评估未完成，不能据此生成可靠结论"] if evidence else ["没有检索到可用于回答的证据"],
        supporting_chunk_ids=[],
        next_search_goal=None if evidence else "补充与问题直接相关的文献证据",
        reason=reason,
        used_llm=True,
        fallback_used=True,
    )


def _sanitize_string_list(value: object, limit: int = 12) -> list[str]:
    if not isinstance(value, list):
        return []
    values = [str(item).strip()[:240] for item in value if isinstance(item, str) and item.strip()]
    return list(dict.fromkeys(values))[:limit]


def _sanitize_confidence(value: object) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    return max(0.0, min(1.0, float(value)))


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
        "LLM 服务暂时不可用，先基于已验证证据给出快速摘要：",
    ]
    for index, item in enumerate(evidence[:3], start=1):
        snippet = re.sub(r"\s+", " ", item.text).strip()[:260]
        lines.append(f"[{index}] p. {item.page_start}-{item.page_end}，{item.section_title}：{snippet}")
    return "\n".join(lines)


def _is_recoverable_llm_error(exc: Exception) -> bool:
    error_name = exc.__class__.__name__.lower()
    error_text = str(exc).lower()
    if any(
        marker in error_name or marker in error_text
        for marker in (
            "timeout",
            "timed out",
            "connection",
            "connecterror",
            "ratelimit",
            "rate limit",
            "serviceunavailable",
            "service unavailable",
            "internalserver",
        )
    ):
        return True
    status_code = getattr(exc, "status_code", None)
    return isinstance(status_code, int) and (status_code in {408, 429} or status_code >= 500)


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
