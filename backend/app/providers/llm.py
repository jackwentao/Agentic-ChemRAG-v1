from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import json
import re

from backend.app.config import Settings
from backend.app.domain.documents import DocumentSection, Evidence, EvidenceImage, SourceRef
from backend.app.domain.rag import (
    AnswerVerification,
    EvidenceAssessment,
    ImageSelectionResult,
    RoutePlan,
    RouteStep,
    TaskDecision,
)
from backend.app.services.profile_schema import ProfileExtraction


class LlmProvider(ABC):
    @abstractmethod
    def answer_with_evidence(self, question: str, evidence: list[Evidence]) -> str:
        raise NotImplementedError

    def uses_llm_for_answer(self) -> bool:
        return True

    def healthcheck(self) -> None:
        """验证当前 provider 是否能执行其配置的问答能力。"""
        return None

    def extract_document_profile(
        self,
        *,
        document_id: str,
        document_text: str,
        outline: list[DocumentSection],
        figure_captions: list[str],
    ) -> ProfileExtraction | None:
        return None

    def plan_route(
        self,
        question: str,
        *,
        conversation_context: str | None,
        scope: str,
        document_ids: list[str] | None,
        basic_facts: list[str] | None = None,
        outline_context: list[Evidence] | None = None,
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

    def refine_search_queries(
        self,
        question: str,
        *,
        assessment: EvidenceAssessment,
        task_decision: TaskDecision | None,
        route: str,
        scope: str,
        attempted_queries: list[str],
        max_queries: int,
        safety_mode: str,
    ) -> dict[str, object] | None:
        return None

    def verify_answer_with_evidence(
        self,
        question: str,
        answer: str,
        evidence: list[Evidence],
        *,
        limitations: list[str],
        task_decision: TaskDecision | None,
        evidence_assessment: EvidenceAssessment | None,
        trigger_reasons: list[str],
    ) -> AnswerVerification:
        return AnswerVerification(
            passed=True,
            triggered=bool(trigger_reasons),
            trigger_reasons=trigger_reasons,
            reason="当前 provider 未实现答案级核验，默认通过。",
            fallback_used=True,
        )

    def revise_answer_with_evidence(
        self,
        question: str,
        answer: str,
        evidence: list[Evidence],
        *,
        limitations: list[str],
        revision_instruction: str,
    ) -> str:
        if limitations and "资料限制" not in answer:
            return "\n".join([answer, "", "资料限制：", *[f"- {item}" for item in limitations]])
        return answer

    def select_images_for_answer(
        self,
        question: str,
        answer: str,
        evidence: list[Evidence],
        max_images: int = 3,
    ) -> ImageSelectionResult:
        return ImageSelectionResult(
            images={},
            strategy="none",
            candidate_count=_image_candidate_count(evidence),
            selected_count=0,
            reason="当前 provider 未实现图片选择。",
        )


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
        basic_facts: list[str] | None = None,
        outline_context: list[Evidence] | None = None,
    ) -> RoutePlan | None:
        context = {
            "scope": scope,
            "document_ids": document_ids or [],
            "conversation_context": (conversation_context or "")[:3000],
            "basic_facts": basic_facts or [],
            "outline_context": [
                _evidence_candidate(item, text_limit=1200)
                for item in (outline_context or [])[:5]
            ],
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
                    "basic_facts 和 outline_context 是已完成的前置问题重写与大纲/画像检索结果，"
                    "必须作为规划依据；不要重复做已经由 outline_context 覆盖的论文定位。"
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
            response = future.result(timeout=self.timeout_seconds)
            payload = json.loads(response.choices[0].message.content or "{}")
        except (FutureTimeoutError, json.JSONDecodeError):
            return None
        except Exception as exc:
            if _is_recoverable_llm_error(exc):
                return None
            raise
        return _sanitize_route_plan(payload, question)

    def answer_with_evidence(self, question: str, evidence: list[Evidence]) -> str:
        context = "\n\n".join(_format_evidence_for_prompt(idx, item) for idx, item in enumerate(evidence))
        messages = [
            {
                "role": "system",
                "content": (
                    "You answer chemistry/materials-science questions using only the provided evidence. "
                    "Cite page numbers when evidence has page spans; when evidence is document-level profile data, "
                    "describe it as document-level profile evidence instead of inventing a page citation."
                ),
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

    def extract_document_profile(
        self,
        *,
        document_id: str,
        document_text: str,
        outline: list[DocumentSection],
        figure_captions: list[str],
    ) -> ProfileExtraction | None:
        context = {
            "document_id": document_id,
            "outline_candidates": [_document_section_payload(section) for section in outline[:80]],
            "figure_captions": figure_captions[:40],
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "你是化学与材料文献的结构化文档画像抽取器。"
                    "只依据给定论文文本、大纲候选和图题抽取，不得编造。"
                    "字段缺失时返回 null 或空数组。"
                    "实体应覆盖材料/化合物/方法/性能/应用/表征技术等开放领域词，而不是只限固定词表。"
                    "source_refs 用于说明字段来自哪里，尽量填 field_name、source_type、page_start/page_end 和短 text。"
                    "严格输出 JSON："
                    '{"title":null或字符串,"abstract":null或字符串,"conclusion":null或字符串,'
                    '"topics":["..."],"entities":["..."],"key_findings":["..."],'
                    '"methods":["..."],"properties":["..."],"applications":["..."],'
                    '"limitations":["..."],"source_refs":[{"source_type":"page_text或figure_caption",'
                    '"page_start":1,"page_end":1,"section_title":"...","field_name":"abstract","text":"..."}],'
                    '"reason":"简短说明"}。'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"抽取上下文：\n{json.dumps(context, ensure_ascii=False)}\n\n"
                    f"论文文本（可能被截断）：\n{document_text[:12000]}"
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
            return None
        except Exception as exc:
            if _is_recoverable_llm_error(exc):
                return None
            raise
        return _sanitize_document_profile_extraction(payload, document_id=document_id, outline=outline)

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
            _evidence_candidate(item, text_limit=2400)
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

    def refine_search_queries(
        self,
        question: str,
        *,
        assessment: EvidenceAssessment,
        task_decision: TaskDecision | None,
        route: str,
        scope: str,
        attempted_queries: list[str],
        max_queries: int,
        safety_mode: str,
    ) -> dict[str, object] | None:
        context = {
            "route": route,
            "scope": scope,
            "task_type": task_decision.task_type if task_decision else "unknown",
            "risk_level": task_decision.risk_level if task_decision else "unknown",
            "missing_aspects": assessment.missing_aspects,
            "conflicts": assessment.conflicts,
            "next_search_goal": assessment.next_search_goal,
            "attempted_queries": attempted_queries[-8:],
            "max_queries": max(1, min(max_queries, 3)),
            "safety_mode": safety_mode,
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "你是化学与材料文献 RAG 系统的补充检索 query 生成器。"
                    "只生成下一轮检索 query，不回答问题，不扩展用户没有要求的材料、文档范围或实验目标。"
                    "query 必须服务于补足 missing_aspects、conflicts 或 next_search_goal。"
                    "如果 safety_mode 是 conservative，禁止生成精确剂量、温度步骤、放大操作或可执行实验流程 query，"
                    "只能检索文献中已报告的限制、风险、概述性条件和证据充分性。"
                    "严格输出 JSON："
                    '{"queries":["..."],"search_goal":"...","reason":"...","safety_notes":["..."]}。'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"用户问题：\n{question}\n\n"
                    f"补充检索上下文：\n{json.dumps(context, ensure_ascii=False)}"
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
            return None
        except Exception as exc:
            if _is_recoverable_llm_error(exc):
                return None
            raise
        return payload if isinstance(payload, dict) else None

    def verify_answer_with_evidence(
        self,
        question: str,
        answer: str,
        evidence: list[Evidence],
        *,
        limitations: list[str],
        task_decision: TaskDecision | None,
        evidence_assessment: EvidenceAssessment | None,
        trigger_reasons: list[str],
    ) -> AnswerVerification:
        candidates = [
            _evidence_candidate(item, text_limit=1800)
            for item in evidence
        ]
        context = {
            "task_type": task_decision.task_type if task_decision else "unknown",
            "risk_level": task_decision.risk_level if task_decision else "unknown",
            "limitations": limitations,
            "trigger_reasons": trigger_reasons,
            "evidence_assessment": {
                "sufficient": evidence_assessment.sufficient if evidence_assessment else None,
                "covered_aspects": evidence_assessment.covered_aspects if evidence_assessment else [],
                "missing_aspects": evidence_assessment.missing_aspects if evidence_assessment else [],
                "conflicts": evidence_assessment.conflicts if evidence_assessment else [],
                "confidence": evidence_assessment.confidence if evidence_assessment else None,
            },
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "你是化学与材料文献问答系统的答案核验器。"
                    "只检查最终答案是否被给定证据直接支撑，不要重新回答问题。"
                    "找出未支撑断言、遗漏的资料限制、引用或证据对应错误、以及超过证据强度的表述。"
                    "证据强度需要按 evidence_type/source_scope 区分：chunk/page_span 是原文页级证据；"
                    "document_profile/document_level 是结构化画像或字段级画像，适合概括、定位和筛选，"
                    "但若答案包含具体数值、实验条件、机制解释或强结论，应检查是否需要原文 chunk 支撑；"
                    "figure 证据主要来自图题或图片索引，解释图中机制/结论时需检查是否有 caption 或原文支持。"
                    "高风险问题必须更保守：不能允许可执行实验步骤、放大建议或安全结论超出证据。"
                    "如果需要修改，给出 revision_instruction；不要输出新答案。"
                    "严格输出 JSON："
                    '{"passed":true或false,"unsupported_claims":["..."],'
                    '"missing_disclosures":["..."],"citation_errors":["..."],'
                    '"overstatements":["..."],"revision_instruction":"..."或null,"reason":"简短理由"}。'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"用户问题：\n{question}\n\n"
                    f"最终答案：\n{answer}\n\n"
                    f"核验上下文：\n{json.dumps(context, ensure_ascii=False)}\n\n"
                    f"已验证证据：\n{json.dumps(candidates, ensure_ascii=False)}"
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
            return _fallback_answer_verification(trigger_reasons, "答案核验响应超时或格式无效，已要求保守保留限制说明。")
        except Exception as exc:
            if _is_recoverable_llm_error(exc):
                return _fallback_answer_verification(trigger_reasons, "答案核验服务暂时不可用，已要求保守保留限制说明。")
            raise
        return _sanitize_answer_verification(payload, trigger_reasons)

    def revise_answer_with_evidence(
        self,
        question: str,
        answer: str,
        evidence: list[Evidence],
        *,
        limitations: list[str],
        revision_instruction: str,
    ) -> str:
        context = "\n\n".join(_format_evidence_for_prompt(idx, item, text_limit=1800) for idx, item in enumerate(evidence))
        messages = [
            {
                "role": "system",
                "content": (
                    "你是化学与材料文献问答系统的答案修订器。"
                    "只能依据给定证据和修订要求改写答案，不能新增证据、不能扩大用户问题范围。"
                    "必须保留必要的资料限制；高风险问题应保守表达。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"用户问题：\n{question}\n\n"
                    f"原答案：\n{answer}\n\n"
                    f"修订要求：\n{revision_instruction}\n\n"
                    f"必须披露的资料限制：\n{json.dumps(limitations, ensure_ascii=False)}\n\n"
                    f"已验证证据：\n{context}"
                ),
            },
        ]
        future = self.executor.submit(self.client.chat.completions.create, model=self.model, messages=messages)
        try:
            response = future.result(timeout=self.timeout_seconds)
            revised = response.choices[0].message.content or ""
        except Exception as exc:
            if _is_recoverable_llm_error(exc):
                return super().revise_answer_with_evidence(
                    question,
                    answer,
                    evidence,
                    limitations=limitations,
                    revision_instruction=revision_instruction,
                )
            raise
        return revised.strip() or answer

    def select_images_for_answer(
        self,
        question: str,
        answer: str,
        evidence: list[Evidence],
        max_images: int = 3,
    ) -> ImageSelectionResult:
        candidate_count = _image_candidate_count(evidence)
        if not getattr(self, "image_selection_enabled", True):
            selected = _fallback_selected_images(question, answer, evidence, max_images)
            return _image_selection_result(
                selected,
                strategy="fallback",
                used_llm=False,
                fallback_used=True,
                candidate_count=candidate_count,
                reason="图片选择模型未启用，使用本地启发式选择。",
            )

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
            return ImageSelectionResult(
                images={},
                strategy="none",
                candidate_count=0,
                selected_count=0,
                reason="没有可选择的候选图片。",
            )

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
                selected = _fallback_selected_images(question, answer, evidence, max_images)
                return _image_selection_result(
                    selected,
                    strategy="fallback",
                    used_llm=False,
                    fallback_used=True,
                    candidate_count=candidate_count,
                    reason="图片选择模型连接不可用，使用本地启发式选择。",
                )
            raise
        try:
            payload = json.loads(response.choices[0].message.content or "{}")
        except json.JSONDecodeError:
            selected = _fallback_selected_images(question, answer, evidence, max_images)
            return _image_selection_result(
                selected,
                strategy="fallback_after_llm",
                used_llm=True,
                fallback_used=True,
                candidate_count=candidate_count,
                reason="图片选择模型返回非 JSON，使用本地启发式选择。",
            )
        selected = _sanitize_selected_images(payload.get("images", []), by_chunk, max_images)
        if selected:
            return _image_selection_result(
                selected,
                strategy="provider_llm",
                used_llm=True,
                fallback_used=False,
                candidate_count=candidate_count,
                reason="图片选择模型从候选图片中选择了相关图片。",
            )
        selected = _fallback_selected_images(question, answer, evidence, max_images)
        return _image_selection_result(
            selected,
            strategy="fallback_after_llm",
            used_llm=True,
            fallback_used=True,
            candidate_count=candidate_count,
            reason="图片选择模型未返回可用图片，使用本地启发式选择。",
        )


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

    def extract_document_profile(
        self,
        *,
        document_id: str,
        document_text: str,
        outline: list[DocumentSection],
        figure_captions: list[str],
    ) -> ProfileExtraction | None:
        first_line = next((line.strip() for line in document_text.splitlines() if line.strip()), None)
        return ProfileExtraction(
            document_id=document_id,
            title=first_line[:300] if first_line else None,
            topics=["fake_llm_profile"],
            entities=[],
            figure_summary="\n".join(figure_captions[:3]) or None,
            outline=outline,
            source_refs=[
                SourceRef(
                    source_type="generated_profile",
                    field_name="title",
                    text=(first_line or "")[:300],
                )
            ] if first_line else [],
            extraction_method="llm",
        )

    def answer_with_evidence(self, question: str, evidence: list[Evidence]) -> str:
        if not evidence:
            return f"I could not find enough evidence to answer: {question}"
        locations = ", ".join(_evidence_location(item) for item in evidence)
        snippets = " | ".join(re.sub(r"\s+", " ", item.text).strip()[:160] for item in evidence[:3])
        return f"Based on the retrieved evidence ({locations}), this is a grounded draft answer to: {question}. Evidence: {snippets}"

    def select_images_for_answer(
        self,
        question: str,
        answer: str,
        evidence: list[Evidence],
        max_images: int = 3,
    ) -> ImageSelectionResult:
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
        return _image_selection_result(
            selected,
            strategy="fake_provider",
            used_llm=False,
            fallback_used=False,
            candidate_count=_image_candidate_count(evidence),
            reason="测试 provider 从候选证据图片中确定性选择。",
        )

    def verify_answer_with_evidence(
        self,
        question: str,
        answer: str,
        evidence: list[Evidence],
        *,
        limitations: list[str],
        task_decision: TaskDecision | None,
        evidence_assessment: EvidenceAssessment | None,
        trigger_reasons: list[str],
    ) -> AnswerVerification:
        unsupported = ["答案包含测试用未支撑断言。"] if "unsupported" in answer.lower() else []
        missing = []
        if limitations and not any(marker in answer for marker in ["资料限制", "不足", "缺口", "无法确认"]):
            missing = limitations
        passed = not unsupported and not missing
        return AnswerVerification(
            passed=passed,
            triggered=bool(trigger_reasons),
            trigger_reasons=trigger_reasons,
            unsupported_claims=unsupported,
            missing_disclosures=missing,
            revision_instruction="删除未支撑断言并补充资料限制。" if not passed else None,
            reason="测试 provider 完成答案核验。",
        )


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

    def extract_document_profile(
        self,
        *,
        document_id: str,
        document_text: str,
        outline: list[DocumentSection],
        figure_captions: list[str],
    ) -> ProfileExtraction | None:
        return self.answer_provider.extract_document_profile(
            document_id=document_id,
            document_text=document_text,
            outline=outline,
            figure_captions=figure_captions,
        )

    def plan_route(
        self,
        question: str,
        *,
        conversation_context: str | None,
        scope: str,
        document_ids: list[str] | None,
        basic_facts: list[str] | None = None,
        outline_context: list[Evidence] | None = None,
    ) -> RoutePlan | None:
        return self.answer_provider.plan_route(
            question,
            conversation_context=conversation_context,
            scope=scope,
            document_ids=document_ids,
            basic_facts=basic_facts,
            outline_context=outline_context,
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

    def refine_search_queries(
        self,
        question: str,
        *,
        assessment: EvidenceAssessment,
        task_decision: TaskDecision | None,
        route: str,
        scope: str,
        attempted_queries: list[str],
        max_queries: int,
        safety_mode: str,
    ) -> dict[str, object] | None:
        return self.answer_provider.refine_search_queries(
            question,
            assessment=assessment,
            task_decision=task_decision,
            route=route,
            scope=scope,
            attempted_queries=attempted_queries,
            max_queries=max_queries,
            safety_mode=safety_mode,
        )

    def verify_answer_with_evidence(
        self,
        question: str,
        answer: str,
        evidence: list[Evidence],
        *,
        limitations: list[str],
        task_decision: TaskDecision | None,
        evidence_assessment: EvidenceAssessment | None,
        trigger_reasons: list[str],
    ) -> AnswerVerification:
        return self.assessment_provider.verify_answer_with_evidence(
            question,
            answer,
            evidence,
            limitations=limitations,
            task_decision=task_decision,
            evidence_assessment=evidence_assessment,
            trigger_reasons=trigger_reasons,
        )

    def revise_answer_with_evidence(
        self,
        question: str,
        answer: str,
        evidence: list[Evidence],
        *,
        limitations: list[str],
        revision_instruction: str,
    ) -> str:
        return self.answer_provider.revise_answer_with_evidence(
            question,
            answer,
            evidence,
            limitations=limitations,
            revision_instruction=revision_instruction,
        )

    def select_images_for_answer(
        self,
        question: str,
        answer: str,
        evidence: list[Evidence],
        max_images: int = 3,
    ) -> ImageSelectionResult:
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


def _sanitize_document_profile_extraction(
    payload: object,
    *,
    document_id: str,
    outline: list[DocumentSection],
) -> ProfileExtraction | None:
    if not isinstance(payload, dict):
        return None
    return ProfileExtraction(
        document_id=document_id,
        title=_sanitize_optional_text(payload.get("title"), max_chars=300),
        abstract=_sanitize_optional_text(payload.get("abstract"), max_chars=1600),
        conclusion=_sanitize_optional_text(payload.get("conclusion"), max_chars=1600),
        topics=_sanitize_string_list(payload.get("topics"), limit=20),
        entities=_sanitize_string_list(payload.get("entities"), limit=40),
        key_findings=_sanitize_string_list(payload.get("key_findings"), limit=12),
        methods=_sanitize_string_list(payload.get("methods"), limit=12),
        properties=_sanitize_string_list(payload.get("properties"), limit=12),
        applications=_sanitize_string_list(payload.get("applications"), limit=12),
        limitations=_sanitize_string_list(payload.get("limitations"), limit=12),
        outline=outline,
        source_refs=_sanitize_source_refs(payload.get("source_refs")),
        extraction_method="llm",
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


def _sanitize_answer_verification(payload: object, trigger_reasons: list[str]) -> AnswerVerification:
    if not isinstance(payload, dict):
        return _fallback_answer_verification(trigger_reasons, "答案核验返回结构无效，已要求保守保留限制说明。")
    unsupported_claims = _sanitize_string_list(payload.get("unsupported_claims"))
    missing_disclosures = _sanitize_string_list(payload.get("missing_disclosures"))
    citation_errors = _sanitize_string_list(payload.get("citation_errors"))
    overstatements = _sanitize_string_list(payload.get("overstatements"))
    revision_instruction = str(payload.get("revision_instruction") or "").strip()[:800] or None
    failed_items = unsupported_claims or missing_disclosures or citation_errors or overstatements
    passed = bool(payload.get("passed", not failed_items)) and not failed_items
    if not passed and revision_instruction is None:
        revision_instruction = "删除或改写未被证据直接支撑的断言，并补充遗漏的资料限制。"
    return AnswerVerification(
        passed=passed,
        triggered=bool(trigger_reasons),
        trigger_reasons=list(trigger_reasons),
        unsupported_claims=unsupported_claims,
        missing_disclosures=missing_disclosures,
        citation_errors=citation_errors,
        overstatements=overstatements,
        revision_instruction=revision_instruction,
        used_llm=True,
        reason=str(payload.get("reason") or ("答案核验通过。" if passed else "答案核验发现需要修订的问题。"))[:500],
    )


def _fallback_answer_verification(trigger_reasons: list[str], reason: str) -> AnswerVerification:
    return AnswerVerification(
        passed=False,
        triggered=bool(trigger_reasons),
        trigger_reasons=list(trigger_reasons),
        missing_disclosures=["答案级核验未完成，输出必须保留资料限制并避免扩大结论。"],
        revision_instruction="保留资料限制，删除任何没有直接证据支撑的强结论。",
        used_llm=True,
        fallback_used=True,
        reason=reason,
    )


def _sanitize_string_list(value: object, limit: int = 12) -> list[str]:
    if not isinstance(value, list):
        return []
    values = [str(item).strip()[:240] for item in value if isinstance(item, str) and item.strip()]
    return list(dict.fromkeys(values))[:limit]


def _sanitize_optional_text(value: object, max_chars: int) -> str | None:
    if not isinstance(value, str):
        return None
    text_value = re.sub(r"\s+", " ", value).strip()
    return text_value[:max_chars] if text_value else None


def _sanitize_source_refs(value: object) -> list[SourceRef]:
    if not isinstance(value, list):
        return []
    refs: list[SourceRef] = []
    for item in value[:40]:
        if not isinstance(item, dict):
            continue
        refs.append(
            SourceRef(
                source_type=str(item.get("source_type") or "generated_profile")[:80],
                page_start=_optional_int(item.get("page_start")),
                page_end=_optional_int(item.get("page_end")),
                section_title=_sanitize_optional_text(item.get("section_title"), max_chars=160),
                field_name=_sanitize_optional_text(item.get("field_name"), max_chars=80),
                text=_sanitize_optional_text(item.get("text"), max_chars=500),
            )
        )
    return refs


def _optional_int(value: object) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


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


def _image_candidate_count(evidence: list[Evidence]) -> int:
    return sum(len(item.images) for item in evidence)


def _selected_image_count(selected: dict[str, list[EvidenceImage]]) -> int:
    return sum(len(images) for images in selected.values())


def _image_selection_result(
    selected: dict[str, list[EvidenceImage]],
    *,
    strategy: str,
    used_llm: bool,
    fallback_used: bool,
    candidate_count: int,
    reason: str,
) -> ImageSelectionResult:
    return ImageSelectionResult(
        images=selected,
        strategy=strategy,
        used_llm=used_llm,
        fallback_used=fallback_used,
        candidate_count=candidate_count,
        selected_count=_selected_image_count(selected),
        reason=reason,
    )


def _meaningful_terms(text: str) -> set[str]:
    terms = {term.lower() for term in re.findall(r"[A-Za-z][A-Za-z0-9-]{3,}", text)}
    terms.update(re.findall(r"[\u4e00-\u9fff]{2,}", text))
    return terms


def _document_section_payload(section: DocumentSection) -> dict[str, object]:
    return {
        "title": section.title,
        "normalized_type": section.normalized_type,
        "page_start": section.page_start,
        "page_end": section.page_end,
        "summary": section.summary,
        "keywords": section.keywords,
        "linked_figures": section.linked_figures,
    }


def _evidence_location(item: Evidence) -> str:
    if item.source_scope == "document_level" or item.evidence_type == "document_profile":
        return "document-level profile"
    return f"page {item.page_start}-{item.page_end}"


def _source_ref_payload(item: Evidence) -> list[dict[str, object]]:
    return [
        {
            "source_type": ref.source_type,
            "page_start": ref.page_start,
            "page_end": ref.page_end,
            "section_title": ref.section_title,
            "field_name": ref.field_name,
            "text": ref.text,
        }
        for ref in item.source_refs
    ]


def _evidence_candidate(item: Evidence, text_limit: int) -> dict[str, object]:
    return {
        "chunk_id": item.chunk_id,
        "document_id": item.document_id,
        "evidence_type": item.evidence_type,
        "source_scope": item.source_scope,
        "location": _evidence_location(item),
        "pages": None if item.source_scope == "document_level" else f"{item.page_start}-{item.page_end}",
        "section_title": item.section_title,
        "source_refs": _source_ref_payload(item),
        "text": item.text[:text_limit],
    }


def _format_evidence_for_prompt(index: int, item: Evidence, text_limit: int | None = None) -> str:
    text = item.text if text_limit is None else item.text[:text_limit]
    return (
        f"[{index + 1}] {item.evidence_type}, {_evidence_location(item)}, {item.section_title}\n"
        f"{text}"
    )


def _fallback_answer_with_evidence(question: str, evidence: list[Evidence]) -> str:
    if not evidence:
        return f"未检索到足够证据回答：{question}"
    lines = [
        "LLM 服务暂时不可用，先基于已验证证据给出快速摘要：",
    ]
    for index, item in enumerate(evidence[:3], start=1):
        snippet = re.sub(r"\s+", " ", item.text).strip()[:260]
        lines.append(f"[{index}] {_evidence_location(item)}，{item.section_title}：{snippet}")
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
