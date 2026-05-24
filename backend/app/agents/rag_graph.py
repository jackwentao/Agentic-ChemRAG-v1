from time import perf_counter

from langgraph.graph import END, StateGraph

from backend.app.agents.state import RagGraphState
from backend.app.domain.documents import Evidence, EvidenceImage
from backend.app.observability.diagnostics import (
    build_llm_diagnostics,
    build_route_diagnostics,
    set_total_latency,
    update_stage_proxy_hit,
)
from backend.app.providers.llm import LlmProvider
from backend.app.routing.query_router import QueryRouter, RouteContext
from backend.app.services.rag import RetrievalService


class MultiAgentRagGraph:
    def __init__(
        self,
        retrieval_service: RetrievalService,
        llm_provider: LlmProvider,
        retrieval_top_k: int = 5,
        figure_top_k: int = 8,
        profile_top_k: int = 5,
        query_router: QueryRouter | None = None,
    ) -> None:
        self.retrieval_service = retrieval_service
        self.llm_provider = llm_provider
        self.retrieval_top_k = retrieval_top_k
        self.figure_top_k = figure_top_k
        self.profile_top_k = profile_top_k
        self.query_router = query_router or QueryRouter(getattr(retrieval_service, "embedding_provider", None))
        self.graph = self._build().compile()

    def invoke(
        self,
        question: str,
        document_ids: list[str] | None = None,
        scope: str | None = None,
    ) -> RagGraphState:
        started_at = perf_counter()
        state = self.graph.invoke(
            {
                "question": question,
                "document_ids": document_ids,
                "scope": scope,
                "warnings": [],
            }
        )
        total_latency_ms = (perf_counter() - started_at) * 1000
        diagnostics = set_total_latency(state.get("diagnostics"), total_latency_ms)
        return {
            **state,
            "total_latency_ms": total_latency_ms,
            "diagnostics": diagnostics,
        }

    def _build(self) -> StateGraph:
        graph = StateGraph(RagGraphState)
        graph.add_node("planner", self._planner)
        graph.add_node("retriever", self._retriever)
        graph.add_node("verifier", self._verifier)
        graph.add_node("answerer", self._answerer)
        graph.add_node("image_selector", self._image_selector)
        graph.add_edge("planner", "retriever")
        graph.add_edge("retriever", "verifier")
        graph.add_edge("verifier", "answerer")
        graph.add_edge("answerer", "image_selector")
        graph.add_edge("image_selector", END)
        graph.set_entry_point("planner")
        return graph

    def _planner(self, state: RagGraphState) -> RagGraphState:
        started_at = perf_counter()
        context = RouteContext(
            scope=state.get("scope") or ("current_document" if state.get("document_ids") else "all_documents"),
            document_ids=state.get("document_ids"),
        )
        decision = self.query_router.route(state["question"], context)
        route_latency_ms = (perf_counter() - started_at) * 1000
        diagnostics = {
            **state.get("diagnostics", {}),
            "route": build_route_diagnostics(
                latency_ms=route_latency_ms,
                selected_route=decision.route,
                confidence=decision.confidence,
                needs_llm_router=decision.needs_llm_router,
                is_multi_step=decision.is_multi_step,
                scores=decision.scores,
                lexical_scores=decision.lexical_scores,
                semantic_scores=decision.semantic_scores,
                rationale=decision.rationale,
            ),
        }
        return {
            **state,
            "route": decision.route,
            "route_scores": decision.scores,
            "route_lexical_scores": decision.lexical_scores,
            "route_semantic_scores": decision.semantic_scores,
            "route_confidence": decision.confidence,
            "route_rationale": decision.rationale,
            "needs_llm_router": decision.needs_llm_router,
            "is_multi_step": decision.is_multi_step,
            "route_latency_ms": route_latency_ms,
            "plan": _plan_for_route(decision.route),
            "diagnostics": diagnostics,
        }

    def _retriever(self, state: RagGraphState) -> RagGraphState:
        started_at = perf_counter()
        route = state.get("route", "chunk_search")
        if route == "figure_index":
            evidence = self.retrieval_service.retrieve_figures(
                state["question"],
                top_k=self.figure_top_k,
                document_ids=state.get("document_ids"),
            )
        elif route == "document_profile":
            evidence = self.retrieval_service.retrieve_document_profiles(
                state["question"],
                top_k=self.profile_top_k,
                document_ids=state.get("document_ids"),
            )
        else:
            evidence = self.retrieval_service.retrieve(
                state["question"],
                top_k=self.retrieval_top_k,
                document_ids=state.get("document_ids"),
            )
        return {
            **state,
            "evidence": evidence,
            "retrieval_latency_ms": (perf_counter() - started_at) * 1000,
            "rerank_latency_ms": getattr(self.retrieval_service, "last_rerank_latency_ms", 0.0),
            "rerank_enabled": getattr(self.retrieval_service, "last_rerank_enabled", False),
            "diagnostics": {
                **state.get("diagnostics", {}),
                "recall": getattr(self.retrieval_service, "last_recall_diagnostics", {}),
                "rerank": getattr(self.retrieval_service, "last_rerank_diagnostics", {}),
            },
        }

    def _verifier(self, state: RagGraphState) -> RagGraphState:
        evidence = state.get("evidence", [])
        verified = [item for item in evidence if item.text.strip()]
        warnings = state.get("warnings", [])
        if not verified:
            warnings = [*warnings, "No evidence was retrieved."]
        diagnostics = update_stage_proxy_hit(state.get("diagnostics"), "route", bool(verified))
        diagnostics = update_stage_proxy_hit(diagnostics, "recall", bool(verified))
        return {**state, "verified_evidence": verified, "warnings": warnings, "diagnostics": diagnostics}

    def _answerer(self, state: RagGraphState) -> RagGraphState:
        started_at = perf_counter()
        used_llm = state.get("route") not in {"figure_index", "document_profile"}
        if state.get("route") == "figure_index":
            answer = _answer_from_figure_index(state.get("verified_evidence", []))
            latency_ms = (perf_counter() - started_at) * 1000
            diagnostics = {
                **state.get("diagnostics", {}),
                "llm": build_llm_diagnostics(
                    latency_ms=latency_ms,
                    used_llm=used_llm,
                    answer_chars=len(answer),
                    evidence_count=len(state.get("verified_evidence", [])),
                    citation_count=len(state.get("verified_evidence", [])),
                ),
            }
            return {**state, "answer": answer, "answer_latency_ms": latency_ms, "diagnostics": diagnostics}
        if state.get("route") == "document_profile":
            answer = _answer_from_document_profiles(state["question"], state.get("verified_evidence", []))
            latency_ms = (perf_counter() - started_at) * 1000
            diagnostics = {
                **state.get("diagnostics", {}),
                "llm": build_llm_diagnostics(
                    latency_ms=latency_ms,
                    used_llm=used_llm,
                    answer_chars=len(answer),
                    evidence_count=len(state.get("verified_evidence", [])),
                    citation_count=len(state.get("verified_evidence", [])),
                ),
            }
            return {**state, "answer": answer, "answer_latency_ms": latency_ms, "diagnostics": diagnostics}
        answer = self.llm_provider.answer_with_evidence(state["question"], state.get("verified_evidence", []))
        latency_ms = (perf_counter() - started_at) * 1000
        diagnostics = {
            **state.get("diagnostics", {}),
            "llm": build_llm_diagnostics(
                latency_ms=latency_ms,
                used_llm=used_llm,
                answer_chars=len(answer),
                evidence_count=len(state.get("verified_evidence", [])),
                citation_count=len(state.get("verified_evidence", [])),
            ),
        }
        return {**state, "answer": answer, "answer_latency_ms": latency_ms, "diagnostics": diagnostics}

    def _image_selector(self, state: RagGraphState) -> RagGraphState:
        if state.get("route") == "figure_index":
            selected = _select_first_figure_images(state.get("verified_evidence", []), max_images=3)
            diagnostics = _with_selected_image_count(state.get("diagnostics"), selected)
            return {**state, "selected_images": selected, "diagnostics": diagnostics}
        selected = self.llm_provider.select_images_for_answer(
            state["question"],
            state.get("answer", ""),
            state.get("verified_evidence", []),
            max_images=3,
        )
        diagnostics = _with_selected_image_count(state.get("diagnostics"), selected)
        return {**state, "selected_images": selected, "diagnostics": diagnostics}


def _plan_for_route(route: str) -> str:
    if route == "figure_index":
        return "search_figure_index_then_answer_with_figure_citations"
    if route == "document_profile":
        return "search_document_profiles_then_answer_with_document_citations"
    return "retrieve_relevant_chunks_then_answer_with_citations"


def _answer_from_figure_index(evidence: list[Evidence]) -> str:
    if not evidence:
        return "没有在 figure 索引中找到足够相关的图片或图题。"
    lines = ["根据 figure 索引，最相关的图包括："]
    for index, item in enumerate(evidence[:8], start=1):
        image = item.images[0] if item.images else None
        label = image.label if image and image.label else ""
        caption = image.caption if image and image.caption else item.text
        lines.append(f"{index}. p. {item.page_start}: {label} {caption}".strip())
    return "\n".join(lines)


def _answer_from_document_profiles(question: str, evidence: list[Evidence]) -> str:
    if not evidence:
        return f"没有检索到足够的文档画像证据来回答：{question}"
    lines = ["根据文档画像，相关论文要点如下："]
    for index, item in enumerate(evidence[:5], start=1):
        title = item.section_title or "Document Profile"
        snippet = " ".join(item.text.split())[:700]
        lines.append(f"{index}. {title} (p. {item.page_start}): {snippet}")
    return "\n".join(lines)


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


def _with_selected_image_count(
    diagnostics: dict[str, object] | None,
    selected_images: dict[str, list[EvidenceImage]],
) -> dict[str, object]:
    payload = dict(diagnostics or {})
    llm = dict(payload.get("llm", {}))
    if not llm:
        return payload
    llm["selected_image_count"] = sum(len(images) for images in selected_images.values())
    payload["llm"] = llm
    return payload
