from dataclasses import replace
import re
from time import perf_counter

from backend.app.domain.documents import Evidence
from backend.app.observability.diagnostics import (
    build_recall_diagnostics,
    build_rerank_diagnostics,
    candidate_char_stats,
    candidate_text,
    score_summary,
    top1_term_overlap,
    top_k_changed,
)
from backend.app.providers.embeddings import EmbeddingProvider
from backend.app.providers.rerankers import RerankCandidate, RerankerProvider


class RetrievalService:
    def __init__(
        self,
        repository,
        embedding_provider: EmbeddingProvider,
        reranker_provider: RerankerProvider | None = None,
        reranker_enabled: bool = True,
        reranker_chunk_candidates: int = 10,
        reranker_figure_candidates: int = 10,
        reranker_min_candidates: int = 10,
    ) -> None:
        self.repository = repository
        self.embedding_provider = embedding_provider
        self.reranker_provider = reranker_provider
        self.reranker_enabled = reranker_enabled and reranker_provider is not None
        self.reranker_chunk_candidates = reranker_chunk_candidates
        self.reranker_figure_candidates = reranker_figure_candidates
        self.reranker_min_candidates = reranker_min_candidates
        self.last_rerank_latency_ms = 0.0
        self.last_rerank_enabled = False
        self.last_recall_diagnostics: dict[str, object] = {}
        self.last_rerank_diagnostics: dict[str, object] = build_rerank_diagnostics(enabled=self.reranker_enabled)

    def retrieve(self, question: str, top_k: int = 5, document_ids: list[str] | None = None):
        self._reset_rerank_state()
        if not hasattr(self.repository, "search"):
            raise TypeError("Repository must implement search.")
        if getattr(self.repository, "expects_text_query", False):
            evidence = self.repository.search(question, self.embedding_provider, top_k)
            top_score, avg_score = score_summary(evidence)
            self.last_recall_diagnostics = build_recall_diagnostics(
                mode="chunk",
                latency_ms=0.0,
                embedding_latency_ms=0.0,
                search_latency_ms=0.0,
                candidate_count=len(evidence),
                returned_count=min(len(evidence), top_k),
                top_score=top_score,
                avg_score=avg_score,
            )
            return evidence
        started_at = perf_counter()
        embedding_started_at = perf_counter()
        query_embedding = self.embedding_provider.embed_query(question)
        embedding_latency_ms = (perf_counter() - embedding_started_at) * 1000
        candidate_k = max(top_k, self.reranker_chunk_candidates) if self.reranker_enabled else top_k
        search_started_at = perf_counter()
        evidence = self.repository.search(query_embedding, candidate_k, document_ids=document_ids)
        search_latency_ms = (perf_counter() - search_started_at) * 1000
        selected = self._rerank_evidence(question, evidence, top_k, mode="chunk") if self.reranker_enabled else evidence[:top_k]
        top_score, avg_score = score_summary(evidence)
        self.last_recall_diagnostics = build_recall_diagnostics(
            mode="chunk",
            latency_ms=(perf_counter() - started_at) * 1000,
            embedding_latency_ms=embedding_latency_ms,
            search_latency_ms=search_latency_ms,
            candidate_count=len(evidence),
            returned_count=len(selected),
            top_score=top_score,
            avg_score=avg_score,
        )
        return selected

    def retrieve_figures(self, question: str, top_k: int = 8, document_ids: list[str] | None = None):
        self._reset_rerank_state()
        if not hasattr(self.repository, "search_figures"):
            return self.retrieve(question, top_k=top_k, document_ids=document_ids)
        started_at = perf_counter()
        embedding_started_at = perf_counter()
        query_embedding = self.embedding_provider.embed_query(question)
        embedding_latency_ms = (perf_counter() - embedding_started_at) * 1000
        candidate_k = max(top_k, self.reranker_figure_candidates) if self.reranker_enabled else top_k
        search_started_at = perf_counter()
        evidence = self.repository.search_figures(query_embedding, candidate_k, document_ids=document_ids)
        search_latency_ms = (perf_counter() - search_started_at) * 1000
        selected = self._rerank_evidence(question, evidence, top_k, mode="figure") if self.reranker_enabled else evidence[:top_k]
        top_score, avg_score = score_summary(evidence)
        self.last_recall_diagnostics = build_recall_diagnostics(
            mode="figure",
            latency_ms=(perf_counter() - started_at) * 1000,
            embedding_latency_ms=embedding_latency_ms,
            search_latency_ms=search_latency_ms,
            candidate_count=len(evidence),
            returned_count=len(selected),
            top_score=top_score,
            avg_score=avg_score,
        )
        return selected

    def retrieve_document_profiles(self, question: str, top_k: int = 5, document_ids: list[str] | None = None):
        self._reset_rerank_state()
        if not hasattr(self.repository, "search_document_profiles"):
            return self.retrieve(question, top_k=top_k, document_ids=document_ids)
        started_at = perf_counter()
        embedding_started_at = perf_counter()
        query_embedding = self.embedding_provider.embed_query(question)
        embedding_latency_ms = (perf_counter() - embedding_started_at) * 1000
        search_started_at = perf_counter()
        evidence = self.repository.search_document_profiles(query_embedding, top_k, document_ids=document_ids)
        search_latency_ms = (perf_counter() - search_started_at) * 1000
        top_score, avg_score = score_summary(evidence)
        self.last_recall_diagnostics = build_recall_diagnostics(
            mode="document_profile",
            latency_ms=(perf_counter() - started_at) * 1000,
            embedding_latency_ms=embedding_latency_ms,
            search_latency_ms=search_latency_ms,
            candidate_count=len(evidence),
            returned_count=len(evidence),
            top_score=top_score,
            avg_score=avg_score,
        )
        return evidence

    def _rerank_evidence(self, question: str, evidence: list[Evidence], top_k: int, mode: str) -> list[Evidence]:
        if not evidence or self.reranker_provider is None:
            return evidence[:top_k]
        if len(evidence) < self.reranker_min_candidates:
            self.last_rerank_diagnostics = build_rerank_diagnostics(
                enabled=False,
                candidate_count=len(evidence),
                returned_count=min(len(evidence), top_k),
                skip_reason="insufficient_candidates",
            )
            return evidence[:top_k]
        started_at = perf_counter()
        candidate_build_started_at = perf_counter()
        candidate_texts = [candidate_text(item, mode) for item in evidence]
        candidates = [
            RerankCandidate(candidate_id=str(index), text=text)
            for index, text in enumerate(candidate_texts)
        ]
        candidate_build_latency_ms = (perf_counter() - candidate_build_started_at) * 1000
        inference_started_at = perf_counter()
        reranked = self.reranker_provider.rerank(question, candidates, top_k=len(candidates))
        inference_latency_ms = (perf_counter() - inference_started_at) * 1000
        by_index = {str(index): item for index, item in enumerate(evidence)}
        postprocess_started_at = perf_counter()
        scored: list[tuple[float, int, Evidence]] = []
        for rank, result in enumerate(reranked):
            item = by_index.get(result.candidate_id)
            if item is None:
                continue
            original_index = int(result.candidate_id)
            adjusted_score = result.score + _metadata_adjustment(question, item, mode) - rank * 0.0001
            scored.append((adjusted_score, original_index, item))
        scored.sort(key=lambda row: (-row[0], row[1]))
        selected: list[Evidence] = []
        seen_images: set[str] = set()
        for score, _index, item in scored:
            if len(selected) >= top_k:
                break
            image_id = item.images[0].image_id if mode == "figure" and item.images else None
            if image_id and image_id in seen_images:
                continue
            if image_id:
                seen_images.add(image_id)
            selected.append(replace(item, score=float(score)))
        self.last_rerank_latency_ms = (perf_counter() - started_at) * 1000
        self.last_rerank_enabled = True
        total_chars, avg_chars = candidate_char_stats(candidate_texts)
        top1_before = top1_term_overlap(question, evidence[:1], mode)
        top1_after = top1_term_overlap(question, selected[:1], mode)
        self.last_rerank_diagnostics = build_rerank_diagnostics(
            enabled=True,
            latency_ms=self.last_rerank_latency_ms,
            candidate_build_latency_ms=candidate_build_latency_ms,
            inference_latency_ms=inference_latency_ms,
            postprocess_latency_ms=(perf_counter() - postprocess_started_at) * 1000,
            candidate_count=len(candidates),
            returned_count=len(selected),
            input_char_count=total_chars,
            avg_candidate_chars=avg_chars,
            changed_top1=bool(selected) and evidence[0].chunk_id != selected[0].chunk_id,
            changed_top_k=top_k_changed(evidence[:top_k], selected),
            top1_term_overlap_before=top1_before,
            top1_term_overlap_after=top1_after,
        )
        return selected

    def _reset_rerank_state(self) -> None:
        self.last_rerank_latency_ms = 0.0
        self.last_rerank_enabled = False
        self.last_recall_diagnostics = {}
        self.last_rerank_diagnostics = build_rerank_diagnostics(enabled=self.reranker_enabled)


def _metadata_adjustment(question: str, item: Evidence, mode: str) -> float:
    if mode != "figure" or not item.images:
        return 0.0
    image = item.images[0]
    label_caption = " ".join(value for value in [image.label, image.caption] if value)
    nearby_text = image.nearby_text or ""
    adjustment = 0.0
    if label_caption:
        adjustment += 0.6
    else:
        adjustment -= 0.8
    if not label_caption and item.page_start <= 1 and nearby_text:
        adjustment -= 1.2
    if _has_alias_match(question, label_caption):
        adjustment += 3.0
    if _term_overlap(question, label_caption) > 0:
        adjustment += min(1.5, 0.4 * _term_overlap(question, label_caption))
    return adjustment


def _has_alias_match(query: str, text: str) -> bool:
    query_normalized = _normalize_aliases(query)
    text_normalized = _normalize_aliases(text)
    return "nonlinear" in query_normalized and "nonlinear" in text_normalized


def _term_overlap(query: str, text: str) -> int:
    return len(_terms(query) & _terms(text))


def _terms(text: str) -> set[str]:
    normalized = _normalize_aliases(text)
    terms = {term.lower() for term in re.findall(r"[A-Za-z][A-Za-z0-9-]{3,}", normalized)}
    terms.update(re.findall(r"[\u4e00-\u9fff]{2,}", normalized))
    return terms


def _normalize_aliases(text: str) -> str:
    return text.lower().replace("nlo", "nonlinear optical")
