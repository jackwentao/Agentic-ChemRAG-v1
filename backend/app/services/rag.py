from dataclasses import replace
import logging
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


logger = logging.getLogger(__name__)


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
        exact_chunk_limit: int = 2000,
        exact_timeout_ms: int = 300,
    ) -> None:
        self.repository = repository
        self.embedding_provider = embedding_provider
        self.reranker_provider = reranker_provider
        self.reranker_enabled = reranker_enabled and reranker_provider is not None
        self.reranker_chunk_candidates = reranker_chunk_candidates
        self.reranker_figure_candidates = reranker_figure_candidates
        self.reranker_min_candidates = reranker_min_candidates
        self.exact_chunk_limit = exact_chunk_limit
        self.exact_timeout_ms = exact_timeout_ms
        self.last_rerank_latency_ms = 0.0
        self.last_rerank_enabled = False
        self.last_recall_diagnostics: dict[str, object] = {}
        self.last_rerank_diagnostics: dict[str, object] = build_rerank_diagnostics(enabled=self.reranker_enabled)

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        document_ids: list[str] | None = None,
        owner_id: str | None = None,
        exclude_keys: set[tuple[str, str]] | None = None,
    ):
        # 文本块召回主入口：
        # 1. 先做 query embedding
        # 2. 从 repository 取候选
        # 3. 视配置决定是否 rerank
        # 4. 顺手记录一份可观测性诊断信息，供 API 层透出
        self._reset_rerank_state()
        if not hasattr(self.repository, "search"):
            raise TypeError("Repository must implement search.")
        if getattr(self.repository, "expects_text_query", False):
            # 某些测试仓储或规则仓储不吃向量，而是直接消费原始文本查询。
            raw_evidence = self.repository.search(question, self.embedding_provider, top_k)
            evidence, excluded_count = _exclude_existing(raw_evidence, exclude_keys)
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
            self.last_rerank_diagnostics = build_rerank_diagnostics(
                enabled=False,
                candidate_count=len(evidence),
                returned_count=len(evidence),
                new_candidate_count=len(evidence),
                excluded_existing_count=excluded_count,
                incremental_only=bool(exclude_keys),
                skip_reason="no_reranker",
            )
            return evidence
        started_at = perf_counter()
        embedding_started_at = perf_counter()
        query_embedding = self.embedding_provider.embed_query(question)
        embedding_latency_ms = (perf_counter() - embedding_started_at) * 1000
        candidate_k = max(top_k, self.reranker_chunk_candidates) if self.reranker_enabled else top_k
        candidate_chunk_count = self._count_searchable_chunks(document_ids=document_ids, owner_id=owner_id)
        search_strategy = self._chunk_search_strategy(candidate_chunk_count)
        search_started_at = perf_counter()
        raw_evidence = _call_search(
            self.repository.search,
            query_embedding,
            candidate_k,
            document_ids=document_ids,
            owner_id=owner_id,
            search_strategy=search_strategy,
            exact_timeout_ms=self.exact_timeout_ms if search_strategy == "exact_then_hnsw" else None,
        )
        search_latency_ms = (perf_counter() - search_started_at) * 1000
        evidence, excluded_count = _exclude_existing(raw_evidence, exclude_keys)
        selected = (
            self._rerank_evidence(question, evidence, top_k, mode="chunk", excluded_count=excluded_count, incremental_only=bool(exclude_keys))
            if self.reranker_enabled
            else evidence[:top_k]
        )
        if not self.reranker_enabled:
            self.last_rerank_diagnostics = build_rerank_diagnostics(
                enabled=False,
                candidate_count=len(evidence),
                returned_count=len(selected),
                new_candidate_count=len(evidence),
                excluded_existing_count=excluded_count,
                incremental_only=bool(exclude_keys),
                skip_reason="disabled",
            )
        top_score, avg_score = score_summary(raw_evidence)
        self.last_recall_diagnostics = build_recall_diagnostics(
            mode="chunk",
            latency_ms=(perf_counter() - started_at) * 1000,
            embedding_latency_ms=embedding_latency_ms,
            search_latency_ms=search_latency_ms,
            candidate_count=len(raw_evidence),
            returned_count=len(selected),
            top_score=top_score,
            avg_score=avg_score,
        )
        self.last_recall_diagnostics["search_strategy"] = search_strategy
        self.last_recall_diagnostics["candidate_chunk_count"] = candidate_chunk_count
        return selected

    def retrieve_figures(
        self,
        question: str,
        top_k: int = 8,
        document_ids: list[str] | None = None,
        owner_id: str | None = None,
        exclude_keys: set[tuple[str, str]] | None = None,
    ):
        # figure 检索与 chunk 检索共用同一套服务接口，
        # 但在 repository 层拆出 search_figures，便于后续替换为专门的图像索引。
        self._reset_rerank_state()
        if not hasattr(self.repository, "search_figures"):
            return self.retrieve(question, top_k=top_k, document_ids=document_ids, owner_id=owner_id, exclude_keys=exclude_keys)
        started_at = perf_counter()
        embedding_started_at = perf_counter()
        query_embedding = self.embedding_provider.embed_query(question)
        embedding_latency_ms = (perf_counter() - embedding_started_at) * 1000
        candidate_k = max(top_k, self.reranker_figure_candidates) if self.reranker_enabled else top_k
        search_started_at = perf_counter()
        raw_evidence = _call_search(
            self.repository.search_figures,
            query_embedding,
            candidate_k,
            document_ids=document_ids,
            owner_id=owner_id,
        )
        search_latency_ms = (perf_counter() - search_started_at) * 1000
        evidence, excluded_count = _exclude_existing(raw_evidence, exclude_keys)
        selected = (
            self._rerank_evidence(question, evidence, top_k, mode="figure", excluded_count=excluded_count, incremental_only=bool(exclude_keys))
            if self.reranker_enabled
            else evidence[:top_k]
        )
        if not self.reranker_enabled:
            self.last_rerank_diagnostics = build_rerank_diagnostics(
                enabled=False,
                candidate_count=len(evidence),
                returned_count=len(selected),
                new_candidate_count=len(evidence),
                excluded_existing_count=excluded_count,
                incremental_only=bool(exclude_keys),
                skip_reason="disabled",
            )
        top_score, avg_score = score_summary(raw_evidence)
        self.last_recall_diagnostics = build_recall_diagnostics(
            mode="figure",
            latency_ms=(perf_counter() - started_at) * 1000,
            embedding_latency_ms=embedding_latency_ms,
            search_latency_ms=search_latency_ms,
            candidate_count=len(raw_evidence),
            returned_count=len(selected),
            top_score=top_score,
            avg_score=avg_score,
        )
        return selected

    def retrieve_document_profiles(
        self,
        question: str,
        top_k: int = 5,
        document_ids: list[str] | None = None,
        owner_id: str | None = None,
        exclude_keys: set[tuple[str, str]] | None = None,
    ):
        # 文档画像检索更偏“论文级概览”，因此这里暂不做 rerank，
        # 避免把原本就较短的画像描述再做额外改写和排序干预。
        self._reset_rerank_state()
        if not hasattr(self.repository, "search_document_profiles"):
            return self.retrieve(question, top_k=top_k, document_ids=document_ids, owner_id=owner_id, exclude_keys=exclude_keys)
        started_at = perf_counter()
        embedding_started_at = perf_counter()
        query_embedding = self.embedding_provider.embed_query(question)
        embedding_latency_ms = (perf_counter() - embedding_started_at) * 1000
        search_started_at = perf_counter()
        raw_evidence = _call_search(
            self.repository.search_document_profiles,
            query_embedding,
            top_k,
            document_ids=document_ids,
            owner_id=owner_id,
        )
        search_latency_ms = (perf_counter() - search_started_at) * 1000
        evidence, excluded_count = _exclude_existing(raw_evidence, exclude_keys)
        self.last_rerank_diagnostics = build_rerank_diagnostics(
            enabled=self.reranker_enabled,
            candidate_count=len(evidence),
            returned_count=0,
            new_candidate_count=len(evidence),
            excluded_existing_count=excluded_count,
            incremental_only=bool(exclude_keys),
            skip_reason="document_profile_not_reranked",
        )
        top_score, avg_score = score_summary(raw_evidence)
        self.last_recall_diagnostics = build_recall_diagnostics(
            mode="document_profile",
            latency_ms=(perf_counter() - started_at) * 1000,
            embedding_latency_ms=embedding_latency_ms,
            search_latency_ms=search_latency_ms,
            candidate_count=len(raw_evidence),
            returned_count=len(evidence),
            top_score=top_score,
            avg_score=avg_score,
        )
        return evidence

    def _rerank_evidence(
        self,
        question: str,
        evidence: list[Evidence],
        top_k: int,
        mode: str,
        excluded_count: int = 0,
        incremental_only: bool = False,
    ) -> list[Evidence]:
        # 重排阶段只关心候选集合内部相对顺序，不改变召回总入口的抽象。
        if not evidence or self.reranker_provider is None:
            self.last_rerank_diagnostics = build_rerank_diagnostics(
                enabled=False,
                candidate_count=len(evidence),
                returned_count=min(len(evidence), top_k),
                new_candidate_count=len(evidence),
                excluded_existing_count=excluded_count,
                incremental_only=incremental_only,
                skip_reason="no_new_candidates" if incremental_only and not evidence else "no_reranker",
            )
            return evidence[:top_k]
        if len(evidence) < self.reranker_min_candidates:
            self.last_rerank_diagnostics = build_rerank_diagnostics(
                enabled=False,
                candidate_count=len(evidence),
                returned_count=min(len(evidence), top_k),
                new_candidate_count=len(evidence),
                excluded_existing_count=excluded_count,
                incremental_only=incremental_only,
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
        try:
            reranked = self.reranker_provider.rerank(question, candidates, top_k=len(candidates))
        except Exception:
            self.last_rerank_latency_ms = (perf_counter() - started_at) * 1000
            inference_latency_ms = (perf_counter() - inference_started_at) * 1000
            total_chars, avg_chars = candidate_char_stats(candidate_texts)
            self.last_rerank_diagnostics = build_rerank_diagnostics(
                enabled=False,
                latency_ms=self.last_rerank_latency_ms,
                candidate_build_latency_ms=candidate_build_latency_ms,
                inference_latency_ms=inference_latency_ms,
                candidate_count=len(candidates),
                returned_count=min(len(evidence), top_k),
                input_char_count=total_chars,
                avg_candidate_chars=avg_chars,
                new_candidate_count=len(candidates),
                excluded_existing_count=excluded_count,
                incremental_only=incremental_only,
                skip_reason="provider_error",
            )
            logger.exception("重排器调用失败，回退为向量召回结果。")
            return evidence[:top_k]
        inference_latency_ms = (perf_counter() - inference_started_at) * 1000
        by_index = {str(index): item for index, item in enumerate(evidence)}
        postprocess_started_at = perf_counter()
        scored: list[tuple[float, int, Evidence]] = []
        for rank, result in enumerate(reranked):
            item = by_index.get(result.candidate_id)
            if item is None:
                continue
            original_index = int(result.candidate_id)
            # 在通用 rerank 分数之外叠加少量业务启发式，专门改善 figure 问题的排序质量。
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
            new_candidate_count=len(candidates),
            excluded_existing_count=excluded_count,
            incremental_only=incremental_only,
        )
        return selected

    def _reset_rerank_state(self) -> None:
        self.last_rerank_latency_ms = 0.0
        self.last_rerank_enabled = False
        self.last_recall_diagnostics = {}
        self.last_rerank_diagnostics = build_rerank_diagnostics(enabled=self.reranker_enabled)

    def _count_searchable_chunks(self, document_ids: list[str] | None, owner_id: str | None) -> int | None:
        if not hasattr(self.repository, "count_searchable_chunks"):
            return None
        try:
            return int(self.repository.count_searchable_chunks(document_ids=document_ids, owner_id=owner_id))
        except TypeError:
            return int(self.repository.count_searchable_chunks(document_ids=document_ids))

    def _chunk_search_strategy(self, candidate_chunk_count: int | None) -> str:
        if candidate_chunk_count is None:
            return "hnsw"
        if candidate_chunk_count <= self.exact_chunk_limit:
            return "exact_then_hnsw"
        return "hnsw"


def _call_search(search_method, query_embedding, top_k: int, **kwargs):
    try:
        return search_method(query_embedding, top_k, **kwargs)
    except TypeError as exc:
        if "unexpected keyword" not in str(exc):
            raise
        supported = dict(kwargs)
        for key in ["exact_timeout_ms", "search_strategy", "owner_id"]:
            supported.pop(key, None)
            try:
                return search_method(query_embedding, top_k, **supported)
            except TypeError as retry_exc:
                if "unexpected keyword" not in str(retry_exc):
                    raise
                exc = retry_exc
        raise exc


def _metadata_adjustment(question: str, item: Evidence, mode: str) -> float:
    # 图像类检索往往依赖 caption、label、邻近文本等结构化元数据，
    # 因此这里对 figure 模式做一层轻量加权，补足纯文本 rerank 的盲区。
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


def _exclude_existing(
    evidence: list[Evidence],
    exclude_keys: set[tuple[str, str]] | None,
) -> tuple[list[Evidence], int]:
    # 多轮检索时要排除历史已经见过的 chunk，避免系统陷入“反复召回同一证据”的空转。
    if not exclude_keys:
        return evidence, 0
    selected = [item for item in evidence if (item.document_id, item.chunk_id) not in exclude_keys]
    return selected, len(evidence) - len(selected)


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
