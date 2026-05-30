from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from time import perf_counter
from hashlib import sha256
from typing import Any

from backend.app.providers.embeddings import EmbeddingProvider


ROUTES = ("chunk_search", "figure_index", "document_profile")
CACHE_SCHEMA_VERSION = 1
CACHE_FILENAME = "query_router_prototypes.json"


@dataclass(frozen=True)
class RouteContext:
    scope: str = "current_document"
    document_ids: list[str] | None = None
    recent_questions: list[str] = field(default_factory=list)
    selected_document_id: str | None = None


@dataclass(frozen=True)
class RouteDecision:
    route: str
    scores: dict[str, float]
    lexical_scores: dict[str, float]
    semantic_scores: dict[str, float]
    confidence: float
    is_multi_step: bool
    needs_llm_router: bool
    rationale: list[str]


@dataclass(frozen=True)
class RouterWarmupStatus:
    cache_enabled: bool
    status: str
    latency_ms: float
    cache_path: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_enabled": self.cache_enabled,
            "status": self.status,
            "latency_ms": self.latency_ms,
            "cache_path": self.cache_path,
            "error": self.error,
        }


class QueryRouter:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider | None = None,
        lexical_weight: float = 0.70,
        semantic_weight: float = 0.30,
        confidence_threshold: float = 0.15,
        score_threshold: float = 0.35,
        cache_dir: Path | None = None,
        cache_enabled: bool = True,
        provider_fingerprint: str | None = None,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.lexical_weight = lexical_weight
        self.semantic_weight = semantic_weight
        self.confidence_threshold = confidence_threshold
        self.score_threshold = score_threshold
        self.cache_dir = cache_dir
        self.cache_enabled = cache_enabled and cache_dir is not None
        self.provider_fingerprint = provider_fingerprint or _provider_fingerprint(embedding_provider)
        self.prototypes_hash = _prototypes_hash()
        self._prototype_vectors: dict[str, list[list[float]]] | None = None
        self._last_cache_status = "not_loaded"
        self._last_cache_error: str | None = None
        self._last_warmup: RouterWarmupStatus | None = None

    @property
    def cache_path(self) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / CACHE_FILENAME

    def warmup(self, warm_embedding_provider: bool = True) -> RouterWarmupStatus:
        started_at = perf_counter()
        try:
            if warm_embedding_provider and self.embedding_provider is not None:
                self.embedding_provider.embed_query("router warmup query")
            self._get_prototype_vectors()
            status = self._last_cache_status
            error = self._last_cache_error
        except Exception as exc:
            status = "error_fallback"
            error = str(exc)
            self._prototype_vectors = {route: [] for route in ROUTES}
        warmup = RouterWarmupStatus(
            cache_enabled=self.cache_enabled,
            status=status,
            latency_ms=(perf_counter() - started_at) * 1000,
            cache_path=str(self.cache_path) if self.cache_path else None,
            error=error,
        )
        self._last_warmup = warmup
        return warmup

    def route(self, question: str, context: RouteContext | None = None) -> RouteDecision:
        context = context or RouteContext()
        normalized = _normalize(question)
        lexical_scores, rationale = self._lexical_scores(normalized, context)
        semantic_scores = self._semantic_scores(question)
        scores = {
            route: round(
                self.lexical_weight * lexical_scores.get(route, 0.0)
                + self.semantic_weight * semantic_scores.get(route, 0.0),
                6,
            )
            for route in ROUTES
        }
        ranked = sorted(scores.items(), key=lambda item: (-item[1], _route_tiebreaker(item[0])))
        top_route, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        confidence = round(top_score - second_score, 6)
        is_multi_step = _is_multi_step(normalized)
        out_of_domain = "likely_out_of_domain_low_confidence" in rationale
        if is_multi_step:
            rationale.append("multi_step_signal")
        needs_llm_router = (
            confidence < self.confidence_threshold
            or top_score < self.score_threshold
            or is_multi_step
            or out_of_domain
        )
        return RouteDecision(
            route=top_route,
            scores=scores,
            lexical_scores=lexical_scores,
            semantic_scores=semantic_scores,
            confidence=confidence,
            is_multi_step=is_multi_step,
            needs_llm_router=needs_llm_router,
            rationale=rationale[:12],
        )

    def _lexical_scores(self, normalized: str, context: RouteContext) -> tuple[dict[str, float], list[str]]:
        scores = {"chunk_search": 0.22, "figure_index": 0.0, "document_profile": 0.0}
        rationale: list[str] = ["default_chunk_search"]

        explicit_figure = _has_any(
            normalized,
            [
                "what figures",
                "which figures",
                "what plots",
                "which plots",
                "list the figures",
                "show figures",
                "figures or plots",
                "figures are shown",
                "plots or figures",
                "有什么图",
                "哪些图",
                "性能图",
                "图表",
                "图片",
                "谱图",
            ],
        ) or _has_chinese_figure_listing_signal(normalized)
        figure_label = bool(
            re.search(
                r"\b(?:fig(?:ure)?|scheme|table)\.?\s*\d+[a-z]?|\b(?:fig(?:ure)?|scheme|table)s?\b|[图表]\s*\d+",
                normalized,
            )
        )
        visual_intent = _has_any(
            normalized,
            [
                "show",
                "shown",
                "list",
                "which",
                "what",
                "describe",
                "contain",
                "包含",
                "列出",
                "列举",
                "展示",
                "显示",
                "查看",
                "看一下",
                "给我看",
                "哪些",
                "有什么",
                "有哪",
            ],
        )
        visual_terms = _has_any(
            normalized,
            [
                "spectrum",
                "spectra",
                "pl",
                "xrd",
                "ftir",
                "uv-vis",
                "transmittance",
                "sem",
                "tem",
                "plot",
                "chart",
                "image",
                "图",
                "谱",
                "曲线",
                "图像",
                "显微",
            ],
        )
        figure_local_fact = figure_label and _has_any(
            normalized,
            ["说明", "表明", "代表", "结果", "含义", "结论", "机制", "峰位", "数值", "多少", "为什么", "如何", "怎么", "是否", "支撑", "证明"],
        )

        if explicit_figure:
            scores["figure_index"] += 1.0
            rationale.append("explicit_figure_query")
        if figure_label:
            scores["figure_index"] += 0.72
            rationale.append("figure_label_or_caption_signal")
        if visual_terms:
            scores["figure_index"] += 0.18
            rationale.append("weak_visual_term")
        if visual_terms and visual_intent:
            scores["figure_index"] += 0.38
            rationale.append("visual_term_with_query_intent")
        if figure_local_fact:
            scores["chunk_search"] += 0.65
            rationale.append("figure_local_fact_question")

        strong_summary_signal = _has_any(
            normalized,
            [
                "what is this paper about",
                "what is the paper about",
                "main contribution",
                "main idea",
                "overview",
                "这篇论文讲什么",
                "该论文讲什么",
                "这个论文讲什么",
                "这篇文献讲什么",
                "该文献讲什么",
                "这篇论文主要研究什么",
                "这篇文献主要研究什么",
                "该论文主要研究什么",
                "该文献主要研究什么",
                "主要研究什么",
            ],
        )
        generic_summary_signal = _has_any(
            normalized,
            ["summarize", "summary", "总结", "概述", "概括", "梳理"],
        )
        whole_document_target = _has_any(normalized, ["paper", "document", "论文", "文献", "文章", "全文", "整篇"])
        summary_signal = strong_summary_signal or (
            generic_summary_signal
            and whole_document_target
            and not explicit_figure
            and not figure_label
            and not figure_local_fact
        )
        cross_doc_signal = bool(
            re.search(r"\bwhich\s+(?:papers|documents)\b|\bpapers?\s+(?:related to|discuss|about)\b", normalized)
            or re.search(r"(?:哪些|哪几篇|找出|筛选|检索).{0,16}(?:论文|文献|文章)", normalized)
            or (
                context.scope == "all_documents"
                and re.search(r"(?:论文|文献|文章).{0,16}(?:相关|有关|讨论|研究|涉及)", normalized)
            )
        ) or _has_any(
            normalized,
            ["哪些论文", "哪些文献", "哪几篇论文", "哪几篇文献", "相关论文", "相关文献"],
        )
        profile_entity_signal = _has_any(normalized, ["porphyrin", "phthalocyanine", "卟啉", "酞菁"])
        all_docs_topic_signal = context.scope == "all_documents" and _has_any(
            normalized,
            ["zno", "photoluminescence", "luminescence", "polymer", "cof", "nonlinear optical", "发光", "荧光"],
        )

        if summary_signal:
            scores["document_profile"] += 1.0
            rationale.append("document_summary_signal")
        if cross_doc_signal:
            scores["document_profile"] += 1.05
            rationale.append("cross_document_signal")
        if profile_entity_signal:
            scores["document_profile"] += 0.6
            rationale.append("profile_entity_signal")
        if all_docs_topic_signal:
            scores["document_profile"] += 0.35
            rationale.append("all_documents_topic_boost")

        fact_signal = _has_any(
            normalized,
            [
                "wavelength",
                "dopant",
                "dopants",
                "solvent",
                "temperature",
                "method",
                "result",
                "property",
                "condition",
                "mechanism",
                "efficiency",
                "reported",
                "covered",
                "discussed",
                "phenomenon",
                "多少",
                "如何",
                "怎么",
                "原因",
                "条件",
                "方法",
                "结果",
                "结论",
                "是什么",
            ],
        )
        fact_question = bool(re.search(r"\b(?:what|how|why|which|when|where)\b", normalized)) or _has_any(
            normalized,
            ["什么", "多少", "如何", "怎么"],
        )
        if fact_signal:
            scores["chunk_search"] += 0.55
            rationale.append("fact_detail_signal")
        if fact_question and not explicit_figure and not summary_signal and not cross_doc_signal:
            scores["chunk_search"] += 0.28
            rationale.append("local_fact_question_signal")

        unrelated_signal = _has_any(normalized, ["crispr", "gene editing", "biomedical", "clinical trial"])
        if unrelated_signal:
            scores["document_profile"] = max(0.0, scores["document_profile"] - 0.35)
            scores["figure_index"] = max(0.0, scores["figure_index"] - 0.2)
            scores["chunk_search"] += 0.15
            rationale.append("likely_out_of_domain_low_confidence")

        return ({route: min(round(score, 6), 1.0) for route, score in scores.items()}, rationale)

    def _semantic_scores(self, question: str) -> dict[str, float]:
        if self.embedding_provider is None:
            return {route: 0.0 for route in ROUTES}
        try:
            query_vector = self.embedding_provider.embed_query(question)
            prototypes = self._get_prototype_vectors()
        except Exception:
            return {route: 0.0 for route in ROUTES}
        scores = {}
        for route, vectors in prototypes.items():
            if not vectors:
                scores[route] = 0.0
                continue
            score = max(_cosine(query_vector, vector) for vector in vectors)
            scores[route] = round(max(0.0, min(1.0, (score + 1.0) / 2.0)), 6)
        return scores

    def _get_prototype_vectors(self) -> dict[str, list[list[float]]]:
        if self._prototype_vectors is not None:
            self._last_cache_status = "memory_hit"
            self._last_cache_error = None
            return self._prototype_vectors
        if self.embedding_provider is None:
            self._prototype_vectors = {route: [] for route in ROUTES}
            self._last_cache_status = "no_provider"
            self._last_cache_error = None
            return self._prototype_vectors

        if self.cache_enabled:
            cached = self._load_cached_vectors()
            if cached is not None:
                self._prototype_vectors = cached
                self._last_cache_status = "disk_hit"
                self._last_cache_error = None
                return self._prototype_vectors

        self._prototype_vectors = self._compute_prototype_vectors()
        if not self.cache_enabled:
            self._last_cache_status = "disabled"
            return self._prototype_vectors

        previous_error = self._last_cache_error
        if self._write_cached_vectors(self._prototype_vectors):
            self._last_cache_status = "miss_written" if previous_error is None else "invalid_recomputed_written"
            self._last_cache_error = previous_error
        else:
            self._last_cache_status = "miss_write_failed" if previous_error is None else "invalid_recomputed_write_failed"
        return self._prototype_vectors

    def _compute_prototype_vectors(self) -> dict[str, list[list[float]]]:
        return {
            route: self.embedding_provider.embed_documents(texts)
            for route, texts in ROUTE_PROTOTYPES.items()
        }

    def _load_cached_vectors(self) -> dict[str, list[list[float]]] | None:
        path = self.cache_path
        if path is None or not path.exists():
            self._last_cache_error = None
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            vectors = _validate_cache_payload(
                payload,
                provider_fingerprint=self.provider_fingerprint,
                prototypes_hash=self.prototypes_hash,
            )
            return vectors
        except Exception as exc:
            self._last_cache_error = str(exc)
            return None

    def _write_cached_vectors(self, vectors: dict[str, list[list[float]]]) -> bool:
        path = self.cache_path
        if path is None:
            return False
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": CACHE_SCHEMA_VERSION,
                "provider_fingerprint": self.provider_fingerprint,
                "prototypes_hash": self.prototypes_hash,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "dimensions": _vector_dimensions(vectors),
                "vectors": vectors,
            }
            tmp_path = path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp_path.replace(path)
            return True
        except Exception as exc:
            self._last_cache_error = str(exc)
            return False


ROUTE_PROTOTYPES = {
    "chunk_search": [
        "What wavelength was used in the measurement?",
        "What solvent was used for synthesis?",
        "What dopants are discussed in this paper?",
        "How does the method affect the material property?",
        "What result is reported for the sample?",
        "What optical property is discussed?",
        "这个实验条件是什么？",
        "论文中某个结果是什么？",
    ],
    "figure_index": [
        "What figures or plots does this paper contain?",
        "Which figure shows the PL spectrum?",
        "List the charts and images in this paper.",
        "Which scheme describes the synthesis route?",
        "What performance figures are shown?",
        "该文献有什么性能图？",
        "哪些图展示了荧光光谱？",
        "图表里有哪些结果？",
    ],
    "document_profile": [
        "What is this paper about?",
        "Summarize the main contribution of this paper.",
        "Which papers discuss ZnO photoluminescence?",
        "Which documents are related to phthalocyanine?",
        "Find papers about porphyrin materials.",
        "这篇论文主要研究什么？",
        "哪些论文是讲卟啉相关的？",
        "总结这篇文献的主题。",
    ],
}


def _validate_cache_payload(
    payload: Any,
    provider_fingerprint: str,
    prototypes_hash: str,
) -> dict[str, list[list[float]]]:
    if not isinstance(payload, dict):
        raise ValueError("Router cache payload must be an object.")
    if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError("Router cache schema version does not match.")
    if payload.get("provider_fingerprint") != provider_fingerprint:
        raise ValueError("Router cache provider fingerprint does not match.")
    if payload.get("prototypes_hash") != prototypes_hash:
        raise ValueError("Router cache prototype hash does not match.")
    vectors = payload.get("vectors")
    if not isinstance(vectors, dict):
        raise ValueError("Router cache vectors must be an object.")
    normalized: dict[str, list[list[float]]] = {}
    for route in ROUTES:
        route_vectors = vectors.get(route)
        if not isinstance(route_vectors, list):
            raise ValueError(f"Router cache missing vectors for {route}.")
        normalized[route] = []
        for vector in route_vectors:
            if not isinstance(vector, list) or not vector or not all(isinstance(value, (int, float)) for value in vector):
                raise ValueError(f"Router cache vector for {route} is invalid.")
            normalized[route].append([float(value) for value in vector])
    dimensions = _vector_dimensions(normalized)
    if int(payload.get("dimensions", dimensions)) != dimensions:
        raise ValueError("Router cache vector dimensions do not match.")
    return normalized


def _vector_dimensions(vectors: dict[str, list[list[float]]]) -> int:
    dimensions = {len(vector) for route_vectors in vectors.values() for vector in route_vectors}
    if not dimensions:
        return 0
    if len(dimensions) != 1:
        raise ValueError("Router prototype vectors have inconsistent dimensions.")
    return dimensions.pop()


def _prototypes_hash() -> str:
    payload = json.dumps(ROUTE_PROTOTYPES, ensure_ascii=False, sort_keys=True)
    return sha256(payload.encode("utf-8")).hexdigest()


def _provider_fingerprint(provider: EmbeddingProvider | None) -> str:
    if provider is None:
        return "none"
    attrs = []
    for name in ("model_name", "device", "local_files_only", "dimensions"):
        if hasattr(provider, name):
            attrs.append(f"{name}={getattr(provider, name)}")
    return f"{provider.__class__.__module__}.{provider.__class__.__name__}:{'|'.join(attrs)}"


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _has_any(text: str, phrases: list[str]) -> bool:
    return any(phrase.lower() in text for phrase in phrases)


def _has_chinese_figure_listing_signal(text: str) -> bool:
    # 中文图表问题常用“有哪些/给我看/列出 + 图/谱图/曲线”的松散表达，
    # 单靠固定短语容易漏召回；但这里仍限定输出对象必须是图像类资产。
    figure_object = r"(?:图表|图片|图像|谱图|曲线|性能图|显微图|sem图|tem图|xrd图|ftir图|uv-vis图|pl图|图)"
    return bool(
        re.search(rf"(?:哪些|有哪(?:些)?|有什么|列出|列举|展示|显示|查看|看一下|给我看).{{0,16}}{figure_object}", text)
        or re.search(rf"{figure_object}.{{0,12}}(?:有哪些|有哪(?:些)?|是什么|列表|清单)", text)
    )


def _is_multi_step(normalized: str) -> bool:
    return bool(
        re.search(r"\b(?:compare|then|and summarize|summarize their|find .* and |list .* and )\b", normalized)
        or re.search(r"先.*再|找出.*并|比较.*总结|筛选.*总结|列出.*并", normalized)
    )


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _route_tiebreaker(route: str) -> int:
    return {"chunk_search": 0, "document_profile": 1, "figure_index": 2}.get(route, 99)
