from __future__ import annotations

import re
from typing import Any

from backend.app.domain.documents import Evidence


def build_route_diagnostics(
    *,
    latency_ms: float,
    selected_route: str,
    confidence: float,
    needs_llm_router: bool,
    is_multi_step: bool,
    scores: dict[str, float],
    lexical_scores: dict[str, float],
    semantic_scores: dict[str, float],
    rationale: list[str],
) -> dict[str, Any]:
    return {
        "latency_ms": float(latency_ms),
        "selected_route": selected_route,
        "confidence": float(confidence),
        "needs_llm_router": bool(needs_llm_router),
        "is_multi_step": bool(is_multi_step),
        "proxy_hit": False,
        "scores": {key: float(value) for key, value in scores.items()},
        "lexical_scores": {key: float(value) for key, value in lexical_scores.items()},
        "semantic_scores": {key: float(value) for key, value in semantic_scores.items()},
        "rationale": list(rationale),
    }


def build_recall_diagnostics(
    *,
    mode: str,
    latency_ms: float,
    embedding_latency_ms: float,
    search_latency_ms: float,
    candidate_count: int,
    returned_count: int,
    top_score: float | None,
    avg_score: float | None,
) -> dict[str, Any]:
    hit = candidate_count > 0
    return {
        "mode": mode,
        "latency_ms": float(latency_ms),
        "embedding_latency_ms": float(embedding_latency_ms),
        "search_latency_ms": float(search_latency_ms),
        "candidate_count": int(candidate_count),
        "returned_count": int(returned_count),
        "hit": hit,
        "proxy_hit": hit and returned_count > 0,
        "top_score": _maybe_float(top_score),
        "avg_score": _maybe_float(avg_score),
    }


def build_rerank_diagnostics(
    *,
    enabled: bool,
    latency_ms: float = 0.0,
    candidate_build_latency_ms: float = 0.0,
    inference_latency_ms: float = 0.0,
    postprocess_latency_ms: float = 0.0,
    candidate_count: int = 0,
    returned_count: int = 0,
    input_char_count: int = 0,
    avg_candidate_chars: float = 0.0,
    changed_top1: bool = False,
    changed_top_k: bool = False,
    top1_term_overlap_before: int = 0,
    top1_term_overlap_after: int = 0,
    skip_reason: str | None = None,
) -> dict[str, Any]:
    proxy_hit = returned_count > 0 and top1_term_overlap_after >= top1_term_overlap_before
    return {
        "enabled": bool(enabled),
        "latency_ms": float(latency_ms),
        "candidate_build_latency_ms": float(candidate_build_latency_ms),
        "inference_latency_ms": float(inference_latency_ms),
        "postprocess_latency_ms": float(postprocess_latency_ms),
        "candidate_count": int(candidate_count),
        "returned_count": int(returned_count),
        "input_char_count": int(input_char_count),
        "avg_candidate_chars": float(avg_candidate_chars),
        "changed_top1": bool(changed_top1),
        "changed_top_k": bool(changed_top_k),
        "top1_term_overlap_before": int(top1_term_overlap_before),
        "top1_term_overlap_after": int(top1_term_overlap_after),
        "skip_reason": skip_reason,
        "proxy_hit": proxy_hit,
    }


def build_llm_diagnostics(
    *,
    latency_ms: float,
    used_llm: bool,
    answer_chars: int,
    evidence_count: int,
    citation_count: int,
    selected_image_count: int = 0,
) -> dict[str, Any]:
    proxy_hit = answer_chars > 0 and (citation_count > 0 or evidence_count > 0 or not used_llm)
    return {
        "latency_ms": float(latency_ms),
        "used_llm": bool(used_llm),
        "answer_chars": int(answer_chars),
        "evidence_count": int(evidence_count),
        "citation_count": int(citation_count),
        "selected_image_count": int(selected_image_count),
        "proxy_hit": proxy_hit,
    }


def update_stage_proxy_hit(diagnostics: dict[str, Any] | None, stage: str, value: bool) -> dict[str, Any]:
    payload = dict(diagnostics or {})
    stage_payload = dict(payload.get(stage, {}))
    if stage_payload:
        stage_payload["proxy_hit"] = bool(value)
        payload[stage] = stage_payload
    return payload


def set_total_latency(diagnostics: dict[str, Any] | None, total_latency_ms: float) -> dict[str, Any]:
    payload = dict(diagnostics or {})
    payload["total_latency_ms"] = float(total_latency_ms)
    return payload


def score_summary(evidence: list[Evidence]) -> tuple[float | None, float | None]:
    if not evidence:
        return None, None
    scores = [float(item.score) for item in evidence]
    return scores[0], sum(scores) / len(scores)


def candidate_char_stats(texts: list[str]) -> tuple[int, float]:
    if not texts:
        return 0, 0.0
    total = sum(len(text) for text in texts)
    return total, total / len(texts)


def top1_term_overlap(question: str, evidence: list[Evidence], mode: str) -> int:
    if not evidence:
        return 0
    return term_overlap(question, candidate_text(evidence[0], mode))


def top_k_changed(before: list[Evidence], after: list[Evidence]) -> bool:
    before_ids = [item.chunk_id for item in before]
    after_ids = [item.chunk_id for item in after]
    return before_ids != after_ids


def candidate_text(item: Evidence, mode: str) -> str:
    if mode == "figure" and item.images:
        image = item.images[0]
        return " ".join(
            value
            for value in [image.label, image.caption, image.nearby_text, item.text]
            if value
        )
    return item.text


def term_overlap(query: str, text: str) -> int:
    return len(_terms(query) & _terms(text))


def _terms(text: str) -> set[str]:
    normalized = _normalize_aliases(text)
    terms = {term.lower() for term in re.findall(r"[A-Za-z][A-Za-z0-9-]{3,}", normalized)}
    terms.update(re.findall(r"[\u4e00-\u9fff]{2,}", normalized))
    return terms


def _normalize_aliases(text: str) -> str:
    return text.lower().replace("nlo", "nonlinear optical")


def _maybe_float(value: float | None) -> float | None:
    return None if value is None else float(value)
