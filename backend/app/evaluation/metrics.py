from __future__ import annotations

from dataclasses import asdict
import math
import re
from statistics import median
from typing import Any

from backend.app.domain.documents import Evidence, EvidenceImage
from backend.app.evaluation.cases import GoldenCase


PDF_SOURCE_URL_PATTERN = re.compile(r"^/api/documents/[^/]+/pdf#page=\d+$")
REFUSAL_MARKERS = [
    "not enough evidence",
    "could not find",
    "cannot answer",
    "insufficient evidence",
    "没有",
    "无法",
    "未检索",
    "证据不足",
]


def evaluate_case_metrics(
    case: GoldenCase,
    actual_route: str | None,
    evidence: list[Evidence],
    answer: str,
    citations: list[dict[str, Any]],
    selected_images: list[dict[str, Any]],
    top_k: int,
    limitations: list[str] | None = None,
    retrieval_round: int = 0,
    stop_reason: str | None = None,
    assessment_rounds: list[dict[str, Any]] | None = None,
    llm_call_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    limitations = limitations or []
    assessment_rounds = assessment_rounds or []
    llm_call_counts = llm_call_counts or {}
    metrics: dict[str, Any] = {
        "route_hit": actual_route == case.expected_route,
        "target_type": case.target_type,
    }
    if case.target_type in {"document", "mixed"}:
        metrics.update(_prefixed("document", ranking_metrics(_document_relevance(evidence, case.expected_documents), len(case.expected_documents), top_k)))
    if case.target_type in {"page", "mixed"}:
        metrics.update(_prefixed("page", ranking_metrics(_page_relevance(evidence, case.expected_pages), len(case.expected_pages), top_k)))
    if case.target_type in {"figure", "mixed"}:
        figure_relevance = _figure_relevance(evidence, case.expected_figures)
        metrics.update(_prefixed("figure", ranking_metrics(figure_relevance, len(case.expected_figures), top_k, include_ndcg=False)))

    target_hits = []
    if "document_recall@k" in metrics:
        target_hits.append(metrics["document_recall@k"] > 0)
    if "page_recall@k" in metrics:
        target_hits.append(metrics["page_recall@k"] > 0)
    if "figure_recall@k" in metrics:
        target_hits.append(metrics["figure_recall@k"] > 0)
    metrics["target_hit"] = any(target_hits) if target_hits else case.target_type == "none"

    metrics["citation_page_hit"] = citation_page_hit(citations, case.expected_pages)
    figure_hit, figure_hit_mode = selected_figure_hit(selected_images, case.expected_figures)
    metrics["figure_hit"] = figure_hit
    metrics["figure_hit_mode"] = figure_hit_mode
    metrics["source_url_valid"] = source_urls_valid(citations, selected_images)
    metrics["answer_term_coverage"] = answer_term_coverage(answer, case.expected_answer_terms)
    metrics["unsupported_claim_risk"] = (
        bool(answer.strip())
        and not citations
        and case.target_type != "none"
        and not _contains_refusal_marker(answer)
    )
    metrics["refusal_correct"] = refusal_correct(answer, case.target_type)
    metrics["aspect_coverage_rate"] = answer_term_coverage(answer, case.expected_aspects)
    metrics["limitation_disclosure_hit"] = limitation_disclosure_hit(limitations, case.required_disclosures)
    metrics["retrieval_round_count"] = int(retrieval_round)
    metrics["first_round_success"] = retrieval_round == 1 and stop_reason == "sufficient_evidence"
    metrics["retry_triggered"] = retrieval_round > 1
    metrics["observed_retry_success"] = retrieval_round > 1 and stop_reason == "sufficient_evidence"
    metrics["wasted_retry"] = retrieval_round > 1 and stop_reason != "sufficient_evidence"
    metrics["max_rounds_exhausted"] = stop_reason == "max_rounds_reached"
    metrics["no_new_evidence_stop"] = stop_reason == "no_new_evidence"
    metrics["no_new_query_stop"] = stop_reason == "no_new_query"
    metrics["retrieval_behavior_hit"] = retrieval_behavior_hit(case, retrieval_round, stop_reason)
    metrics["assessment_round_count"] = len(assessment_rounds)
    metrics["assessment_llm_calls"] = int(llm_call_counts.get("evidence_assessment", 0))
    metrics["answer_llm_calls"] = int(llm_call_counts.get("answer_generation", 0))
    metrics["total_tracked_llm_calls"] = metrics["assessment_llm_calls"] + metrics["answer_llm_calls"]
    return metrics


def ranking_metrics(
    relevance: list[bool],
    expected_count: int,
    top_k: int,
    include_ndcg: bool = True,
) -> dict[str, float]:
    considered = relevance[:top_k]
    hits = sum(1 for item in considered if item)
    denominator = max(1, min(top_k, len(considered) or top_k))
    metrics = {
        "recall@k": min(hits, expected_count) / expected_count if expected_count else 0.0,
        "precision@k": hits / denominator,
        "mrr@k": _mrr(considered),
    }
    if include_ndcg:
        metrics["ndcg@k"] = _ndcg(considered, expected_count, top_k)
    return metrics


def citation_page_hit(citations: list[dict[str, Any]], expected_pages: list[int]) -> bool:
    if not expected_pages:
        return True
    expected = set(expected_pages)
    for citation in citations:
        pages = set(range(int(citation.get("page_start", 0)), int(citation.get("page_end", 0)) + 1))
        if expected & pages:
            return True
    return False


def selected_figure_hit(
    selected_images: list[dict[str, Any]],
    expected_figures: list[dict[str, Any] | str],
) -> tuple[bool, str]:
    if not expected_figures:
        return True, "exact_id"
    expected_ids = _expected_figure_ids(expected_figures)
    selected_ids = {str(image.get("image_id") or "") for image in selected_images}
    if expected_ids and expected_ids & selected_ids:
        return True, "exact_id"
    if _page_caption_figure_hit(selected_images, expected_figures):
        return True, "page_caption"
    return False, "none"


def source_urls_valid(citations: list[dict[str, Any]], selected_images: list[dict[str, Any]]) -> bool:
    urls = [citation.get("source_url") for citation in citations]
    urls.extend(image.get("source_url") for image in selected_images)
    return all(isinstance(url, str) and PDF_SOURCE_URL_PATTERN.match(url) for url in urls)


def answer_term_coverage(answer: str, expected_terms: list[str]) -> float:
    if not expected_terms:
        return 1.0
    normalized = answer.lower()
    hits = sum(1 for term in expected_terms if term.lower() in normalized)
    return hits / len(expected_terms)


def refusal_correct(answer: str, target_type: str) -> bool:
    if target_type != "none":
        return True
    return _contains_refusal_marker(answer)


def _contains_refusal_marker(answer: str) -> bool:
    normalized = answer.lower()
    return any(marker.lower() in normalized for marker in REFUSAL_MARKERS)


def limitation_disclosure_hit(limitations: list[str], required_disclosures: list[str]) -> bool:
    if not required_disclosures:
        return True
    text = "\n".join(limitations).lower()
    return all(disclosure.lower() in text for disclosure in required_disclosures)


def retrieval_behavior_hit(case: GoldenCase, retrieval_round: int, stop_reason: str | None) -> bool:
    if retrieval_round <= 0 or retrieval_round > case.max_acceptable_rounds:
        return False
    if case.expected_retrieval_behavior == "single_round_expected":
        return retrieval_round == 1 and stop_reason == "sufficient_evidence"
    if case.expected_retrieval_behavior == "insufficient_should_stop":
        return stop_reason in {
            "max_rounds_reached",
            "no_new_query",
            "no_new_evidence",
            "assessment_fallback",
            "no_search_direction",
        }
    return stop_reason == "sufficient_evidence"


def summarize_latencies(results: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for key in [
        "latency_total_ms",
        "latency_retrieval_ms",
        "latency_assessment_ms",
        "latency_rerank_ms",
        "latency_answer_ms",
    ]:
        values = [float(result.get(key, 0.0)) for result in results]
        summary[key] = _latency_stats(values)
    return summary


def aggregate_metric_average(results: list[dict[str, Any]], metric_name: str) -> float:
    values = [result["metrics"][metric_name] for result in results if metric_name in result.get("metrics", {})]
    if not values:
        return 0.0
    numeric = [float(value) for value in values if isinstance(value, (bool, int, float))]
    if not numeric:
        return 0.0
    return sum(numeric) / len(numeric)


def case_to_dict(case: GoldenCase) -> dict[str, Any]:
    return asdict(case)


def _prefixed(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{name}": value for name, value in metrics.items()}


def _document_relevance(evidence: list[Evidence], expected_documents: list[str]) -> list[bool]:
    expected = set(expected_documents)
    return [item.document_id in expected for item in evidence]


def _page_relevance(evidence: list[Evidence], expected_pages: list[int]) -> list[bool]:
    expected = set(expected_pages)
    return [bool(expected & set(range(item.page_start, item.page_end + 1))) for item in evidence]


def _figure_relevance(evidence: list[Evidence], expected_figures: list[dict[str, Any] | str]) -> list[bool]:
    return [any(_image_matches_expected(image, expected_figures) for image in item.images) for item in evidence]


def _image_matches_expected(image: EvidenceImage, expected_figures: list[dict[str, Any] | str]) -> bool:
    image_dict = {
        "image_id": image.image_id,
        "page": image.page,
        "caption": image.caption,
        "label": image.label,
    }
    if image.image_id in _expected_figure_ids(expected_figures):
        return True
    return _page_caption_figure_hit([image_dict], expected_figures)


def _expected_figure_ids(expected_figures: list[dict[str, Any] | str]) -> set[str]:
    ids: set[str] = set()
    for figure in expected_figures:
        if isinstance(figure, str):
            ids.add(figure)
        else:
            for key in ["image_id", "figure_id"]:
                if figure.get(key):
                    ids.add(str(figure[key]))
    return ids


def _page_caption_figure_hit(
    selected_images: list[dict[str, Any]],
    expected_figures: list[dict[str, Any] | str],
) -> bool:
    for expected in expected_figures:
        if not isinstance(expected, dict):
            continue
        expected_page = expected.get("page")
        caption_terms = [_normalize_figure_aliases(str(term)) for term in expected.get("caption_terms", [])]
        if expected_page is None or not caption_terms:
            continue
        for image in selected_images:
            if int(image.get("page") or -1) != int(expected_page):
                continue
            text = " ".join(
                str(value)
                for value in [image.get("label"), image.get("caption"), image.get("nearby_text"), image.get("reason")]
                if value
            )
            text = _normalize_figure_aliases(text)
            if all(term in text for term in caption_terms):
                return True
    return False


def _normalize_figure_aliases(text: str) -> str:
    return text.lower().replace("nlo", "nonlinear optical")


def _mrr(relevance: list[bool]) -> float:
    for index, is_relevant in enumerate(relevance, start=1):
        if is_relevant:
            return 1 / index
    return 0.0


def _ndcg(relevance: list[bool], expected_count: int, top_k: int) -> float:
    capped = _cap_relevance(relevance[:top_k], expected_count)
    dcg = sum((1.0 if rel else 0.0) / math.log2(index + 1) for index, rel in enumerate(capped, start=1))
    ideal_hits = min(expected_count, top_k)
    if ideal_hits <= 0:
        return 0.0
    ideal = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    return dcg / ideal if ideal else 0.0


def _cap_relevance(relevance: list[bool], expected_count: int) -> list[bool]:
    remaining = expected_count
    capped = []
    for item in relevance:
        if item and remaining > 0:
            capped.append(True)
            remaining -= 1
        else:
            capped.append(False)
    return capped


def _latency_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "max": 0.0}
    sorted_values = sorted(values)
    return {
        "p50": float(median(sorted_values)),
        "p95": _percentile(sorted_values, 0.95),
        "max": float(max(sorted_values)),
    }


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    index = math.ceil(percentile * len(sorted_values)) - 1
    index = min(max(index, 0), len(sorted_values) - 1)
    return float(sorted_values[index])
