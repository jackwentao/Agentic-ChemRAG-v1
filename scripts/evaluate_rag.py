import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.agents.rag_graph import MultiAgentRagGraph
from backend.app.config import get_settings
from backend.app.database import build_database_engine
from backend.app.domain.documents import Evidence, EvidenceImage
from backend.app.evaluation.cases import GoldenCase, load_golden_cases
from backend.app.evaluation.judge import LlmJudge
from backend.app.evaluation.metrics import (
    aggregate_metric_average,
    case_to_dict,
    evaluate_case_metrics,
    summarize_latencies,
)
from backend.app.providers.embeddings import build_embedding_provider
from backend.app.providers.llm import FakeLlmProvider, build_llm_provider
from backend.app.providers.rerankers import build_reranker_provider
from backend.app.repositories.postgres import PgVectorRepository
from backend.app.routing.query_router import QueryRouter
from backend.app.services.rag import RetrievalService


def main() -> None:
    args = _parse_args()
    settings = get_settings()
    cases = load_golden_cases(args.cases)
    embedding_provider = build_embedding_provider(settings)
    query_router = QueryRouter(
        embedding_provider=embedding_provider,
        cache_dir=settings.router_cache_dir,
        cache_enabled=settings.router_cache_enabled,
        provider_fingerprint=_router_provider_fingerprint(settings),
    )
    warmup_info = _warmup_router(query_router) if args.warmup == "on" and settings.evaluation_warmup else _warmup_disabled()
    reranker_provider = build_reranker_provider(settings)
    if args.warmup == "on" and reranker_provider is not None and settings.reranker_warmup_on_startup:
        rerank_started_at = perf_counter()
        reranker_provider.warmup()
        warmup_info["reranker_warmup_latency_ms"] = (perf_counter() - rerank_started_at) * 1000
    else:
        warmup_info["reranker_warmup_latency_ms"] = 0.0
    repository = PgVectorRepository(build_database_engine(settings.database_url))
    graph = MultiAgentRagGraph(
        RetrievalService(
            repository,
            embedding_provider,
            reranker_provider=reranker_provider,
            reranker_enabled=settings.reranker_enabled,
            reranker_chunk_candidates=settings.reranker_chunk_candidates,
            reranker_figure_candidates=settings.reranker_figure_candidates,
            reranker_min_candidates=settings.reranker_min_candidates,
        ),
        FakeLlmProvider() if args.answer_provider == "fake" or not settings.deepseek_api_key else build_llm_provider(settings),
        retrieval_top_k=args.top_k,
        figure_top_k=args.top_k,
        profile_top_k=args.top_k,
        query_router=query_router,
    )
    judge = LlmJudge(settings) if args.llm_judge == "on" else None
    report = evaluate_cases(
        cases=cases,
        graph=graph,
        top_k=args.top_k,
        scope_override=None if args.scope == "from_case" else args.scope,
        judge=judge,
        warmup_info=warmup_info,
    )
    write_reports(report, args.output_dir)
    print(f"Wrote {args.output_dir / 'latest.json'}")
    print(f"Wrote {args.output_dir / 'latest.md'}")


def evaluate_cases(
    cases: list[GoldenCase],
    graph,
    top_k: int = 5,
    scope_override: str | None = None,
    judge: LlmJudge | None = None,
    warmup_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for case in cases:
        document_ids = _document_ids_for_case(case, scope_override)
        started_at = perf_counter()
        state = graph.invoke(case.question, document_ids=document_ids)
        total_latency_ms = float(state.get("total_latency_ms", (perf_counter() - started_at) * 1000))
        evidence = list(state.get("verified_evidence", []))
        selected_images = _flatten_selected_images(state.get("selected_images", {}), evidence)
        citations = _citations_from_evidence(evidence)
        answer = str(state.get("answer", ""))
        metrics = evaluate_case_metrics(
            case=case,
            actual_route=state.get("route"),
            evidence=evidence,
            answer=answer,
            citations=citations,
            selected_images=selected_images,
            top_k=top_k,
        )
        result: dict[str, Any] = {
            "case": case_to_dict(case),
            "actual_route": state.get("route"),
            "route_scores": state.get("route_scores", {}),
            "route_lexical_scores": state.get("route_lexical_scores", {}),
            "route_semantic_scores": state.get("route_semantic_scores", {}),
            "route_confidence": float(state.get("route_confidence", 0.0)),
            "route_rationale": state.get("route_rationale", []),
            "needs_llm_router": bool(state.get("needs_llm_router", False)),
            "is_multi_step": bool(state.get("is_multi_step", False)),
            "answer": answer,
            "citations": citations,
            "selected_images": selected_images,
            "retrieved_evidence": [_evidence_to_dict(item) for item in evidence],
            "diagnostics": state.get("diagnostics", {}),
            "metrics": metrics,
            "latency_total_ms": total_latency_ms,
            "latency_retrieval_ms": float(state.get("retrieval_latency_ms", 0.0)),
            "latency_rerank_ms": float(state.get("rerank_latency_ms", 0.0)),
            "rerank_enabled": bool(state.get("rerank_enabled", False)),
            "latency_answer_ms": float(state.get("answer_latency_ms", 0.0)),
        }
        if judge is not None:
            try:
                result["judge"] = judge.score(
                    question=case.question,
                    gold_hints=_gold_hints(case),
                    answer=answer,
                    citations=citations,
                )
            except Exception as exc:
                result["judge_error"] = str(exc)
        results.append(result)
    return {
        "warmup": warmup_info or _warmup_not_requested(),
        "summary": summarize_results(results),
        "results": results,
    }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = sorted(
        {
            metric
            for result in results
            for metric, value in result.get("metrics", {}).items()
            if isinstance(value, (bool, int, float))
        }
    )
    summary = {
        "case_count": len(results),
        "metrics": {name: aggregate_metric_average(results, name) for name in metric_names},
        "latencies": summarize_latencies(results),
        "by_answer_type": _group_summary(results, "answer_type"),
        "by_expected_route": _group_summary(results, "expected_route"),
        "by_target_type": _group_summary(results, "target_type"),
    }
    return summary


def write_reports(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "latest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "latest.md").write_text(render_markdown_report(report), encoding="utf-8")


def render_markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    warmup = report.get("warmup", _warmup_not_requested())
    results = report["results"]
    lines = [
        "# RAG Evaluation Report",
        "",
        f"- Cases: {summary['case_count']}",
        f"- Warmup latency: {_fmt(warmup.get('warmup_latency_ms', 0.0))} ms",
        f"- Router cache status: {warmup.get('router_cache_status', 'not_requested')}",
        f"- Reranker warmup latency: {_fmt(warmup.get('reranker_warmup_latency_ms', 0.0))} ms",
        "",
        "## Overall Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for metric, value in summary["metrics"].items():
        lines.append(f"| {metric} | {_fmt(value)} |")
    lines.extend(["", "## Latency", "", "| Metric | p50 | p95 | max |", "| --- | ---: | ---: | ---: |"])
    for metric, values in summary["latencies"].items():
        lines.append(f"| {metric} | {_fmt(values['p50'])} | {_fmt(values['p95'])} | {_fmt(values['max'])} |")

    lines.extend(_group_lines("By Answer Type", summary["by_answer_type"]))
    lines.extend(_group_lines("By Expected Route", summary["by_expected_route"]))
    lines.extend(_group_lines("By Target Type", summary["by_target_type"]))
    lines.extend(_failure_lines("Route Failures", results, lambda item: not item["metrics"].get("route_hit", False)))
    lines.extend(_low_confidence_lines(results))
    lines.extend(_needs_llm_router_lines(results))
    lines.extend(_failure_lines("Retrieval Target Failures", results, lambda item: not item["metrics"].get("target_hit", False)))
    lines.extend(_failure_lines("Figure Misses", results, lambda item: item["metrics"].get("figure_hit_mode") == "none"))
    lines.extend(_slowest_lines(results))
    lines.extend(_judge_low_score_lines(results))
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval, answer quality, and latency.")
    parser.add_argument("--cases", type=Path, default=Path("eval/golden_cases.jsonl"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("eval/reports"))
    parser.add_argument("--llm-judge", choices=["off", "on"], default="off")
    parser.add_argument("--answer-provider", choices=["configured", "fake"], default="configured")
    parser.add_argument("--scope", choices=["current_document", "all_documents", "from_case"], default="from_case")
    parser.add_argument("--warmup", choices=["on", "off"], default="on")
    return parser.parse_args()


def _warmup_router(query_router: QueryRouter) -> dict[str, Any]:
    status = query_router.warmup()
    return {
        "warmup_latency_ms": status.latency_ms,
        "router_cache_status": status.status,
        "router_cache_enabled": status.cache_enabled,
        "router_cache_path": status.cache_path,
        "router_cache_error": status.error,
    }


def _warmup_disabled() -> dict[str, Any]:
    return {
        "warmup_latency_ms": 0.0,
        "router_cache_status": "disabled",
        "router_cache_enabled": False,
        "router_cache_path": None,
        "router_cache_error": None,
    }


def _warmup_not_requested() -> dict[str, Any]:
    return {
        "warmup_latency_ms": 0.0,
        "router_cache_status": "not_requested",
        "router_cache_enabled": False,
        "router_cache_path": None,
        "router_cache_error": None,
    }


def _router_provider_fingerprint(settings) -> str:
    return (
        f"{settings.embedding_provider}:"
        f"{settings.embedding_model}:"
        f"{settings.embedding_device}:"
        f"{settings.embedding_dim}:"
        f"local_files_only={settings.embedding_local_files_only}"
    )


def _document_ids_for_case(case: GoldenCase, scope_override: str | None) -> list[str] | None:
    scope = scope_override or case.scope
    if scope == "current_document":
        return case.document_ids
    return None


def _citations_from_evidence(evidence: list[Evidence]) -> list[dict[str, Any]]:
    return [
        {
            "document_id": item.document_id,
            "chunk_id": item.chunk_id,
            "page_start": item.page_start,
            "page_end": item.page_end,
            "source_url": _pdf_page_url(item.document_id, item.page_start),
            "section_title": item.section_title,
            "quote": item.text[:500],
            "score": item.score,
        }
        for item in evidence
    ]


def _flatten_selected_images(
    selected_images: dict[str, list[EvidenceImage]],
    evidence: list[Evidence],
) -> list[dict[str, Any]]:
    document_by_chunk = {item.chunk_id: item.document_id for item in evidence}
    rows: list[dict[str, Any]] = []
    for chunk_id, images in selected_images.items():
        document_id = document_by_chunk.get(chunk_id)
        for image in images:
            rows.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "image_id": image.image_id,
                    "page": image.page,
                    "label": image.label,
                    "caption": image.caption,
                    "nearby_text": image.nearby_text,
                    "reason": image.reason,
                    "source_url": _pdf_page_url(document_id, image.page) if document_id else None,
                }
            )
    return rows


def _evidence_to_dict(item: Evidence) -> dict[str, Any]:
    return {
        "chunk_id": item.chunk_id,
        "document_id": item.document_id,
        "page_start": item.page_start,
        "page_end": item.page_end,
        "section_title": item.section_title,
        "score": item.score,
        "text_preview": item.text[:500],
        "images": [asdict(image) for image in item.images],
    }


def _gold_hints(case: GoldenCase) -> dict[str, Any]:
    return {
        "expected_route": case.expected_route,
        "target_type": case.target_type,
        "expected_documents": case.expected_documents,
        "expected_pages": case.expected_pages,
        "expected_figures": case.expected_figures,
        "expected_answer_terms": case.expected_answer_terms,
        "answer_type": case.answer_type,
        "notes": case.notes,
    }


def _pdf_page_url(document_id: str | None, page: int) -> str:
    return f"/api/documents/{document_id}/pdf#page={page}"


def _group_summary(results: list[dict[str, Any]], case_key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        groups.setdefault(str(result["case"][case_key]), []).append(result)
    return {
        name: {
            "case_count": len(items),
            "route_hit": aggregate_metric_average(items, "route_hit"),
            "target_hit": aggregate_metric_average(items, "target_hit"),
            "citation_page_hit": aggregate_metric_average(items, "citation_page_hit"),
            "answer_term_coverage": aggregate_metric_average(items, "answer_term_coverage"),
        }
        for name, items in groups.items()
    }


def _group_lines(title: str, groups: dict[str, dict[str, Any]]) -> list[str]:
    lines = ["", f"## {title}", "", "| Group | Cases | Route Hit | Target Hit | Citation Hit | Term Coverage |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, values in groups.items():
        lines.append(
            f"| {name} | {values['case_count']} | {_fmt(values['route_hit'])} | {_fmt(values['target_hit'])} | "
            f"{_fmt(values['citation_page_hit'])} | {_fmt(values['answer_term_coverage'])} |"
        )
    return lines


def _failure_lines(title: str, results: list[dict[str, Any]], predicate) -> list[str]:
    failures = [result for result in results if predicate(result)]
    lines = ["", f"## {title}", ""]
    if not failures:
        return [*lines, "None."]
    for result in failures:
        lines.append(
            f"- `{result['case']['case_id']}` route={result.get('actual_route')} "
            f"target={result['case']['target_type']} confidence={_fmt(result.get('route_confidence', 0.0))} "
            f"scores={result.get('route_scores', {})}"
        )
    return lines


def _low_confidence_lines(results: list[dict[str, Any]]) -> list[str]:
    low = [result for result in results if float(result.get("route_confidence", 0.0)) < 0.15]
    lines = ["", "## Low Confidence Routes", ""]
    if not low:
        return [*lines, "None."]
    for result in low:
        lines.append(
            f"- `{result['case']['case_id']}` route={result.get('actual_route')} "
            f"confidence={_fmt(result.get('route_confidence', 0.0))} rationale={result.get('route_rationale', [])}"
        )
    return lines


def _needs_llm_router_lines(results: list[dict[str, Any]]) -> list[str]:
    flagged = [result for result in results if result.get("needs_llm_router")]
    lines = ["", "## Needs LLM Router", ""]
    if not flagged:
        return [*lines, "None."]
    for result in flagged:
        lines.append(
            f"- `{result['case']['case_id']}` route={result.get('actual_route')} "
            f"multi_step={result.get('is_multi_step')} confidence={_fmt(result.get('route_confidence', 0.0))}"
        )
    return lines


def _slowest_lines(results: list[dict[str, Any]]) -> list[str]:
    lines = ["", "## Slowest Cases", ""]
    for result in sorted(results, key=lambda item: item["latency_total_ms"], reverse=True)[:5]:
        rerank = result.get("diagnostics", {}).get("rerank", {})
        lines.append(
            f"- `{result['case']['case_id']}` total={_fmt(result['latency_total_ms'])} ms "
            f"rerank={_fmt(rerank.get('latency_ms', 0.0))} ms candidates={rerank.get('candidate_count', 0)}"
        )
    return lines


def _judge_low_score_lines(results: list[dict[str, Any]]) -> list[str]:
    judged = [result for result in results if "judge" in result]
    lines = ["", "## LLM Judge Low Scores", ""]
    low = [
        result
        for result in judged
        if min(
            result["judge"].get("faithfulness", 5),
            result["judge"].get("completeness", 5),
            result["judge"].get("relevance", 5),
        )
        <= 2
    ]
    if not low:
        return [*lines, "None."]
    for result in low:
        judge = result["judge"]
        lines.append(
            f"- `{result['case']['case_id']}` faithfulness={judge.get('faithfulness')} "
            f"completeness={judge.get('completeness')} relevance={judge.get('relevance')}"
        )
    return lines


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "1.000" if value else "0.000"
    if isinstance(value, (int, float)):
        return f"{float(value):.3f}"
    return str(value)


if __name__ == "__main__":
    main()
