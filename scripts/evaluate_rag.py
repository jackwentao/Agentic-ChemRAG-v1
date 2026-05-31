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
from backend.app.providers.llm import ComposedLlmProvider, FakeLlmProvider, LlmProvider, build_llm_provider
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
    reranker_provider = build_reranker_provider(settings)
    warmup_info = _warmup_evaluation_components(
        embedding_provider=embedding_provider,
        query_router=query_router,
        reranker_provider=reranker_provider,
        enabled=args.warmup == "on" and settings.evaluation_warmup,
        warm_reranker=settings.reranker_warmup_on_startup,
    )
    repository = PgVectorRepository(build_database_engine(settings.database_url))
    llm_provider = _build_evaluation_llm_provider(
        settings,
        answer_provider=args.answer_provider,
        assessment_provider=args.assessment_provider,
    )
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
        llm_provider,
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
        evaluation_info={
            "answer_provider": args.answer_provider,
            "assessment_provider": args.assessment_provider,
            "llm_judge": args.llm_judge,
            "embedding_provider": settings.embedding_provider,
            "reranker_provider": settings.reranker_provider if settings.reranker_enabled else "disabled",
        },
    )
    write_reports(report, args.output_dir)
    print(f"已写入 {args.output_dir / 'latest.json'}")
    print(f"已写入 {args.output_dir / 'latest.md'}")


def evaluate_cases(
    cases: list[GoldenCase],
    graph,
    top_k: int = 5,
    scope_override: str | None = None,
    judge: LlmJudge | None = None,
    warmup_info: dict[str, Any] | None = None,
    evaluation_info: dict[str, str] | None = None,
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
        limitations = list(state.get("limitations", []))
        diagnostics = dict(state.get("diagnostics", {}))
        assessment_rounds = list(diagnostics.get("assessment_rounds", []))
        answer_verification = dict(diagnostics.get("answer_verification", {}))
        llm_call_counts = dict(diagnostics.get("llm_call_counts", {}))
        metrics = evaluate_case_metrics(
            case=case,
            actual_route=state.get("route"),
            evidence=evidence,
            answer=answer,
            citations=citations,
            selected_images=selected_images,
            top_k=top_k,
            limitations=limitations,
            retrieval_round=int(state.get("retrieval_round", 0)),
            stop_reason=state.get("stop_reason"),
            assessment_rounds=assessment_rounds,
            llm_call_counts=llm_call_counts,
            answer_verification=answer_verification,
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
            "retrieval_round": int(state.get("retrieval_round", 0)),
            "stop_reason": state.get("stop_reason"),
            "answer": answer,
            "limitations": limitations,
            "citations": citations,
            "selected_images": selected_images,
            "retrieved_evidence": [_evidence_to_dict(item) for item in evidence],
            "assessment_rounds": assessment_rounds,
            "llm_call_counts": llm_call_counts,
            "diagnostics": diagnostics,
            "metrics": metrics,
            "latency_total_ms": total_latency_ms,
            "latency_retrieval_ms": float(state.get("retrieval_latency_ms", 0.0)),
            "latency_rerank_ms": float(state.get("rerank_latency_ms", 0.0)),
            "latency_assessment_ms": sum(float(item.get("latency_ms", 0.0)) for item in assessment_rounds),
            "latency_answer_verification_ms": float(answer_verification.get("latency_ms", 0.0)),
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
        "evaluation_info": evaluation_info or {
            "answer_provider": "unspecified",
            "assessment_provider": "unspecified",
            "llm_judge": "off",
            "embedding_provider": "unspecified",
            "reranker_provider": "unspecified",
        },
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
        "by_task_category": _group_summary(results, "task_category"),
        "by_risk_level": _group_summary(results, "risk_level"),
        "stop_reasons": _value_counts(results, "stop_reason"),
        "adaptive_retrieval_cost": {
            "average_retrieval_rounds": aggregate_metric_average(results, "retrieval_round_count"),
            "retry_trigger_rate": aggregate_metric_average(results, "retry_triggered"),
            "observed_retry_success_rate": aggregate_metric_average(results, "observed_retry_success"),
            "wasted_retry_rate": aggregate_metric_average(results, "wasted_retry"),
            "average_tracked_llm_calls": aggregate_metric_average(results, "total_tracked_llm_calls"),
            "average_assessment_llm_calls": aggregate_metric_average(results, "assessment_llm_calls"),
            "average_answer_verification_llm_calls": aggregate_metric_average(results, "answer_verification_llm_calls"),
            "average_answer_revision_llm_calls": aggregate_metric_average(results, "answer_revision_llm_calls"),
            "average_image_selection_llm_calls": aggregate_metric_average(results, "image_selection_llm_calls"),
        },
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
    evaluation_info = report.get("evaluation_info", {})
    lines = [
        "# RAG 离线评测报告",
        "",
        f"- 案例数量：{summary['case_count']}",
        f"- 回答生成提供器：`{evaluation_info.get('answer_provider', 'unspecified')}`",
        f"- 证据评估提供器：`{evaluation_info.get('assessment_provider', 'unspecified')}`",
        f"- LLM 裁判：`{evaluation_info.get('llm_judge', 'off')}`",
        f"- 嵌入提供器：`{evaluation_info.get('embedding_provider', 'unspecified')}`",
        f"- 重排提供器：`{evaluation_info.get('reranker_provider', 'unspecified')}`",
        f"- 是否启用预热：{'是' if warmup.get('enabled', False) else '否'}",
        f"- 嵌入预热时延：{_fmt(warmup.get('embedding_warmup_latency_ms', 0.0))} ms",
        f"- 路由预热时延：{_fmt(warmup.get('warmup_latency_ms', 0.0))} ms",
        f"- 路由缓存状态：{_status_label(warmup.get('router_cache_status', 'not_requested'))}",
        f"- 重排器预热时延：{_fmt(warmup.get('reranker_warmup_latency_ms', 0.0))} ms",
        "",
        "> 说明：`configured` 表示使用已配置的真实模型；`fake` 表示对应阶段使用模拟实现。"
        "当证据评估提供器为 `configured` 时，检索循环仍由真实模型判断驱动。"
        "`observed_retry_success` 是观察到的追加检索后成功，不代表反事实增益。",
        "",
        "## 总体指标",
        "",
        "| 指标 | 数值 |",
        "| --- | ---: |",
    ]
    for metric, value in summary["metrics"].items():
        lines.append(f"| {_metric_label(metric)} | {_fmt(value)} |")
    lines.extend(["", "## 时延", "", "| 指标 | p50 | p95 | 最大值 |", "| --- | ---: | ---: | ---: |"])
    for metric, values in summary["latencies"].items():
        lines.append(f"| {_metric_label(metric)} | {_fmt(values['p50'])} | {_fmt(values['p95'])} | {_fmt(values['max'])} |")

    lines.extend(["", "## 自适应检索成本", "", "| 指标 | 数值 |", "| --- | ---: |"])
    for metric, value in summary["adaptive_retrieval_cost"].items():
        lines.append(f"| {_metric_label(metric)} | {_fmt(value)} |")
    lines.extend(["", "## 停止原因", "", "| 原因 | 案例数 |", "| --- | ---: |"])
    for reason, count in summary["stop_reasons"].items():
        lines.append(f"| {_stop_reason_label(reason)} | {count} |")

    lines.extend(_group_lines("按答案类型", summary["by_answer_type"]))
    lines.extend(_group_lines("按预期路由", summary["by_expected_route"]))
    lines.extend(_group_lines("按目标类型", summary["by_target_type"]))
    lines.extend(_group_lines("按任务类别", summary["by_task_category"]))
    lines.extend(_group_lines("按风险等级", summary["by_risk_level"]))
    lines.extend(_failure_lines("路由失败案例", results, lambda item: not item["metrics"].get("route_hit", False)))
    lines.extend(_low_confidence_lines(results))
    lines.extend(_needs_llm_router_lines(results))
    lines.extend(_failure_lines("检索目标未命中案例", results, lambda item: not item["metrics"].get("target_hit", False)))
    lines.extend(_failure_lines("图片未命中案例", results, lambda item: item["metrics"].get("figure_hit_mode") == "none"))
    lines.extend(_slowest_lines(results))
    lines.extend(_judge_low_score_lines(results))
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评测 RAG 的检索质量、回答质量与运行时延。")
    parser.add_argument("--cases", type=Path, default=Path("eval/golden_cases.jsonl"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("eval/reports"))
    parser.add_argument("--llm-judge", choices=["off", "on"], default="off")
    parser.add_argument("--answer-provider", choices=["configured", "fake"], default="configured")
    parser.add_argument(
        "--assessment-provider",
        choices=["configured", "fake"],
        default="configured",
        help="证据充分性评估提供器；默认使用真实已配置模型，fake 仅用于确定性回归。",
    )
    parser.add_argument("--scope", choices=["current_document", "all_documents", "from_case"], default="from_case")
    parser.add_argument("--warmup", choices=["on", "off"], default="on")
    return parser.parse_args()


def _build_evaluation_llm_provider(
    settings,
    *,
    answer_provider: str,
    assessment_provider: str,
) -> LlmProvider:
    needs_configured = answer_provider == "configured" or assessment_provider == "configured"
    configured: LlmProvider | None = None
    if needs_configured:
        if not settings.deepseek_api_key:
            raise ValueError(
                "真实回答或证据评估需要配置 DEEPSEEK_API_KEY；"
                "仅运行确定性回归时请传入 --answer-provider fake --assessment-provider fake。"
            )
        configured = build_llm_provider(settings)
    answer = configured if answer_provider == "configured" else FakeLlmProvider()
    assessment = configured if assessment_provider == "configured" else FakeLlmProvider()
    if answer is assessment:
        return answer
    return ComposedLlmProvider(answer_provider=answer, assessment_provider=assessment)


def _warmup_router(query_router: QueryRouter, warm_embedding_provider: bool = True) -> dict[str, Any]:
    status = query_router.warmup(warm_embedding_provider=warm_embedding_provider)
    return {
        "warmup_latency_ms": status.latency_ms,
        "router_cache_status": status.status,
        "router_cache_enabled": status.cache_enabled,
        "router_cache_path": status.cache_path,
        "router_cache_error": status.error,
        "embedding_warmup_latency_ms": 0.0,
        "enabled": True,
    }


def _warmup_evaluation_components(
    *,
    embedding_provider,
    query_router: QueryRouter,
    reranker_provider,
    enabled: bool,
    warm_reranker: bool,
) -> dict[str, Any]:
    if not enabled:
        return _warmup_disabled()
    embedding_started_at = perf_counter()
    embedding_provider.embed_query("evaluation embedding warmup probe")
    embedding_warmup_latency_ms = (perf_counter() - embedding_started_at) * 1000
    warmup_info = _warmup_router(query_router, warm_embedding_provider=False)
    warmup_info["embedding_warmup_latency_ms"] = embedding_warmup_latency_ms
    warmup_info["enabled"] = True
    if reranker_provider is not None and warm_reranker:
        rerank_started_at = perf_counter()
        reranker_provider.warmup()
        warmup_info["reranker_warmup_latency_ms"] = (perf_counter() - rerank_started_at) * 1000
    else:
        warmup_info["reranker_warmup_latency_ms"] = 0.0
    return warmup_info


def _warmup_disabled() -> dict[str, Any]:
    return {
        "warmup_latency_ms": 0.0,
        "router_cache_status": "disabled",
        "router_cache_enabled": False,
        "router_cache_path": None,
        "router_cache_error": None,
        "embedding_warmup_latency_ms": 0.0,
        "enabled": False,
    }


def _warmup_not_requested() -> dict[str, Any]:
    return {
        "warmup_latency_ms": 0.0,
        "router_cache_status": "not_requested",
        "router_cache_enabled": False,
        "router_cache_path": None,
        "router_cache_error": None,
        "embedding_warmup_latency_ms": 0.0,
        "enabled": False,
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
        "task_category": case.task_category,
        "risk_level": case.risk_level,
        "expected_aspects": case.expected_aspects,
        "required_disclosures": case.required_disclosures,
        "expected_retrieval_behavior": case.expected_retrieval_behavior,
        "max_acceptable_rounds": case.max_acceptable_rounds,
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
            "retrieval_round_count": aggregate_metric_average(items, "retrieval_round_count"),
            "total_tracked_llm_calls": aggregate_metric_average(items, "total_tracked_llm_calls"),
        }
        for name, items in groups.items()
    }


def _value_counts(results: list[dict[str, Any]], result_key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        name = str(result.get(result_key) or "unknown")
        counts[name] = counts.get(name, 0) + 1
    return counts


def _group_lines(title: str, groups: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "",
        f"## {title}",
        "",
        "| 分组 | 案例数 | 路由命中 | 目标命中 | 引用命中 | 术语覆盖 | 平均轮次 | 平均 LLM 调用 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, values in groups.items():
        lines.append(
            f"| {_group_value_label(name)} | {values['case_count']} | {_fmt(values['route_hit'])} | {_fmt(values['target_hit'])} | "
            f"{_fmt(values['citation_page_hit'])} | {_fmt(values['answer_term_coverage'])} | "
            f"{_fmt(values['retrieval_round_count'])} | {_fmt(values['total_tracked_llm_calls'])} |"
        )
    return lines


def _failure_lines(title: str, results: list[dict[str, Any]], predicate) -> list[str]:
    failures = [result for result in results if predicate(result)]
    lines = ["", f"## {title}", ""]
    if not failures:
        return [*lines, "无。"]
    for result in failures:
        lines.append(
            f"- `{result['case']['case_id']}` 路由={_group_value_label(str(result.get('actual_route')))} "
            f"目标={_group_value_label(result['case']['target_type'])} 置信度={_fmt(result.get('route_confidence', 0.0))} "
            f"得分={result.get('route_scores', {})}"
        )
    return lines


def _low_confidence_lines(results: list[dict[str, Any]]) -> list[str]:
    low = [result for result in results if float(result.get("route_confidence", 0.0)) < 0.15]
    lines = ["", "## 低置信度路由", ""]
    if not low:
        return [*lines, "无。"]
    for result in low:
        lines.append(
            f"- `{result['case']['case_id']}` 路由={_group_value_label(str(result.get('actual_route')))} "
            f"置信度={_fmt(result.get('route_confidence', 0.0))} 原因={result.get('route_rationale', [])}"
        )
    return lines


def _needs_llm_router_lines(results: list[dict[str, Any]]) -> list[str]:
    flagged = [result for result in results if result.get("needs_llm_router")]
    lines = ["", "## 可能需要 LLM 路由的案例", ""]
    if not flagged:
        return [*lines, "无。"]
    for result in flagged:
        lines.append(
            f"- `{result['case']['case_id']}` 路由={_group_value_label(str(result.get('actual_route')))} "
            f"多步骤={result.get('is_multi_step')} 置信度={_fmt(result.get('route_confidence', 0.0))}"
        )
    return lines


def _slowest_lines(results: list[dict[str, Any]]) -> list[str]:
    lines = ["", "## 最慢案例", ""]
    for result in sorted(results, key=lambda item: item["latency_total_ms"], reverse=True)[:5]:
        rerank = result.get("diagnostics", {}).get("rerank", {})
        lines.append(
            f"- `{result['case']['case_id']}` 总时延={_fmt(result['latency_total_ms'])} ms "
            f"重排={_fmt(rerank.get('latency_ms', 0.0))} ms 候选数={rerank.get('candidate_count', 0)}"
        )
    return lines


def _judge_low_score_lines(results: list[dict[str, Any]]) -> list[str]:
    judged = [result for result in results if "judge" in result]
    lines = ["", "## LLM 裁判低分案例", ""]
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
        return [*lines, "无。"]
    for result in low:
        judge = result["judge"]
        lines.append(
            f"- `{result['case']['case_id']}` 忠实性={judge.get('faithfulness')} "
            f"完整性={judge.get('completeness')} 相关性={judge.get('relevance')}"
        )
    return lines


def _metric_label(name: str) -> str:
    labels = {
        "answer_llm_calls": "回答生成 LLM 调用数",
        "answer_revision_llm_calls": "答案修订 LLM 调用数",
        "answer_revision_applied": "答案已修订比例",
        "answer_revision_required": "答案需修订比例",
        "answer_term_coverage": "答案术语覆盖率",
        "answer_verification_llm_calls": "答案核验 LLM 调用数",
        "answer_verification_passed": "答案核验通过率",
        "answer_verification_triggered": "答案核验触发率",
        "image_selection_llm_calls": "图片选择 LLM 调用数",
        "aspect_coverage_rate": "预期方面覆盖率",
        "assessment_llm_calls": "证据评估 LLM 调用数",
        "assessment_round_count": "证据评估轮数",
        "citation_page_hit": "引用页命中率",
        "figure_hit": "图片命中率",
        "first_round_success": "首轮成功率",
        "limitation_disclosure_hit": "限制说明披露命中率",
        "missing_disclosure_count": "遗漏限制数量",
        "max_rounds_exhausted": "达到最大轮次比例",
        "no_new_evidence_stop": "无新证据停止比例",
        "no_new_query_stop": "无新查询停止比例",
        "observed_retry_success": "观察到的重试成功率",
        "refusal_correct": "正确拒答率",
        "retrieval_behavior_hit": "预期检索行为命中率",
        "retrieval_round_count": "检索轮次数",
        "retry_triggered": "重试触发率",
        "route_hit": "路由命中率",
        "source_url_valid": "来源链接有效率",
        "target_hit": "检索目标命中率",
        "total_tracked_llm_calls": "已追踪 LLM 调用总数",
        "unsupported_claim_risk": "无引用结论风险率",
        "unsupported_claim_count": "未支撑断言数量",
        "wasted_retry": "无收益重试率",
        "latency_total_ms": "总时延（毫秒）",
        "latency_retrieval_ms": "检索时延（毫秒）",
        "latency_assessment_ms": "证据评估时延（毫秒）",
        "latency_answer_verification_ms": "答案核验时延（毫秒）",
        "latency_rerank_ms": "重排时延（毫秒）",
        "latency_answer_ms": "回答时延（毫秒）",
        "average_retrieval_rounds": "平均检索轮次",
        "retry_trigger_rate": "重试触发率",
        "observed_retry_success_rate": "观察到的重试成功率",
        "wasted_retry_rate": "无收益重试率",
        "average_tracked_llm_calls": "平均已追踪 LLM 调用数",
        "average_assessment_llm_calls": "平均证据评估 LLM 调用数",
        "average_answer_verification_llm_calls": "平均答案核验 LLM 调用数",
        "average_answer_revision_llm_calls": "平均答案修订 LLM 调用数",
        "average_image_selection_llm_calls": "平均图片选择 LLM 调用数",
    }
    if name in labels:
        return labels[name]
    suffix_labels = {
        "_recall@k": "召回率@k",
        "_precision@k": "准确率@k",
        "_mrr@k": "MRR@k",
        "_ndcg@k": "NDCG@k",
    }
    prefix_labels = {"document": "文档", "page": "页面", "figure": "图片"}
    for suffix, label in suffix_labels.items():
        if name.endswith(suffix):
            prefix = name[: -len(suffix)]
            return f"{prefix_labels.get(prefix, prefix)}{label}"
    return name


def _status_label(status: str) -> str:
    return {
        "disabled": "已关闭",
        "not_requested": "未请求",
        "disk_hit": "磁盘缓存命中",
        "memory_hit": "内存缓存命中",
        "rebuilt": "已重建",
    }.get(str(status), str(status))


def _stop_reason_label(reason: str) -> str:
    return {
        "sufficient_evidence": "证据充分",
        "max_rounds_reached": "达到最大检索轮次",
        "no_new_query": "没有新的检索查询",
        "no_new_evidence": "没有新增证据",
        "assessment_fallback": "证据评估回退",
        "no_search_direction": "没有继续检索方向",
        "unknown": "未知",
    }.get(reason, reason)


def _group_value_label(name: str) -> str:
    return {
        "fact_qa": "事实问答",
        "figure_query": "图片查询",
        "document_summary": "文档总结",
        "cross_document_filter": "跨文档筛选",
        "refusal": "资料不足拒答",
        "chunk_search": "片段检索",
        "figure_index": "图片索引",
        "document_profile": "文档画像",
        "document": "文档",
        "page": "页面",
        "figure": "图片",
        "mixed": "混合",
        "none": "无证据目标",
        "simple_fact": "简单事实",
        "figure_or_summary": "图片或总结",
        "multi_condition": "多条件",
        "comparison": "比较",
        "context_followup": "上下文追问",
        "high_risk": "高风险",
        "insufficient_evidence": "资料不足",
        "low": "低",
        "medium": "中",
        "high": "高",
    }.get(name, name)


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "1.000" if value else "0.000"
    if isinstance(value, (int, float)):
        return f"{float(value):.3f}"
    return str(value)


if __name__ == "__main__":
    main()
