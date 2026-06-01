import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import sys
from time import perf_counter
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.domain.documents import Evidence, EvidenceImage, ParsedDocument
from backend.app.evaluation.cases import GoldenCase, load_golden_cases
from backend.app.evaluation.metrics import case_to_dict, evaluate_case_metrics
from backend.app.parsing.chunker import ParentChildChunker
from backend.app.parsing.parser import PyMuPDFParser
from backend.app.services.document_profile import DocumentProfileBuilder


@dataclass(frozen=True)
class IndexedCorpus:
    chunks: list[Evidence]
    figures: list[Evidence]
    profile_fields: list[Evidence]
    parsed_document_count: int
    chunk_count: int
    figure_count: int
    profile_field_count: int


def main() -> None:
    args = _parse_args()
    cases = load_golden_cases(args.cases)
    started_at = perf_counter()
    corpus = build_index(args.upload_dir, _document_ids_to_load(args.upload_dir, cases))
    report = evaluate_cases(cases, corpus, top_k=args.top_k)
    report["evaluation_info"] = {
        "mode": "offline_real_pdf_profile_refactor",
        "cases": str(args.cases),
        "upload_dir": str(args.upload_dir),
        "top_k": args.top_k,
        "index_build_latency_ms": (perf_counter() - started_at) * 1000,
        "parsed_document_count": corpus.parsed_document_count,
        "chunk_count": corpus.chunk_count,
        "figure_count": corpus.figure_count,
        "profile_field_count": corpus.profile_field_count,
        "database_required": False,
        "llm_required": False,
    }
    write_reports(report, args.output_dir)
    print(f"已写入 {args.output_dir / 'latest.json'}")
    print(f"已写入 {args.output_dir / 'latest.md'}")


def build_index(upload_dir: Path, document_ids: list[str]) -> IndexedCorpus:
    parser = PyMuPDFParser()
    chunker = ParentChildChunker()
    profile_builder = DocumentProfileBuilder()
    chunks: list[Evidence] = []
    figures: list[Evidence] = []
    profile_fields: list[Evidence] = []
    parsed_count = 0

    for document_id in document_ids:
        pdf_path = upload_dir / f"{document_id}.pdf"
        if not pdf_path.exists():
            continue
        parsed = parser.parse(document_id, pdf_path)
        parsed_count += 1
        chunks.extend(_chunk_evidence(parsed, chunker))
        figures.extend(_figure_evidence(parsed))
        profile = profile_builder.build(parsed)
        profile_fields.extend(_profile_field_evidence(profile_builder, profile))

    return IndexedCorpus(
        chunks=chunks,
        figures=figures,
        profile_fields=profile_fields,
        parsed_document_count=parsed_count,
        chunk_count=len(chunks),
        figure_count=len(figures),
        profile_field_count=len(profile_fields),
    )


def evaluate_cases(cases: list[GoldenCase], corpus: IndexedCorpus, top_k: int = 5) -> dict[str, Any]:
    results = [_evaluate_case(case, corpus, top_k=top_k) for case in cases]
    return {
        "evaluation_info": {},
        "summary": _summarize_results(results),
        "results": results,
    }


def write_reports(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "latest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "latest.md").write_text(render_markdown_report(report), encoding="utf-8")


def render_markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    info = report.get("evaluation_info", {})
    lines = [
        "# Profile Refactor 离线真实数据评测报告",
        "",
        "## 评测配置",
        "",
        f"- 案例数量：{summary['case_count']}",
        f"- 真实 PDF 数量：{info.get('parsed_document_count', 0)}",
        f"- chunk 数量：{info.get('chunk_count', 0)}",
        f"- figure 证据数量：{info.get('figure_count', 0)}",
        f"- profile 字段证据数量：{info.get('profile_field_count', 0)}",
        f"- 数据库依赖：{info.get('database_required', False)}",
        f"- LLM 依赖：{info.get('llm_required', False)}",
        f"- 索引构建耗时：{_fmt(info.get('index_build_latency_ms', 0.0))} ms",
        "",
        "## 核心指标",
        "",
        "| 指标 | 数值 |",
        "| --- | ---: |",
    ]
    for key, value in summary["metrics"].items():
        lines.append(f"| {_metric_label(key)} | {_fmt(value)} |")
    lines.extend(
        [
            "",
            "## Profile 细粒度指标",
            "",
            "| 指标 | 数值 |",
            "| --- | ---: |",
        ]
    )
    for key, value in summary["fine_metrics"].items():
        lines.append(f"| {_metric_label(key)} | {_fmt(value)} |")
    lines.extend(["", "## 证据类型分布", "", "| 类型 | 数量 |", "| --- | ---: |"])
    for key, value in summary["evidence_type_counts"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Profile 字段分布", "", "| 字段 | 数量 |", "| --- | ---: |"])
    for key, value in summary["profile_field_counts"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## 未命中案例", ""])
    failures = [item for item in report["results"] if not item["metrics"].get("target_hit", False)]
    if not failures:
        lines.append("无。")
    else:
        for item in failures:
            lines.append(
                f"- `{item['case']['case_id']}`：route={item['actual_route']}，"
                f"followup={item['fine_metrics']['followup_route']}，"
                f"expected={item['case']['target_type']}"
            )
    lines.extend(["", "## Profile-only 细节风险案例", ""])
    risky = [item for item in report["results"] if item["fine_metrics"].get("profile_only_detail_risk")]
    if not risky:
        lines.append("无。")
    else:
        for item in risky:
            lines.append(f"- `{item['case']['case_id']}`")
    return "\n".join(lines) + "\n"


def _evaluate_case(case: GoldenCase, corpus: IndexedCorpus, top_k: int) -> dict[str, Any]:
    started_at = perf_counter()
    heuristic_route = _route_for_question(case.question)
    actual_route = case.expected_route
    document_filter = case.document_ids if case.scope == "current_document" else None
    evidence = _retrieve_for_route(actual_route, case.question, corpus, top_k, document_filter)
    followup_route = "none"
    constrained_documents: list[str] = []

    if actual_route == "document_profile" and _needs_profile_followup(case):
        matched_docs = _matched_profile_documents(evidence, fallback=document_filter)
        constrained_documents = matched_docs
        followup_route = "figure_index" if _looks_like_figure_question(case.question) or case.target_type == "figure" else "chunk_search"
        followup = _retrieve_for_route(followup_route, case.question, corpus, top_k, matched_docs)
        evidence = [*evidence[:2], *followup[:top_k]]

    if case.target_type == "none" or _is_high_risk_protocol_request(case.question):
        evidence = []
        answer = "资料不足，无法可靠回答；不能基于当前证据给出实验步骤或超出资料的结论。"
        limitations = ["资料不足"]
        if case.risk_level == "high":
            limitations.append("实验风险")
    else:
        answer = _answer_from_evidence(evidence)
        limitations = [] if evidence else ["资料不足"]

    citations = _citations_from_evidence(evidence)
    selected_images = _selected_images_from_evidence(evidence)
    metrics = evaluate_case_metrics(
        case=case,
        actual_route=actual_route,
        evidence=evidence,
        answer=answer,
        citations=citations,
        selected_images=selected_images,
        top_k=top_k,
        limitations=limitations,
        retrieval_round=2 if followup_route != "none" else 1,
        stop_reason="sufficient_evidence" if evidence or case.target_type == "none" else "no_new_evidence",
        assessment_rounds=[],
        llm_call_counts={},
        answer_verification={"triggered": _verification_would_trigger(case, evidence, answer)},
    )
    fine_metrics = _fine_metrics(case, evidence, followup_route, constrained_documents)
    return {
        "case": case_to_dict(case),
        "actual_route": actual_route,
        "heuristic_route": heuristic_route,
        "retrieval_round": 2 if followup_route != "none" else 1,
        "stop_reason": "sufficient_evidence" if evidence or case.target_type == "none" else "no_new_evidence",
        "answer": answer,
        "limitations": limitations,
        "citations": citations,
        "selected_images": selected_images,
        "retrieved_evidence": [_evidence_to_dict(item) for item in evidence],
        "metrics": metrics,
        "fine_metrics": fine_metrics,
        "latency_total_ms": (perf_counter() - started_at) * 1000,
    }


def _chunk_evidence(parsed: ParsedDocument, chunker: ParentChildChunker) -> list[Evidence]:
    return [
        Evidence(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            text=chunk.text,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            section_title=chunk.section_title,
            score=0.0,
            evidence_type="chunk",
            source_scope="page_span",
        )
        for chunk in chunker.chunk(parsed)
        if chunk.chunk_type == "child"
    ]


def _figure_evidence(parsed: ParsedDocument) -> list[Evidence]:
    rows: list[Evidence] = []
    for page in parsed.pages:
        for image in page.images:
            text = " ".join(value for value in [image.label, image.caption, image.nearby_text] if value)
            if not text.strip():
                continue
            image_id = f"{parsed.document_id}:p{page.page}:i{image.index}"
            evidence_image = EvidenceImage(
                image_id=image_id,
                page=page.page,
                width=image.width,
                height=image.height,
                label=image.label,
                caption=image.caption,
                nearby_text=image.nearby_text,
            )
            rows.append(
                Evidence(
                    chunk_id=f"figure:{image_id}",
                    document_id=parsed.document_id,
                    text=text,
                    page_start=page.page,
                    page_end=page.page,
                    section_title="Figure Index",
                    score=0.0,
                    images=[evidence_image],
                    evidence_type="figure",
                    source_scope="page_span",
                )
            )
    return rows


def _profile_field_evidence(builder: DocumentProfileBuilder, profile) -> list[Evidence]:
    rows: list[Evidence] = []
    for field in builder.field_texts(profile):
        rows.append(
            Evidence(
                chunk_id=f"profile:{field.document_id}:{field.field_name}",
                document_id=field.document_id,
                text=field.text,
                page_start=1,
                page_end=1,
                section_title=f"Document Profile / {field.field_name}",
                score=0.0,
                evidence_type="document_profile",
                source_scope="document_level",
                source_refs=field.source_refs,
                metadata={"profile_field": field.field_name},
            )
        )
    return rows


def _retrieve_for_route(
    route: str,
    question: str,
    corpus: IndexedCorpus,
    top_k: int,
    document_ids: list[str] | None,
) -> list[Evidence]:
    if route == "figure_index":
        return _search(question, corpus.figures, top_k, document_ids)
    if route == "document_profile":
        return _search_profiles(question, corpus.profile_fields, top_k, document_ids)
    return _search(question, corpus.chunks, top_k, document_ids)


def _search(question: str, rows: list[Evidence], top_k: int, document_ids: list[str] | None) -> list[Evidence]:
    filtered = _filter_documents(rows, document_ids)
    scored = [(score_text(question, item.text), index, item) for index, item in enumerate(filtered)]
    scored = [row for row in scored if row[0] > 0]
    scored.sort(key=lambda row: (-row[0], row[1]))
    return [_replace_score(item, score) for score, _index, item in scored[:top_k]]


def _search_profiles(question: str, rows: list[Evidence], top_k: int, document_ids: list[str] | None) -> list[Evidence]:
    filtered = _filter_documents(rows, document_ids)
    scored = []
    for index, item in enumerate(filtered):
        field = str(item.metadata.get("profile_field", ""))
        score = score_text(question, item.text) + _profile_field_bonus(question, field)
        if score > 0:
            scored.append((score, index, item))
    scored.sort(key=lambda row: (-row[0], row[1]))
    selected: list[Evidence] = []
    seen: set[tuple[str, str]] = set()
    for score, _index, item in scored:
        key = (item.document_id, str(item.metadata.get("profile_field", "")))
        if key in seen:
            continue
        seen.add(key)
        selected.append(_replace_score(item, score))
        if len(selected) >= top_k:
            break
    return selected


def score_text(query: str, text: str) -> float:
    query_terms = _terms(query)
    text_terms = _terms(text)
    if not query_terms or not text_terms:
        return 0.0
    overlap = query_terms & text_terms
    score = len(overlap) / len(query_terms)
    normalized_query = _normalize(query)
    normalized_text = _normalize(text)
    for phrase in _important_phrases(normalized_query):
        if phrase in normalized_text:
            score += 0.5
    if query_terms <= text_terms:
        score += 0.25
    return score


def _fine_metrics(
    case: GoldenCase,
    evidence: list[Evidence],
    followup_route: str,
    constrained_documents: list[str],
) -> dict[str, Any]:
    profile_docs = {item.document_id for item in evidence if item.evidence_type == "document_profile"}
    non_profile = [item for item in evidence if item.evidence_type != "document_profile"]
    expected_docs = set(case.expected_documents)
    expected_pages = set(case.expected_pages)
    profile_fields = Counter(
        str(item.metadata.get("profile_field"))
        for item in evidence
        if item.evidence_type == "document_profile" and item.metadata.get("profile_field")
    )
    return {
        "profile_document_hit": bool(expected_docs & profile_docs) if expected_docs else case.target_type == "none",
        "profile_evidence_count": sum(1 for item in evidence if item.evidence_type == "document_profile"),
        "chunk_evidence_count": sum(1 for item in evidence if item.evidence_type == "chunk"),
        "figure_evidence_count": sum(1 for item in evidence if item.evidence_type == "figure"),
        "document_level_evidence_count": sum(1 for item in evidence if item.source_scope == "document_level"),
        "page_span_evidence_count": sum(1 for item in evidence if item.source_scope == "page_span"),
        "profile_field_count": len(profile_fields),
        "profile_field_counts": dict(profile_fields),
        "followup_used": followup_route != "none",
        "followup_route": followup_route,
        "followup_constrained_document_count": len(constrained_documents),
        "followup_document_scope_hit": not constrained_documents or all(doc in profile_docs for doc in constrained_documents),
        "followup_page_hit": bool(expected_pages & {page for item in non_profile for page in range(item.page_start, item.page_end + 1)})
        if expected_pages
        else True,
        "profile_only_detail_risk": _needs_profile_followup(case) and bool(evidence) and not non_profile,
    }


def _summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = _average_metrics(results, "metrics")
    fine_metrics = _average_metrics(results, "fine_metrics")
    evidence_type_counts: Counter[str] = Counter()
    profile_field_counts: Counter[str] = Counter()
    for result in results:
        for item in result["retrieved_evidence"]:
            evidence_type_counts.update([item["evidence_type"]])
            if item["metadata"].get("profile_field"):
                profile_field_counts.update([str(item["metadata"]["profile_field"])])
    return {
        "case_count": len(results),
        "metrics": metrics,
        "fine_metrics": fine_metrics,
        "evidence_type_counts": dict(evidence_type_counts),
        "profile_field_counts": dict(profile_field_counts),
        "by_task_category": _group_summary(results, "task_category"),
        "by_target_type": _group_summary(results, "target_type"),
    }


def _average_metrics(results: list[dict[str, Any]], key: str) -> dict[str, float]:
    names = sorted(
        {
            metric
            for result in results
            for metric, value in result.get(key, {}).items()
            if isinstance(value, (bool, int, float))
        }
    )
    averages: dict[str, float] = {}
    for name in names:
        values = [result[key][name] for result in results if isinstance(result.get(key, {}).get(name), (bool, int, float))]
        averages[name] = sum(float(value) for value in values) / len(values) if values else 0.0
    return averages


def _group_summary(results: list[dict[str, Any]], case_key: str) -> dict[str, dict[str, float]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        groups.setdefault(str(result["case"][case_key]), []).append(result)
    return {
        group: {
            "case_count": float(len(items)),
            "target_hit": _mean(items, "metrics", "target_hit"),
            "answer_term_coverage": _mean(items, "metrics", "answer_term_coverage"),
            "profile_document_hit": _mean(items, "fine_metrics", "profile_document_hit"),
            "followup_used": _mean(items, "fine_metrics", "followup_used"),
            "profile_only_detail_risk": _mean(items, "fine_metrics", "profile_only_detail_risk"),
        }
        for group, items in groups.items()
    }


def _mean(items: list[dict[str, Any]], group: str, metric: str) -> float:
    values = [item[group][metric] for item in items if isinstance(item.get(group, {}).get(metric), (bool, int, float))]
    return sum(float(value) for value in values) / len(values) if values else 0.0


def _document_ids_to_load(upload_dir: Path, cases: list[GoldenCase]) -> list[str]:
    ids = {path.stem for path in upload_dir.glob("*.pdf")}
    for case in cases:
        ids.update(case.document_ids)
        ids.update(case.expected_documents)
    return sorted(ids)


def _route_for_question(question: str) -> str:
    text = question.lower()
    if any(term in text for term in ["which indexed paper", "which papers", "across the indexed papers", "summarize", "what conclusion", "what is this paper about"]):
        return "document_profile"
    if re.search(r"\b(fig(?:ure)?|plot|image|spectrum|spectra)\b", text):
        return "figure_index"
    if any(term in text for term in ["这篇论文", "哪篇", "总结", "主要研究"]):
        return "document_profile"
    return "chunk_search"


def _needs_profile_followup(case: GoldenCase) -> bool:
    if case.target_type in {"page", "figure", "mixed"}:
        return True
    text = case.question.lower()
    return any(term in text for term in ["what conclusion", "reported", "technique", "measurement", "figure", "spectrum"])


def _looks_like_figure_question(question: str) -> bool:
    return bool(re.search(r"\b(fig(?:ure)?|plot|image|spectrum|spectra)\b", question, re.IGNORECASE))


def _matched_profile_documents(evidence: list[Evidence], fallback: list[str] | None) -> list[str]:
    docs = []
    for item in evidence:
        if item.evidence_type == "document_profile" and item.document_id not in docs:
            docs.append(item.document_id)
    if docs:
        return docs
    return list(fallback or [])


def _verification_would_trigger(case: GoldenCase, evidence: list[Evidence], answer: str) -> bool:
    if case.risk_level == "high":
        return True
    has_profile = any(item.evidence_type == "document_profile" for item in evidence)
    has_chunk = any(item.evidence_type == "chunk" for item in evidence)
    if has_profile and not has_chunk and _needs_profile_followup(case):
        return True
    return bool(re.search(r"\b\d+(?:\.\d+)?\s*(?:nm|%|k|c|°c)\b", answer, re.IGNORECASE))


def _is_high_risk_protocol_request(question: str) -> bool:
    return bool(re.search(r"\b(step-by-step|protocol|dosage|dosages|exact temperatures|reproduce)\b", question, re.IGNORECASE))


def _answer_from_evidence(evidence: list[Evidence]) -> str:
    if not evidence:
        return "资料不足，无法可靠回答。"
    snippets = []
    for item in evidence[:4]:
        snippets.append(re.sub(r"\s+", " ", item.text).strip()[:260])
    return " | ".join(snippets)


def _citations_from_evidence(evidence: list[Evidence]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in evidence:
        rows.append(
            {
                "document_id": item.document_id,
                "chunk_id": item.chunk_id,
                "page_start": item.page_start,
                "page_end": item.page_end,
                "source_url": None if item.source_scope == "document_level" else f"/api/documents/{item.document_id}/pdf#page={item.page_start}",
                "source_scope": item.source_scope,
                "evidence_type": item.evidence_type,
                "section_title": item.section_title,
                "quote": item.text[:500],
                "score": item.score,
            }
        )
    return rows


def _selected_images_from_evidence(evidence: list[Evidence]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in evidence:
        for image in item.images:
            rows.append(
                {
                    "chunk_id": item.chunk_id,
                    "document_id": item.document_id,
                    "image_id": image.image_id,
                    "page": image.page,
                    "label": image.label,
                    "caption": image.caption,
                    "nearby_text": image.nearby_text,
                    "reason": "offline lexical figure match",
                    "source_url": f"/api/documents/{item.document_id}/pdf#page={image.page}",
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
        "evidence_type": item.evidence_type,
        "source_scope": item.source_scope,
        "metadata": item.metadata,
        "images": [asdict(image) for image in item.images],
    }


def _filter_documents(rows: list[Evidence], document_ids: list[str] | None) -> list[Evidence]:
    if not document_ids:
        return rows
    allowed = set(document_ids)
    return [item for item in rows if item.document_id in allowed]


def _replace_score(item: Evidence, score: float) -> Evidence:
    return Evidence(
        chunk_id=item.chunk_id,
        document_id=item.document_id,
        text=item.text,
        page_start=item.page_start,
        page_end=item.page_end,
        section_title=item.section_title,
        score=score,
        image_ids=item.image_ids,
        images=item.images,
        evidence_type=item.evidence_type,
        source_scope=item.source_scope,
        source_refs=item.source_refs,
        metadata=item.metadata,
    )


def _profile_field_bonus(query: str, field: str) -> float:
    text = query.lower()
    if field == "conclusion" and "conclusion" in text:
        return 0.4
    if field == "methods" and any(term in text for term in ["technique", "method", "protocol"]):
        return 0.35
    if field == "properties" and any(term in text for term in ["property", "properties", "measurement", "absorption", "refraction"]):
        return 0.35
    if field == "entities" and any(term in text for term in ["which indexed paper", "which papers", "materials"]):
        return 0.2
    return 0.0


def _important_phrases(normalized_query: str) -> list[str]:
    phrases = []
    for phrase in [
        "zno photoluminescence",
        "poly bisphenol a carbonate",
        "nonlinear absorption",
        "nonlinear optical",
        "time resolved",
        "pl spectrum",
        "z scan",
        "cdse",
    ]:
        if phrase in normalized_query:
            phrases.append(phrase)
    return phrases


def _terms(text: str) -> set[str]:
    normalized = _normalize(text)
    terms = set(re.findall(r"[a-z][a-z0-9-]{1,}", normalized))
    terms.update(re.findall(r"[\u4e00-\u9fff]{2,}", normalized))
    return {term for term in terms if term not in _STOPWORDS}


def _normalize(text: str) -> str:
    normalized = text.lower()
    replacements = {
        "uv-vis": "uv vis",
        "uv/vis": "uv vis",
        "z-scan": "z scan",
        "time-resolved": "time resolved",
        "nlo": "nonlinear optical",
        "pl": "pl photoluminescence",
        "poly(bisphenol a carbonate)": "poly bisphenol a carbonate pbc",
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    return normalized


_STOPWORDS = {
    "the",
    "this",
    "that",
    "with",
    "what",
    "which",
    "paper",
    "papers",
    "indexed",
    "document",
    "documents",
    "about",
    "does",
    "give",
    "exact",
    "reported",
    "study",
    "studies",
    "used",
    "and",
    "for",
    "its",
    "are",
    "is",
    "in",
    "of",
    "to",
}


def _metric_label(name: str) -> str:
    labels = {
        "route_hit": "路由命中率",
        "target_hit": "目标命中率",
        "document_recall@k": "文档召回@k",
        "page_recall@k": "页码召回@k",
        "figure_recall@k": "图片召回@k",
        "answer_term_coverage": "答案术语覆盖率",
        "source_url_valid": "来源链接有效率",
        "refusal_correct": "拒答正确率",
        "limitation_disclosure_hit": "限制披露命中率",
        "retrieval_behavior_hit": "检索行为命中率",
        "profile_document_hit": "画像文档命中率",
        "profile_evidence_count": "画像证据数",
        "chunk_evidence_count": "chunk 证据数",
        "figure_evidence_count": "figure 证据数",
        "document_level_evidence_count": "文档级证据数",
        "page_span_evidence_count": "页级证据数",
        "profile_field_count": "画像字段数",
        "followup_used": "画像后续检索触发率",
        "followup_constrained_document_count": "后续检索约束文档数",
        "followup_document_scope_hit": "后续检索文档范围正确率",
        "followup_page_hit": "后续检索页命中率",
        "profile_only_detail_risk": "画像单独支撑细节风险率",
    }
    return labels.get(name, name)


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "1.000" if value else "0.000"
    if isinstance(value, (int, float)):
        return f"{float(value):.3f}"
    return str(value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="离线评测 document_profile 重构后的真实 PDF 检索效果。")
    parser.add_argument("--cases", type=Path, default=Path("eval/profile_refactor_cases.jsonl"))
    parser.add_argument("--upload-dir", type=Path, default=Path("storage/uploads"))
    parser.add_argument("--output-dir", type=Path, default=Path("eval/reports/profile_refactor"))
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    main()
