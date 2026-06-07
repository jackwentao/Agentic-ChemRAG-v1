import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.config import get_settings
from backend.app.database import build_database_engine
from backend.app.evaluation.cases import parse_golden_case
from backend.app.evaluation.ragas_adapter import _build_ragas_embeddings, _build_ragas_llm
from backend.app.parsing.parser import PyMuPDFParser
from backend.app.providers.embeddings import build_embedding_provider
from backend.app.repositories.postgres import PgVectorRepository


@dataclass(frozen=True)
class PdfPageSource:
    document_id: str
    file_hash: str
    filename: str
    page: int
    text: str


@dataclass(frozen=True)
class PdfFileInfo:
    path: str
    sha256: str


def main() -> None:
    args = _parse_args()
    if args.refresh == "off" and args.output.exists():
        print(f"复用已有 Ragas 测试集：{args.output}")
        print("如需主动新增一批问题答案，请使用 --refresh on；如需追加而不是覆盖，请同时使用 --append on。")
        if args.run_evaluation == "on":
            run_evaluation(args)
        return

    pdf_paths = discover_pdfs(args.pdf_dir, args.pdf_glob, args.limit_pdfs)
    if not pdf_paths:
        raise FileNotFoundError(f"没有在 {args.pdf_dir} 中找到匹配 {args.pdf_glob} 的 PDF。")

    settings = get_settings()
    embedding_provider = build_embedding_provider(settings)
    pdf_infos = collect_pdf_file_info(pdf_paths)
    cache_signature = build_cache_signature(pdf_infos, args, settings)
    page_sources, langchain_docs = load_pdf_documents_with_cache(
        pdf_paths,
        pdf_infos=pdf_infos,
        document_id_mode=args.document_id_mode,
        owner_id=args.owner_id,
        db_lookup=args.db_lookup == "on",
        page_cache=args.page_cache,
        cache_manifest=args.cache_manifest,
        cache_signature=cache_signature,
    )
    knowledge_graph = load_or_build_knowledge_graph(
        langchain_docs,
        settings=settings,
        embedding_provider=embedding_provider,
        kg_cache=args.kg_cache,
        cache_manifest=args.cache_manifest,
        cache_signature=cache_signature,
        force_rebuild=args.force_rebuild_kg == "on",
    )
    testset = generate_ragas_testset_from_knowledge_graph(
        knowledge_graph,
        testset_size=args.testset_size,
        settings=settings,
        embedding_provider=embedding_provider,
        raise_exceptions=args.raise_exceptions == "on",
    )
    rows = rows_from_testset(testset)
    existing_cases = load_jsonl(args.output) if args.append == "on" and args.output.exists() else []
    cases = build_golden_cases(
        rows,
        page_sources,
        scope=args.scope,
        case_prefix=args.case_prefix,
        max_answer_terms=args.max_answer_terms,
        start_index=next_case_index(existing_cases, args.case_prefix),
    )
    output_cases = [*existing_cases, *cases] if args.append == "on" else cases
    write_jsonl(args.output, output_cases)
    write_generation_manifest(args.manifest, pdf_infos, page_sources, rows, args, appended_case_count=len(existing_cases))
    print(f"已生成 {len(cases)} 条新的 Ragas 测试 case：{args.output}")
    if existing_cases:
        print(f"已追加到现有测试集；当前总 case 数：{len(output_cases)}")
    print(f"已写入生成清单：{args.manifest}")

    if args.run_evaluation == "on":
        run_evaluation(args)


def discover_pdfs(pdf_dir: Path, pattern: str, limit: int | None = None) -> list[Path]:
    paths = sorted(path for path in pdf_dir.glob(pattern) if path.is_file() and path.suffix.lower() == ".pdf")
    return paths[:limit] if limit else paths


def collect_pdf_file_info(pdf_paths: list[Path]) -> list[PdfFileInfo]:
    return [PdfFileInfo(path=str(path), sha256=sha256(path.read_bytes()).hexdigest()) for path in pdf_paths]


def load_pdf_documents_with_cache(
    pdf_paths: list[Path],
    *,
    pdf_infos: list[PdfFileInfo],
    document_id_mode: str,
    owner_id: str,
    db_lookup: bool,
    page_cache: Path,
    cache_manifest: Path,
    cache_signature: dict[str, Any],
) -> tuple[list[PdfPageSource], list[Any]]:
    if cache_is_valid(cache_manifest, cache_signature) and page_cache.exists():
        page_sources = read_page_sources(page_cache)
        return page_sources, langchain_docs_from_page_sources(page_sources)

    file_hashes = {info.path: info.sha256 for info in pdf_infos}
    page_sources, langchain_docs = load_pdf_documents(
        pdf_paths,
        document_id_mode=document_id_mode,
        owner_id=owner_id,
        db_lookup=db_lookup,
        file_hashes=file_hashes,
    )
    write_page_sources(page_cache, page_sources)
    write_cache_manifest(cache_manifest, cache_signature)
    return page_sources, langchain_docs


def load_pdf_documents(
    pdf_paths: list[Path],
    *,
    document_id_mode: str,
    owner_id: str,
    db_lookup: bool,
    file_hashes: dict[str, str] | None = None,
) -> tuple[list[PdfPageSource], list[Any]]:
    try:
        from langchain_core.documents import Document
    except ImportError as exc:
        raise RuntimeError("Ragas 测试集生成需要安装可选依赖：python -m pip install -e \".[eval]\"") from exc

    parser = PyMuPDFParser(max_extracted_images=0)
    page_sources: list[PdfPageSource] = []
    langchain_docs: list[Any] = []
    for pdf_path in pdf_paths:
        file_hash = (file_hashes or {}).get(str(pdf_path)) or sha256(pdf_path.read_bytes()).hexdigest()
        document_id = resolve_document_id(
            pdf_path,
            file_hash=file_hash,
            document_id_mode=document_id_mode,
            owner_id=owner_id,
            db_lookup=db_lookup,
        )
        parsed = parser.parse(document_id, pdf_path)
        for page in parsed.pages:
            if not page.text.strip():
                continue
            source = PdfPageSource(
                document_id=document_id,
                file_hash=file_hash,
                filename=pdf_path.name,
                page=page.page,
                text=page.text,
            )
            page_sources.append(source)
            langchain_docs.append(
                Document(
                    page_content=page.text,
                    metadata={
                        "source": str(pdf_path),
                        "filename": pdf_path.name,
                        "document_id": document_id,
                        "file_hash": file_hash,
                        "page": page.page,
                    },
                )
            )
    return page_sources, langchain_docs


def langchain_docs_from_page_sources(page_sources: list[PdfPageSource]) -> list[Any]:
    try:
        from langchain_core.documents import Document
    except ImportError as exc:
        raise RuntimeError("Ragas 测试集生成需要安装可选依赖：python -m pip install -e \".[eval]\"") from exc

    return [
        Document(
            page_content=source.text,
            metadata={
                "filename": source.filename,
                "document_id": source.document_id,
                "file_hash": source.file_hash,
                "page": source.page,
            },
        )
        for source in page_sources
    ]


def resolve_document_id(
    pdf_path: Path,
    *,
    file_hash: str,
    document_id_mode: str,
    owner_id: str,
    db_lookup: bool,
) -> str:
    if document_id_mode == "stem":
        return _safe_id(pdf_path.stem)
    if document_id_mode == "hash":
        return file_hash[:32]
    if db_lookup:
        document_id = _lookup_document_id_by_hash(file_hash, owner_id)
        if document_id:
            return document_id
    return file_hash[:32]


def load_or_build_knowledge_graph(
    documents: list[Any],
    *,
    settings: Any,
    embedding_provider: Any,
    kg_cache: Path,
    cache_manifest: Path,
    cache_signature: dict[str, Any],
    force_rebuild: bool,
) -> Any:
    try:
        from ragas.testset.graph import KnowledgeGraph, Node, NodeType
        from ragas.testset.transforms import apply_transforms, default_transforms
        from ragas.run_config import RunConfig
    except ImportError as exc:
        raise RuntimeError("Ragas KnowledgeGraph 缓存需要安装可选依赖：python -m pip install -e \".[eval]\"") from exc

    if not force_rebuild and cache_is_valid(cache_manifest, cache_signature) and kg_cache.exists():
        return KnowledgeGraph.load(kg_cache)
    if not documents:
        raise ValueError("没有可用于构建 KnowledgeGraph 的 PDF 文本。")

    llm = _build_ragas_llm(settings)
    embeddings = _build_ragas_embeddings(embedding_provider)
    transforms = default_transforms(documents=list(documents), llm=llm, embedding_model=embeddings)
    nodes = [
        Node(
            type=NodeType.DOCUMENT,
            properties={
                "page_content": document.page_content,
                "document_metadata": document.metadata,
            },
        )
        for document in documents
    ]
    knowledge_graph = KnowledgeGraph(nodes=nodes)
    apply_transforms(knowledge_graph, transforms, run_config=RunConfig())
    kg_cache.parent.mkdir(parents=True, exist_ok=True)
    knowledge_graph.save(kg_cache)
    write_cache_manifest(cache_manifest, cache_signature)
    return knowledge_graph


def generate_ragas_testset_from_knowledge_graph(
    knowledge_graph: Any,
    *,
    testset_size: int,
    settings: Any,
    embedding_provider: Any,
    raise_exceptions: bool,
) -> Any:
    try:
        from ragas.testset import TestsetGenerator
    except ImportError as exc:
        raise RuntimeError("Ragas 测试集生成需要安装可选依赖：python -m pip install -e \".[eval]\"") from exc

    generator = TestsetGenerator(
        llm=_build_ragas_llm(settings),
        embedding_model=_build_ragas_embeddings(embedding_provider),
        knowledge_graph=knowledge_graph,
    )
    return generator.generate(
        testset_size=testset_size,
        raise_exceptions=raise_exceptions,
    )


def rows_from_testset(testset: Any) -> list[dict[str, Any]]:
    if hasattr(testset, "to_pandas"):
        return testset.to_pandas().to_dict("records")
    if hasattr(testset, "to_list"):
        rows = testset.to_list()
        return [dict(row) if isinstance(row, dict) else _object_to_dict(row) for row in rows]
    if isinstance(testset, list):
        return [dict(row) if isinstance(row, dict) else _object_to_dict(row) for row in testset]
    raise TypeError("无法解析 Ragas testset，请确认当前 Ragas 版本仍支持 to_pandas() 或 to_list()。")


def build_golden_cases(
    rows: list[dict[str, Any]],
    page_sources: list[PdfPageSource],
    *,
    scope: str,
    case_prefix: str,
    max_answer_terms: int,
    start_index: int = 1,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=start_index):
        question = _first_text(row, "user_input", "question", "query")
        reference = _first_text(row, "reference", "answer", "ground_truth")
        reference_contexts = _list_text(row.get("reference_contexts") or row.get("contexts") or row.get("retrieved_contexts"))
        matches = match_reference_contexts(reference_contexts, page_sources)
        expected_documents = list(dict.fromkeys(match.document_id for match in matches))
        expected_pages = sorted({match.page for match in matches})
        document_ids = expected_documents if scope == "current_document" and len(expected_documents) == 1 else []
        case = {
            "case_id": f"{case_prefix}-{index:04d}",
            "question": question,
            "scope": scope,
            "document_ids": document_ids,
            "expected_route": "chunk_search",
            "target_type": _target_type(expected_documents, expected_pages),
            "expected_documents": expected_documents,
            "expected_pages": expected_pages,
            "expected_figures": [],
            "expected_answer_terms": extract_answer_terms(reference, max_terms=max_answer_terms),
            "answer_type": "fact_qa",
            "task_category": "simple_fact",
            "risk_level": "low",
            "expected_aspects": [],
            "required_disclosures": [],
            "expected_retrieval_behavior": "single_round_expected",
            "max_acceptable_rounds": 1,
            "notes": "由 Ragas testset generation 基于项目 PDF 自动生成。",
            "reference_answer": reference,
            "reference_contexts": reference_contexts,
            "ragas_tags": ["ragas_generated", "pdf", "chemistry"],
        }
        parse_golden_case(case, source=case["case_id"])
        cases.append(case)
    return cases


def match_reference_contexts(reference_contexts: list[str], page_sources: list[PdfPageSource]) -> list[PdfPageSource]:
    matches: list[PdfPageSource] = []
    seen: set[tuple[str, int]] = set()
    for context in reference_contexts:
        match = best_page_match(context, page_sources)
        if match is None:
            continue
        key = (match.document_id, match.page)
        if key in seen:
            continue
        seen.add(key)
        matches.append(match)
    return matches


def best_page_match(context: str, page_sources: list[PdfPageSource]) -> PdfPageSource | None:
    normalized_context = _normalize_text(context)
    if not normalized_context:
        return None
    best_score = 0.0
    best_source: PdfPageSource | None = None
    context_terms = _terms(normalized_context)
    for source in page_sources:
        normalized_page = _normalize_text(source.text)
        if normalized_context in normalized_page:
            score = 1.0
        else:
            page_terms = _terms(normalized_page)
            score = _jaccard(context_terms, page_terms)
        if score > best_score:
            best_score = score
            best_source = source
    return best_source if best_score >= 0.08 else None


def extract_answer_terms(reference: str, *, max_terms: int = 5) -> list[str]:
    if max_terms <= 0:
        return []
    candidates = re.findall(r"\b[A-Z][A-Za-z0-9/-]{1,}\b|\b[A-Za-z][A-Za-z0-9/-]{4,}\b|[\u4e00-\u9fff]{2,}", reference)
    stopwords = {
        "about",
        "after",
        "based",
        "between",
        "could",
        "found",
        "paper",
        "report",
        "reports",
        "reported",
        "shows",
        "study",
        "their",
        "there",
        "the",
        "these",
        "this",
        "using",
        "were",
        "which",
    }
    terms: list[str] = []
    for candidate in candidates:
        term = candidate.strip(".,;:()[]{}")
        if not term or term.lower() in stopwords:
            continue
        if term.lower() in {item.lower() for item in terms}:
            continue
        terms.append(term)
        if len(terms) >= max_terms:
            break
    return terms


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def next_case_index(existing_cases: list[dict[str, Any]], case_prefix: str) -> int:
    max_index = 0
    pattern = re.compile(rf"^{re.escape(case_prefix)}-(\d+)$")
    for case in existing_cases:
        match = pattern.match(str(case.get("case_id", "")))
        if match:
            max_index = max(max_index, int(match.group(1)))
    return max_index + 1


def write_generation_manifest(
    path: Path,
    pdf_infos: list[PdfFileInfo],
    page_sources: list[PdfPageSource],
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    appended_case_count: int = 0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pdfs": [info.__dict__ for info in pdf_infos],
        "page_count": len(page_sources),
        "generated_row_count": len(rows),
        "appended_case_count": appended_case_count,
        "output": str(args.output),
        "scope": args.scope,
        "testset_size": args.testset_size,
        "document_id_mode": args.document_id_mode,
        "kg_cache": str(args.kg_cache),
        "page_cache": str(args.page_cache),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_page_sources(path: Path) -> list[PdfPageSource]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [PdfPageSource(**item) for item in payload.get("pages", [])]


def write_page_sources(path: Path, page_sources: list[PdfPageSource]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"pages": [source.__dict__ for source in page_sources]}
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def build_cache_signature(pdf_infos: list[PdfFileInfo], args: argparse.Namespace, settings: Any) -> dict[str, Any]:
    return {
        "version": 1,
        "pdfs": [info.__dict__ for info in pdf_infos],
        "document_id_mode": args.document_id_mode,
        "owner_id": args.owner_id,
        "db_lookup": args.db_lookup,
        "embedding_provider": getattr(settings, "embedding_provider", ""),
        "embedding_model": getattr(settings, "embedding_model", ""),
        "deepseek_model": getattr(settings, "deepseek_model", ""),
    }


def cache_is_valid(path: Path, expected_signature: dict[str, Any]) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return payload.get("signature") == expected_signature


def write_cache_manifest(path: Path, signature: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"signature": signature}, ensure_ascii=False, indent=2), encoding="utf-8")


def run_evaluation(args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        str(Path(__file__).resolve().with_name("evaluate_rag.py")),
        "--cases",
        str(args.output),
        "--output-dir",
        str(args.evaluation_output_dir),
        "--answer-provider",
        args.answer_provider,
        "--assessment-provider",
        args.assessment_provider,
        "--ragas",
        args.ragas,
        "--ragas-metrics",
        args.ragas_metrics,
        "--scope",
        args.evaluation_scope,
    ]
    if args.ragas_sample_limit is not None:
        command.extend(["--ragas-sample-limit", str(args.ragas_sample_limit)])
    subprocess.run(command, check=True)


def _lookup_document_id_by_hash(file_hash: str, owner_id: str) -> str | None:
    try:
        repository = PgVectorRepository(build_database_engine(get_settings().database_url))
        row = repository.find_document_by_hash(owner_id, file_hash)
    except Exception:
        return None
    return str(row["document_id"]) if row else None


def _target_type(expected_documents: list[str], expected_pages: list[int]) -> str:
    if expected_documents and expected_pages:
        return "mixed"
    if expected_pages:
        return "page"
    if expected_documents:
        return "document"
    return "page"


def _first_text(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(f"Ragas 输出缺少文本字段：{keys}")


def _list_text(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)] if str(value).strip() else []


def _object_to_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    raise TypeError(f"无法转换 Ragas 样本：{type(value)!r}")


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _terms(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z][a-zA-Z0-9/-]{2,}|[\u4e00-\u9fff]{2,}", text.lower()))


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _safe_id(text: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-._")
    return safe[:80] or "document"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 Ragas 基于项目 PDF 自动生成测试问答，并可选运行现有 RAG 评测。")
    parser.add_argument("--pdf-dir", type=Path, default=Path("test_pdf"))
    parser.add_argument("--pdf-glob", default="*.pdf")
    parser.add_argument("--limit-pdfs", type=int, default=None)
    parser.add_argument("--testset-size", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("eval/ragas_generated_cases.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("eval/ragas_generated_manifest.json"))
    parser.add_argument("--refresh", choices=["on", "off"], default="off")
    parser.add_argument("--append", choices=["on", "off"], default="off")
    parser.add_argument("--kg-cache", type=Path, default=Path("eval/ragas_cache/knowledge_graph.json"))
    parser.add_argument("--page-cache", type=Path, default=Path("eval/ragas_cache/page_sources.json"))
    parser.add_argument("--cache-manifest", type=Path, default=Path("eval/ragas_cache/cache_manifest.json"))
    parser.add_argument("--force-rebuild-kg", choices=["on", "off"], default="off")
    parser.add_argument("--scope", choices=["all_documents", "current_document"], default="all_documents")
    parser.add_argument("--case-prefix", default="ragas-pdf")
    parser.add_argument("--max-answer-terms", type=int, default=5)
    parser.add_argument("--owner-id", default="demo")
    parser.add_argument("--document-id-mode", choices=["db_or_hash", "hash", "stem"], default="db_or_hash")
    parser.add_argument("--db-lookup", choices=["on", "off"], default="on")
    parser.add_argument("--raise-exceptions", choices=["on", "off"], default="off")
    parser.add_argument("--run-evaluation", choices=["on", "off"], default="off")
    parser.add_argument("--evaluation-output-dir", type=Path, default=Path("eval/reports/ragas_generated"))
    parser.add_argument("--evaluation-scope", choices=["from_case", "all_documents", "current_document"], default="from_case")
    parser.add_argument("--answer-provider", choices=["configured", "fake"], default="configured")
    parser.add_argument("--assessment-provider", choices=["configured", "fake"], default="configured")
    parser.add_argument("--ragas", choices=["on", "off"], default="on")
    parser.add_argument("--ragas-metrics", choices=["basic", "retrieval", "all"], default="all")
    parser.add_argument("--ragas-sample-limit", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
