from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import Any, Literal


Route = Literal["chunk_search", "figure_index", "document_profile"]
TargetType = Literal["document", "page", "figure", "mixed", "none"]
AnswerType = Literal["fact_qa", "figure_query", "document_summary", "cross_document_filter", "refusal"]
Scope = Literal["current_document", "all_documents"]

ROUTES = {"chunk_search", "figure_index", "document_profile"}
TARGET_TYPES = {"document", "page", "figure", "mixed", "none"}
ANSWER_TYPES = {"fact_qa", "figure_query", "document_summary", "cross_document_filter", "refusal"}
SCOPES = {"current_document", "all_documents"}


@dataclass(frozen=True)
class GoldenCase:
    case_id: str
    question: str
    scope: Scope
    document_ids: list[str]
    expected_route: Route
    target_type: TargetType
    expected_documents: list[str] = field(default_factory=list)
    expected_pages: list[int] = field(default_factory=list)
    expected_figures: list[dict[str, Any] | str] = field(default_factory=list)
    expected_answer_terms: list[str] = field(default_factory=list)
    answer_type: AnswerType = "fact_qa"
    notes: str | None = None


def load_golden_cases(path: Path) -> list[GoldenCase]:
    cases: list[GoldenCase] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            cases.append(parse_golden_case(payload, source=f"{path}:{line_number}"))
    return cases


def parse_golden_case(payload: dict[str, Any], source: str = "<case>") -> GoldenCase:
    required = [
        "case_id",
        "question",
        "scope",
        "document_ids",
        "expected_route",
        "target_type",
        "expected_documents",
        "expected_pages",
        "expected_figures",
        "expected_answer_terms",
        "answer_type",
        "notes",
    ]
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(f"{source}: missing required fields: {', '.join(missing)}")

    case_id = _non_empty_string(payload["case_id"], "case_id", source)
    question = _non_empty_string(payload["question"], "question", source)
    scope = _enum(payload["scope"], SCOPES, "scope", source)
    expected_route = _enum(payload["expected_route"], ROUTES, "expected_route", source)
    target_type = _enum(payload["target_type"], TARGET_TYPES, "target_type", source)
    answer_type = _enum(payload["answer_type"], ANSWER_TYPES, "answer_type", source)
    document_ids = _string_list(payload["document_ids"], "document_ids", source)
    expected_documents = _string_list(payload["expected_documents"], "expected_documents", source)
    expected_pages = _int_list(payload["expected_pages"], "expected_pages", source)
    expected_figures = _figure_list(payload["expected_figures"], source)
    expected_answer_terms = _string_list(payload["expected_answer_terms"], "expected_answer_terms", source)
    notes = payload["notes"]
    if notes is not None and not isinstance(notes, str):
        raise ValueError(f"{source}: notes must be a string or null")

    return GoldenCase(
        case_id=case_id,
        question=question,
        scope=scope,  # type: ignore[arg-type]
        document_ids=document_ids,
        expected_route=expected_route,  # type: ignore[arg-type]
        target_type=target_type,  # type: ignore[arg-type]
        expected_documents=expected_documents,
        expected_pages=expected_pages,
        expected_figures=expected_figures,
        expected_answer_terms=expected_answer_terms,
        answer_type=answer_type,  # type: ignore[arg-type]
        notes=notes,
    )


def _non_empty_string(value: Any, field_name: str, source: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source}: {field_name} must be a non-empty string")
    return value


def _enum(value: Any, allowed: set[str], field_name: str, source: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{source}: {field_name} must be one of {sorted(allowed)}")
    return value


def _string_list(value: Any, field_name: str, source: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{source}: {field_name} must be a list of strings")
    return value


def _int_list(value: Any, field_name: str, source: str) -> list[int]:
    if not isinstance(value, list) or not all(isinstance(item, int) and item > 0 for item in value):
        raise ValueError(f"{source}: {field_name} must be a list of positive integers")
    return value


def _figure_list(value: Any, source: str) -> list[dict[str, Any] | str]:
    if not isinstance(value, list):
        raise ValueError(f"{source}: expected_figures must be a list")
    figures: list[dict[str, Any] | str] = []
    for index, item in enumerate(value):
        if isinstance(item, str):
            figures.append(item)
            continue
        if not isinstance(item, dict):
            raise ValueError(f"{source}: expected_figures[{index}] must be a string or object")
        allowed = {"image_id", "figure_id", "page", "caption_terms"}
        unknown = set(item) - allowed
        if unknown:
            raise ValueError(f"{source}: expected_figures[{index}] has unknown fields: {sorted(unknown)}")
        if "page" in item and (not isinstance(item["page"], int) or item["page"] <= 0):
            raise ValueError(f"{source}: expected_figures[{index}].page must be a positive integer")
        if "caption_terms" in item and not (
            isinstance(item["caption_terms"], list) and all(isinstance(term, str) for term in item["caption_terms"])
        ):
            raise ValueError(f"{source}: expected_figures[{index}].caption_terms must be a list of strings")
        figures.append(item)
    return figures
