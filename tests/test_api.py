from io import BytesIO

from fastapi.testclient import TestClient
from fastapi import HTTPException, Request
import pytest

import backend.app.api.dependencies as dependencies
import backend.app.main as main_module
from backend.app.api.routers.maintenance import task_status
from backend.app.domain.documents import Evidence, EvidenceImage
from backend.app.main import app
from backend.app.providers.embeddings import FakeEmbeddingProvider
from backend.app.providers.llm import FakeLlmProvider
from backend.app.routing.query_router import QueryRouter, RouterWarmupStatus
from backend.app.schemas import DocumentUploadResponse
from backend.app.services.readiness import ReadinessCheck, ReadinessReport


@pytest.fixture(autouse=True)
def clear_cached_app_dependencies():
    _clear_dependency_caches()
    yield
    _clear_dependency_caches()


def _clear_dependency_caches():
    app.dependency_overrides.clear()
    for name in ("get_repository", "get_object_storage", "get_embedding_provider", "get_llm_provider", "get_query_router", "get_reranker_provider"):
        dependency = getattr(dependencies, name)
        if hasattr(dependency, "cache_clear"):
            dependency.cache_clear()


def test_health_endpoint():
    client = TestClient(app)

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readiness_endpoint_reports_complete_question_answer_capability():
    class FakeReadinessService:
        def inspect(self):
            return ReadinessReport(
                status="ready",
                complete_qa_online=True,
                can_serve_degraded=True,
                checks={"llm": ReadinessCheck(status="ready", required=True, detail="可用。")},
            )

    app.dependency_overrides[dependencies.get_readiness_service] = FakeReadinessService
    response = TestClient(app).get("/api/ready")

    assert response.status_code == 200
    assert response.json()["complete_qa_online"] is True
    assert response.json()["checks"]["llm"]["status"] == "ready"


def test_readiness_endpoint_returns_503_for_degraded_question_answer_capability():
    class FakeReadinessService:
        def inspect(self):
            return ReadinessReport(
                status="degraded",
                complete_qa_online=False,
                can_serve_degraded=True,
                checks={"llm": ReadinessCheck(status="failed", required=True, detail="探测失败：ConnectionError")},
            )

    app.dependency_overrides[dependencies.get_readiness_service] = FakeReadinessService
    response = TestClient(app).get("/api/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "degraded"
    assert response.json()["can_serve_degraded"] is True


def test_batch_upload_accepts_valid_pdfs_and_rejects_invalid_files():
    class FakeDocumentService:
        def store_uploaded_pdf(self, filename, content):
            return DocumentUploadResponse(document_id=f"doc-{filename}", task_id="task-1", status="pending")

    app.dependency_overrides[dependencies.get_document_service] = FakeDocumentService
    client = TestClient(app)

    response = client.post(
        "/api/documents/batch",
        files=[
            ("files", ("paper.pdf", b"pdf bytes", "application/pdf")),
            ("files", ("notes.txt", b"text bytes", "text/plain")),
        ],
    )

    assert response.status_code == 200
    assert response.json()["results"] == [
        {
            "filename": "paper.pdf",
            "document_id": "doc-paper.pdf",
            "task_id": "task-1",
            "status": "pending",
            "reused": False,
            "error": None,
        },
        {
            "filename": "notes.txt",
            "document_id": None,
            "task_id": None,
            "status": "rejected",
            "reused": False,
            "error": "Only PDF files are supported.",
        },
    ]


def test_task_status_returns_failure_error():
    class FailedResult:
        status = "FAILURE"
        result = RuntimeError("parse failed")

        def failed(self):
            return True

        def successful(self):
            return False

    class FakeTask:
        def AsyncResult(self, task_id):
            return FailedResult()

    response = task_status("task-1", FakeTask())

    assert response.status == "failure"
    assert response.error == "parse failed"


def test_task_status_raises_503_when_result_backend_is_unavailable():
    class FakeTask:
        def AsyncResult(self, task_id):
            raise RuntimeError("redis unavailable")

    try:
        task_status("task-1", FakeTask())
    except HTTPException as exc:
        assert exc.status_code == 503
        assert "Task status backend is unavailable" in exc.detail
    else:
        raise AssertionError("Expected HTTPException")


def test_mark_stale_indexing_failed_uses_demo_owner():
    class FakeRepository:
        def __init__(self, database_url):
            self.database_url = database_url

        def mark_stale_indexing_documents_failed(self, owner_id: str, stale_after_minutes: int):
            assert owner_id == "demo"
            assert stale_after_minutes == 30
            return ["doc1", "doc2"]

    app.dependency_overrides[dependencies.get_repository] = lambda: FakeRepository("test-database")
    client = TestClient(app)

    response = client.post("/api/maintenance/indexing/mark-stale-failed?timeout_minutes=30")

    assert response.status_code == 200
    assert response.json() == {"updated": 2, "document_ids": ["doc1", "doc2"]}


def test_pdf_asset_streams_object_storage_content():
    class FakeRepository:
        def get_document_asset(self, document_id: str):
            assert document_id == "doc1"
            return {"filename": "paper.pdf", "pdf_object_key": "documents/doc1.pdf"}

    class FakeStorage:
        def exists(self, key: str):
            assert key == "documents/doc1.pdf"
            return True

        def open(self, key: str):
            return BytesIO(b"%PDF stored")

    app.dependency_overrides[dependencies.get_repository] = FakeRepository
    app.dependency_overrides[dependencies.get_object_storage] = FakeStorage
    response = TestClient(app).get("/api/documents/doc1/pdf")

    assert response.status_code == 200
    assert response.content == b"%PDF stored"
    assert response.headers["content-type"] == "application/pdf"
    assert "paper.pdf" in response.headers["content-disposition"]


def test_image_asset_streams_object_storage_content():
    class FakeRepository:
        def get_image_asset(self, image_id: str):
            assert image_id == "img1"
            return {"path": "images/img1.png", "extension": "png"}

    class FakeStorage:
        def exists(self, key: str):
            assert key == "images/img1.png"
            return True

        def open(self, key: str):
            return BytesIO(b"png bytes")

    app.dependency_overrides[dependencies.get_repository] = FakeRepository
    app.dependency_overrides[dependencies.get_object_storage] = FakeStorage
    response = TestClient(app).get("/api/images/img1")

    assert response.status_code == 200
    assert response.content == b"png bytes"
    assert response.headers["content-type"] == "image/png"


def test_repository_dependency_uses_fastapi_managed_instance():
    repository = object()
    app.state.repository = repository
    request = Request({"type": "http", "app": app})

    try:
        assert dependencies.get_repository(request) is repository
    finally:
        del app.state.repository


def test_chat_response_includes_selected_citation_images():
    embedding_provider = FakeEmbeddingProvider(dimensions=8)

    class FakeRepository:
        def __init__(self, database_url):
            self.database_url = database_url

        def search(self, query_embedding, top_k=5, document_ids=None):
            return [
                Evidence(
                    chunk_id="c1",
                    document_id="doc1",
                    text="The SEM image shows porous nanosheets.",
                    page_start=4,
                    page_end=4,
                    section_title="Results",
                    score=0.9,
                    image_ids=["img1"],
                    images=[
                        EvidenceImage(
                            image_id="img1",
                            page=4,
                            width=320,
                            height=240,
                            caption="Figure 1. SEM image of porous nanosheets.",
                        )
                    ],
                )
            ]

    app.dependency_overrides[dependencies.get_repository] = lambda: FakeRepository("test-database")
    app.dependency_overrides[dependencies.get_embedding_provider] = lambda: embedding_provider
    app.dependency_overrides[dependencies.get_llm_provider] = lambda: FakeLlmProvider()
    app.dependency_overrides[dependencies.get_query_router] = lambda: QueryRouter(embedding_provider, cache_enabled=False)
    app.dependency_overrides[dependencies.get_reranker_provider] = lambda: None
    client = TestClient(app)

    response = client.post(
        "/api/chat",
        json={"question": "What morphology is shown?", "scope": "current_document", "document_ids": ["doc1"]},
    )

    assert response.status_code == 200
    payload = response.json()
    citation = payload["citations"][0]
    assert citation["image_ids"] == ["img1"]
    assert citation["source_url"] == "/api/documents/doc1/pdf#page=4"
    assert citation["images"][0] == {
        "image_id": "img1",
        "url": "/api/images/img1",
        "source_url": "/api/documents/doc1/pdf#page=4",
        "page": 4,
        "width": 320,
        "height": 240,
        "label": None,
        "caption": "Figure 1. SEM image of porous nanosheets.",
        "reason": "Selected by fake provider from candidate evidence images.",
    }
    assert payload["diagnostics"]["route"]["selected_route"] == "chunk_search"
    assert payload["diagnostics"]["recall"]["candidate_count"] == 1
    assert payload["limitations"] == []
    assert payload["diagnostics"]["verifier"]["sufficient"] is True
    assert payload["diagnostics"]["verifier"]["answerable"] is True
    assert payload["diagnostics"]["verifier"]["verified_evidence_count"] == 1
    assert payload["diagnostics"]["retrieval_round"] == 1
    assert payload["diagnostics"]["max_retrieval_rounds"] == 1
    assert payload["diagnostics"]["task"]["task_type"] == "simple_fact"
    assert payload["diagnostics"]["stop_reason"] == "sufficient_evidence"
    assert payload["diagnostics"]["retrieval_rounds"][0]["new_evidence_count"] == 1
    assert payload["diagnostics"]["assessment_rounds"][0]["sufficient"] is True
    assert payload["diagnostics"]["llm_call_counts"] == {
        "evidence_assessment": 0,
        "answer_generation": 0,
        "answer_verification": 0,
        "answer_revision": 0,
        "image_selection": 0,
    }
    assert payload["diagnostics"]["llm"]["proxy_hit"] is True
    assert payload["diagnostics"]["image_selection"]["used_llm"] is False


def test_chat_response_exposes_evidence_limitations():
    class FakeRepository:
        pass

    class LimitedGraph:
        def invoke(self, question, document_ids=None, scope=None):
            return {
                "answer": "检索到的证据不足以可靠回答该问题。",
                "verified_evidence": [],
                "selected_images": {},
                "limitations": ["资料缺口：高温水相稳定性"],
                "warnings": ["检索证据不足以回答问题。"],
            }

    app.dependency_overrides[dependencies.get_repository] = FakeRepository
    app.dependency_overrides[dependencies.get_rag_graph] = LimitedGraph
    response = TestClient(app).post(
        "/api/chat",
        json={"question": "高温水相中稳定吗？", "scope": "current_document", "document_ids": ["doc1"]},
    )

    assert response.status_code == 200
    assert response.json()["limitations"] == ["资料缺口：高温水相稳定性"]


def test_startup_warmup_uses_cached_query_router(monkeypatch):
    calls = []

    class FakeRouter:
        def warmup(self):
            calls.append("warmup")
            return RouterWarmupStatus(cache_enabled=True, status="disk_hit", latency_ms=1.0)

    monkeypatch.setattr(main_module.settings, "router_warmup_on_startup", True)
    monkeypatch.setattr(main_module, "get_query_router", lambda: FakeRouter())

    main_module.warmup_router_on_startup()

    assert calls == ["warmup"]


def test_startup_warmup_failure_does_not_block_startup(monkeypatch):
    class FakeRouter:
        def warmup(self):
            raise RuntimeError("cache unavailable")

    monkeypatch.setattr(main_module.settings, "router_warmup_on_startup", True)
    monkeypatch.setattr(main_module, "get_query_router", lambda: FakeRouter())

    main_module.warmup_router_on_startup()


def test_startup_reranker_warmup_uses_cached_provider(monkeypatch):
    calls = []

    class FakeReranker:
        def warmup(self):
            calls.append("warmup")

    monkeypatch.setattr(main_module.settings, "reranker_enabled", True)
    monkeypatch.setattr(main_module.settings, "reranker_warmup_on_startup", True)
    monkeypatch.setattr(main_module, "get_reranker_provider", lambda: FakeReranker())

    main_module.warmup_reranker_on_startup()

    assert calls == ["warmup"]


def test_startup_reranker_warmup_failure_does_not_block_startup(monkeypatch):
    class FakeReranker:
        def warmup(self):
            raise RuntimeError("reranker unavailable")

    monkeypatch.setattr(main_module.settings, "reranker_enabled", True)
    monkeypatch.setattr(main_module.settings, "reranker_warmup_on_startup", True)
    monkeypatch.setattr(main_module, "get_reranker_provider", lambda: FakeReranker())

    main_module.warmup_reranker_on_startup()


def test_lifespan_prepares_database_and_manages_repository(monkeypatch):
    events = []

    class FakeEngine:
        def dispose(self):
            events.append("dispose")

    fake_engine = FakeEngine()

    monkeypatch.setattr(main_module, "build_database_engine", lambda database_url: fake_engine)
    monkeypatch.setattr(main_module, "prepare_database", lambda engine: events.append(("prepare", engine)))
    monkeypatch.setattr(main_module, "warmup_router_on_startup", lambda: None)
    monkeypatch.setattr(main_module, "warmup_reranker_on_startup", lambda: None)

    with TestClient(app):
        assert isinstance(app.state.repository, main_module.PgVectorRepository)
        assert app.state.repository.engine is fake_engine
        assert events == [("prepare", fake_engine)]

    assert not hasattr(app.state, "repository")
    assert events == [("prepare", fake_engine), "dispose"]
