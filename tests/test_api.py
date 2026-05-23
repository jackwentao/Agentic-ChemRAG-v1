from fastapi.testclient import TestClient
from fastapi import HTTPException
import pytest

import backend.app.main as main_module
from backend.app.domain.documents import Evidence, EvidenceImage
from backend.app.main import app, task_status
from backend.app.providers.embeddings import FakeEmbeddingProvider
from backend.app.providers.llm import FakeLlmProvider
from backend.app.routing.query_router import QueryRouter, RouterWarmupStatus
from backend.app.schemas import DocumentUploadResponse


@pytest.fixture(autouse=True)
def clear_cached_app_dependencies():
    _clear_dependency_caches()
    yield
    _clear_dependency_caches()


def _clear_dependency_caches():
    for name in ("_get_embedding_provider", "_get_llm_provider", "_get_query_router", "_get_reranker_provider"):
        dependency = getattr(main_module, name)
        if hasattr(dependency, "cache_clear"):
            dependency.cache_clear()


def test_health_endpoint():
    client = TestClient(app)

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_batch_upload_accepts_valid_pdfs_and_rejects_invalid_files(monkeypatch):
    def fake_store(filename, content, repository):
        return DocumentUploadResponse(document_id=f"doc-{filename}", task_id="task-1", status="pending")

    monkeypatch.setattr(main_module, "_store_uploaded_pdf", fake_store)
    monkeypatch.setattr(main_module, "PgVectorRepository", lambda database_url: object())
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


def test_task_status_returns_failure_error(monkeypatch):
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

    monkeypatch.setattr(main_module, "ingest_document", FakeTask())

    response = task_status("task-1")

    assert response.status == "failure"
    assert response.error == "parse failed"


def test_task_status_raises_503_when_result_backend_is_unavailable(monkeypatch):
    class FakeTask:
        def AsyncResult(self, task_id):
            raise RuntimeError("redis unavailable")

    monkeypatch.setattr(main_module, "ingest_document", FakeTask())

    try:
        task_status("task-1")
    except HTTPException as exc:
        assert exc.status_code == 503
        assert "Task status backend is unavailable" in exc.detail
    else:
        raise AssertionError("Expected HTTPException")


def test_mark_stale_indexing_failed_uses_demo_owner(monkeypatch):
    class FakeRepository:
        def __init__(self, database_url):
            self.database_url = database_url

        def mark_stale_indexing_documents_failed(self, owner_id: str, stale_after_minutes: int):
            assert owner_id == "demo"
            assert stale_after_minutes == 30
            return ["doc1", "doc2"]

    monkeypatch.setattr(main_module, "PgVectorRepository", FakeRepository)
    client = TestClient(app)

    response = client.post("/api/maintenance/indexing/mark-stale-failed?timeout_minutes=30")

    assert response.status_code == 200
    assert response.json() == {"updated": 2, "document_ids": ["doc1", "doc2"]}


def test_chat_response_includes_selected_citation_images(monkeypatch):
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

    monkeypatch.setattr(main_module, "PgVectorRepository", FakeRepository)
    monkeypatch.setattr(main_module, "_get_embedding_provider", lambda: embedding_provider)
    monkeypatch.setattr(main_module, "_get_llm_provider", lambda: FakeLlmProvider())
    monkeypatch.setattr(main_module, "_get_query_router", lambda: QueryRouter(embedding_provider, cache_enabled=False))
    monkeypatch.setattr(main_module, "_get_reranker_provider", lambda: None)
    client = TestClient(app)

    response = client.post(
        "/api/chat",
        json={"question": "What morphology is shown?", "scope": "current_document", "document_ids": ["doc1"]},
    )

    assert response.status_code == 200
    citation = response.json()["citations"][0]
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


def test_startup_warmup_uses_cached_query_router(monkeypatch):
    calls = []

    class FakeRouter:
        def warmup(self):
            calls.append("warmup")
            return RouterWarmupStatus(cache_enabled=True, status="disk_hit", latency_ms=1.0)

    monkeypatch.setattr(main_module.settings, "router_warmup_on_startup", True)
    monkeypatch.setattr(main_module, "_get_query_router", lambda: FakeRouter())

    main_module.warmup_router_on_startup()

    assert calls == ["warmup"]


def test_startup_warmup_failure_does_not_block_startup(monkeypatch):
    class FakeRouter:
        def warmup(self):
            raise RuntimeError("cache unavailable")

    monkeypatch.setattr(main_module.settings, "router_warmup_on_startup", True)
    monkeypatch.setattr(main_module, "_get_query_router", lambda: FakeRouter())

    main_module.warmup_router_on_startup()


def test_startup_reranker_warmup_uses_cached_provider(monkeypatch):
    calls = []

    class FakeReranker:
        def warmup(self):
            calls.append("warmup")

    monkeypatch.setattr(main_module.settings, "reranker_enabled", True)
    monkeypatch.setattr(main_module.settings, "reranker_warmup_on_startup", True)
    monkeypatch.setattr(main_module, "_get_reranker_provider", lambda: FakeReranker())

    main_module.warmup_reranker_on_startup()

    assert calls == ["warmup"]


def test_startup_reranker_warmup_failure_does_not_block_startup(monkeypatch):
    class FakeReranker:
        def warmup(self):
            raise RuntimeError("reranker unavailable")

    monkeypatch.setattr(main_module.settings, "reranker_enabled", True)
    monkeypatch.setattr(main_module.settings, "reranker_warmup_on_startup", True)
    monkeypatch.setattr(main_module, "_get_reranker_provider", lambda: FakeReranker())

    main_module.warmup_reranker_on_startup()
