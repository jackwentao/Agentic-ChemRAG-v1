import pytest
from fastapi import HTTPException

from backend.app.main import _resolve_chat_document_ids
from backend.app.schemas import ChatRequest


class FakeDocumentRepository:
    def __init__(self, indexed_document_ids: list[str]) -> None:
        self.indexed_document_ids = indexed_document_ids
        self.owner_id = None

    def list_indexed_document_ids(self, owner_id: str = "demo") -> list[str]:
        self.owner_id = owner_id
        return self.indexed_document_ids


def test_current_document_scope_requires_document_ids():
    request = ChatRequest(question="What is PL?", scope="current_document")

    with pytest.raises(HTTPException) as exc_info:
        _resolve_chat_document_ids(request, FakeDocumentRepository(["doc1"]))

    assert exc_info.value.status_code == 400


def test_current_document_scope_uses_given_document_ids():
    request = ChatRequest(question="What is PL?", scope="current_document", document_ids=["doc1"])

    document_ids = _resolve_chat_document_ids(request, FakeDocumentRepository(["doc2"]))

    assert document_ids == ["doc1"]


def test_all_documents_scope_uses_demo_indexed_documents():
    repository = FakeDocumentRepository(["doc1", "doc2"])
    request = ChatRequest(question="What is PL?", scope="all_documents")

    document_ids = _resolve_chat_document_ids(request, repository)

    assert document_ids == ["doc1", "doc2"]
    assert repository.owner_id == "demo"


def test_all_documents_scope_rejects_empty_library():
    request = ChatRequest(question="What is PL?", scope="all_documents")

    with pytest.raises(HTTPException) as exc_info:
        _resolve_chat_document_ids(request, FakeDocumentRepository([]))

    assert exc_info.value.status_code == 400
