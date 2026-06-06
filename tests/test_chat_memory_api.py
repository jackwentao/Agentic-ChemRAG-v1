from fastapi.testclient import TestClient

import backend.app.api.dependencies as dependencies
from backend.app.main import app
from backend.app.repositories.conversation import InMemoryConversationRepository


def setup_function():
    app.dependency_overrides.clear()
    if hasattr(app.state, "conversation_repository"):
        del app.state.conversation_repository


def teardown_function():
    app.dependency_overrides.clear()
    if hasattr(app.state, "conversation_repository"):
        del app.state.conversation_repository


def test_chat_creates_session_persists_messages_and_injects_context_on_next_turn():
    class FakeRepository:
        def list_indexed_document_ids(self, owner_id="demo"):
            return ["doc1"]

    class RecordingGraph:
        def __init__(self):
            self.calls = []

        def invoke(self, question, document_ids=None, scope=None, session_id=None, conversation_context=None):
            self.calls.append(
                {
                    "question": question,
                    "document_ids": document_ids,
                    "scope": scope,
                    "session_id": session_id,
                    "conversation_context": conversation_context,
                }
            )
            return {
                "answer": f"answer for {question}",
                "verified_evidence": [],
                "selected_images": {},
                "limitations": [],
                "warnings": [],
                "diagnostics": {"route": {"selected_route": "chunk_search"}, "stop_reason": "sufficient_evidence"},
            }

    repository = InMemoryConversationRepository()
    graph = RecordingGraph()
    app.state.conversation_repository = repository
    app.dependency_overrides[dependencies.get_repository] = lambda: FakeRepository()
    app.dependency_overrides[dependencies.get_rag_graph] = lambda: graph
    client = TestClient(app)

    first = client.post("/api/chat", json={"question": "这篇论文研究什么？", "scope": "all_documents"})
    assert first.status_code == 200
    first_payload = first.json()
    session_id = first_payload["session_id"]
    assert first_payload["user_message"]["role"] == "user"
    assert first_payload["user_message"]["session_id"] == session_id
    assert first_payload["assistant_message"]["role"] == "assistant"
    assert first_payload["assistant_message"]["session_id"] == session_id
    assert first_payload["assistant_message"]["content"].startswith("answer for ")

    second = client.post(
        "/api/chat",
        json={"question": "它稳定吗？", "scope": "all_documents", "session_id": session_id},
    )

    assert second.status_code == 200
    assert second.json()["session_id"] == session_id
    assert graph.calls[0]["conversation_context"] == ""
    assert "这篇论文研究什么" in graph.calls[1]["conversation_context"]
    assert graph.calls[1]["session_id"] == session_id

    messages = client.get(f"/api/chat/sessions/{session_id}/messages")
    assert messages.status_code == 200
    payload = messages.json()
    assert [message["role"] for message in payload["messages"]] == ["user", "assistant", "user", "assistant"]
    assert payload["messages"][0]["message_id"] == first_payload["user_message"]["message_id"]
    assert payload["messages"][1]["message_id"] == first_payload["assistant_message"]["message_id"]


def test_chat_sessions_endpoint_lists_persisted_sessions():
    class FakeRepository:
        def list_indexed_document_ids(self, owner_id="demo"):
            return ["doc1"]

    class SimpleGraph:
        def invoke(self, question, document_ids=None, scope=None, session_id=None, conversation_context=None):
            return {
                "answer": "ok",
                "verified_evidence": [],
                "selected_images": {},
                "limitations": [],
                "warnings": [],
            }

    app.state.conversation_repository = InMemoryConversationRepository()
    app.dependency_overrides[dependencies.get_repository] = lambda: FakeRepository()
    app.dependency_overrides[dependencies.get_rag_graph] = lambda: SimpleGraph()
    client = TestClient(app)

    client.post("/api/chat", json={"question": "第一轮问题", "scope": "all_documents"})
    response = client.get("/api/chat/sessions")

    assert response.status_code == 200
    assert response.json()["sessions"][0]["title"] == "第一轮问题"


def test_chat_sessions_can_be_pinned_and_deleted():
    class FakeRepository:
        def list_indexed_document_ids(self, owner_id="demo"):
            return ["doc1"]

    class SimpleGraph:
        def invoke(self, question, document_ids=None, scope=None, session_id=None, conversation_context=None):
            return {
                "answer": "ok",
                "verified_evidence": [],
                "selected_images": {},
                "limitations": [],
                "warnings": [],
            }

    app.state.conversation_repository = InMemoryConversationRepository()
    app.dependency_overrides[dependencies.get_repository] = lambda: FakeRepository()
    app.dependency_overrides[dependencies.get_rag_graph] = lambda: SimpleGraph()
    client = TestClient(app)

    first = client.post("/api/chat", json={"question": "第一轮问题", "scope": "all_documents"}).json()
    second = client.post("/api/chat", json={"question": "第二轮问题", "scope": "all_documents"}).json()

    pinned = client.patch(f"/api/chat/sessions/{first['session_id']}/pin", json={"pinned": True})
    assert pinned.status_code == 200
    assert pinned.json()["pinned"] is True

    sessions = client.get("/api/chat/sessions").json()["sessions"]
    assert sessions[0]["session_id"] == first["session_id"]
    assert sessions[0]["pinned"] is True

    deleted = client.delete(f"/api/chat/sessions/{first['session_id']}")
    assert deleted.status_code == 204

    remaining = client.get("/api/chat/sessions").json()["sessions"]
    assert [session["session_id"] for session in remaining] == [second["session_id"]]
    messages = client.get(f"/api/chat/sessions/{first['session_id']}/messages")
    assert messages.status_code == 404


def test_chat_stream_emits_answer_deltas_and_persists_messages():
    class FakeRepository:
        def list_indexed_document_ids(self, owner_id="demo"):
            return ["doc1"]

    class StreamingGraph:
        def invoke(
            self,
            question,
            document_ids=None,
            scope=None,
            session_id=None,
            conversation_context=None,
            answer_delta_callback=None,
        ):
            if answer_delta_callback is not None:
                answer_delta_callback("hello ")
                answer_delta_callback("world")
            return {
                "answer": "hello world",
                "verified_evidence": [],
                "selected_images": {},
                "limitations": [],
                "warnings": [],
                "diagnostics": {"route": {"selected_route": "chunk_search"}, "stop_reason": "sufficient_evidence"},
            }

    repository = InMemoryConversationRepository()
    app.state.conversation_repository = repository
    app.dependency_overrides[dependencies.get_repository] = lambda: FakeRepository()
    app.dependency_overrides[dependencies.get_rag_graph] = lambda: StreamingGraph()
    client = TestClient(app)

    with client.stream("POST", "/api/chat/stream", json={"question": "stream?", "scope": "all_documents"}) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert "event: answer_delta" in body
    assert '"delta": "hello "' in body
    assert '"delta": "world"' in body
    assert "event: final" in body
    assert '"answer": "hello world"' in body

    session_id = repository.list_sessions("demo")[0].session_id
    messages = client.get(f"/api/chat/sessions/{session_id}/messages")
    assert messages.status_code == 200
    assert [message["role"] for message in messages.json()["messages"]] == ["user", "assistant"]
    assert messages.json()["messages"][1]["content"] == "hello world"
