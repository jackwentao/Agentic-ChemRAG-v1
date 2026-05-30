from pathlib import Path

import scripts.smoke_api_flow as smoke_api_flow


class FakeResponse:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self.payload


class FakeClient:
    def __init__(self) -> None:
        self.task_checks = 0
        self.chat_payload = None

    def post(self, path: str, **kwargs) -> FakeResponse:
        if path == "/api/documents":
            return FakeResponse({"document_id": "doc1", "task_id": "task1", "status": "pending"})
        if path == "/api/chat":
            self.chat_payload = kwargs["json"]
            return FakeResponse({"answer": "有依据的回答", "citations": [{"page_start": 1}]})
        raise AssertionError(path)

    def get(self, path: str) -> FakeResponse:
        if path == "/api/tasks/task1":
            self.task_checks += 1
            return FakeResponse({"status": "success"})
        if path == "/api/documents":
            return FakeResponse(
                [{"document_id": "doc1", "is_indexed": True, "indexing_status": "succeeded", "chunk_count": 4}]
            )
        raise AssertionError(path)


def test_smoke_flow_uploads_waits_for_indexing_and_asks_question(tmp_path: Path):
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF fake")
    client = FakeClient()

    result = smoke_api_flow.run_smoke_flow(client, pdf_path, "论文结论是什么？", sleep=lambda _: None)

    assert result["document"]["chunk_count"] == 4
    assert result["chat"]["answer"] == "有依据的回答"
    assert client.task_checks == 1
    assert client.chat_payload == {
        "question": "论文结论是什么？",
        "scope": "current_document",
        "document_ids": ["doc1"],
    }


def test_smoke_script_configures_utf8_console_output(monkeypatch):
    class RecordingStream:
        def __init__(self) -> None:
            self.calls = []

        def reconfigure(self, **kwargs) -> None:
            self.calls.append(kwargs)

    stdout = RecordingStream()
    stderr = RecordingStream()
    monkeypatch.setattr(smoke_api_flow.sys, "stdout", stdout)
    monkeypatch.setattr(smoke_api_flow.sys, "stderr", stderr)

    smoke_api_flow.configure_console_output()

    expected = [{"encoding": "utf-8", "errors": "replace"}]
    assert stdout.calls == expected
    assert stderr.calls == expected
