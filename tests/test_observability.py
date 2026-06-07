from fastapi.testclient import TestClient

from backend.app.main import app


def test_request_id_header_is_returned():
    response = TestClient(app).get("/api/health", headers={"X-Request-ID": "request-123"})

    assert response.status_code == 200
    assert response.headers["x-request-id"] == "request-123"


def test_metrics_endpoint_returns_text():
    response = TestClient(app).get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
