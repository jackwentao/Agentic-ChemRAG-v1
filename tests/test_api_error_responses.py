from fastapi.testclient import TestClient

from backend.app.main import app


def test_unhandled_error_returns_json_payload():
    @app.get("/__test_unhandled_error")
    def _raise_unhandled_error():
        raise RuntimeError("database exploded")

    response = TestClient(app, raise_server_exceptions=False).get(
        "/__test_unhandled_error",
        headers={"X-Request-ID": "req-500"},
    )

    assert response.status_code == 500
    assert response.json() == {
        "error": {
            "code": "INTERNAL_SERVER_ERROR",
            "message": "Server error. Please check the API logs.",
            "request_id": "req-500",
        }
    }
