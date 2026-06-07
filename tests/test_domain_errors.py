from fastapi.testclient import TestClient

from backend.app.domain.errors import DomainError
from backend.app.main import app


def test_domain_error_response_contains_stable_error_shape():
    @app.get("/__test_domain_error")
    def _raise_domain_error():
        raise DomainError(code="TEST_ERROR", user_message="Test error.", status_code=418)

    response = TestClient(app).get("/__test_domain_error", headers={"X-Request-ID": "req-1"})

    assert response.status_code == 418
    assert response.headers["x-request-id"] == "req-1"
    assert response.json() == {
        "error": {
            "code": "TEST_ERROR",
            "message": "Test error.",
            "request_id": "req-1",
        }
    }
