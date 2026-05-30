from backend.app.services.readiness import ReadinessService


class FakeRepository:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def healthcheck(self) -> None:
        if self.fail:
            raise RuntimeError("database down")


class FakeStorage:
    def healthcheck(self) -> None:
        return None


class FakeEmbedding:
    def embed_query(self, text: str) -> list[float]:
        return [1.0]


class FakeReranker:
    def warmup(self) -> None:
        return None


class FakeLlm:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def healthcheck(self) -> None:
        if self.fail:
            raise ConnectionError("llm offline")


def build_service(*, database_fail: bool = False, llm_fail: bool = False, reranker_enabled: bool = True) -> ReadinessService:
    return ReadinessService(
        repository=FakeRepository(fail=database_fail),
        object_storage_factory=FakeStorage,
        embedding_provider_factory=FakeEmbedding,
        reranker_provider_factory=FakeReranker,
        llm_provider_factory=lambda: FakeLlm(fail=llm_fail),
        task_backend_probe=lambda: None,
        reranker_enabled=reranker_enabled,
    )


def test_readiness_reports_complete_qa_online_when_all_components_work():
    report = build_service().inspect()

    assert report.status == "ready"
    assert report.complete_qa_online is True
    assert report.can_serve_degraded is True
    assert all(check.status == "ready" for check in report.checks.values())


def test_readiness_reports_degraded_when_llm_is_offline_but_fallback_prerequisites_work():
    report = build_service(llm_fail=True, reranker_enabled=False).inspect()

    assert report.status == "degraded"
    assert report.complete_qa_online is False
    assert report.can_serve_degraded is True
    assert report.checks["llm"].status == "failed"
    assert report.checks["reranker"].status == "disabled"


def test_readiness_reports_unavailable_when_database_is_offline():
    report = build_service(database_fail=True).inspect()

    assert report.status == "unavailable"
    assert report.complete_qa_online is False
    assert report.can_serve_degraded is False
    assert report.checks["database"].status == "failed"
