from dataclasses import dataclass
from typing import Callable

from backend.app.providers.embeddings import EmbeddingProvider
from backend.app.providers.llm import LlmProvider
from backend.app.providers.rerankers import RerankerProvider
from backend.app.repositories.postgres import PgVectorRepository
from backend.app.services.object_storage import ObjectStorage


@dataclass(frozen=True)
class ReadinessCheck:
    status: str
    required: bool
    detail: str


@dataclass(frozen=True)
class ReadinessReport:
    status: str
    complete_qa_online: bool
    can_serve_degraded: bool
    checks: dict[str, ReadinessCheck]


class ReadinessService:
    """按一次真实能力探测汇总问答链路所依赖的组件状态。"""

    def __init__(
        self,
        repository: PgVectorRepository,
        object_storage_factory: Callable[[], ObjectStorage],
        embedding_provider_factory: Callable[[], EmbeddingProvider],
        reranker_provider_factory: Callable[[], RerankerProvider | None],
        llm_provider_factory: Callable[[], LlmProvider],
        task_backend_probe: Callable[[], None],
        reranker_enabled: bool,
    ) -> None:
        self.repository = repository
        self.object_storage_factory = object_storage_factory
        self.embedding_provider_factory = embedding_provider_factory
        self.reranker_provider_factory = reranker_provider_factory
        self.llm_provider_factory = llm_provider_factory
        self.task_backend_probe = task_backend_probe
        self.reranker_enabled = reranker_enabled

    def inspect(self) -> ReadinessReport:
        checks = {
            "database": self._probe(self.repository.healthcheck),
            "object_storage": self._probe(lambda: self.object_storage_factory().healthcheck()),
            "task_backend": self._probe(self.task_backend_probe),
            "embedding": self._probe(self._probe_embedding),
            "reranker": self._probe_reranker(),
            "llm": self._probe(lambda: self.llm_provider_factory().healthcheck()),
        }
        complete_qa_online = all(not item.required or item.status == "ready" for item in checks.values())
        degraded_prerequisites = ("database", "object_storage", "task_backend", "embedding")
        can_serve_degraded = all(checks[name].status == "ready" for name in degraded_prerequisites)
        if complete_qa_online:
            status = "ready"
        elif can_serve_degraded:
            status = "degraded"
        else:
            status = "unavailable"
        return ReadinessReport(
            status=status,
            complete_qa_online=complete_qa_online,
            can_serve_degraded=can_serve_degraded,
            checks=checks,
        )

    def _probe_embedding(self) -> None:
        vector = self.embedding_provider_factory().embed_query("readiness probe")
        if not vector:
            raise RuntimeError("Embedding provider returned an empty vector.")

    def _probe_reranker(self) -> ReadinessCheck:
        if not self.reranker_enabled:
            return ReadinessCheck(status="disabled", required=False, detail="配置已禁用。")
        return self._probe(lambda: self._require_reranker().warmup())

    def _require_reranker(self) -> RerankerProvider:
        provider = self.reranker_provider_factory()
        if provider is None:
            raise RuntimeError("Reranker provider is not configured.")
        return provider

    def _probe(self, action: Callable[[], object]) -> ReadinessCheck:
        try:
            action()
        except Exception as exc:
            return ReadinessCheck(status="failed", required=True, detail=f"探测失败：{type(exc).__name__}")
        return ReadinessCheck(status="ready", required=True, detail="可用。")
