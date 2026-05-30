from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from backend.app.config import Settings


@dataclass(frozen=True)
class RerankCandidate:
    candidate_id: str
    text: str


@dataclass(frozen=True)
class RerankResult:
    candidate_id: str
    score: float


class RerankerProvider(ABC):
    @abstractmethod
    def rerank(self, query: str, candidates: list[RerankCandidate], top_k: int) -> list[RerankResult]:
        raise NotImplementedError

    def warmup(self) -> None:
        self.rerank("warmup", [RerankCandidate(candidate_id="warmup", text="warmup document")], top_k=1)


class LocalBgeRerankerProvider(RerankerProvider):
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cpu",
        cache_dir: Path | None = None,
        local_files_only: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self._model = None
        self._model_lock = Lock()

    @property
    def model(self):
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    self._configure_hf_environment()
                    from sentence_transformers import CrossEncoder

                    try:
                        self._model = CrossEncoder(
                            self.model_name,
                            device=self.device,
                            local_files_only=self.local_files_only,
                        )
                    except Exception as exc:
                        raise RuntimeError(
                            f"Failed to load reranker model {self.model_name!r}. "
                            "Make sure it exists under MODEL_CACHE_DIR or disable local-files-only mode."
                        ) from exc
        return self._model

    def _configure_hf_environment(self) -> None:
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_value = str(self.cache_dir.resolve())
            os.environ.setdefault("HF_HOME", cache_value)
            os.environ["HUGGINGFACE_HUB_CACHE"] = cache_value
            os.environ.pop("SENTENCE_TRANSFORMERS_HOME", None)
        if self.local_files_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        else:
            # Embedding warmup may have set global offline switches earlier in the same process.
            # When reranker downloads are explicitly allowed, remove those switches before load.
            os.environ.pop("HF_HUB_OFFLINE", None)
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
            try:
                import huggingface_hub.constants as hub_constants

                hub_constants.HF_HUB_OFFLINE = False
            except Exception:
                pass

    def rerank(self, query: str, candidates: list[RerankCandidate], top_k: int) -> list[RerankResult]:
        if not candidates or top_k <= 0:
            return []
        pairs = [(query, candidate.text) for candidate in candidates]
        scores = self.model.predict(pairs)
        rows = [
            RerankResult(candidate_id=candidate.candidate_id, score=float(score))
            for candidate, score in zip(candidates, scores, strict=True)
        ]
        rows.sort(key=lambda item: item.score, reverse=True)
        return rows[:top_k]


class FakeRerankerProvider(RerankerProvider):
    def rerank(self, query: str, candidates: list[RerankCandidate], top_k: int) -> list[RerankResult]:
        query_terms = _terms(query)
        rows = []
        for index, candidate in enumerate(candidates):
            text_terms = _terms(candidate.text)
            overlap = len(query_terms & text_terms)
            alias_bonus = _alias_bonus(query, candidate.text)
            score = overlap + alias_bonus - index * 0.001
            rows.append(RerankResult(candidate_id=candidate.candidate_id, score=float(score)))
        rows.sort(key=lambda item: item.score, reverse=True)
        return rows[:top_k]


def build_reranker_provider(settings: Settings) -> RerankerProvider | None:
    if not settings.reranker_enabled:
        return None
    if settings.env == "test" or settings.reranker_provider == "fake":
        return FakeRerankerProvider()
    return LocalBgeRerankerProvider(
        model_name=settings.reranker_model,
        device=settings.reranker_device,
        cache_dir=settings.model_cache_dir,
        local_files_only=settings.reranker_local_files_only,
    )


def _terms(text: str) -> set[str]:
    import re

    normalized = _normalize_aliases(text)
    terms = {term.lower() for term in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", normalized)}
    terms.update(re.findall(r"[\u4e00-\u9fff]{2,}", normalized))
    return terms


def _alias_bonus(query: str, text: str) -> float:
    query_normalized = _normalize_aliases(query)
    text_normalized = _normalize_aliases(text)
    if "nonlinear" in query_normalized and "nonlinear" in text_normalized:
        return 3.0
    return 0.0


def _normalize_aliases(text: str) -> str:
    lowered = text.lower()
    return lowered.replace("nlo", "nonlinear optical")
