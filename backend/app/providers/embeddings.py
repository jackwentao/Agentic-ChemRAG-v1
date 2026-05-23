import os
from threading import Lock
from abc import ABC, abstractmethod
from pathlib import Path
from hashlib import sha256
from random import Random

from backend.app.config import Settings


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError


class LocalBgeM3EmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
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
                    if self.cache_dir is not None:
                        self.cache_dir.mkdir(parents=True, exist_ok=True)
                        cache_value = str(self.cache_dir.resolve())
                        os.environ.setdefault("HF_HOME", cache_value)
                        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str((self.cache_dir / "hub").resolve()))
                        os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str((self.cache_dir / "sentence-transformers").resolve()))
                    if self.local_files_only:
                        os.environ.setdefault("HF_HUB_OFFLINE", "1")
                        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                    from sentence_transformers import SentenceTransformer

                    self._model = SentenceTransformer(
                        self.model_name,
                        device=self.device,
                        cache_folder=str(self.cache_dir) if self.cache_dir else None,
                        local_files_only=self.local_files_only,
                    )
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class FakeEmbeddingProvider(EmbeddingProvider):
    def __init__(self, dimensions: int = 8) -> None:
        self.dimensions = dimensions

    def _embed(self, text: str) -> list[float]:
        seed = int.from_bytes(sha256(text.encode("utf-8")).digest()[:8], "big")
        rng = Random(seed)
        vector = [rng.uniform(-1, 1) for _ in range(self.dimensions)]
        norm = sum(value * value for value in vector) ** 0.5 or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_provider == "fake":
        return FakeEmbeddingProvider(settings.embedding_dim)
    return LocalBgeM3EmbeddingProvider(
        settings.embedding_model,
        settings.embedding_device,
        settings.model_cache_dir,
        local_files_only=settings.embedding_local_files_only,
    )
