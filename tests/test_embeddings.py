import sys
from types import SimpleNamespace

from backend.app.providers.embeddings import FakeEmbeddingProvider
from backend.app.providers.embeddings import LocalBgeM3EmbeddingProvider


def test_fake_embedding_provider_is_deterministic_and_normalized():
    provider = FakeEmbeddingProvider(dimensions=8)

    first = provider.embed_query("chemical bonding")
    second = provider.embed_query("chemical bonding")

    assert first == second
    assert len(first) == 8
    assert abs(sum(value * value for value in first) - 1.0) < 1e-6


def test_local_embedding_passes_cache_dir_through_model_kwargs(monkeypatch, tmp_path):
    captured = {}

    class FakeVector:
        def tolist(self):
            return [1.0]

    class FakeSentenceTransformer:
        def __init__(self, model_name, **kwargs):
            captured.update(kwargs)

        def encode(self, texts, normalize_embeddings=True):
            return [FakeVector() for _text in texts]

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    provider = LocalBgeM3EmbeddingProvider(cache_dir=tmp_path, local_files_only=True)

    provider.embed_query("readiness")

    expected = {"cache_dir": str(tmp_path.resolve())}
    assert "cache_folder" not in captured
    assert captured["model_kwargs"] == expected
    assert captured["processor_kwargs"] == expected
    assert captured["config_kwargs"] == expected
