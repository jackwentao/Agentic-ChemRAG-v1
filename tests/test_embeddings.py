from backend.app.providers.embeddings import FakeEmbeddingProvider


def test_fake_embedding_provider_is_deterministic_and_normalized():
    provider = FakeEmbeddingProvider(dimensions=8)

    first = provider.embed_query("chemical bonding")
    second = provider.embed_query("chemical bonding")

    assert first == second
    assert len(first) == 8
    assert abs(sum(value * value for value in first) - 1.0) < 1e-6
