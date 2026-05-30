import os
import sys
from types import SimpleNamespace

import pytest

from backend.app.config import Settings
from backend.app.providers.rerankers import (
    FakeRerankerProvider,
    LocalBgeRerankerProvider,
    RerankCandidate,
    build_reranker_provider,
)


def test_fake_reranker_orders_by_query_overlap_and_aliases():
    provider = FakeRerankerProvider()
    candidates = [
        RerankCandidate("title", "Ionic covalent organic framework title page."),
        RerankCandidate("figure", "Fig. 1. The packing underlying principles between NLO properties and iCOFs structures."),
    ]

    results = provider.rerank("figures describe nonlinear optical responses", candidates, top_k=1)

    assert results[0].candidate_id == "figure"


def test_build_reranker_provider_returns_none_when_disabled():
    settings = Settings(reranker_enabled=False)

    assert build_reranker_provider(settings) is None


def test_build_reranker_provider_uses_fake_in_test_env():
    settings = Settings(env="test", reranker_enabled=True)

    assert isinstance(build_reranker_provider(settings), FakeRerankerProvider)


def test_local_reranker_reports_clear_model_load_error(tmp_path):
    provider = LocalBgeRerankerProvider(
        model_name="missing-reranker-model",
        cache_dir=tmp_path,
        local_files_only=True,
    )

    with pytest.raises(RuntimeError, match="Failed to load reranker model"):
        provider.warmup()


def test_online_reranker_clears_global_offline_flags(monkeypatch, tmp_path):
    import huggingface_hub.constants as hub_constants

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(hub_constants, "HF_HUB_OFFLINE", True)
    provider = LocalBgeRerankerProvider(cache_dir=tmp_path, local_files_only=False)

    provider._configure_hf_environment()

    assert "HF_HUB_OFFLINE" not in os.environ
    assert "TRANSFORMERS_OFFLINE" not in os.environ
    assert hub_constants.HF_HUB_OFFLINE is False


def test_local_reranker_uses_huggingface_cache_without_transformer_cache_argument(monkeypatch, tmp_path):
    captured = {}

    class FakeCrossEncoder:
        def __init__(self, model_name, **kwargs):
            captured.update(kwargs)

        def predict(self, pairs):
            return [1.0 for _pair in pairs]

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=FakeCrossEncoder))
    provider = LocalBgeRerankerProvider(cache_dir=tmp_path, local_files_only=True)

    provider.warmup()

    assert "cache_folder" not in captured
    assert "model_kwargs" not in captured
    assert "processor_kwargs" not in captured
    assert "config_kwargs" not in captured
    assert os.environ["HUGGINGFACE_HUB_CACHE"] == str(tmp_path.resolve())
    assert "SENTENCE_TRANSFORMERS_HOME" not in os.environ
