from pathlib import Path

from backend.app.config import Settings


def test_default_infrastructure_ports_match_docker_compose_exposed_ports():
    settings = Settings(_env_file=None)
    compose_text = Path("docker-compose.yml").read_text(encoding="utf-8")

    assert settings.redis_url == "redis://localhost:6380/0"
    assert settings.minio_endpoint == "localhost:9100"
    assert '"6380:6379"' in compose_text
    assert '"9100:9000"' in compose_text


def test_default_models_allow_first_run_download():
    settings = Settings(_env_file=None)

    assert settings.embedding_local_files_only is False
    assert settings.reranker_local_files_only is False


def test_windows_setup_starts_minio_with_other_dependencies():
    script_text = Path("scripts/setup_windows.ps1").read_text(encoding="utf-8")

    assert "docker compose up -d postgres redis minio" in script_text
