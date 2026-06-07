from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Agentic-ChemRAG"
    env: str = "dev"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    database_auto_prepare: bool = True

    database_url: str = "postgresql+psycopg://chemrag:chemrag@localhost:5432/chemrag"
    redis_url: str = "redis://localhost:6380/0"
    upload_dir: Path = Path("storage/uploads")
    image_dir: Path = Path("storage/images")
    model_cache_dir: Path = Path("storage/models")
    router_cache_dir: Path = Path("storage/router_cache")
    router_cache_enabled: bool = True
    router_warmup_on_startup: bool = True
    evaluation_warmup: bool = True
    object_storage_provider: str = "minio"
    minio_endpoint: str = "localhost:9100"
    minio_access_key: str = "chemrag"
    minio_secret_key: str = "chemrag-secret"
    minio_bucket: str = "chemrag"
    minio_secure: bool = False

    deepseek_api_key: str | None = None
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-flash"
    deepseek_timeout_seconds: float = 25.0
    llm_image_selection_enabled: bool = False
    llm_profile_extraction_enabled: bool = True

    embedding_provider: str = "local_bge_m3"
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = Field(default=1024, gt=0)
    embedding_device: str = "cpu"
    embedding_local_files_only: bool = False

    reranker_enabled: bool = True
    reranker_provider: str = "local_bge"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str = "cpu"
    reranker_local_files_only: bool = False
    reranker_chunk_candidates: int = Field(default=10, gt=0)
    reranker_figure_candidates: int = Field(default=10, gt=0)
    reranker_min_candidates: int = Field(default=10, gt=0)
    retrieval_exact_chunk_limit: int = Field(default=2000, ge=0)
    retrieval_exact_timeout_ms: int = Field(default=300, gt=0)
    reranker_warmup_on_startup: bool = True
    conversation_recent_turns: int = Field(default=3, ge=0)
    conversation_max_context_chars: int = Field(default=8000, gt=0)
    conversation_user_excerpt_chars: int = Field(default=160, gt=0)
    conversation_assistant_excerpt_chars: int = Field(default=240, gt=0)
    conversation_citation_limit_per_turn: int = Field(default=3, ge=0)
    conversation_limitation_limit_per_turn: int = Field(default=2, ge=0)
    upload_max_file_bytes: int = Field(default=50 * 1024 * 1024, gt=0)
    upload_max_batch_files: int = Field(default=10, gt=0)
    upload_spool_max_memory_bytes: int = Field(default=8 * 1024 * 1024, gt=0)
    upload_allowed_mime_types: str = "application/pdf"
    pdf_max_pages: int = Field(default=300, gt=0)
    pdf_parse_probe_timeout_seconds: float = Field(default=10.0, gt=0)
    pdf_max_extracted_images: int = Field(default=500, ge=0)
    pdf_max_image_bytes: int = Field(default=20 * 1024 * 1024, gt=0)
    malware_scanner_provider: str = "noop"
    request_id_header: str = "X-Request-ID"
    log_format: str = "plain"
    metrics_enabled: bool = True
    app_version: str = "0.1.0"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.image_dir.mkdir(parents=True, exist_ok=True)
    settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
    settings.router_cache_dir.mkdir(parents=True, exist_ok=True)
    return settings
