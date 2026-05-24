from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from backend.app.agents.rag_graph import MultiAgentRagGraph
from backend.app.config import Settings, get_settings
from backend.app.providers.embeddings import EmbeddingProvider, build_embedding_provider
from backend.app.providers.llm import LlmProvider, build_llm_provider
from backend.app.providers.rerankers import RerankerProvider, build_reranker_provider
from backend.app.repositories.postgres import PgVectorRepository
from backend.app.routing.query_router import QueryRouter
from backend.app.services.documents import DocumentService
from backend.app.services.object_storage import ObjectStorage, build_object_storage
from backend.app.services.rag import RetrievalService
from backend.app.tasks.ingestion_tasks import ingest_document

DEFAULT_OWNER_ID = "demo"


@lru_cache(maxsize=1)
def get_repository() -> PgVectorRepository:
    """复用数据库引擎与建表检查，避免每个请求重复初始化。"""
    return PgVectorRepository(get_settings().database_url)


@lru_cache(maxsize=1)
def get_object_storage() -> ObjectStorage:
    return build_object_storage(get_settings())


@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    return build_embedding_provider(get_settings())


@lru_cache(maxsize=1)
def get_llm_provider() -> LlmProvider:
    return build_llm_provider(get_settings())


@lru_cache(maxsize=1)
def get_reranker_provider() -> RerankerProvider | None:
    return build_reranker_provider(get_settings())


def _router_provider_fingerprint(settings: Settings) -> str:
    return (
        f"{settings.embedding_provider}:"
        f"{settings.embedding_model}:"
        f"{settings.embedding_device}:"
        f"{settings.embedding_dim}:"
        f"local_files_only={settings.embedding_local_files_only}"
    )


@lru_cache(maxsize=1)
def get_query_router() -> QueryRouter:
    settings = get_settings()
    return QueryRouter(
        embedding_provider=get_embedding_provider(),
        cache_dir=settings.router_cache_dir,
        cache_enabled=settings.router_cache_enabled,
        provider_fingerprint=_router_provider_fingerprint(settings),
    )


def get_ingestion_task():
    return ingest_document


def get_document_service(
    repository: Annotated[PgVectorRepository, Depends(get_repository)],
    object_storage: Annotated[ObjectStorage, Depends(get_object_storage)],
    settings: Annotated[Settings, Depends(get_settings)],
    ingestion_task=Depends(get_ingestion_task),
) -> DocumentService:
    return DocumentService(
        repository=repository,
        object_storage=object_storage,
        upload_dir=settings.upload_dir,
        ingestion_task=ingestion_task,
        owner_id=DEFAULT_OWNER_ID,
    )


def get_retrieval_service(
    repository: Annotated[PgVectorRepository, Depends(get_repository)],
    embedding_provider: Annotated[EmbeddingProvider, Depends(get_embedding_provider)],
    reranker_provider: Annotated[RerankerProvider | None, Depends(get_reranker_provider)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> RetrievalService:
    # RetrievalService 会记录单次检索诊断，按请求实例化以避免并发请求互相覆盖。
    return RetrievalService(
        repository=repository,
        embedding_provider=embedding_provider,
        reranker_provider=reranker_provider,
        reranker_enabled=settings.reranker_enabled,
        reranker_chunk_candidates=settings.reranker_chunk_candidates,
        reranker_figure_candidates=settings.reranker_figure_candidates,
        reranker_min_candidates=settings.reranker_min_candidates,
    )


def get_rag_graph(
    retrieval_service: Annotated[RetrievalService, Depends(get_retrieval_service)],
    llm_provider: Annotated[LlmProvider, Depends(get_llm_provider)],
    query_router: Annotated[QueryRouter, Depends(get_query_router)],
) -> MultiAgentRagGraph:
    return MultiAgentRagGraph(
        retrieval_service=retrieval_service,
        llm_provider=llm_provider,
        query_router=query_router,
    )
