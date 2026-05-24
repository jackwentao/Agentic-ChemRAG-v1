from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.dependencies import get_query_router, get_reranker_provider
from backend.app.api.routers import assets, chat, documents, maintenance
from backend.app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


def warmup_router_on_startup() -> None:
    if not settings.router_warmup_on_startup:
        return
    try:
        status = get_query_router().warmup()
        logger.info("Query router warmup finished: %s", status.to_dict())
    except Exception:
        logger.exception("Query router warmup failed; continuing without blocking startup.")


def warmup_reranker_on_startup() -> None:
    if not settings.reranker_enabled or not settings.reranker_warmup_on_startup:
        return
    try:
        reranker = get_reranker_provider()
        if reranker is not None:
            reranker.warmup()
            logger.info("Reranker warmup finished.")
    except Exception:
        logger.exception("Reranker warmup failed; continuing startup but reranked retrieval may fail.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    warmup_router_on_startup()
    warmup_reranker_on_startup()
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(assets.router)
app.include_router(maintenance.router)
