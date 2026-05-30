from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.dependencies import get_query_router, get_reranker_provider
from backend.app.api.routers import assets, chat, documents, maintenance
from backend.app.config import get_settings
from backend.app.database import build_database_engine, prepare_database
from backend.app.repositories.conversation import PgConversationRepository
from backend.app.repositories.postgres import PgVectorRepository

settings = get_settings()
logger = logging.getLogger(__name__)


def warmup_router_on_startup() -> None:
    # 路由器预热用于提前加载分类/向量化依赖，避免首个请求承担冷启动延迟。
    if not settings.router_warmup_on_startup:
        return
    try:
        status = get_query_router().warmup()
        logger.info("Query router warmup finished: %s", status.to_dict())
    except Exception:
        logger.exception("Query router warmup failed; continuing without blocking startup.")


def warmup_reranker_on_startup() -> None:
    # 重排器通常比召回更重，是否预热交由配置控制，避免阻塞轻量部署场景。
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
    # 将数据库初始化、Repository 装配和模型预热放到统一生命周期里，
    # 这样 API 层只关心依赖注入，不直接承担基础设施编排逻辑。
    engine = build_database_engine(settings.database_url)
    try:
        prepare_database(engine)
        app.state.repository = PgVectorRepository(engine)
        app.state.conversation_repository = PgConversationRepository(engine)
        warmup_router_on_startup()
        warmup_reranker_on_startup()
        yield
    finally:
        if hasattr(app.state, "repository"):
            del app.state.repository
        if hasattr(app.state, "conversation_repository"):
            del app.state.conversation_repository
        engine.dispose()


app = FastAPI(title=settings.app_name, lifespan=lifespan)

# 这里使用最宽松的 CORS 配置，便于本地前后端联调；如果走生产部署，建议收紧来源名单。
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
