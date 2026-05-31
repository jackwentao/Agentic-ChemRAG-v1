from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


def build_database_engine(database_url: str) -> Engine:
    """创建由应用生命周期或独立进程负责释放的数据库引擎。"""
    return create_engine(database_url, pool_pre_ping=True)


def prepare_database(engine: Engine) -> None:
    """在应用启动或部署准备阶段统一补齐数据库结构。"""
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                ALTER TABLE documents
                ADD COLUMN IF NOT EXISTS owner_id TEXT NOT NULL DEFAULT 'demo'
                """
            )
        )
        conn.execute(text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS file_hash TEXT"))
        conn.execute(text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS file_size BIGINT"))
        conn.execute(text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS original_filename TEXT"))
        conn.execute(text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS indexing_status TEXT NOT NULL DEFAULT 'pending'"))
        conn.execute(text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS indexing_error TEXT"))
        conn.execute(text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS indexed_at TIMESTAMPTZ"))
        conn.execute(text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS indexing_updated_at TIMESTAMPTZ NOT NULL DEFAULT now()"))
        conn.execute(text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS pdf_object_key TEXT"))
        conn.execute(text("ALTER TABLE images ADD COLUMN IF NOT EXISTS label TEXT"))
        conn.execute(text("ALTER TABLE images ADD COLUMN IF NOT EXISTS caption TEXT"))
        conn.execute(text("ALTER TABLE images ADD COLUMN IF NOT EXISTS nearby_text TEXT"))
        conn.execute(text("ALTER TABLE images ADD COLUMN IF NOT EXISTS bbox JSONB"))
        conn.execute(text("ALTER TABLE images ADD COLUMN IF NOT EXISTS source_type TEXT NOT NULL DEFAULT 'embedded'"))
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS figures (
                    figure_id TEXT PRIMARY KEY,
                    image_id TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    page INTEGER NOT NULL,
                    label TEXT,
                    caption TEXT,
                    nearby_text TEXT,
                    bbox JSONB,
                    path TEXT NOT NULL,
                    extension TEXT NOT NULL,
                    width INTEGER,
                    height INTEGER,
                    source_type TEXT NOT NULL DEFAULT 'embedded',
                    embedding VECTOR(1024)
                )
                """
            )
        )
        conn.execute(text("ALTER TABLE figures ADD COLUMN IF NOT EXISTS label TEXT"))
        conn.execute(text("ALTER TABLE figures ADD COLUMN IF NOT EXISTS caption TEXT"))
        conn.execute(text("ALTER TABLE figures ADD COLUMN IF NOT EXISTS nearby_text TEXT"))
        conn.execute(text("ALTER TABLE figures ADD COLUMN IF NOT EXISTS bbox JSONB"))
        conn.execute(text("ALTER TABLE figures ADD COLUMN IF NOT EXISTS source_type TEXT NOT NULL DEFAULT 'embedded'"))
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS document_profiles (
                    document_id TEXT PRIMARY KEY,
                    title TEXT,
                    abstract TEXT,
                    summary_text TEXT NOT NULL,
                    topics JSONB NOT NULL DEFAULT '[]'::jsonb,
                    entities JSONB NOT NULL DEFAULT '[]'::jsonb,
                    figure_summary TEXT,
                    embedding VECTOR(1024),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL DEFAULT 'demo',
                    title TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    message_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    scope TEXT,
                    document_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
                    citations_json JSONB NOT NULL DEFAULT '[]'::jsonb,
                    images_json JSONB NOT NULL DEFAULT '[]'::jsonb,
                    limitations_json JSONB NOT NULL DEFAULT '[]'::jsonb,
                    diagnostics_summary_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS chat_contexts (
                    session_id TEXT PRIMARY KEY REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
                    recent_context TEXT NOT NULL DEFAULT '',
                    compressed_context TEXT NOT NULL DEFAULT '',
                    context_char_count INTEGER NOT NULL DEFAULT 0,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS documents_owner_file_hash_idx
                ON documents (owner_id, file_hash)
                WHERE file_hash IS NOT NULL
                """
            )
        )
        conn.execute(
            text(
                """
                UPDATE documents
                SET indexing_status = 'succeeded',
                    indexing_error = NULL,
                    indexed_at = COALESCE(indexed_at, now())
                WHERE indexing_status = 'pending'
                  AND EXISTS (
                      SELECT 1
                      FROM chunks
                      WHERE chunks.document_id = documents.document_id
                  )
                """
            )
        )
        conn.execute(text("CREATE INDEX IF NOT EXISTS figures_document_page_idx ON figures (document_id, page)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS chat_sessions_owner_updated_idx ON chat_sessions (owner_id, updated_at DESC)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS chat_messages_session_created_idx ON chat_messages (session_id, created_at ASC)"))
        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS figures_embedding_idx
                ON figures USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
                """
            )
        )
        conn.execute(
            text(
                """
                -- document_profiles 是单用户文档级画像，默认规模较小。
                -- 删除历史 ivfflat 索引，让查询保持精确全表向量排序，避免近似召回遗漏。
                DROP INDEX IF EXISTS document_profiles_embedding_idx
                """
            )
        )
