CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL DEFAULT 'demo',
    file_hash TEXT,
    file_size BIGINT,
    original_filename TEXT,
    indexing_status TEXT NOT NULL DEFAULT 'pending',
    indexing_error TEXT,
    indexing_updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    indexed_at TIMESTAMPTZ,
    pdf_object_key TEXT,
    filename TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE documents
ADD COLUMN IF NOT EXISTS owner_id TEXT NOT NULL DEFAULT 'demo';

ALTER TABLE documents
ADD COLUMN IF NOT EXISTS file_hash TEXT;

ALTER TABLE documents
ADD COLUMN IF NOT EXISTS file_size BIGINT;

ALTER TABLE documents
ADD COLUMN IF NOT EXISTS original_filename TEXT;

ALTER TABLE documents
ADD COLUMN IF NOT EXISTS indexing_status TEXT NOT NULL DEFAULT 'pending';

ALTER TABLE documents
ADD COLUMN IF NOT EXISTS indexing_error TEXT;

ALTER TABLE documents
ADD COLUMN IF NOT EXISTS indexing_updated_at TIMESTAMPTZ NOT NULL DEFAULT now();

ALTER TABLE documents
ADD COLUMN IF NOT EXISTS indexed_at TIMESTAMPTZ;

ALTER TABLE documents
ADD COLUMN IF NOT EXISTS pdf_object_key TEXT;

CREATE UNIQUE INDEX IF NOT EXISTS documents_owner_file_hash_idx
ON documents (owner_id, file_hash)
WHERE file_hash IS NOT NULL;

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    owner_id TEXT NOT NULL DEFAULT 'demo',
    parent_id TEXT,
    text TEXT NOT NULL,
    page_start INTEGER NOT NULL,
    page_end INTEGER NOT NULL,
    section_title TEXT NOT NULL,
    section_path TEXT NOT NULL,
    chunk_type TEXT NOT NULL,
    embedding VECTOR(1024)
);

ALTER TABLE chunks
ADD COLUMN IF NOT EXISTS owner_id TEXT NOT NULL DEFAULT 'demo';

CREATE TABLE IF NOT EXISTS images (
    image_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    page INTEGER NOT NULL,
    path TEXT NOT NULL,
    extension TEXT NOT NULL,
    width INTEGER,
    height INTEGER,
    label TEXT,
    caption TEXT,
    nearby_text TEXT,
    bbox JSONB,
    source_type TEXT NOT NULL DEFAULT 'embedded'
);

ALTER TABLE images
ADD COLUMN IF NOT EXISTS label TEXT;

ALTER TABLE images
ADD COLUMN IF NOT EXISTS caption TEXT;

ALTER TABLE images
ADD COLUMN IF NOT EXISTS nearby_text TEXT;

ALTER TABLE images
ADD COLUMN IF NOT EXISTS bbox JSONB;

ALTER TABLE images
ADD COLUMN IF NOT EXISTS source_type TEXT NOT NULL DEFAULT 'embedded';

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
);

CREATE TABLE IF NOT EXISTS document_profiles (
    document_id TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL DEFAULT 'demo',
    title TEXT,
    abstract TEXT,
    conclusion TEXT,
    summary_text TEXT NOT NULL,
    topics JSONB NOT NULL DEFAULT '[]'::jsonb,
    entities JSONB NOT NULL DEFAULT '[]'::jsonb,
    profile_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    outline_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    source_refs_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    figure_summary TEXT,
    embedding VECTOR(1024),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE document_profiles
ADD COLUMN IF NOT EXISTS owner_id TEXT NOT NULL DEFAULT 'demo';

CREATE TABLE IF NOT EXISTS document_profile_embeddings (
    profile_field_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    owner_id TEXT NOT NULL DEFAULT 'demo',
    field_name TEXT NOT NULL,
    field_text TEXT NOT NULL,
    source_refs_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding VECTOR(1024),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE document_profile_embeddings
ADD COLUMN IF NOT EXISTS owner_id TEXT NOT NULL DEFAULT 'demo';

CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL DEFAULT 'demo',
    title TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

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
);

CREATE TABLE IF NOT EXISTS chat_contexts (
    session_id TEXT PRIMARY KEY REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
    recent_context TEXT NOT NULL DEFAULT '',
    compressed_context TEXT NOT NULL DEFAULT '',
    context_char_count INTEGER NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx
ON chunks USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS chunks_owner_document_idx
ON chunks (owner_id, document_id, chunk_type);

CREATE INDEX IF NOT EXISTS document_profiles_owner_document_idx
ON document_profiles (owner_id, document_id);

CREATE INDEX IF NOT EXISTS document_profile_embeddings_owner_document_idx
ON document_profile_embeddings (owner_id, document_id, field_name);

UPDATE documents
SET indexing_status = 'succeeded',
    indexing_error = NULL,
    indexed_at = COALESCE(indexed_at, now())
WHERE indexing_status = 'pending'
  AND EXISTS (
      SELECT 1
      FROM chunks
      WHERE chunks.document_id = documents.document_id
  );

CREATE INDEX IF NOT EXISTS images_document_page_idx
ON images (document_id, page);

CREATE INDEX IF NOT EXISTS figures_document_page_idx
ON figures (document_id, page);

CREATE INDEX IF NOT EXISTS document_profile_embeddings_document_idx
ON document_profile_embeddings (document_id, field_name);

CREATE INDEX IF NOT EXISTS chat_sessions_owner_updated_idx
ON chat_sessions (owner_id, updated_at DESC);

CREATE INDEX IF NOT EXISTS chat_messages_session_created_idx
ON chat_messages (session_id, created_at ASC);

CREATE INDEX IF NOT EXISTS figures_embedding_idx
ON figures USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- document_profiles 是单用户文档级画像，默认规模较小。
-- 这里不创建 ivfflat 近似索引，保留精确全表向量排序，避免文档级召回遗漏。
DROP INDEX IF EXISTS document_profiles_embedding_idx;
