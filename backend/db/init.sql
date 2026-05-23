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
    parent_id TEXT,
    text TEXT NOT NULL,
    page_start INTEGER NOT NULL,
    page_end INTEGER NOT NULL,
    section_title TEXT NOT NULL,
    section_path TEXT NOT NULL,
    chunk_type TEXT NOT NULL,
    embedding VECTOR(1024)
);

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
    title TEXT,
    abstract TEXT,
    summary_text TEXT NOT NULL,
    topics JSONB NOT NULL DEFAULT '[]'::jsonb,
    entities JSONB NOT NULL DEFAULT '[]'::jsonb,
    figure_summary TEXT,
    embedding VECTOR(1024),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx
ON chunks USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

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

CREATE INDEX IF NOT EXISTS figures_embedding_idx
ON figures USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

CREATE INDEX IF NOT EXISTS document_profiles_embedding_idx
ON document_profiles USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
