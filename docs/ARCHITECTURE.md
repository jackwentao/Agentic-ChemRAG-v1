# Agentic-ChemRAG Architecture

## Goal

The first version is a runnable portfolio project: PDF ingestion, parent-child chunking, local bge-m3 embeddings, pgvector retrieval, LangGraph multi-agent QA, persistent demo-user document library, and citation-aware UI.

## Module Boundaries

- `backend/app/api` is intentionally not created yet; route handlers live in `main.py` while the API surface is small.
- `domain` owns pure data structures and has no infrastructure dependency.
- `parsing` owns PDF parsing and chunking strategies.
- `providers` owns LLM and embedding adapters.
- `repositories` owns storage implementations.
- `services` composes parsers, chunkers, providers, and repositories into use cases.
- `agents` owns LangGraph state and graph nodes.
- `tasks` owns Celery integration.

Dependency direction:

```text
FastAPI route -> service -> parser/chunker/provider/repository
LangGraph node -> service/provider
domain -> no project dependency
```

## Current RAG Flow

```text
PDF upload
  -> document is registered under owner_id=demo
  -> same owner_id + file_hash reuses existing document
  -> Celery ingest_document task
  -> PyMuPDFParser extracts page text and images
  -> LocalImageStore saves page-linked images
  -> ParentChildChunker creates section parents and retrieval children
  -> LocalBgeM3EmbeddingProvider embeds retrieval child chunks
  -> PgVectorRepository stores documents, images, child chunks, vectors
  -> /api/chat invokes MultiAgentRagGraph
  -> Planner -> Retriever -> Verifier -> Answerer
  -> UI renders answer and citations
```

## Document Library Flow

The first version uses fixed owner `demo`.

```text
GET /api/documents
  -> returns demo user's documents with chunk_count and is_indexed

POST /api/chat scope=current_document
  -> requires document_ids from the selected document

POST /api/chat scope=all_documents
  -> backend resolves all indexed demo-owned document ids
```

## Parser Strategy

The first version uses PyMuPDF with section-aware parent-child chunking instead of depending on RAGFlow.

Reasoning:

- PyMuPDF keeps the dependency surface small and easy to debug.
- Parent-child chunking preserves paper structure better than fixed windows.
- The `DocumentParser` and `ChunkingStrategy` interfaces leave room for a later `RagflowParserAdapter`.
- References/Bibliography sections are not emitted as child chunks, so citations lists do not pollute retrieval.

## Local Embedding

The default embedding provider is `LocalBgeM3EmbeddingProvider`, backed by `sentence-transformers` and `BAAI/bge-m3`.

The first model download may take time. Tests use `FakeEmbeddingProvider` so CI and architecture tests do not depend on model downloads.

## Next Upgrade Points

- Move route handlers from `main.py` into `api/routes/*.py` once endpoints grow.
- Add cross-encoder reranking after pgvector retrieval.
- Add a real evidence verifier prompt instead of the current non-empty evidence filter.
- Store parent chunk expansion and return both precise child evidence and parent context.
- Add RAGFlow/MinerU parser adapters for difficult PDFs.
- Add evaluation cases with manually labeled expected pages.
