# Agentic-ChemRAG

Agentic-ChemRAG is a portfolio-oriented multi-agent RAG system for chemistry and materials-science papers.

## Architecture

- FastAPI exposes single and batch document ingestion, task status, document listing, image, and chat APIs.
- Celery + Redis runs PDF ingestion in the background.
- PyMuPDF parses PDF text and extracts page-linked images.
- Parent-child chunking preserves section context while indexing smaller child chunks.
- Local `BAAI/bge-m3` embeddings power PostgreSQL + pgvector retrieval.
- LangGraph orchestrates Planner, Retriever, Verifier, Answer, and Citation nodes.
- React + Vite provides a simple upload and citation-aware chat UI.
- Uploaded PDFs are persisted under the default demo user, so users can select an indexed document later and ask questions without uploading it again.

## First Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
copy .env.example .env
docker compose up -d postgres redis
uvicorn backend.app.main:app --reload
```

In another terminal:

```powershell
celery -A backend.app.tasks.celery_app worker --loglevel=INFO --pool=solo
```

Frontend:

```powershell
cd frontend
npm install
npm run dev
```

## Tests

```powershell
pytest
```

The early tests use fake providers so the architecture remains testable before external services are configured.

More detail:

- [Architecture](docs/ARCHITECTURE.md)
- [Testing Guide](docs/TESTING.md)

## Current Product Flow

The first product version uses a fixed `demo` owner instead of login.

1. Upload one or more PDFs.
2. The backend hashes each PDF and reuses an existing `demo` document when the same file was already uploaded.
3. New files are stored and indexed by Celery in the background; reused pending/running documents do not enqueue duplicate tasks.
4. The UI refreshes "My documents", shows per-file upload results, and lets you select an indexed PDF.
5. Chat can search either the current document or all indexed documents owned by `demo`.
