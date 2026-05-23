# Testing Guide

## Fast Checks

```powershell
pytest
```

Current coverage:

- parent-child chunking
- deterministic fake embeddings
- ingestion service composition
- LangGraph RAG workflow
- FastAPI health endpoint
- batch upload accepts valid PDFs while rejecting invalid files per item
- duplicate pending/running uploads reuse existing documents without duplicate tasks

## Frontend Build

```powershell
cd frontend
npm run build
```

## RAG Quality Evaluation

The evaluation suite uses manually labeled golden cases to report retrieval quality,
answer quality, figure hit quality, route accuracy, and latency. It reports metrics
only; it does not fail on quality thresholds in v1.

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_rag.py --cases eval\golden_cases.jsonl --output-dir eval\reports --llm-judge off
```

For offline retrieval/route regression without calling the configured answer LLM:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_rag.py --cases eval\golden_cases.jsonl --output-dir eval\reports --llm-judge off --answer-provider fake
```

Router prototype embeddings are warmed before timed cases by default. To measure cold-start latency explicitly:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_rag.py --cases eval\golden_cases.jsonl --output-dir eval\reports --llm-judge off --answer-provider fake --warmup off
```

Optional LLM-as-judge mode reuses the configured DeepSeek/OpenAI-compatible settings:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_rag.py --cases eval\golden_cases.jsonl --output-dir eval\reports --llm-judge on
```

Outputs:

- `eval/reports/latest.json`
- `eval/reports/latest.md`

## Manual Smoke Test

Windows setup with all runtime data under this project directory:

```powershell
.\scripts\setup_windows.ps1
```

Start infrastructure:

```powershell
docker compose up -d postgres redis
```

Redis is mapped to `localhost:6380` to avoid conflicts with other local RAG stacks using `6379`.

Start API:

```powershell
.\scripts\start_api.ps1
```

Start Celery:

```powershell
.\scripts\start_worker.ps1
```

Start UI:

```powershell
.\scripts\start_frontend.ps1
```

Then open `http://127.0.0.1:5173`.

## Required Environment

Copy `.env.example` to `.env`, then set:

- `DEEPSEEK_API_KEY`
- `DEEPSEEK_MODEL`
- `EMBEDDING_MODEL=BAAI/bge-m3`
- `EMBEDDING_DEVICE=cpu` or `cuda`
- `MODEL_CACHE_DIR=storage/models`
- `ROUTER_CACHE_DIR=storage/router_cache`
- `RERANKER_MODEL=BAAI/bge-reranker-v2-m3`
- `RERANKER_DEVICE=cpu` or `cuda`

For Windows, keep the project on `D:` if you do not want runtime data on `C:`.
The current compose file stores Postgres data in `./storage/postgres`, not a Docker named volume.

## Environment Check

```powershell
.\.venv\Scripts\python.exe scripts/check_env.py
```

## Real PDF Smoke Test

After Postgres and Redis are running:

```powershell
.\.venv\Scripts\python.exe scripts/smoke_ingest_and_chat.py ".\Photoluminescence and Nonlinear Optical Properties of Transition Metal (Ag, Ni, Mn) Doped ZnO Nanoparticles(科研通-ablesci.com).pdf"
```

## Acceptance Baseline

Before using this project in a resume, verify:

- Uploading a PDF returns a task id.
- Uploading multiple PDFs returns one result per file, and one invalid file does not fail the whole batch.
- Task status eventually becomes `success`.
- `/api/documents` shows the indexed document and chunk count.
- Refreshing the UI shows previously uploaded documents without uploading again.
- Re-uploading the same PDF returns the existing document instead of creating a duplicate.
- References/Bibliography text is not inserted as retrieval child chunks.
- Current-document chat refuses to run without a selected document.
- Asking a question returns an answer.
- Every answer includes at least one citation with page number and quote.
- `pytest` and `npm run build` pass.
