# Agentic-ChemRAG 工程化加固计划

## 文档定位

本文档用于规划下一阶段工程化改造，覆盖以下七项目标：

1. 引入 Alembic，所有数据库结构变更走版本化 migration。
2. PDF 上传增加文件大小限制、页数限制、MIME 校验、恶意或畸形 PDF 防护，并改为流式落盘和对象存储。
3. Worker 从 MinIO 读取 PDF，不再依赖 API 节点本地文件路径。
4. 建立更清晰的领域错误体系。
5. 增强可观测性。
6. 增加 Playwright 端到端测试。
7. 补齐完整容器化部署。

本计划只定义改造方案、阶段顺序、测试手段和验收标准，不直接改变当前运行逻辑。后续实施时应按本文拆分任务，逐项提交和验证。

## 总体原则

- 文档、注释和用户可见说明默认使用中文。
- 保持模块解耦，上传、存储、任务、解析、错误、观测、部署各自有清晰边界。
- 数据库结构变更必须可追踪、可回滚、可在部署流水线中独立执行。
- 上传链路先保证安全和资源可控，再追求体验。
- Worker 不依赖 API 本地磁盘，所有跨进程数据通过数据库和对象存储传递。
- 错误响应对前端稳定，对日志和诊断足够具体，对用户不暴露敏感内部细节。
- 每个模块改造后都必须补对应自动化测试，并提供手动验证方式。

## 当前问题基线

### 数据库

当前数据库结构由 `backend/db/init.sql` 和 `backend/app/database.py::prepare_database()` 共同维护。该方式便于本地快速启动，但存在以下问题：

- schema 来源重复，长期容易漂移。
- 缺少版本号，无法明确当前数据库处于哪个结构版本。
- 启动时执行结构补齐会让应用启动承担迁移职责。
- 缺少 downgrade 或回滚策略。

### 上传与解析

当前上传接口将文件一次性读入内存，再写入本地文件和对象存储。该方式存在以下风险：

- 大文件和批量上传可能导致 API 进程内存压力过大。
- 只通过扩展名判断 PDF，不能可靠识别 MIME 和文件头。
- 缺少文件大小、页数、解析时间、图片数量等资源预算。
- 缺少对加密 PDF、损坏 PDF、嵌套对象异常、超大图片、极端页数 PDF 的清晰处理。

### Worker 输入

当前 Celery 任务接收本地 `file_path`。该方式在 API 与 Worker 不在同一机器或同一挂载卷时不可用，也让后续容器化和横向扩展受限。

### 错误处理

当前部分接口直接返回异常字符串，错误类型和用户提示混杂。需要建立稳定错误码、HTTP 状态码、日志细节和前端展示信息之间的映射关系。

### 可观测性

项目已有 `/api/health`、`/api/ready` 和部分诊断字段，但还缺少统一请求 ID、结构化日志、任务指标、上传指标、检索和 LLM 成本指标，以及容器环境下的健康检查和日志规范。

### 端到端测试

当前后端测试覆盖较多，前端主要依赖构建检查。还需要 Playwright 覆盖真实用户链路：上传、等待索引、提问、查看引用、刷新恢复会话、异常提示。

### 容器化

当前 `docker-compose.yml` 只包含 PostgreSQL、Redis、MinIO。API、Worker、Frontend 仍依赖宿主机环境，不利于一致部署和 CI 验证。

## 目标架构

### 数据流

```text
前端选择 PDF
  -> API 流式接收
  -> UploadGuard 校验大小、MIME、文件头、页数和安全预算
  -> ObjectStorage 写入 MinIO 临时对象
  -> DocumentService 注册文档和对象 key
  -> Celery 任务只接收 document_id
  -> Worker 从数据库读取 pdf_object_key
  -> Worker 从 MinIO 拉取 PDF 到受控临时文件
  -> Parser 在资源预算内解析
  -> Repository 保存 chunks、images、figures、profiles
  -> Worker 更新索引状态和诊断摘要
```

### 部署流

```text
docker compose up postgres redis minio
  -> migrate 服务执行 alembic upgrade head
  -> api 服务启动
  -> worker 服务启动
  -> frontend 或 nginx 服务启动
  -> Playwright smoke 验证完整链路
```

## 阶段计划

### 阶段 0：基线固定与任务拆分

目标：在改造前固定当前可运行基线，避免迁移过程中无法区分旧问题和新问题。

工作项：

- 运行后端测试：`.\.venv\Scripts\python.exe -m pytest`。
- 运行前端构建：`cd frontend; npm run build`。
- 记录当前 `.env.example`、数据库结构、启动脚本和 smoke 脚本行为。
- 确认一个小型测试 PDF 可用于 E2E，不依赖私有论文。

产出：

- 当前测试结果记录。
- 后续 Playwright 使用的最小 PDF fixture。
- 若现有环境无法跑通，先记录阻断项，不在本阶段大改代码。

验收：

- 能明确当前失败是否属于已存在问题。
- 后续每个阶段都有可比较的测试基线。

## 阶段 1：引入 Alembic

### 目标

用 Alembic 取代应用启动时的隐式 schema 演进。所有新增表、字段、索引和数据修复都通过 migration 版本管理。

### 设计

新增结构：

```text
alembic.ini
migrations/
  env.py
  script.py.mako
  versions/
```

迁移策略：

- `backend/db/init.sql` 保留为本地全新数据库初始化参考，后续不再作为主要变更入口。
- `backend/app/database.py::prepare_database()` 改为只做轻量连接检查或最终删除。
- `scripts/prepare_database.py` 改为调用 `alembic upgrade head`，或新增 `scripts/migrate_database.py`。
- 首个 Alembic revision 需要表达当前完整 schema，作为 baseline。
- 如果现有数据库已经有表，需要支持 `alembic stamp head` 或幂等 baseline 流程。

### 任务

1. 添加 `alembic` 依赖。
2. 初始化 Alembic 配置，读取项目 `Settings.database_url`。
3. 创建 `0001_baseline_schema`，包含当前表、索引、pgvector extension。
4. 将后续 schema 变更拆成独立 revision，例如上传安全字段、任务诊断字段。
5. 调整 API 启动，不在请求服务启动阶段执行结构升级。
6. 更新 README 和 `docs/TESTING.md` 的数据库准备命令。

### 测试

自动化测试：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_config.py tests\test_api.py
```

迁移测试：

```powershell
docker compose down
Remove-Item -Recurse -Force storage\postgres
docker compose up -d postgres
.\.venv\Scripts\alembic.exe upgrade head
.\.venv\Scripts\python.exe -m pytest tests\test_api.py tests\test_readiness.py
```

验收：

- 新数据库可通过 `alembic upgrade head` 创建完整 schema。
- 已有数据库可通过 stamp 或 baseline 流程接入 Alembic。
- API 启动不再负责执行结构升级。
- migration 文件可以被 CI 单独执行。

### 风险

- pgvector extension、HNSW、ivfflat 索引在 Alembic 中需要显式 SQL。
- 已有本地数据库可能已存在结构，需要提供接入说明。
- 测试库和开发库要避免互相污染。

## 阶段 2：PDF 上传安全与流式存储

### 目标

上传链路资源可控、错误清晰、安全边界明确，并避免一次性把 PDF 读入内存。

### 新增配置

建议添加以下环境变量：

```dotenv
UPLOAD_MAX_FILE_BYTES=52428800
UPLOAD_MAX_BATCH_FILES=10
PDF_MAX_PAGES=300
PDF_PARSE_PROBE_TIMEOUT_SECONDS=10
PDF_MAX_EXTRACTED_IMAGES=500
PDF_MAX_IMAGE_BYTES=20971520
UPLOAD_SPOOL_MAX_MEMORY_BYTES=8388608
UPLOAD_ALLOWED_MIME_TYPES=application/pdf
```

默认值可按本地研发体验保守设置，生产部署再调大。

### 模块设计

新增或调整模块：

```text
backend/app/services/upload_validation.py
backend/app/services/pdf_safety.py
backend/app/domain/errors.py
backend/app/api/error_handlers.py
```

核心职责：

- `UploadValidator`：校验文件名、扩展名、MIME、文件头、大小、批量数量。
- `PdfSafetyInspector`：使用 PyMuPDF 做轻量 probe，检查页数、加密状态、是否可打开、基础元信息。
- `StreamingUploadStore`：将 UploadFile 分块写入临时文件，同时计算 SHA256。
- `DocumentService`：接收已验证的上传结果，不直接处理原始 `UploadFile`。
- `ObjectStorage`：支持从文件或流写入 MinIO，避免只支持 `bytes`。

### MIME 与文件头策略

校验顺序：

1. 文件名存在且扩展名为 `.pdf`。
2. HTTP `content_type` 必须属于允许列表，但不能只信任该字段。
3. 文件头必须以 `%PDF-` 开始，允许前面少量空白字节。
4. 文件大小不能超过配置上限。
5. PyMuPDF 能在 probe 阶段打开。
6. 加密 PDF 默认拒绝，除非后续明确支持密码。
7. 页数不能超过 `PDF_MAX_PAGES`。

### 病毒 PDF 保护范围

本阶段不实现完整杀毒引擎，但要建立多层防护：

- 拒绝损坏、加密、超大、超页数、无法解析 PDF。
- 解析 probe 和正式解析都设置预算和异常映射。
- 对图片提取数量和单图大小设置上限。
- 所有临时文件放入受控目录，使用随机文件名。
- 后续预留 `MalwareScanner` 接口，可接入 ClamAV 或云扫描服务。

接口预留：

```text
MalwareScanner.scan(path) -> ScanResult
NoopMalwareScanner
ClamAvMalwareScanner
```

### 流式落盘和对象存储

上传流程：

```text
UploadFile.stream/read chunk
  -> 临时文件
  -> 同步计算 sha256 和 size
  -> probe PDF
  -> object_storage.put_file(pdf_key, temp_path)
  -> repository.add_document(...)
  -> enqueue ingestion(document_id)
  -> 清理临时文件
```

注意：

- API 仍可保留 `storage/uploads` 作为临时缓冲目录，但不作为 Worker 输入来源。
- 文件去重使用流式计算出的 SHA256。
- 对象存储写入成功后再注册或更新 `pdf_object_key`。
- 若任务入队失败，文档状态必须进入 `failed`，并保留可诊断错误码。

### 测试

单元测试：

- 正常 PDF 上传成功。
- 非 PDF 扩展名拒绝。
- MIME 不匹配拒绝。
- 文件头不是 `%PDF-` 拒绝。
- 超过大小限制拒绝。
- 超过页数限制拒绝。
- 加密 PDF 拒绝。
- 损坏 PDF 返回领域错误。
- 批量数量超过限制拒绝。
- 上传过程中对象存储失败时状态正确。

推荐命令：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_upload_dedup.py tests\test_api.py
```

需要新增：

```text
tests/test_upload_validation.py
tests/test_pdf_safety.py
tests/test_streaming_upload.py
```

手动验证：

```powershell
Invoke-WebRequest -Uri http://127.0.0.1:8000/api/documents/batch -Method Post -Form @{ files = Get-Item ".\test_pdf\sample.pdf" }
```

验收：

- 上传大文件时 API 内存不随文件大小线性暴涨。
- 非法 PDF 返回稳定错误码和用户可读信息。
- 合法 PDF 写入 MinIO，并能进入索引队列。

## 阶段 3：Worker 从 MinIO 读取 PDF

### 目标

Celery 任务只依赖数据库和对象存储，不依赖 API 节点本地文件路径。

### 任务接口

当前：

```text
ingest_document(document_id, file_path, filename, file_hash, file_size)
```

目标：

```text
ingest_document(document_id)
```

Worker 内部流程：

```text
repository.get_document_for_ingestion(document_id)
  -> 获取 pdf_object_key、filename、file_hash、file_size、owner_id
  -> object_storage.open(pdf_object_key)
  -> 写入 worker 临时文件
  -> IngestionService.ingest(document_id, temp_path, ...)
  -> 清理临时文件
```

### Repository 调整

新增方法：

```text
get_document_for_ingestion(document_id) -> DocumentIngestionRecord | None
```

该方法返回：

- `document_id`
- `owner_id`
- `filename`
- `original_filename`
- `file_hash`
- `file_size`
- `pdf_object_key`
- `indexing_status`

### 兼容策略

为降低一次性改造风险，可以先临时兼容旧任务参数：

```text
ingest_document(document_id, file_path=None, ...)
```

但最终应删除 `file_path` 参数和相关本地路径依赖。

### 测试

新增：

```text
tests/test_ingestion_task_object_storage.py
```

覆盖：

- Worker 从对象存储读取 PDF。
- 对象不存在时文档状态为 `failed`。
- 下载过程中异常时错误码明确。
- 临时文件最终清理。
- API 和 Worker 使用不同本地目录时仍可索引。

手动验证：

```powershell
.\.venv\Scripts\python.exe scripts\smoke_api_flow.py ".\test_pdf\sample.pdf" --question "这篇论文的主要结论是什么？"
```

验收：

- Celery 任务参数不包含本地 PDF 路径。
- 删除 API 本地上传文件后，Worker 仍可从 MinIO 完成索引。
- 容器化 API 和 Worker 不共享上传目录时链路仍通过。

## 阶段 4：领域错误体系

### 目标

统一领域错误、HTTP 响应、日志细节和前端提示，避免直接暴露内部异常字符串。

### 错误模型

新增 `backend/app/domain/errors.py`：

```text
DomainError
  code: str
  message: str
  status_code: int
  user_message: str
  details: dict
  cause: Exception | None
```

建议错误码：

```text
UPLOAD_INVALID_TYPE
UPLOAD_FILE_TOO_LARGE
UPLOAD_BATCH_TOO_LARGE
PDF_INVALID
PDF_ENCRYPTED
PDF_TOO_MANY_PAGES
PDF_PARSE_FAILED
OBJECT_STORAGE_UNAVAILABLE
OBJECT_NOT_FOUND
TASK_QUEUE_UNAVAILABLE
INGESTION_FAILED
DOCUMENT_NOT_FOUND
DOCUMENT_NOT_INDEXED
LLM_UNAVAILABLE
RERANKER_UNAVAILABLE
CONFIG_INVALID
```

API 响应格式：

```json
{
  "error": {
    "code": "PDF_TOO_MANY_PAGES",
    "message": "PDF 页数超过限制。",
    "request_id": "..."
  }
}
```

批量上传响应中每个文件也使用同一错误结构。

### FastAPI 集成

新增异常处理器：

```text
app.add_exception_handler(DomainError, handle_domain_error)
app.add_exception_handler(RequestValidationError, handle_validation_error)
```

处理策略：

- 用户可修复错误返回 400 或 413。
- 资源不存在返回 404。
- 外部依赖不可用返回 503。
- 任务超时返回 504。
- 未预期异常返回 500，但日志记录详细堆栈。

### 前端适配

前端不再假设 `detail` 是字符串，应兼容：

```text
data.error.code
data.error.message
data.detail
```

### 测试

新增：

```text
tests/test_domain_errors.py
tests/test_api_error_responses.py
```

覆盖：

- 每类领域错误映射到正确 HTTP 状态。
- 响应包含 `error.code`、`error.message`、`request_id`。
- 日志有内部异常，响应不泄露堆栈和密钥。
- 前端构建通过。

验收：

- API 错误响应结构稳定。
- 前端能显示用户可读错误。
- 日志可通过 request_id 追踪到具体异常。

## 阶段 5：可观测性增强

### 目标

让上传、索引、检索、LLM、对象存储、任务队列和请求链路都可追踪、可计量、可排障。

### 请求追踪

新增中间件：

```text
RequestIdMiddleware
```

行为：

- 接收 `X-Request-ID`，没有则生成。
- 响应头返回 `X-Request-ID`。
- 写入日志上下文。
- 传入 Celery task headers 或 task kwargs。

### 结构化日志

建议统一字段：

```text
timestamp
level
logger
request_id
task_id
document_id
session_id
event
latency_ms
status
error_code
```

日志事件：

- `upload.started`
- `upload.validated`
- `upload.stored`
- `ingestion.queued`
- `ingestion.started`
- `ingestion.parsed`
- `ingestion.embedded`
- `ingestion.completed`
- `ingestion.failed`
- `chat.started`
- `retrieval.completed`
- `llm.completed`
- `chat.completed`

### 指标

新增 `/metrics` 或预留 Prometheus 集成。

核心指标：

```text
http_requests_total
http_request_duration_seconds
upload_bytes_total
upload_rejected_total
ingestion_tasks_total
ingestion_task_duration_seconds
ingestion_pages_total
ingestion_chunks_total
object_storage_operations_total
retrieval_duration_seconds
llm_calls_total
llm_call_duration_seconds
llm_failures_total
```

### Readiness 调整

保持：

- `/api/health` 只做存活检查。
- `/api/ready` 做完整依赖检查。

新增：

- `/api/version` 返回 git sha、应用版本、migration head。
- readiness 输出当前 Alembic revision。

### 测试

新增：

```text
tests/test_observability.py
```

覆盖：

- 响应包含 `X-Request-ID`。
- 错误响应包含同一 request_id。
- 日志记录包含 request_id 和 error_code。
- `/api/ready` 包含 migration 状态。
- `/metrics` 可返回基础指标。

验收：

- 任意失败上传可通过 request_id 在日志中定位。
- Celery 索引失败有 document_id、task_id、error_code。
- smoke 流程可观察关键阶段耗时。

## 阶段 6：Playwright 端到端测试

### 目标

覆盖用户真实路径，避免只验证后端单点能力。

### 前端测试结构

新增：

```text
frontend/playwright.config.ts
frontend/e2e/
  smoke-upload-chat.spec.ts
  upload-errors.spec.ts
  session-restore.spec.ts
```

建议 npm scripts：

```json
{
  "test:e2e": "playwright test",
  "test:e2e:headed": "playwright test --headed"
}
```

### 测试场景

`smoke-upload-chat.spec.ts`：

- 打开前端。
- 上传小型 PDF。
- 等待文档状态变为 chunks。
- 提问。
- 断言助手回答出现。
- 断言引用出现。
- 点击引用 PDF 链接，确认打开成功。

`upload-errors.spec.ts`：

- 上传非 PDF 文件，显示类型错误。
- 上传超大文件 fixture 或 mock API，显示大小错误。
- 上传损坏 PDF，显示 PDF 无效。

`session-restore.spec.ts`：

- 提问后刷新页面。
- 会话列表仍能看到历史会话。
- 点击会话能恢复消息。

### 环境策略

优先使用真实 API、真实 Worker、真实 MinIO、真实 Postgres 的 smoke 测试。为 CI 控制成本，可以再提供 mock 模式：

- `E2E_REAL_BACKEND=true`：真实链路。
- `E2E_REAL_BACKEND=false`：前端 API mock，只验证 UI 行为。

### 测试命令

本地真实链路：

```powershell
docker compose up -d postgres redis minio
.\.venv\Scripts\alembic.exe upgrade head
.\scripts\start_api.ps1
.\scripts\start_worker.ps1
.\scripts\start_frontend.ps1
cd frontend
npm run test:e2e
```

CI 链路：

```powershell
docker compose -f docker-compose.yml -f docker-compose.e2e.yml up --build --abort-on-container-exit e2e
```

验收：

- Playwright 可在无头模式跑通至少一个完整上传问答链路。
- 错误上传场景有可见提示。
- 会话恢复场景通过。

## 阶段 7：完整容器化部署

### 目标

API、Worker、Frontend、Migration、E2E 都能通过 Docker Compose 启动，减少宿主机依赖。

### 镜像设计

新增：

```text
backend/Dockerfile
frontend/Dockerfile
docker-compose.override.yml
docker-compose.prod.yml
docker-compose.e2e.yml
.dockerignore
```

服务：

```text
postgres
redis
minio
migrate
api
worker
frontend
e2e
```

### API 镜像

职责：

- 安装 Python 依赖。
- 暴露 `8000`。
- 启动 `uvicorn backend.app.main:app`。
- 不执行 migration。

### Worker 镜像

职责：

- 与 API 共享同一后端镜像或同一基础镜像。
- 启动 `celery -A backend.app.tasks.celery_app worker`。
- 从 MinIO 读取 PDF。
- 使用独立临时目录。

### Migration 服务

职责：

- 等待 Postgres 可用。
- 执行 `alembic upgrade head`。
- 成功退出。
- API 和 Worker 依赖该服务成功。

### Frontend 镜像

开发模式：

- Vite dev server。

生产模式：

- `npm run build`。
- 使用 nginx 或静态服务托管 `dist`。
- API 地址通过环境变量或反向代理配置注入。

### Compose 配置

开发：

```powershell
docker compose up --build postgres redis minio migrate api worker frontend
```

生产近似：

```powershell
docker compose -f docker-compose.yml -f docker-compose.prod.yml up --build -d
```

E2E：

```powershell
docker compose -f docker-compose.yml -f docker-compose.e2e.yml up --build --abort-on-container-exit e2e
```

### 配置管理

`.env.example` 需要区分本地宿主机和容器网络：

宿主机：

```dotenv
DATABASE_URL=postgresql+psycopg://chemrag:chemrag@localhost:5432/chemrag
REDIS_URL=redis://localhost:6380/0
MINIO_ENDPOINT=localhost:9100
```

容器内：

```dotenv
DATABASE_URL=postgresql+psycopg://chemrag:chemrag@postgres:5432/chemrag
REDIS_URL=redis://redis:6379/0
MINIO_ENDPOINT=minio:9000
```

建议新增：

```text
.env.example
.env.docker.example
```

### 测试

自动化：

```powershell
docker compose up --build migrate api worker frontend
docker compose ps
```

容器内测试：

```powershell
docker compose run --rm api python -m pytest
docker compose run --rm frontend npm run build
```

验收：

- 新机器只安装 Docker Desktop，也能启动完整系统。
- API、Worker、Frontend 都在容器中运行。
- Worker 不依赖 API 容器本地文件。
- `migrate` 服务可重复执行且幂等。
- E2E compose 可以一键跑通 smoke。

## 推荐实施顺序

建议按以下顺序推进：

1. 阶段 0：基线固定。
2. 阶段 1：Alembic baseline。
3. 阶段 4：领域错误体系基础，因为后续上传、Worker、观测都要复用错误码。
4. 阶段 2：上传安全和流式存储。
5. 阶段 3：Worker 从 MinIO 读取 PDF。
6. 阶段 5：可观测性增强。
7. 阶段 7：完整容器化部署。
8. 阶段 6：Playwright E2E，先本地跑，再接入容器和 CI。

说明：Playwright 可以提前搭脚手架，但完整真实链路建议等容器化和 Worker MinIO 改造完成后再作为强验收门禁。

## 建议提交拆分

### PR 1：Alembic 基线

范围：

- 新增 Alembic 配置。
- 新增 baseline migration。
- 调整数据库准备脚本。
- 更新文档。

测试：

- `alembic upgrade head`
- `pytest tests/test_api.py tests/test_readiness.py`

### PR 2：领域错误基础

范围：

- 新增 `DomainError`。
- 新增 FastAPI 错误处理器。
- 改造已有上传和任务状态部分错误响应。

测试：

- `pytest tests/test_domain_errors.py tests/test_api_error_responses.py`
- `npm run build`

### PR 3：上传安全与流式存储

范围：

- 新增上传配置。
- 新增 `UploadValidator`、`PdfSafetyInspector`。
- `DocumentService` 改为接收已验证上传对象。
- `ObjectStorage` 增加文件流写入。

测试：

- `pytest tests/test_upload_validation.py tests/test_pdf_safety.py tests/test_upload_dedup.py tests/test_api.py`

### PR 4：Worker MinIO 输入

范围：

- Celery 任务参数改为 `document_id`。
- Worker 从 MinIO 下载 PDF 到临时目录。
- 删除 API/Worker 共享本地上传路径假设。

测试：

- `pytest tests/test_ingestion_task_object_storage.py tests/test_ingestion.py tests/test_smoke_api_flow.py`
- 手动跑 `scripts/smoke_api_flow.py`。

### PR 5：可观测性

范围：

- 请求 ID 中间件。
- 结构化日志。
- 基础指标。
- readiness 增加 migration 状态。

测试：

- `pytest tests/test_observability.py tests/test_readiness.py tests/test_api.py`

### PR 6：容器化

范围：

- Backend Dockerfile。
- Frontend Dockerfile。
- migrate/api/worker/frontend compose 服务。
- `.env.docker.example`。

测试：

- `docker compose up --build migrate api worker frontend`
- `docker compose run --rm api python -m pytest`
- `docker compose run --rm frontend npm run build`

### PR 7：Playwright E2E

范围：

- Playwright 配置。
- smoke、错误上传、会话恢复测试。
- E2E compose 服务。

测试：

- `cd frontend; npm run test:e2e`
- `docker compose -f docker-compose.yml -f docker-compose.e2e.yml up --build --abort-on-container-exit e2e`

## 数据库变更清单

可能新增或调整字段：

```text
documents.upload_error_code
documents.upload_error_detail
documents.uploaded_at
documents.page_count
documents.mime_type
documents.pdf_object_key NOT NULL after backfill
documents.indexing_diagnostics_json
documents.indexing_task_id
```

可能新增表：

```text
ingestion_events
  event_id
  document_id
  task_id
  event_type
  status
  error_code
  detail_json
  created_at

app_metadata
  key
  value
  updated_at
```

是否新增 `ingestion_events` 需要按实现复杂度决定。如果短期只用日志和 `documents.indexing_diagnostics_json`，可以先不建表。

## 配置变更清单

新增建议配置：

```dotenv
UPLOAD_MAX_FILE_BYTES=52428800
UPLOAD_MAX_BATCH_FILES=10
UPLOAD_SPOOL_MAX_MEMORY_BYTES=8388608
UPLOAD_ALLOWED_MIME_TYPES=application/pdf
PDF_MAX_PAGES=300
PDF_PARSE_PROBE_TIMEOUT_SECONDS=10
PDF_MAX_EXTRACTED_IMAGES=500
PDF_MAX_IMAGE_BYTES=20971520
MALWARE_SCANNER_PROVIDER=noop
REQUEST_ID_HEADER=X-Request-ID
LOG_FORMAT=json
METRICS_ENABLED=true
APP_VERSION=0.1.0
```

容器配置：

```dotenv
DATABASE_URL=postgresql+psycopg://chemrag:chemrag@postgres:5432/chemrag
REDIS_URL=redis://redis:6379/0
MINIO_ENDPOINT=minio:9000
FRONTEND_API_BASE_URL=http://api:8000
```

## 测试矩阵

| 模块 | 自动化测试 | 手动验证 |
| --- | --- | --- |
| Alembic | `alembic upgrade head`、迁移测试 | 删除本地数据库后重建 |
| 上传校验 | `tests/test_upload_validation.py` | 上传非 PDF、损坏 PDF、超大 PDF |
| PDF 安全 | `tests/test_pdf_safety.py` | 上传加密 PDF、超页数 PDF |
| 对象存储 | `tests/test_streaming_upload.py` | MinIO 控制台查看 PDF 对象 |
| Worker | `tests/test_ingestion_task_object_storage.py` | 删除 API 本地文件后索引仍成功 |
| 错误处理 | `tests/test_domain_errors.py` | 前端显示稳定中文错误 |
| 可观测性 | `tests/test_observability.py` | 根据 request_id 查日志 |
| 容器化 | compose smoke | 新机器一键启动 |
| E2E | Playwright | 浏览器完整上传问答 |

## 完整验收命令

本地宿主机模式：

```powershell
docker compose up -d postgres redis minio
.\.venv\Scripts\alembic.exe upgrade head
.\.venv\Scripts\python.exe -m pytest
cd frontend
npm run build
npm run test:e2e
```

容器模式：

```powershell
docker compose up --build migrate api worker frontend
docker compose run --rm api python -m pytest
docker compose run --rm frontend npm run build
docker compose -f docker-compose.yml -f docker-compose.e2e.yml up --build --abort-on-container-exit e2e
```

真实链路 smoke：

```powershell
.\.venv\Scripts\python.exe scripts\smoke_api_flow.py ".\test_pdf\sample.pdf" --question "这篇论文的主要结论是什么？"
```

## 风险与缓解

| 风险 | 影响 | 缓解 |
| --- | --- | --- |
| Alembic baseline 与已有数据库不一致 | 本地开发库迁移失败 | 提供 `stamp head` 接入步骤和全新库验证 |
| PDF probe 过严 | 合法论文被拒绝 | 错误码可观测，限制值配置化，先保守再根据样本调优 |
| PyMuPDF 无法强制中断所有异常 PDF | Worker 卡住或超时 | Celery time limit、解析预算、临时文件隔离 |
| Worker MinIO 下载增加 IO | 索引耗时增加 | 使用本地临时文件，记录下载耗时，后续评估流式解析 |
| 错误响应改造影响前端 | 前端提示异常 | 前端兼容旧 `detail` 和新 `error` 结构 |
| 指标和日志引入过多复杂度 | 开发成本上升 | 先落 request_id 和关键事件日志，再接 Prometheus |
| 容器模型下载慢 | 首次启动体验差 | 模型缓存 volume、预下载脚本、离线模式说明 |
| E2E 真实链路耗时长 | CI 不稳定 | 区分 mock E2E 和真实 smoke E2E |

## 最终完成标准

全部阶段完成后，应满足：

- 数据库结构由 Alembic 管理，部署时显式迁移。
- API 上传大文件时不一次性把完整文件放入内存。
- PDF 上传具备大小、页数、MIME、文件头、加密和损坏文件防护。
- Worker 仅通过数据库和 MinIO 获取 PDF，不依赖 API 本地路径。
- API 错误响应使用稳定领域错误码。
- 请求、上传、索引、问答和外部依赖调用可通过 request_id 追踪。
- Playwright 能跑通上传、索引、问答、引用和会话恢复。
- Docker Compose 能启动完整应用栈，并能在容器内跑测试。
