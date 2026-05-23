# Agentic-ChemRAG

Agentic-ChemRAG 是面向化学与材料科学论文的多智能体 RAG 系统。它支持 PDF 上传、异步解析与索引、基于文档范围的提问、页码引用以及关联图片返回。

## 架构概览

- FastAPI 提供上传、任务状态、文档列表、图片和问答接口。
- Celery 使用 Redis 执行后台 PDF 索引任务。
- PostgreSQL + pgvector 保存文档、切块、向量与图片元数据。
- MinIO 保存原始 PDF 与抽取图片，默认通过 `localhost:9100` 访问。
- 本地 `BAAI/bge-m3` 与 `BAAI/bge-reranker-v2-m3` 完成向量检索和重排。
- LangGraph 编排检索、验证、回答与引用流程。
- React + Vite 提供上传和引用感知的问答界面。

## 一次跑通上传与提问

以下步骤以 Windows PowerShell 为例。首次运行需要 Docker Desktop、Python 3.11+、Node.js、可用的 DeepSeek API Key，以及下载本地模型所需的网络。

### 1. 安装依赖并启动基础服务

在项目根目录执行：

```powershell
.\scripts\setup_windows.ps1
```

该脚本会创建虚拟环境、安装后端与前端依赖、在缺少时生成 `.env`、启动 `postgres`、`redis`、`minio` 三个容器，并运行后端测试。运行数据均保存在 `storage/` 下。

### 2. 配置 `.env`

打开 `.env`，至少填写真实的 `DEEPSEEK_API_KEY`。若 `.env` 是此前创建的文件，请同时确认以下默认链路配置：

```dotenv
REDIS_URL=redis://localhost:6380/0
OBJECT_STORAGE_PROVIDER=minio
MINIO_ENDPOINT=localhost:9100
MINIO_ACCESS_KEY=chemrag
MINIO_SECRET_KEY=chemrag-secret
MINIO_BUCKET=chemrag
MINIO_SECURE=false
EMBEDDING_LOCAL_FILES_ONLY=false
RERANKER_LOCAL_FILES_ONLY=false
```

`*_LOCAL_FILES_ONLY=false` 允许第一次启动时把模型下载到 `storage/models/`。模型已缓存后，可改为 `true` 以使用离线模式。

### 3. 确认容器与配置

```powershell
docker compose ps
.\.venv\Scripts\python.exe scripts\check_env.py
```

应能看到 `postgres`、`redis`、`minio` 处于运行状态；配置输出中 `Redis 连接` 应为 `redis://localhost:6380/0`，`MinIO 端点` 应为 `localhost:9100`，`DeepSeek API Key 已配置` 应为 `True`。

### 4. 启动应用

分别打开三个 PowerShell 终端，均在项目根目录执行：

```powershell
# 终端 1：API
.\scripts\start_api.ps1
```

```powershell
# 终端 2：后台索引 Worker
.\scripts\start_worker.ps1
```

```powershell
# 终端 3：前端界面
.\scripts\start_frontend.ps1
```

首次运行时 API 或 Worker 可能需要下载并加载模型，请等待其完成后再上传 PDF。

### 5. 在界面完成上传与提问

1. 打开 `http://127.0.0.1:5173`。
2. 点击 `Upload PDFs` 上传一篇 PDF。
3. 等待文档状态从 `indexing` 变为切块数量，例如 `42 chunks`。
4. 保持范围为 `Current`，输入问题并发送。
5. 确认页面返回答案、页码引用，并可打开对应 PDF 页。

### 6. 用脚本自动验证真实链路

在 API 与 Worker 正在运行时，可用一条命令验证“上传 -> MinIO 保存 -> Celery 索引 -> 问答 -> 引用”全过程：

```powershell
.\.venv\Scripts\python.exe scripts\smoke_api_flow.py ".\你的论文.pdf" --question "这篇论文的主要结论是什么？"
```

命令成功时会输出文档 ID、索引切块数、回答和引用数量。

## 常用验证

```powershell
.\.venv\Scripts\python.exe -m pytest
cd frontend
npm run build
```

更多测试与排错命令见 [docs/TESTING.md](docs/TESTING.md)，架构说明见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。
