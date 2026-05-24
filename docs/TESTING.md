# 测试与启动验证指南

## 自动化检查

后端单元与集成测试：

```powershell
.\.venv\Scripts\python.exe -m pytest
```

前端生产构建：

```powershell
cd frontend
npm run build
```

测试当前覆盖父子切块、嵌入提供器、上传去重、图片元数据、数据库启动准备与 FastAPI 仓储生命周期、路由与重排、评估指标、问答接口，以及真实 API 冒烟脚本的请求流程。

## 默认基础服务配置

本地应用通过 Docker 对外暴露的端口访问基础设施，以下值必须一致：

| 服务 | Docker 映射 | 应用配置 |
| --- | --- | --- |
| PostgreSQL | `localhost:5432` | `DATABASE_URL=postgresql+psycopg://chemrag:chemrag@localhost:5432/chemrag` |
| Redis | `localhost:6380 -> 6379` | `REDIS_URL=redis://localhost:6380/0` |
| MinIO API | `localhost:9100 -> 9000` | `MINIO_ENDPOINT=localhost:9100` |
| MinIO Console | `localhost:9101 -> 9001` | 浏览器管理界面，可选使用 |

检查当前读取到的 `.env` 配置：

```powershell
.\.venv\Scripts\python.exe scripts\check_env.py
```

## 从零启动并验证上传问答

### 1. 初始化

```powershell
.\scripts\setup_windows.ps1
```

脚本会安装后端与前端依赖、创建 `.env`（仅在文件不存在时）、启动 `postgres`、`redis`、`minio`，并运行后端测试。真实 `.env` 不进入 Git。

### 2. 编辑环境变量

在 `.env` 中配置真实的 `DEEPSEEK_API_KEY`，并确认：

```dotenv
REDIS_URL=redis://localhost:6380/0
OBJECT_STORAGE_PROVIDER=minio
MINIO_ENDPOINT=localhost:9100
EMBEDDING_LOCAL_FILES_ONLY=false
RERANKER_LOCAL_FILES_ONLY=false
```

首次运行需要下载嵌入与重排模型，因此应保留两项 `false`。模型下载完成后，离线运行时可将其改为 `true`。

### 3. 确认基础设施已运行

```powershell
docker compose up -d postgres redis minio
docker compose ps
.\.venv\Scripts\python.exe scripts\check_env.py
```

若上传立即失败，请优先检查 MinIO 是否启动以及 `.env` 的 `MINIO_ENDPOINT` 是否仍为旧端口；若任务一直无法开始，请检查 Redis URL 是否为 `6380`。

### 4. 分终端启动服务

在部署流程中可提前执行数据库结构准备；API 启动时也会幂等执行同一准备函数：

```powershell
.\.venv\Scripts\python.exe scripts\prepare_database.py
```

```powershell
# 终端 1
.\scripts\start_api.ps1
```

```powershell
# 终端 2
.\scripts\start_worker.ps1
```

```powershell
# 终端 3
.\scripts\start_frontend.ps1
```

API 健康检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
```

预期返回：

```text
status
------
ok
```

### 5. 验证真实上传与提问

推荐先运行自动冒烟验证，它会实际经过 API、MinIO、Redis/Celery、PostgreSQL 和问答流程：

```powershell
.\.venv\Scripts\python.exe scripts\smoke_api_flow.py ".\你的论文.pdf" --question "该论文报告了哪些关键实验结果？"
```

验收条件：

- 输出 `上传成功` 与 `索引完成`。
- 输出的切块数大于 `0`。
- 输出非空回答。
- 输出的引用数量大于 `0`。

随后可打开 `http://127.0.0.1:5173`，上传或选择文档并在界面重复提问。

## 质量评估

离线检索与路由回归评估：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_rag.py --cases eval\golden_cases.jsonl --output-dir eval\reports --llm-judge off --answer-provider fake
```

使用已配置回答模型进行评估：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_rag.py --cases eval\golden_cases.jsonl --output-dir eval\reports --llm-judge off
```

评估结果输出至被 Git 忽略的 `eval/reports/`，不会污染提交基线。

## 排错顺序

1. `docker compose ps`：确认 `postgres`、`redis`、`minio` 正在运行。
2. `scripts\check_env.py`：确认 Redis 指向 `6380`、MinIO 指向 `9100`、API Key 已配置。
3. API 终端日志：确认模型是否仍在首次下载或加载失败。
4. Worker 终端日志：确认 Celery 能连接 Redis 并完成索引任务。
5. `scripts\smoke_api_flow.py`：重新执行完整链路验证，获得最直接的失败步骤。
