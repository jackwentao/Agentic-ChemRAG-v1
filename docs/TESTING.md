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

测试当前覆盖父子切块、嵌入提供器、上传去重、图片元数据、数据库启动准备与 FastAPI 仓储生命周期、路由与重排、结构化证据充分性评估、readiness 组件探测、资料限制输出、评估指标、问答接口，以及真实 API 冒烟脚本的请求流程。

## 多轮会话记忆验证

本项目将完整展示历史和推理注入上下文分开测试：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_conversation_context.py tests\test_chat_memory_api.py tests\test_rag_graph_memory.py
```

验收重点：

- `chat_messages` 保存完整用户与助手文本、引用和资料限制，不保存工具调用或 LangGraph 中间节点状态。
- `chat_contexts` 使用确定性策略重建：最近轮次完整保留，较早轮次按结构化字段截取，不调用 LLM。
- `/api/chat` 未携带 `session_id` 时返回新会话 ID；后续请求携带同一 ID 时会读取并注入上下文。
- `/api/chat/sessions` 可列出会话，`/api/chat/sessions/{session_id}/messages` 可恢复完整消息流。
- LangGraph 首轮检索 query 能看到压缩上下文，但最终答案仍只依据 `verified_evidence`。

## 就绪探测与冒烟脚本回归

针对 Windows 输出、模型缓存参数与完整问答能力就绪状态的确定性回归：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_smoke_api_flow.py tests\test_embeddings.py tests\test_rerankers.py tests\test_readiness.py tests\test_api.py
```

验收重点：

- 冒烟脚本将标准输出和标准错误显式配置为 UTF-8，模型答案包含上标等字符时不再因打印失败返回错误退出码。
- embedding provider 通过 `model_kwargs`、`processor_kwargs`、`config_kwargs` 传递 Transformer 缓存目录；reranker 使用统一的 Hugging Face 缓存根定位模型，避免 `CrossEncoder` 内部将 `cache_folder` 转交为弃用参数。两者均应以真实 warmup 确认无对应弃用警告。
- `/api/health` 只表示 API 进程可响应；`/api/ready` 实际探测数据库、对象存储、任务后端、嵌入、重排器与 LLM 能力。
- `/api/ready` 仅在完整问答能力在线时返回 HTTP `200`；有降级路径但非完整能力时返回 HTTP `503` 且 `status=degraded`。

## 第一批架构契约验证

第一批引入的 `EvidenceAssessment` 与 `limitations` 契约已被第二批闭环复用。可执行以下针对性测试确认结构化评估与输出链路：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_llm_verifier.py tests\test_rag_graph.py tests\test_api.py tests\test_evaluate_rag.py
```

验收重点：

- 证据评估返回 `sufficient`、覆盖项、缺失项、冲突项与下一检索目标等结构化字段。
- 首轮证据不足时，回答不会调用无证据生成路径，且 API 返回可展示的 `limitations`。
- 评估响应无效或超时时，回退状态与资料限制能够被观察到，且不会基于候选证据生成回答。
- 证据评估、回答生成或图片选择遇到外部模型连接异常时，接口走保守降级路径而不是返回未处理异常。
- 前端生产构建通过，并能渲染 API 返回的“资料限制”区域。

## 第二批多轮检索验证

第二批新增程序控制的补充检索闭环，并保持查询生成不产生额外 LLM 调用。针对性验证命令：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_rag_graph.py tests\test_api.py tests\test_evaluate_rag.py
```

验收重点：

- 首轮证据充分时立即停止，`stop_reason=sufficient_evidence` 且只有一条检索历史。
- 第一轮缺少关键方面时，第二轮查询由 `EvidenceAssessment` 的缺口生成，且评估使用累计证据。
- 检索最多运行三轮；达到预算时以 `stop_reason=max_rounds_reached` 停止并输出限制。
- 规则生成重复查询时以 `stop_reason=no_new_query` 停止。
- 后续检索未产生新证据时以 `stop_reason=no_new_evidence` 停止，且不增加一次无意义的证据评估调用。
- 证据评估回退时以 `stop_reason=assessment_fallback` 停止，不依据不可靠缺口追加检索。
- API 诊断返回 `retrieval_rounds`、当前轮次、最大轮次与停止原因，便于统计追加检索成本。

## 第三批评测基线验证

第三批不增加新的推理节点，而是为当前自适应闭环增加可度量的成本与效果基线。针对性验证命令：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_evaluation_cases.py tests\test_evaluation_metrics.py tests\test_evaluate_rag.py tests\test_rag_graph.py tests\test_api.py
```

验收重点：

- API 诊断提供 `assessment_rounds` 与 `llm_call_counts`，分别用于逐轮解释证据判断和统计按节点路径标记使用 LLM 的阶段调用；后者是架构比较指标，不是账单计数。
- 黄金样例兼容旧字段，并可新增任务类别、风险等级、预期覆盖面、必须披露限制、预期检索行为和最大可接受轮次。
- 离线报告输出首轮成功率、重试触发率、观察到的重试成功率、无收益重试率、平均检索轮次及平均已追踪 LLM 调用。
- 默认质量评测即使使用 `fake` 回答，也由已配置真实模型判断证据是否充分；纯 fake 判断仅供无需网络和 API 调用的确定性回归。
- `latest.md` 使用中文说明评测模式、指标、停止原因、分类结果和失败案例。
- 图片选择的模型调用尚未纳入 `llm_call_counts`，因为当前 provider 未暴露可可靠区分模型调用与回退选择的状态。

## 第四批任务预算与增量重排验证

第四批将简单问题的单轮目标落实为程序预算，并将多轮成本优化到重排输入层。针对性验证命令：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_task_policy.py tests\test_retrieval_service.py tests\test_llm_verifier.py tests\test_rag_graph.py tests\test_api.py tests\test_evaluate_rag.py
```

节点级验收重点：

| 节点或模块 | 固定输入与关注输出 | 对应测试 |
| --- | --- | --- |
| `TaskClassifier` / `BudgetPolicy` | 简单事实为一轮；图片/总结为两轮；多条件与高风险为三轮 | `tests/test_task_policy.py` |
| `Retriever` / `Reranker` | 第二轮传入历史证据键，只重排新增候选；诊断记录被排除数量 | `tests/test_retrieval_service.py` |
| `EvidenceAssessor` | 真实模型输入包含任务类型与明确覆盖边界，不扩大用户要求 | `tests/test_llm_verifier.py` |
| `RetrievalController` | 简单事实证据不足首轮停止；复杂问题仍允许补查 | `tests/test_rag_graph.py` |
| `Planner` / `QueryRouter` | 固定问题落入正确检索路由，缓存预热不改变判断结果 | `tests/test_query_router.py` |
| `QueryRefiner` | 固定缺口生成补充查询；重复查询及时停止 | `tests/test_rag_graph.py` |
| `Answerer` / `ImageSelector` | 仅使用已验证证据生成回答与选择图片；不足时保守输出 | `tests/test_rag_graph.py`、`tests/test_image_selection.py` |
| `ChatDiagnostics` | API 输出任务类型、动态预算与阶段诊断 | `tests/test_api.py` |
| `EvaluationWarmup` / `ReportRenderer` | embedding、路由、重排独立预热并在中文报告展示 | `tests/test_evaluate_rag.py` |

下一轮提醒：优先实现**图片查询**和**文档画像查询**的专用轻路径，再比较是否需要引入 LLM 查询重写。

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

问答完整能力就绪检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/ready
```

该接口会实际触发本地模型与外部 LLM 探测，首次调用可能包含模型初始化耗时。返回 `status=ready` 且 `complete_qa_online=True` 才表示完整问答能力在线；若返回 HTTP `503` 与 `status=degraded`，说明进程仍可服务保守回退，但不应被视为完整能力健康实例。

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

低成本回答生成、真实证据充分性评估的质量基线：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_rag.py --cases eval\golden_cases.jsonl --output-dir eval\reports --llm-judge off --answer-provider fake
```

该命令需要配置 `DEEPSEEK_API_KEY`。其中回答文本使用模拟实现降低生成成本，但每轮证据是否足以回答问题由真实模型判断，因此可观察追加检索行为。
默认启用预热，报告会分别显示嵌入预热、路由预热与重排器预热时延；案例时延统计不再包含这些显式预热步骤。

使用真实证据评估与真实回答模型进行端到端评估：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_rag.py --cases eval\golden_cases.jsonl --output-dir eval\reports --llm-judge off
```

无需真实模型、只验证评测代码链路与报告格式的确定性回归：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_rag.py --cases eval\golden_cases.jsonl --output-dir eval\reports --llm-judge off --answer-provider fake --assessment-provider fake
```

评估结果输出至被 Git 忽略的 `eval/reports/`，不会污染提交基线。中文报告 `latest.md` 的“自适应检索成本”、“停止原因”、“按任务类别”与“按风险等级”区块用于比较多轮检索的收益和成本；其中“观察到的重试成功率”表示追加检索后最终取得充分证据的观察结果，不等价于严格反事实提升。

## 排错顺序

1. `docker compose ps`：确认 `postgres`、`redis`、`minio` 正在运行。
2. `scripts\check_env.py`：确认 Redis 指向 `6380`、MinIO 指向 `9100`、API Key 已配置。
3. API 终端日志：确认模型是否仍在首次下载或加载失败。
4. Worker 终端日志：确认 Celery 能连接 Redis 并完成索引任务。
5. `scripts\smoke_api_flow.py`：重新执行完整链路验证，获得最直接的失败步骤。
