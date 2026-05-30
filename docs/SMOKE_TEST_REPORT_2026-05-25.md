# Agentic-ChemRAG 冒烟测试报告

## 1. 报告信息

| 项目 | 内容 |
| --- | --- |
| 测试日期 | 2026-05-25 |
| 测试环境 | Windows PowerShell，本地开发环境 |
| 测试对象 | FastAPI API、Celery Worker、PostgreSQL/pgvector、Redis、MinIO、BGE 向量/重排模型、React 前端构建、DeepSeek 问答调用 |
| 测试目的 | 验证基础回归、应用启动、上传索引及问答主链路，暴露阻断运行的问题 |
| 测试结论 | 通过。基础回归、本地重排、DeepSeek 在线问答以及新 PDF 上传、异步索引、回答与引用完整链路均已验证；外部模型不可达时亦可保守降级 |

## 2. 测试范围

本轮冒烟覆盖以下路径：

1. 后端自动化测试全量执行。
2. 前端生产构建。
3. 配置读取与关键本地依赖端口连通性检查。
4. API 启动、`/api/health` 存活检查及 `/api/ready` 完整问答能力检查。
5. Celery Worker 启动及 Redis 连接。
6. 真实 PDF 上传、异步索引、检索重排与问答请求。
7. 故障定位后的重排器本地缓存加载验证与降级回归测试。
8. DeepSeek 连接异常时的证据评估、回答生成和图片选择降级验证。
9. Windows UTF-8 输出保护、readiness 状态契约以及模型缓存参数兼容性验证。

## 3. 环境与前置条件

| 依赖项 | 验证结果 | 说明 |
| --- | --- | --- |
| PostgreSQL | 端口 `5432` 可连接 | 用于文档、切块和向量数据 |
| Redis | 端口 `6380` 可连接 | Worker 启动后成功连接 |
| MinIO | 端口 `9100` 可连接 | 用于 PDF 与图片对象存储 |
| API | 可短时启动并返回健康检查 | 测试后进程已结束，不持续占用 `8000` 端口 |
| BGE Embedding | 可从本地缓存加载 | 索引过程成功执行 |
| BGE Reranker | 设置离线加载后可从本地缓存加载 | 缓存本身完整可用 |
| DeepSeek | 在线可连接并成功完成证据评估及回答生成 | 未记录任何密钥内容 |

说明：本地 `.env` 已设置 `RERANKER_LOCAL_FILES_ONLY=true` 以复用现有模型缓存；`.env` 为忽略文件，该运行配置需要在部署或其他开发环境中独立同步。

## 4. 执行记录

| 编号 | 验证项 | 执行方式 | 结果 |
| --- | --- | --- | --- |
| T01 | 后端完整回归 | `.\.venv\Scripts\python.exe -m pytest` | 通过，修复前 `122 passed`；LLM/重排降级后 `126 passed`；本轮 readiness、UTF-8 与缓存参数验证后 `135 passed` |
| T02 | 前端生产构建 | `cd frontend; npm run build` | 通过，Vite 成功生成 `dist` 产物 |
| T03 | 配置检查 | `.\.venv\Scripts\python.exe scripts\check_env.py` | 通过，关键连接配置可读取 |
| T04 | 基础设施探测 | 探测 `5432`、`6380`、`9100` | 通过，三个服务端口可连接 |
| T05 | API 健康检查 | 短时启动 API 并请求 `/api/health` | 通过，返回 `{"status":"ok"}` |
| T06 | Worker 启动 | 启动 Celery `--pool=solo` | 通过，连接 Redis 并进入 `ready` |
| T07 | 修复前真实上传索引 | `scripts\smoke_api_flow.py` 上传测试 PDF | 上传与索引通过，生成 `24` 个切块，Worker 单次索引约 `71.8s` |
| T08 | 修复前真实问答 | 同一脚本调用 `/api/chat` | 失败，HTTP `500`，重排器加载时尝试联网失败 |
| T09 | 重排器离线 warmup | 使用 `RERANKER_LOCAL_FILES_ONLY=true` 加载重排模型 | 通过，确认缓存可用而非模型损坏 |
| T10 | 重排异常降级回归 | `tests\test_retrieval_service.py` 新增 provider 故障场景 | 通过，重排失败时回退到向量召回结果 |
| T11 | 修复后 API 启动 | 以更新后的本地配置启动 API | 通过，重排模型从本地缓存加载，不再出现 Hugging Face 连接错误 |
| T12 | 重排修复后真实问答复验 | 再次执行真实上传/问答脚本 | 失败，HTTP `500`；定位出 DeepSeek 证据评估外部连接未处理 |
| T13 | LLM 容错修复后 HTTP 复验 | 临时关闭已单独验证的路由预热/重排开销，请求已有索引文档 `/api/chat` | 通过，HTTP `200`；`stop_reason=assessment_fallback`、`fallback_used=true`，返回资料限制且不生成未经验证的引用 |
| T14 | 获得网络权限后的在线问答复验 | 使用已有索引文档请求真实 `/api/chat` | 通过，返回 `1247` 字符回答、`2` 条引用；`stop_reason=sufficient_evidence`、`fallback_used=false` |
| T15 | 新 PDF 在线全链路复验 | `PYTHONIOENCODING=utf-8` 下执行 `scripts\smoke_api_flow.py` | 通过，首次上传 `status=pending`，异步索引产生 `27` 个切块，真实回答返回 `3` 条引用 |
| T16 | 冒烟脚本 UTF-8 防误判回归 | `tests\test_smoke_api_flow.py` | 通过，脚本在开始输出前将标准输出/错误显式配置为 UTF-8，并以安全替换处理极端字符 |
| T17 | readiness 契约回归 | `tests\test_readiness.py tests\test_api.py` | 通过，完整能力为 `ready/200`；LLM 不在线但可降级时为 `degraded/503` |
| T18 | 模型缓存参数真实 warmup | 本地缓存加载 embedding 与 reranker 并捕获 `sentence_transformers` 日志 | 通过，embedding 输出 `1024` 维向量，reranker 完成推理；两者 `cache_dir` 弃用日志均为 `0` |

## 5. 已发现问题

### ST-001：重排模型在线探测失败导致问答接口返回 500

| 项目 | 内容 |
| --- | --- |
| 严重级别 | P1 |
| 状态 | 已修复并验证 |
| 现象 | API 可健康启动，但 `/api/chat` 在检索重排阶段返回 `500` |
| 根因 | 本地已有 `BAAI/bge-reranker-v2-m3` 缓存，但运行配置未启用离线加载；重排器尝试访问 Hugging Face，受网络限制后抛出异常 |
| 影响 | 上传与索引成功后，用户无法完成问答 |

处置：

- 本地运行配置增加 `RERANKER_LOCAL_FILES_ONLY=true`，直接加载已有缓存。
- [backend/app/services/rag.py](../backend/app/services/rag.py) 增加重排失败降级逻辑；重排调用异常时返回向量召回结果，并记录 `skip_reason=provider_error`。
- [tests/test_retrieval_service.py](../tests/test_retrieval_service.py) 增加重排 provider 故障回退测试。

验证证据：

- 重排器离线 warmup 成功。
- API 启动日志显示本地重排权重加载成功，不再出现 Hugging Face 网络连接失败。
- 完整后端测试由 `122 passed` 增至 `123 passed`。

### ST-002：DeepSeek 连接异常导致证据评估节点返回 500

| 项目 | 内容 |
| --- | --- |
| 严重级别 | P1 |
| 状态 | 已修复接口中断问题；外部模型可用性仍取决于运行环境 |
| 现象 | 修复重排链路后，`/api/chat` 仍返回 HTTP `500` |
| 根因 | 证据评估节点调用 DeepSeek 时出现 `openai.APIConnectionError: Connection error.`，底层为外部套接字访问受限；当前代码只对超时或无效 JSON 做回退，对连接异常继续抛出 |
| 定位路径 | `backend/app/agents/rag_graph.py` 的 `_evidence_assessor` -> `backend/app/providers/llm.py` 的 `_assess_evidence` |
| 影响 | 修复前外部模型暂时不可达会导致真实问答整体失败；修复后会返回保守降级响应，无法生成需要模型验证的有据结论 |

处置：

1. `DeepSeekLlmProvider._assess_evidence` 已将连接失败、限流、服务端临时错误等可恢复异常转换为结构化 fallback assessment，并由 `limitations` 与诊断字段对外暴露。
2. `answer_with_evidence` 与图片选择调用已实施一致的可恢复异常回退，避免后续模型调用中断完整接口。
3. 已新增 provider 回归测试，覆盖证据评估连接失败、已验证证据回答连接失败和图片选择连接失败三条路径。
4. 已用真实 HTTP 请求验证当前无外部连接场景返回 `200`、`assessment_fallback` 与限制说明。

获得网络访问权限后已完成复验：真实 DeepSeek 证据评估与回答生成成功，且新 PDF 上传、异步索引、问答和引用链路均已通过。

### ST-003：健康检查未体现关键问答依赖降级状态

| 项目 | 内容 |
| --- | --- |
| 严重级别 | P2 |
| 状态 | 已修复并验证 |
| 现象 | `/api/health` 返回 `ok`，但随后 `/api/chat` 因模型依赖不可用返回 `500` |
| 影响 | 部署探活可能判断实例健康，实际用户链路不可用 |

处理结果：保留 `/api/health` 作为进程存活检查，新增 `/api/ready` 实际探测数据库、对象存储、任务后端、embedding、reranker 与 LLM。只有完整问答能力在线时返回 `200 ready`；仍可保守降级但能力不完整时返回 `503 degraded`，部署层不会误接收降级实例为完整健康。

### ST-004：模型加载参数存在依赖弃用警告

| 项目 | 内容 |
| --- | --- |
| 严重级别 | P3 |
| 状态 | 已修复并验证 |
| 现象 | 加载重排模型时输出 `cache_dir` 参数弃用警告 |
| 影响 | 未来升级 `sentence-transformers` 后可能产生兼容性风险 |

处理结果：embedding 按新版接口将缓存目录传入模型/处理器/配置参数；统一 Hugging Face cache 根目录为现有 `storage/models`，reranker 不再让 `CrossEncoder` 将 `cache_folder` 转交到旧 Transformer 参数。真实本地 warmup 捕获日志确认弃用提示不再出现。

### ST-005：Windows 默认控制台编码可能导致冒烟脚本打印回答失败

| 项目 | 内容 |
| --- | --- |
| 严重级别 | P3 |
| 状态 | 已修复并验证 |
| 现象 | 回答包含上标等非 GBK 字符时，`scripts\smoke_api_flow.py` 在默认 PowerShell 输出阶段触发 `UnicodeEncodeError` |
| 影响 | 实际 HTTP 问答已成功，但脚本最终退出码为失败，容易误判链路状态 |

此前可通过环境变量临时规避：

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe scripts\smoke_api_flow.py ".\test_pdf\<任一PDF>" --question "What are the main reported results?"
```

处理结果：`scripts\smoke_api_flow.py` 进入执行入口时显式将标准输出与标准错误设置为 UTF-8，并启用安全替换。成功的 HTTP 问答不再因为 PowerShell 默认区域编码无法打印个别字符而被脚本错误判失败。

## 6. 修复后当前状态

| 能力 | 状态 | 说明 |
| --- | --- | --- |
| 后端单元/集成测试 | 通过 | `126 passed` |
| 前端生产构建 | 通过 | 可生成生产静态资源 |
| API 存活与 readiness 契约 | 通过 | `/api/health` 仅表示进程存活；`/api/ready` 区分完整问答在线与降级可服务 |
| Worker 与 Redis 链路 | 通过 | Worker 可启动并接受任务 |
| PDF 上传与索引 | 通过 | 已真实完成一次索引，得到 `24` 个切块 |
| 检索重排模块 | 通过/可降级 | 本地重排加载成功；故障时已具备向量召回回退 |
| 真实问答异常降级 | 通过 | DeepSeek 不可连接时 `/api/chat` 返回 `200`、`assessment_fallback` 与限制说明 |
| 已索引文档真实有据回答 | 通过 | DeepSeek 在线返回有效回答与 `2` 条引用，未触发 fallback |
| 新 PDF 上传到问答端到端 | 通过 | 首次上传、Worker 新索引 `27` 个切块、真实回答及 `3` 条引用均完成 |
| Windows 冒烟输出稳定性 | 通过 | 脚本自行配置 UTF-8，不再要求调用方预设控制台编码变量 |
| 模型缓存参数兼容性 | 通过 | embedding 与 reranker 真实 warmup 均无 `cache_dir` 弃用日志 |

## 7. 结论与验收判断

本轮冒烟成功暴露并关闭了两类导致问答 `500` 的故障：重排器配置/调用异常，以及外部 LLM 暂时不可达。系统现已能在 LLM 连接失败时保守降级并返回可解释限制说明，不会将候选证据误当成已验证结论；readiness 能把这种降级与完整能力在线区分开。Windows 输出误判和模型加载弃用警告也已闭环。在获得网络访问权限后，真实 DeepSeek 在线问答与新 PDF 上传、异步索引、回答和引用链路均通过，因此本轮核心业务链路可判定为冒烟通过。

本报告发现的阻断项与紧随其后的运行稳定性项均已处理并配套自动化验证；后续可继续推进图片查询和文档画像查询的专用轻路径评测。

## 8. 复验命令

基础回归：

```powershell
.\.venv\Scripts\python.exe -m pytest
cd frontend
npm run build
```

模型与服务验证：

```powershell
.\.venv\Scripts\python.exe -c "from backend.app.config import Settings; from backend.app.providers.rerankers import build_reranker_provider; p=build_reranker_provider(Settings()); p.warmup(); print('reranker warmup ok')"
.\scripts\start_api.ps1
.\scripts\start_worker.ps1
Invoke-RestMethod http://127.0.0.1:8000/api/health
Invoke-RestMethod http://127.0.0.1:8000/api/ready
```

真实端到端复验：

```powershell
.\.venv\Scripts\python.exe scripts\smoke_api_flow.py ".\test_pdf\<任一PDF>" --question "What are the main reported results?"
```
