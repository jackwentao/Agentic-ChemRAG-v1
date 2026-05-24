# Agentic-ChemRAG 架构

## 目标

当前版本提供可运行的化学/材料文献问答链路：PDF 上传、父子切块、本地嵌入、pgvector 检索、LangGraph 多代理问答、文档库、资产展示与带引用回答。

## 模块边界

- `backend/app/main.py` 只装配 FastAPI 应用、中间件、路由、数据库生命周期和启动预热，不承载具体接口业务。
- `backend/app/database.py` 管理数据库引擎创建与模式准备；同一准备函数可由应用启动和部署脚本复用。
- `backend/app/api/dependencies.py` 统一管理依赖注入；读取 FastAPI 托管的 repository，复用对象存储和 provider，并按请求构建带可变诊断状态的检索服务与 graph。
- `backend/app/api/routers/documents.py` 提供上传、批量上传和文档列表接口。
- `backend/app/api/routers/chat.py` 提供问答编排、作用域解析和引用响应映射。
- `backend/app/api/routers/assets.py` 提供 PDF 与图片资产读取接口。
- `backend/app/api/routers/maintenance.py` 提供健康检查、任务状态和索引维护接口。
- `domain` 维护纯数据结构，不依赖基础设施。
- `parsing` 维护 PDF 解析与切块策略。
- `providers` 维护 LLM、嵌入和重排适配器。
- `repositories` 维护持久化实现。
- `services` 组合业务用例，其中 `DocumentService` 隔离上传去重与索引排队逻辑。
- `agents` 维护 LangGraph 状态与节点。
- `tasks` 维护 Celery 集成。

依赖方向：

```text
FastAPI router -> injected service/repository -> provider/repository/storage
LangGraph node -> RetrievalService/provider
domain -> no project dependency
```

## 依赖生命周期

FastAPI `lifespan` 在接收请求前建立数据库引擎、执行 `prepare_database()` 并将单个 `PgVectorRepository` 存放在 `app.state` 中；请求依赖只读取该对象。停止应用时统一释放引擎，因此接口请求既不创建连接池，也不执行模式升级。

部署流水线也可执行 `scripts/prepare_database.py`，使用相同的准备函数先行升级数据库。Celery 任务与离线脚本只按工作需要创建数据访问对象，不隐式改变数据库结构。

对象存储客户端、embedding/LLM/reranker provider 与 `QueryRouter` 通过缓存依赖复用，从而避免在每个 HTTP 请求中重复初始化客户端或模型。

`RetrievalService` 与 `MultiAgentRagGraph` 按请求创建，因为检索服务保存本次请求的 `last_*_diagnostics`；如果全局复用，会使并发请求相互覆盖观测数据。

## 当前 RAG 流程

```text
PDF upload
  -> DocumentService registers owner_id=demo and deduplicates by file_hash
  -> Celery ingest_document task
  -> PyMuPDFParser extracts page text and images
  -> ObjectImageStore stores page-linked image assets
  -> ParentChildChunker creates retrieval child chunks
  -> embedding provider generates vectors
  -> PgVectorRepository stores documents, images, chunks and vectors
  -> /api/chat invokes request-scoped MultiAgentRagGraph
  -> Planner -> Retriever -> Verifier -> Answerer -> ImageSelector
  -> UI renders answers, citations and assets
```

## 扩展点

- 将固定的 `demo` owner 替换为认证后的租户/用户依赖，无需改动各路由业务。
- 增加 RAGFlow/MinerU 等解析适配器，处理更复杂的 PDF。
- 引入更强的证据验证器和手工标注的评估案例。
- 将 parent context 扩展与精确 child evidence 一并返回。
