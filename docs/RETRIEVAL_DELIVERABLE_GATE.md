# 检索节点可交付门禁

本文档定义检索节点进入可交付状态前必须通过的最小质量门禁。门禁目标不是替代人工评审，而是把 Ragas 暴露出的召回、路由、证据清洗和回答核验问题固化为可重复运行的验收命令。

## 门禁范围

当前门禁优先覆盖以下风险：

- 精确事实召回：作者、机构、参考文献编号等低语义但高精确度信息必须能被词法/混合检索命中。
- 混合问题召回：一个问题同时包含参考文献、材料体系和谱图/图片线索时，不能只走单一路由导致证据缺口。
- 拼写与 OCR 噪声：用户问题中的拼写噪声和 PDF 图注中的轻量 OCR 噪声不能直接击穿检索或回答。
- 可观测性：报告必须暴露向量、词法和 figure sidecar 候选数，便于定位失败属于哪一路召回。
- 保守回答：无引用断言风险、最大轮次耗尽和答案核验失败不能被平均指标掩盖。

## 本地门禁命令

先确保数据库结构包含词法索引：

```powershell
.\.venv\Scripts\python.exe scripts\prepare_database.py
```

运行 Ragas 暴露问题的 hybrid 回归。该命令临时关闭 reranker，用于隔离验证召回层本身：

```powershell
$env:RERANKER_ENABLED='false'
.\.venv\Scripts\python.exe scripts\evaluate_rag.py `
  --cases eval\retrieval_hybrid_regression_cases.jsonl `
  --output-dir eval\reports\retrieval_hybrid_regression_configured `
  --answer-provider configured `
  --assessment-provider configured `
  --ragas off `
  --scope from_case `
  --warmup off
```

然后执行召回层质量门禁：

```powershell
.\.venv\Scripts\python.exe scripts\assert_retrieval_quality.py `
  --report eval\reports\retrieval_hybrid_regression_configured\latest.json `
  --min-case-count 2 `
  --require-diagnostic-key diagnostics.recall.vector_candidate_count `
  --require-diagnostic-key diagnostics.recall.lexical_candidate_count `
  --require-diagnostic-key diagnostics.recall.figure_sidecar_candidate_count
```

再运行 reranker 开启版回归，验证强词法召回不会在重排阶段被挤出最终候选：

```powershell
$env:RERANKER_ENABLED='true'
.\.venv\Scripts\python.exe scripts\evaluate_rag.py `
  --cases eval\retrieval_hybrid_regression_cases.jsonl `
  --output-dir eval\reports\retrieval_hybrid_regression_reranker_configured `
  --answer-provider configured `
  --assessment-provider configured `
  --ragas off `
  --scope from_case `
  --warmup off
```

reranker 组合门禁额外要求暴露 `diagnostics.rerank.enabled` 和 `diagnostics.rerank.protected_lexical_count`：

```powershell
.\.venv\Scripts\python.exe scripts\assert_retrieval_quality.py `
  --report eval\reports\retrieval_hybrid_regression_reranker_configured\latest.json `
  --min-case-count 2 `
  --require-diagnostic-key diagnostics.recall.vector_candidate_count `
  --require-diagnostic-key diagnostics.recall.lexical_candidate_count `
  --require-diagnostic-key diagnostics.recall.figure_sidecar_candidate_count `
  --require-diagnostic-key diagnostics.rerank.enabled `
  --require-diagnostic-key diagnostics.rerank.protected_lexical_count
```

默认门禁要求：

| 指标 | 默认要求 |
| --- | ---: |
| `route_hit` | >= 0.95 |
| `target_hit` | >= 0.95 |
| `citation_page_hit` | >= 0.90 |
| `source_url_valid` | >= 1.00 |
| `retrieval_behavior_hit` | >= 0.95 |
| `answer_term_coverage` | >= 0.80 |
| `final_answer_accepted` | >= 0.95 |
| `unsupported_claim_risk` | <= 0.00 |
| `max_rounds_exhausted` | <= 0.00 |

逐 case 还会强制检查 `route_hit`、`target_hit`、`source_url_valid`、`retrieval_behavior_hit`、`final_answer_accepted` 必须为 `true`，并要求单 case `answer_term_coverage >= 0.50`。

## 扩展验收

在提交检索节点前，建议继续跑更大集合：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_rag.py `
  --cases eval\golden_cases_extended.jsonl `
  --output-dir eval\reports\golden_cases_extended_configured `
  --answer-provider configured `
  --assessment-provider configured `
  --ragas off `
  --scope from_case
```

扩展集合可以先使用较宽松阈值定位剩余失败，再逐步收紧：

```powershell
.\.venv\Scripts\python.exe scripts\assert_retrieval_quality.py `
  --report eval\reports\golden_cases_extended_configured\latest.json `
  --min-case-count 20 `
  --min-metric route_hit=0.90 `
  --min-metric target_hit=0.85 `
  --min-metric source_url_valid=1.0 `
  --max-metric unsupported_claim_risk=0.0
```

## 当前已知边界

- hybrid 最小回归已覆盖关闭 reranker 的召回层验收和开启 reranker 的组合验收；更大 golden/Ragas 集合仍需分别跑两种配置，确认排序稳定性不是只在两条回归样本上成立。
- `Si(111)` 归一化是窄规则，仅处理 `TPP/ZnO/S(111)`、`TPP-ML/ZnO/S(111)` 这类已观察到的 ZnO/Si 衬底 OCR 噪声，不做泛化化学实体改写。
- 作者/机构回答 fast path 只在问题和证据都出现作者名、且证据可解析机构标号时触发；它只输出 co-author/affiliation 和“不含个人贡献详情”的保守说明。
- Ragas 自动生成集仍建议定期扩容，新增失败样本应先沉淀到 `eval/retrieval_hybrid_regression_cases.jsonl` 或扩展 golden cases，再进入门禁。
