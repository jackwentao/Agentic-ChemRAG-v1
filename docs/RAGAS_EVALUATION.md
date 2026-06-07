# Ragas 评测接入说明

本项目将 Ragas 作为可选的 LLM 评测层使用，用于补充答案忠实性、回答相关性和上下文质量评估。Ragas 不替代现有 pytest、golden case 命中率、路由命中、页码/图片命中、多轮检索成本和安全拒答策略指标。

## 安装依赖

Ragas 依赖放在 `eval` 可选依赖组中，不进入生产主依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,eval]"
```

启用真实 Ragas 评测还需要 `.env` 中配置 `DEEPSEEK_API_KEY`，并复用项目的 DeepSeek OpenAI-compatible 配置。

## 运行方式

默认离线评测不启用 Ragas：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_rag.py `
  --cases eval\golden_cases.jsonl `
  --output-dir eval\reports `
  --answer-provider fake `
  --assessment-provider fake `
  --ragas off
```

小样本启用 Ragas：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_rag.py `
  --cases eval\golden_cases.jsonl `
  --output-dir eval\reports `
  --answer-provider configured `
  --assessment-provider configured `
  --ragas on `
  --ragas-metrics basic `
  --ragas-sample-limit 10
```

指标分层：

- `basic`：`faithfulness`、`answer_relevancy`。
- `retrieval`：在 `basic` 基础上增加 `context_precision`。
- `all`：在 `retrieval` 基础上尝试增加 `context_recall`、`answer_correctness`；只有 case 显式提供 `reference_answer` 时才启用这些需要标准答案的指标。

## Golden Case 扩展字段

Ragas 可读取以下可选字段：

```json
{
  "reference_answer": "The paper reports ZnO photoluminescence behavior.",
  "reference_contexts": ["Fig. 2 PL spectrum of ZnO nanoparticles."],
  "ragas_tags": ["figure", "chemistry"]
}
```

`expected_answer_terms` 和 `expected_aspects` 只作为项目自身的确定性指标使用，不会冒充 Ragas 的标准答案。

## 报告输出

启用 Ragas 后，`eval/reports/latest.json` 会增加：

- 顶层 `ragas.summary`：平均分。
- 每个案例的 `result.ragas.metrics`：逐 case 分数。
- `ragas.skipped_metrics`：因缺少 `reference_answer` 跳过的指标。
- `ragas.errors`：Ragas 执行错误；默认不会中断基础 RAG 报告生成。

`latest.md` 会追加 `Ragas LLM 评测` 区块。

## 测试手段

新增和相关测试：

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests\test_ragas_adapter.py `
  tests\test_evaluation_cases.py `
  tests\test_evaluation_metrics.py `
  tests\test_evaluation_judge.py `
  tests\test_evaluate_rag.py
```

语法检查：

```powershell
.\.venv\Scripts\python.exe -m compileall backend\app\evaluation scripts\evaluate_rag.py
```

全量测试仍使用：

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## 基于项目 PDF 自动生成测试集

可以直接使用 `test_pdf/` 中的论文 PDF 调用 Ragas testset generation，生成项目评测脚本可读取的 golden case：

```powershell
.\.venv\Scripts\python.exe scripts\generate_ragas_testset.py `
  --pdf-dir test_pdf `
  --testset-size 20 `
  --output eval\ragas_generated_cases.jsonl `
  --manifest eval\ragas_generated_manifest.json
```

生成流程会把 PDF 按页解析为 LangChain Document，交给 Ragas 自动构建 KnowledgeGraph、采样 scenario 并生成问题与参考答案。随后脚本会把 Ragas 输出转换为本项目的 `GoldenCase` JSONL，并尝试用 `reference_contexts` 回查 PDF 页文本，填充 `expected_documents` 与 `expected_pages`，便于继续评估检索命中率。

如果要生成后立刻测试当前系统的大模型回答能力和检索能力：

```powershell
.\.venv\Scripts\python.exe scripts\generate_ragas_testset.py `
  --pdf-dir test_pdf `
  --testset-size 20 `
  --output eval\ragas_generated_cases.jsonl `
  --run-evaluation on `
  --answer-provider configured `
  --assessment-provider configured `
  --ragas on `
  --ragas-metrics all
```

输出文件：

- `eval/ragas_generated_cases.jsonl`：Ragas 自动生成并转换后的测试问题、参考答案和期望检索目标。
- `eval/ragas_generated_manifest.json`：参与生成的 PDF、哈希、样本数量等生成清单。
- `eval/reports/ragas_generated/latest.json` 与 `latest.md`：当前系统在这些自动生成问题上的检索、回答和 Ragas 指标报告。

注意：Ragas 生成和 Ragas LLM 评测都需要安装 `.[eval]` 可选依赖，并配置 `.env` 中的 `DEEPSEEK_API_KEY`。默认 `--scope all_documents` 不要求生成时的 PDF document_id 与数据库中已上传 UUID 完全一致；如果已经确认文件哈希能在数据库中匹配，也可以使用 `--scope current_document` 做单文档范围评测。

新增脚本的快速测试：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_generate_ragas_testset.py
```

## Ragas 生成缓存策略

`scripts/generate_ragas_testset.py` 默认不会重复生成测试集：

```powershell
.\.venv\Scripts\python.exe scripts\generate_ragas_testset.py `
  --output eval\ragas_generated_cases.jsonl
```

如果 `eval\ragas_generated_cases.jsonl` 已存在，脚本会直接复用该文件，不解析 PDF、不重建 Ragas KnowledgeGraph，也不会重新调用大模型生成问题答案。

只有主动刷新时才生成新问题：

```powershell
.\.venv\Scripts\python.exe scripts\generate_ragas_testset.py `
  --pdf-dir test_pdf `
  --testset-size 20 `
  --refresh on `
  --output eval\ragas_generated_cases.jsonl
```

如果是“加一批问题答案”，而不是覆盖旧测试集：

```powershell
.\.venv\Scripts\python.exe scripts\generate_ragas_testset.py `
  --pdf-dir test_pdf `
  --testset-size 20 `
  --refresh on `
  --append on `
  --output eval\ragas_generated_cases.jsonl
```

缓存文件默认位置：

- `eval/ragas_cache/page_sources.json`：PDF 按页解析后的文本缓存。
- `eval/ragas_cache/knowledge_graph.json`：Ragas KnowledgeGraph 缓存。
- `eval/ragas_cache/cache_manifest.json`：缓存签名，包含 PDF hash、embedding 模型、生成模型和 document_id 策略。

当 PDF hash、embedding 模型、DeepSeek 模型或 document_id 策略变化时，缓存会自动失效。需要强制重建 KG 时使用：

```powershell
.\.venv\Scripts\python.exe scripts\generate_ragas_testset.py `
  --refresh on `
  --force-rebuild-kg on
```
