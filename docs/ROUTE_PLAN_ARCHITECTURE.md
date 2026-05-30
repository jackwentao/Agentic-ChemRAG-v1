# Route Plan 架构图

本阶段将原来的“单 route 路由”升级为“有界 route_plan 执行”。LLM Planner 默认负责理解中文问题和会话上下文，规则 Router 作为超时、格式错误或不可用时的兜底。旧的单路由问题会被规范化为只有一个 step 的计划，因此兼容现有 `chunk_search`、`figure_index` 与 `document_profile` 链路。

```mermaid
flowchart TD
    A["用户中文问题 + 会话上下文 + 文档范围"] --> B["LLM Planner: 生成有界 route_plan"]
    B -->|成功| C["Plan Guardrails: route 白名单 / 最多 3 步 / 每步预算"]
    B -->|超时或格式无效| D["规则 Router 兜底"]
    D --> E["单步 route_plan"]
    C --> F["Step Controller: 选择当前 step"]
    E --> F

    F --> G{"当前 step.route"}
    G -->|chunk_search| H["文本 chunk 检索"]
    G -->|figure_index| I["图片/图题索引检索"]
    G -->|document_profile| J["文档画像检索"]

    H --> K["Evidence Assessor: 证据充分性判断"]
    I --> K
    J --> K

    K --> L{"当前 step 是否需要补检索"}
    L -->|证据不足且仍有预算| M["Query Refiner: 规则式补充查询"]
    M --> F
    L -->|停止当前 step| N["Step Finalizer: 保存本步证据、限制与诊断"]

    N --> O{"是否还有下一个 step"}
    O -->|有| F
    O -->|无| P["Answerer: 汇总多步证据并生成回答"]
    P --> Q["Image Selector: 选择可展示图片"]
    Q --> R["返回答案、引用、图片、限制说明与 route_plan 诊断"]
```

## 关键约束

- `route_plan.steps` 最多 3 步，避免模型制造无界工具循环。
- 每个 step 只能使用白名单 route：`chunk_search`、`figure_index`、`document_profile`。
- 单步问题保持旧行为：`route_plan.steps` 长度为 1。
- 同一个 step 内仍保留原有“检索 -> 证据评估 -> 补检索”的循环。
- 跨 step 的证据会在 `Step Finalizer` 汇总，最终由 `Answerer` 统一输出。
- 规则 Router 只作为 fallback，不再作为中文复杂问题的默认理解入口。

