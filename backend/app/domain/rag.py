from dataclasses import dataclass, field
from typing import Literal

from backend.app.domain.documents import EvidenceImage


StopReason = Literal[
    "sufficient_evidence",
    "max_rounds_reached",
    "no_new_query",
    "no_new_evidence",
    "assessment_fallback",
    "no_search_direction",
]


@dataclass(frozen=True)
class TaskDecision:
    # 任务分类结果用于控制预算、风险策略和后续是否需要多轮检索。
    task_type: str
    requires_retrieval: bool
    risk_level: str
    resolved_entities: list[str] = field(default_factory=list)
    rationale: str = ""


@dataclass(frozen=True)
class RouteStep:
    # LLM planner 输出的有界执行步骤；单路由问题会退化为只有一个 step 的计划。
    step_id: str
    route: str
    task_type: str
    query: str
    purpose: str = ""
    max_rounds: int | None = None
    requires_assessment: bool = True
    requires_answer_verification: bool = False


@dataclass(frozen=True)
class RoutePlan:
    # route_plan 用于表达“先做文档画像，再找图，再补证据”这类中文复合问题。
    steps: list[RouteStep]
    plan_type: str = "single_step"
    rationale: str = ""
    used_llm: bool = False
    fallback_used: bool = False


@dataclass(frozen=True)
class QueryPlan:
    # 每一轮检索都会沉淀一个 QueryPlan，便于回放“问题是如何被改写与推进的”。
    original_question: str
    resolved_question: str
    queries: list[str]
    search_goal: str
    round_index: int


@dataclass(frozen=True)
class EvidenceAssessment:
    # 证据评估同时承担“能否回答”和“下一轮该往哪里搜”两个职责。
    sufficient: bool
    covered_aspects: list[str] = field(default_factory=list)
    missing_aspects: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    supporting_chunk_ids: list[str] = field(default_factory=list)
    next_search_goal: str | None = None
    confidence: float | None = None
    reason: str = ""
    used_llm: bool = False
    fallback_used: bool = False


@dataclass(frozen=True)
class AnswerVerification:
    # 答案级核验只检查最终答案是否忠实于已验证证据，不负责自由生成新答案。
    passed: bool
    triggered: bool = False
    trigger_reasons: list[str] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    missing_disclosures: list[str] = field(default_factory=list)
    citation_errors: list[str] = field(default_factory=list)
    overstatements: list[str] = field(default_factory=list)
    revision_instruction: str | None = None
    used_llm: bool = False
    fallback_used: bool = False
    reason: str = ""


@dataclass(frozen=True)
class ImageSelectionResult:
    # 图片选择结果同时携带诊断信息，避免编排层猜测 provider 是否实际调用模型。
    images: dict[str, list[EvidenceImage]] = field(default_factory=dict)
    strategy: str = "none"
    used_llm: bool = False
    fallback_used: bool = False
    candidate_count: int = 0
    selected_count: int = 0
    latency_ms: float = 0.0
    reason: str = ""
