from time import perf_counter

from langgraph.graph import END, StateGraph

from backend.app.agents.state import RagGraphState
from backend.app.domain.documents import Evidence, EvidenceImage
from backend.app.domain.rag import AnswerVerification, EvidenceAssessment, QueryPlan, RoutePlan, RouteStep
from backend.app.observability.diagnostics import (
    build_answer_verification_diagnostics,
    build_assessment_round_diagnostics,
    build_image_selection_diagnostics,
    build_llm_diagnostics,
    build_retrieval_round_diagnostics,
    build_route_diagnostics,
    build_verifier_diagnostics,
    increment_llm_call_count,
    set_total_latency,
    update_stage_proxy_hit,
)
from backend.app.providers.llm import LlmProvider
from backend.app.retrieval.refinement import (
    HybridQueryRefiner,
    LlmQueryRefiner,
    QueryRefinementPlan,
    RetrievalStrategyDecision,
    RetrievalStrategySelector,
    RuleQueryRefiner,
    build_retrieval_policy_context,
    normalized_query,
    refinement_diagnostics,
)
from backend.app.routing.query_router import QueryRouter, RouteContext
from backend.app.routing.answer_verification import AnswerVerificationPolicy
from backend.app.routing.task_policy import BudgetPolicy, TaskClassifier, assessment_requirements
from backend.app.services.image_selection import ImageSelectionContext, ImageSelectionService
from backend.app.services.rag import RetrievalService


class MultiAgentRagGraph:
    def __init__(
        self,
        retrieval_service: RetrievalService,
        llm_provider: LlmProvider,
        retrieval_top_k: int = 5,
        figure_top_k: int = 8,
        profile_top_k: int = 5,
        query_router: QueryRouter | None = None,
        task_classifier: TaskClassifier | None = None,
        budget_policy: BudgetPolicy | None = None,
        answer_verification_policy: AnswerVerificationPolicy | None = None,
        strategy_selector: RetrievalStrategySelector | None = None,
        query_refiner: HybridQueryRefiner | None = None,
        image_selection_service: ImageSelectionService | None = None,
    ) -> None:
        self.retrieval_service = retrieval_service
        self.llm_provider = llm_provider
        self.retrieval_top_k = retrieval_top_k
        self.figure_top_k = figure_top_k
        self.profile_top_k = profile_top_k
        self.query_router = query_router or QueryRouter(getattr(retrieval_service, "embedding_provider", None))
        self.task_classifier = task_classifier or TaskClassifier()
        self.budget_policy = budget_policy or BudgetPolicy()
        self.answer_verification_policy = answer_verification_policy or AnswerVerificationPolicy()
        self.strategy_selector = strategy_selector or RetrievalStrategySelector()
        self.image_selection_service = image_selection_service or ImageSelectionService(llm_provider)
        self.query_refiner = query_refiner or HybridQueryRefiner(
            RuleQueryRefiner(),
            LlmQueryRefiner(llm_provider),
        )
        self.graph = self._build().compile()

    def invoke(
        self,
        question: str,
        document_ids: list[str] | None = None,
        scope: str | None = None,
        session_id: str | None = None,
        conversation_context: str | None = None,
    ) -> RagGraphState:
        # invoke 负责初始化首轮状态，并把执行总耗时回填到 diagnostics。
        started_at = perf_counter()
        current_query = _build_contextual_question(question, conversation_context)
        state = self.graph.invoke(
            {
                "question": question,
                "original_question": question,
                "session_id": session_id,
                "conversation_context": conversation_context,
                "document_ids": document_ids,
                "scope": scope,
                "query_plans": [
                    QueryPlan(
                        original_question=question,
                        resolved_question=current_query,
                        queries=[current_query],
                        search_goal="结合对话上下文完成首轮检索" if conversation_context else "使用原始问题完成首轮检索",
                        round_index=1,
                    )
                ],
                "current_query": current_query,
                "retrieval_round": 1,
                "max_retrieval_rounds": 1,
                "route_plan": None,
                "current_step_index": 0,
                "completed_route_steps": [],
                "plan_verified_evidence": [],
                "step_has_assessment": False,
                "continue_retrieval": False,
                "stop_reason": None,
                "accumulated_evidence": [],
                "retrieval_history": [],
                "assessment_history": [],
                "llm_call_counts": {"evidence_assessment": 0, "answer_generation": 0, "image_selection": 0},
                "limitations": [],
                "warnings": [],
                "answer_revision_count": 0,
            }
        )
        total_latency_ms = (perf_counter() - started_at) * 1000
        diagnostics = set_total_latency(state.get("diagnostics"), total_latency_ms)
        return {
            **state,
            "total_latency_ms": total_latency_ms,
            "diagnostics": diagnostics,
        }

    def _build(self) -> StateGraph:
        # 图结构被拆成若干单一职责节点：
        # planner 负责路由，retriever 负责召回，assessor 负责判断证据是否足够，
        # controller/refiner 决定是否继续检索，answerer/image_selector 负责最终输出。
        graph = StateGraph(RagGraphState)
        graph.add_node("planner", self._planner)
        graph.add_node("task_classifier", self._task_classifier)
        graph.add_node("budget_policy", self._budget_policy)
        graph.add_node("retriever", self._retriever)
        graph.add_node("evidence_assessor", self._evidence_assessor)
        graph.add_node("retrieval_controller", self._retrieval_controller)
        graph.add_node("query_refiner", self._query_refiner)
        graph.add_node("step_finalizer", self._step_finalizer)
        graph.add_node("answerer", self._answerer)
        graph.add_node("answer_verifier", self._answer_verifier)
        graph.add_node("answer_reviser", self._answer_reviser)
        graph.add_node("image_selector", self._image_selector)
        graph.add_edge("planner", "task_classifier")
        graph.add_edge("task_classifier", "budget_policy")
        graph.add_edge("budget_policy", "retriever")
        graph.add_conditional_edges(
            "retriever",
            self._next_after_retriever,
            {"assess": "evidence_assessor", "control": "retrieval_controller"},
        )
        graph.add_edge("evidence_assessor", "retrieval_controller")
        graph.add_conditional_edges(
            "retrieval_controller",
            self._next_after_controller,
            {"refine_query": "query_refiner", "finalize": "step_finalizer"},
        )
        graph.add_conditional_edges(
            "query_refiner",
            self._next_after_query_refiner,
            {"retrieve": "retriever", "finalize": "step_finalizer"},
        )
        graph.add_conditional_edges(
            "step_finalizer",
            self._next_after_step_finalizer,
            {"next_step": "task_classifier", "answer": "answerer"},
        )
        graph.add_edge("answerer", "answer_verifier")
        graph.add_conditional_edges(
            "answer_verifier",
            self._next_after_answer_verifier,
            {"revise": "answer_reviser", "images": "image_selector"},
        )
        graph.add_edge("answer_reviser", "image_selector")
        graph.add_edge("image_selector", END)
        graph.set_entry_point("planner")
        return graph

    def _planner(self, state: RagGraphState) -> RagGraphState:
        # planner 先尝试让 LLM 输出有界 route_plan；失败时回退到规则路由。
        # 单步问题会被规范化为只有一个 step 的 route_plan，兼容旧执行链路。
        started_at = perf_counter()
        context = RouteContext(
            scope=state.get("scope") or ("current_document" if state.get("document_ids") else "all_documents"),
            document_ids=state.get("document_ids"),
        )
        contextual_question = _build_contextual_question(state["question"], state.get("conversation_context"))
        decision = self.query_router.route(contextual_question, context)
        route_plan = self.llm_provider.plan_route(
            state["question"],
            conversation_context=state.get("conversation_context"),
            scope=context.scope,
            document_ids=state.get("document_ids"),
        )
        if route_plan is None:
            route_plan = _route_plan_from_rule_decision(decision.route, contextual_question, decision.rationale)
        first_step = route_plan.steps[0]
        route_latency_ms = (perf_counter() - started_at) * 1000
        diagnostics = {
            **state.get("diagnostics", {}),
            "route": build_route_diagnostics(
                latency_ms=route_latency_ms,
                selected_route=first_step.route,
                confidence=decision.confidence,
                needs_llm_router=decision.needs_llm_router,
                is_multi_step=decision.is_multi_step or len(route_plan.steps) > 1,
                scores=decision.scores,
                lexical_scores=decision.lexical_scores,
                semantic_scores=decision.semantic_scores,
                rationale=[*decision.rationale, *([f"llm_plan:{route_plan.rationale}"] if route_plan.used_llm and route_plan.rationale else [])],
            ),
            "route_plan": _route_plan_diagnostics(route_plan, current_step_index=0, completed_steps=[]),
        }
        return {
            **state,
            "route": first_step.route,
            "route_plan": route_plan,
            "current_step_index": 0,
            "completed_route_steps": [],
            "plan_verified_evidence": [],
            "route_scores": decision.scores,
            "route_lexical_scores": decision.lexical_scores,
            "route_semantic_scores": decision.semantic_scores,
            "route_confidence": decision.confidence,
            "route_rationale": diagnostics["route"]["rationale"],
            "needs_llm_router": decision.needs_llm_router,
            "is_multi_step": decision.is_multi_step or len(route_plan.steps) > 1,
            "route_latency_ms": route_latency_ms,
            "plan": _plan_for_route(first_step.route),
            "current_query": first_step.query,
            "query_plans": [
                QueryPlan(
                    original_question=state["original_question"],
                    resolved_question=first_step.query,
                    queries=[first_step.query],
                    search_goal=first_step.purpose or _plan_for_route(first_step.route),
                    round_index=1,
                )
            ],
            "diagnostics": diagnostics,
        }

    def _retriever(self, state: RagGraphState) -> RagGraphState:
        # 这里根据 planner 的 route 决定调用哪一种 retrieval 接口，
        # 同时把本轮结果并入 accumulated_evidence，供后续多轮评估使用。
        started_at = perf_counter()
        route = state.get("route", "chunk_search")
        query = state.get("current_query", state["original_question"])
        exclude_keys = (
            {(item.document_id, item.chunk_id) for item in state.get("accumulated_evidence", [])}
            if state.get("retrieval_round", 1) > 1
            else None
        )
        if route == "figure_index":
            evidence = self.retrieval_service.retrieve_figures(
                query,
                top_k=self.figure_top_k,
                document_ids=state.get("document_ids"),
                exclude_keys=exclude_keys,
            )
        elif route == "document_profile":
            evidence = self.retrieval_service.retrieve_document_profiles(
                query,
                top_k=self.profile_top_k,
                document_ids=state.get("document_ids"),
                exclude_keys=exclude_keys,
            )
        else:
            evidence = self.retrieval_service.retrieve(
                query,
                top_k=self.retrieval_top_k,
                document_ids=state.get("document_ids"),
                exclude_keys=exclude_keys,
            )
        latency_ms = (perf_counter() - started_at) * 1000
        accumulated, new_evidence_count = _merge_evidence(state.get("accumulated_evidence", []), evidence)
        recall_diagnostics = dict(getattr(self.retrieval_service, "last_recall_diagnostics", {}))
        rerank_diagnostics = dict(getattr(self.retrieval_service, "last_rerank_diagnostics", {}))
        history = [
            *state.get("retrieval_history", []),
            build_retrieval_round_diagnostics(
                round_index=state.get("retrieval_round", 1),
                query=query,
                latency_ms=latency_ms,
                candidate_count=int(recall_diagnostics.get("candidate_count", len(evidence))),
                returned_count=len(evidence),
                new_evidence_count=new_evidence_count,
                accumulated_evidence_count=len(accumulated),
                rerank_new_candidate_count=int(rerank_diagnostics.get("new_candidate_count", 0)),
                rerank_excluded_existing_count=int(rerank_diagnostics.get("excluded_existing_count", 0)),
                rerank_incremental_only=bool(rerank_diagnostics.get("incremental_only", False)),
            ),
        ]
        return {
            **state,
            "evidence": evidence,
            "accumulated_evidence": accumulated,
            "current_new_evidence_count": new_evidence_count,
            "retrieval_history": history,
            "retrieval_latency_ms": state.get("retrieval_latency_ms", 0.0) + latency_ms,
            "rerank_latency_ms": state.get("rerank_latency_ms", 0.0)
            + getattr(self.retrieval_service, "last_rerank_latency_ms", 0.0),
            "rerank_enabled": state.get("rerank_enabled", False)
            or getattr(self.retrieval_service, "last_rerank_enabled", False),
            "diagnostics": {
                **state.get("diagnostics", {}),
                "retrieval_round": state.get("retrieval_round", 1),
                "max_retrieval_rounds": state.get("max_retrieval_rounds", 3),
                "retrieval_rounds": history,
                "recall": recall_diagnostics,
                "rerank": rerank_diagnostics,
            },
        }

    def _evidence_assessor(self, state: RagGraphState) -> RagGraphState:
        # assessor 的关键目标不是“总结证据”，而是筛出真正支撑答案的 chunk，
        # 并明确指出当前缺失信息，从而驱动下一轮检索。
        started_at = perf_counter()
        candidates = [item for item in state.get("accumulated_evidence", []) if item.text.strip()]
        decision = state.get("task_decision")
        assessment = self.llm_provider.assess_evidence_for_task(
            _build_contextual_question(state["original_question"], state.get("conversation_context")),
            candidates,
            task_decision=decision,
            requirements=assessment_requirements(decision) if decision else [],
            retrieval_round=state.get("retrieval_round", 1),
            max_retrieval_rounds=state.get("max_retrieval_rounds", 1),
        )
        supporting_ids = set(assessment.supporting_chunk_ids)
        assessment_allows_answer = assessment.sufficient and not assessment.fallback_used
        verified = [item for item in candidates if item.chunk_id in supporting_ids] if assessment_allows_answer else []
        limitations = _limitations_from_assessment(assessment)
        latency_ms = (perf_counter() - started_at) * 1000
        assessment_history = [
            *state.get("assessment_history", []),
            build_assessment_round_diagnostics(
                round_index=state.get("retrieval_round", 1),
                latency_ms=latency_ms,
                used_llm=assessment.used_llm,
                sufficient=assessment.sufficient,
                fallback_used=assessment.fallback_used,
                input_evidence_count=len(candidates),
                verified_evidence_count=len(verified),
                covered_aspects=assessment.covered_aspects,
                missing_aspects=assessment.missing_aspects,
                conflicts=assessment.conflicts,
                next_search_goal=assessment.next_search_goal,
                confidence=assessment.confidence,
            ),
        ]
        llm_call_counts = increment_llm_call_count(
            state.get("llm_call_counts"),
            "evidence_assessment",
            assessment.used_llm,
        )
        diagnostics = update_stage_proxy_hit(state.get("diagnostics"), "route", bool(verified))
        diagnostics = update_stage_proxy_hit(diagnostics, "recall", bool(verified))
        diagnostics = {
            **diagnostics,
            "assessment_rounds": assessment_history,
            "llm_call_counts": llm_call_counts,
            "verifier": build_verifier_diagnostics(
                latency_ms=latency_ms,
                used_llm=assessment.used_llm,
                sufficient=assessment_allows_answer,
                fallback_used=assessment.fallback_used,
                input_evidence_count=len(candidates),
                verified_evidence_count=len(verified),
                covered_aspects=assessment.covered_aspects,
                missing_aspects=assessment.missing_aspects,
                conflicts=assessment.conflicts,
                next_search_goal=assessment.next_search_goal,
                confidence=assessment.confidence,
                reason=assessment.reason,
            ),
        }
        return {
            **state,
            "verified_evidence": verified,
            "evidence_assessment": assessment,
            "assessment_history": assessment_history,
            "llm_call_counts": llm_call_counts,
            # 兼容现有诊断消费者，后续以 evidence_assessment 为正式契约。
            "verification_answerable": assessment_allows_answer,
            "verification_reason": assessment.reason,
            "verification_used_llm": assessment.used_llm,
            "verification_fallback_used": assessment.fallback_used,
            "verification_latency_ms": latency_ms,
            "step_has_assessment": True,
            "limitations": limitations,
            "diagnostics": diagnostics,
        }

    def _retrieval_controller(self, state: RagGraphState) -> RagGraphState:
        # controller 统一收敛多轮检索的停止条件，避免停止逻辑散落在各节点。
        assessment = state["evidence_assessment"]
        stop_reason: str | None = None
        if assessment.fallback_used:
            stop_reason = "assessment_fallback"
        elif assessment.sufficient:
            stop_reason = "sufficient_evidence"
        elif state.get("retrieval_round", 1) >= state.get("max_retrieval_rounds", 3):
            stop_reason = "max_rounds_reached"
        elif state.get("retrieval_round", 1) > 1 and state.get("current_new_evidence_count", 0) == 0:
            stop_reason = "no_new_evidence"
        elif not (assessment.missing_aspects or assessment.conflicts or assessment.next_search_goal):
            stop_reason = "no_search_direction"
        should_continue = stop_reason is None
        diagnostics = {
            **state.get("diagnostics", {}),
            "stop_reason": stop_reason,
            "retrieval_round": state.get("retrieval_round", 1),
            "max_retrieval_rounds": state.get("max_retrieval_rounds", 3),
        }
        return {
            **state,
            "continue_retrieval": should_continue,
            "stop_reason": stop_reason,
            "warnings": state.get("warnings", []) if should_continue else _final_warnings(state),
            "diagnostics": diagnostics,
        }

    def _task_classifier(self, state: RagGraphState) -> RagGraphState:
        scope = state.get("scope") or ("current_document" if state.get("document_ids") else "all_documents")
        decision = self.task_classifier.classify(
            _build_contextual_question(state["original_question"], state.get("conversation_context")),
            route=state.get("route", "chunk_search"),
            scope=scope,
            is_multi_step=state.get("is_multi_step", False),
            route_rationale=state.get("route_rationale", []),
        )
        diagnostics = {
            **state.get("diagnostics", {}),
            "task": {
                "task_type": decision.task_type,
                "requires_retrieval": decision.requires_retrieval,
                "risk_level": decision.risk_level,
                "rationale": decision.rationale,
            },
        }
        return {**state, "task_decision": decision, "diagnostics": diagnostics}

    def _budget_policy(self, state: RagGraphState) -> RagGraphState:
        decision = state["task_decision"]
        max_rounds = self.budget_policy.max_retrieval_rounds(decision)
        current_step = _current_route_step(state)
        if current_step and current_step.max_rounds is not None:
            max_rounds = max(1, min(max_rounds, current_step.max_rounds))
        diagnostics = {
            **state.get("diagnostics", {}),
            "max_retrieval_rounds": max_rounds,
        }
        return {**state, "max_retrieval_rounds": max_rounds, "diagnostics": diagnostics}

    def _query_refiner(self, state: RagGraphState) -> RagGraphState:
        # query_refiner 只负责补充检索策略，不改变当前路由节点给出的 route。
        assessment = state["evidence_assessment"]
        contextual_question = _build_contextual_question(state["original_question"], state.get("conversation_context"))
        context = build_retrieval_policy_context(state)
        decision = self.strategy_selector.select(context)
        plan = self.query_refiner.refine(
            contextual_question,
            assessment,
            context,
            decision,
            state.get("task_decision"),
        )
        attempted_queries = {
            normalized_query(query)
            for plan in state.get("query_plans", [])
            for query in plan.queries
        }
        next_query = _first_new_query(plan, attempted_queries)
        diagnostics = {
            **state.get("diagnostics", {}),
            **refinement_diagnostics(context, decision, plan),
        }
        if decision.strategy == "stop":
            diagnostics = {**diagnostics, "stop_reason": _stop_reason_from_strategy(decision)}
            stopped = {
                **state,
                "continue_retrieval": False,
                "stop_reason": _stop_reason_from_strategy(decision),
                "diagnostics": diagnostics,
            }
            return {**stopped, "warnings": _final_warnings(stopped)}
        if not next_query:
            diagnostics = {**diagnostics, "stop_reason": "no_new_query"}
            stopped = {**state, "continue_retrieval": False, "stop_reason": "no_new_query", "diagnostics": diagnostics}
            return {**stopped, "warnings": _final_warnings(stopped)}
        round_index = state.get("retrieval_round", 1) + 1
        query_plan = QueryPlan(
            original_question=state["original_question"],
            resolved_question=next_query,
            queries=plan.queries or [next_query],
            search_goal=plan.search_goal or assessment.next_search_goal or "补充尚缺证据",
            round_index=round_index,
        )
        return {
            **state,
            "current_query": next_query,
            "retrieval_round": round_index,
            "query_plans": [*state.get("query_plans", []), query_plan],
            "continue_retrieval": True,
            "stop_reason": None,
            "diagnostics": diagnostics,
        }

    def _step_finalizer(self, state: RagGraphState) -> RagGraphState:
        # 一个 route step 完成后，把本步证据沉淀到 plan 级结果。
        # 如果还有后续 step，则重置“本步检索状态”，继续复用现有检索/评估节点。
        route_plan = state.get("route_plan")
        steps = route_plan.steps if route_plan else []
        current_index = state.get("current_step_index", 0)
        current_step = steps[current_index] if current_index < len(steps) else None
        step_verified = list(state.get("verified_evidence", []))
        plan_verified = _merge_verified_evidence(state.get("plan_verified_evidence", []), step_verified)
        completed_steps = [
            *state.get("completed_route_steps", []),
            {
                "step_index": current_index,
                "step_id": current_step.step_id if current_step else f"step-{current_index + 1}",
                "route": state.get("route", "chunk_search"),
                "task_type": state.get("task_decision").task_type if state.get("task_decision") else "",
                "query": state.get("current_query", state["original_question"]),
                "stop_reason": state.get("stop_reason"),
                "verified_evidence_count": len(step_verified),
                "limitations": list(state.get("limitations", [])),
            },
        ]
        next_index = current_index + 1
        diagnostics = {
            **state.get("diagnostics", {}),
            "route_plan": _route_plan_diagnostics(route_plan, current_step_index=next_index, completed_steps=completed_steps)
            if route_plan
            else None,
        }
        if route_plan and next_index < len(route_plan.steps):
            next_step = route_plan.steps[next_index]
            next_query = next_step.query or state["original_question"]
            return {
                **state,
                "route": next_step.route,
                "plan": _plan_for_route(next_step.route),
                "current_step_index": next_index,
                "current_query": next_query,
                "query_plans": [
                    *state.get("query_plans", []),
                    QueryPlan(
                        original_question=state["original_question"],
                        resolved_question=next_query,
                        queries=[next_query],
                        search_goal=next_step.purpose or _plan_for_route(next_step.route),
                        round_index=1,
                    ),
                ],
                "retrieval_round": 1,
                "max_retrieval_rounds": 1,
                "continue_retrieval": False,
                "step_has_assessment": False,
                "stop_reason": None,
                "evidence": [],
                "accumulated_evidence": [],
                "verified_evidence": [],
                "current_new_evidence_count": 0,
                "limitations": [],
                "warnings": [],
                "completed_route_steps": completed_steps,
                "plan_verified_evidence": plan_verified,
                "diagnostics": diagnostics,
            }
        return {
            **state,
            "current_step_index": next_index,
            "completed_route_steps": completed_steps,
            "plan_verified_evidence": plan_verified,
            "verified_evidence": plan_verified,
            "diagnostics": diagnostics,
        }

    def _next_after_step_finalizer(self, state: RagGraphState) -> str:
        route_plan = state.get("route_plan")
        if route_plan and state.get("current_step_index", 0) < len(route_plan.steps):
            return "next_step"
        return "answer"

    def _next_after_controller(self, state: RagGraphState) -> str:
        return "refine_query" if state.get("continue_retrieval") else "finalize"

    def _next_after_retriever(self, state: RagGraphState) -> str:
        has_previous_assessment = bool(state.get("step_has_assessment"))
        no_new_followup_evidence = state.get("retrieval_round", 1) > 1 and state.get("current_new_evidence_count", 0) == 0
        return "control" if has_previous_assessment and no_new_followup_evidence else "assess"

    def _next_after_query_refiner(self, state: RagGraphState) -> str:
        return "retrieve" if state.get("continue_retrieval") else "finalize"

    def _answerer(self, state: RagGraphState) -> RagGraphState:
        # answerer 只消费 verified_evidence，保证输出答案一定来自已通过评估的证据。
        started_at = perf_counter()
        if not state.get("verified_evidence"):
            answer = "检索到的证据不足以可靠回答该问题。请尝试补充限定条件或换一种提问方式。"
            latency_ms = (perf_counter() - started_at) * 1000
            llm_call_counts = increment_llm_call_count(state.get("llm_call_counts"), "answer_generation", False)
            diagnostics = {
                **state.get("diagnostics", {}),
                "llm_call_counts": llm_call_counts,
                "llm": build_llm_diagnostics(
                    latency_ms=latency_ms,
                    used_llm=False,
                    answer_chars=len(answer),
                    evidence_count=0,
                    citation_count=0,
                ),
            }
            return {
                **state,
                "answer": answer,
                "answer_latency_ms": latency_ms,
                "llm_call_counts": llm_call_counts,
                "diagnostics": diagnostics,
            }
        route_plan = state.get("route_plan")
        if route_plan and len(route_plan.steps) > 1:
            used_llm = self.llm_provider.uses_llm_for_answer()
            answer = (
                self.llm_provider.answer_with_evidence(
                    _build_contextual_question(state["question"], state.get("conversation_context")),
                    state.get("verified_evidence", []),
                )
                if used_llm
                else _answer_from_route_plan(route_plan, state.get("completed_route_steps", []), state.get("verified_evidence", []))
            )
            latency_ms = (perf_counter() - started_at) * 1000
            llm_call_counts = increment_llm_call_count(state.get("llm_call_counts"), "answer_generation", used_llm)
            diagnostics = {
                **state.get("diagnostics", {}),
                "llm_call_counts": llm_call_counts,
                "llm": build_llm_diagnostics(
                    latency_ms=latency_ms,
                    used_llm=used_llm,
                    answer_chars=len(answer),
                    evidence_count=len(state.get("verified_evidence", [])),
                    citation_count=len(state.get("verified_evidence", [])),
                ),
            }
            return {
                **state,
                "answer": answer,
                "answer_latency_ms": latency_ms,
                "llm_call_counts": llm_call_counts,
                "diagnostics": diagnostics,
            }
        used_llm = (
            state.get("route") not in {"figure_index", "document_profile"}
            and self.llm_provider.uses_llm_for_answer()
        )
        if state.get("route") == "figure_index":
            answer = _answer_from_figure_index(state.get("verified_evidence", []))
            latency_ms = (perf_counter() - started_at) * 1000
            llm_call_counts = increment_llm_call_count(state.get("llm_call_counts"), "answer_generation", used_llm)
            diagnostics = {
                **state.get("diagnostics", {}),
                "llm_call_counts": llm_call_counts,
                "llm": build_llm_diagnostics(
                    latency_ms=latency_ms,
                    used_llm=used_llm,
                    answer_chars=len(answer),
                    evidence_count=len(state.get("verified_evidence", [])),
                    citation_count=len(state.get("verified_evidence", [])),
                ),
            }
            return {
                **state,
                "answer": answer,
                "answer_latency_ms": latency_ms,
                "llm_call_counts": llm_call_counts,
                "diagnostics": diagnostics,
            }
        if state.get("route") == "document_profile":
            answer = _answer_from_document_profiles(
                _build_contextual_question(state["question"], state.get("conversation_context")),
                state.get("verified_evidence", []),
            )
            latency_ms = (perf_counter() - started_at) * 1000
            llm_call_counts = increment_llm_call_count(state.get("llm_call_counts"), "answer_generation", used_llm)
            diagnostics = {
                **state.get("diagnostics", {}),
                "llm_call_counts": llm_call_counts,
                "llm": build_llm_diagnostics(
                    latency_ms=latency_ms,
                    used_llm=used_llm,
                    answer_chars=len(answer),
                    evidence_count=len(state.get("verified_evidence", [])),
                    citation_count=len(state.get("verified_evidence", [])),
                ),
            }
            return {
                **state,
                "answer": answer,
                "answer_latency_ms": latency_ms,
                "llm_call_counts": llm_call_counts,
                "diagnostics": diagnostics,
            }
        answer = self.llm_provider.answer_with_evidence(
            _build_contextual_question(state["question"], state.get("conversation_context")),
            state.get("verified_evidence", []),
        )
        latency_ms = (perf_counter() - started_at) * 1000
        llm_call_counts = increment_llm_call_count(state.get("llm_call_counts"), "answer_generation", used_llm)
        diagnostics = {
            **state.get("diagnostics", {}),
            "llm_call_counts": llm_call_counts,
            "llm": build_llm_diagnostics(
                latency_ms=latency_ms,
                used_llm=used_llm,
                answer_chars=len(answer),
                evidence_count=len(state.get("verified_evidence", [])),
                citation_count=len(state.get("verified_evidence", [])),
            ),
        }
        return {
            **state,
            "answer": answer,
            "answer_latency_ms": latency_ms,
            "llm_call_counts": llm_call_counts,
            "diagnostics": diagnostics,
        }

    def _answer_verifier(self, state: RagGraphState) -> RagGraphState:
        # 答案核验发生在生成之后，只检查答案是否忠实于 verified_evidence。
        # 未触发时写入轻量诊断，便于评测区分“未核验”和“核验通过”。
        started_at = perf_counter()
        decision = self.answer_verification_policy.decide(
            answer=state.get("answer", ""),
            evidence=state.get("verified_evidence", []),
            limitations=state.get("limitations", []),
            task_decision=state.get("task_decision"),
            evidence_assessment=state.get("evidence_assessment"),
            route=state.get("route", "chunk_search"),
            scope=state.get("scope") or ("current_document" if state.get("document_ids") else "all_documents"),
            is_multi_step=bool(state.get("is_multi_step", False)),
        )
        if not decision.should_verify:
            verification = AnswerVerification(
                passed=True,
                triggered=False,
                reason="未满足答案级核验触发条件。",
            )
            latency_ms = (perf_counter() - started_at) * 1000
            diagnostics = {
                **state.get("diagnostics", {}),
                "answer_verification": build_answer_verification_diagnostics(
                    latency_ms=latency_ms,
                    triggered=False,
                    passed=True,
                    trigger_reasons=[],
                    unsupported_claims=[],
                    missing_disclosures=[],
                    citation_errors=[],
                    overstatements=[],
                    revision_instruction=None,
                    used_llm=False,
                    fallback_used=False,
                    reason=verification.reason,
                ),
            }
            return {**state, "answer_verification": verification, "diagnostics": diagnostics}

        verification = self.llm_provider.verify_answer_with_evidence(
            _build_contextual_question(state["question"], state.get("conversation_context")),
            state.get("answer", ""),
            state.get("verified_evidence", []),
            limitations=state.get("limitations", []),
            task_decision=state.get("task_decision"),
            evidence_assessment=state.get("evidence_assessment"),
            trigger_reasons=decision.reasons,
        )
        latency_ms = (perf_counter() - started_at) * 1000
        llm_call_counts = increment_llm_call_count(
            state.get("llm_call_counts"),
            "answer_verification",
            verification.used_llm,
        )
        diagnostics = {
            **state.get("diagnostics", {}),
            "llm_call_counts": llm_call_counts,
            "answer_verification": build_answer_verification_diagnostics(
                latency_ms=latency_ms,
                triggered=verification.triggered,
                passed=verification.passed,
                trigger_reasons=verification.trigger_reasons,
                unsupported_claims=verification.unsupported_claims,
                missing_disclosures=verification.missing_disclosures,
                citation_errors=verification.citation_errors,
                overstatements=verification.overstatements,
                revision_instruction=verification.revision_instruction,
                used_llm=verification.used_llm,
                fallback_used=verification.fallback_used,
                reason=verification.reason,
            ),
        }
        warnings = list(state.get("warnings", []))
        if verification.triggered and not verification.passed:
            warnings.append(f"答案核验未通过：{verification.reason or '存在未支撑断言或遗漏限制。'}")
        return {
            **state,
            "answer_verification": verification,
            "answer_verification_latency_ms": latency_ms,
            "llm_call_counts": llm_call_counts,
            "warnings": list(dict.fromkeys(warnings)),
            "diagnostics": diagnostics,
        }

    def _next_after_answer_verifier(self, state: RagGraphState) -> str:
        verification = state.get("answer_verification")
        if (
            isinstance(verification, AnswerVerification)
            and verification.triggered
            and not verification.passed
            and verification.revision_instruction
            and state.get("answer_revision_count", 0) < 1
            and state.get("verified_evidence")
        ):
            return "revise"
        return "images"

    def _answer_reviser(self, state: RagGraphState) -> RagGraphState:
        # 修订节点最多执行一次，且仍只消费原始问题、已验证证据和 verifier 的修订要求。
        verification = state.get("answer_verification")
        if not isinstance(verification, AnswerVerification) or not verification.revision_instruction:
            return state
        started_at = perf_counter()
        used_llm = self.llm_provider.uses_llm_for_answer()
        revised = self.llm_provider.revise_answer_with_evidence(
            _build_contextual_question(state["question"], state.get("conversation_context")),
            state.get("answer", ""),
            state.get("verified_evidence", []),
            limitations=state.get("limitations", []),
            revision_instruction=verification.revision_instruction,
        )
        latency_ms = (perf_counter() - started_at) * 1000
        llm_call_counts = increment_llm_call_count(
            state.get("llm_call_counts"),
            "answer_revision",
            used_llm,
        )
        diagnostics = dict(state.get("diagnostics", {}))
        answer_verification = dict(diagnostics.get("answer_verification", {}))
        answer_verification["revision_applied"] = True
        diagnostics["answer_verification"] = answer_verification
        diagnostics["llm_call_counts"] = llm_call_counts
        diagnostics["answer_revision"] = {
            "latency_ms": latency_ms,
            "used_llm": used_llm,
            "answer_chars": len(revised),
            "revision_count": state.get("answer_revision_count", 0) + 1,
        }
        return {
            **state,
            "answer": revised,
            "answer_revision_count": state.get("answer_revision_count", 0) + 1,
            "llm_call_counts": llm_call_counts,
            "diagnostics": diagnostics,
        }

    def _image_selector(self, state: RagGraphState) -> RagGraphState:
        # 图像选择与答案生成解耦，并基于完整 route_plan 而不是最终 route 字符串决策。
        result = self.image_selection_service.select(
            ImageSelectionContext(
                question=state["question"],
                answer=state.get("answer", ""),
                conversation_context=state.get("conversation_context"),
                route_plan=state.get("route_plan"),
                completed_route_steps=state.get("completed_route_steps", []),
                task_decision=state.get("task_decision"),
                verified_evidence=state.get("verified_evidence", []),
                max_images=3,
            )
        )
        llm_call_counts = increment_llm_call_count(
            state.get("llm_call_counts"),
            "image_selection",
            result.used_llm,
        )
        diagnostics = {
            **state.get("diagnostics", {}),
            "llm_call_counts": llm_call_counts,
            "image_selection": build_image_selection_diagnostics(
                strategy=result.strategy,
                used_llm=result.used_llm,
                fallback_used=result.fallback_used,
                candidate_count=result.candidate_count,
                selected_count=result.selected_count,
                latency_ms=result.latency_ms,
                reason=result.reason,
            ),
        }
        diagnostics = _with_selected_image_count(diagnostics, result.images)
        return {
            **state,
            "selected_images": result.images,
            "llm_call_counts": llm_call_counts,
            "diagnostics": diagnostics,
        }


def _plan_for_route(route: str) -> str:
    if route == "figure_index":
        return "search_figure_index_then_answer_with_figure_citations"
    if route == "document_profile":
        return "search_document_profiles_then_answer_with_document_citations"
    return "retrieve_relevant_chunks_then_answer_with_citations"


def _route_plan_from_rule_decision(route: str, query: str, rationale: list[str]) -> RoutePlan:
    return RoutePlan(
        steps=[
            RouteStep(
                step_id="step-1",
                route=route,
                task_type=_task_type_for_route(route),
                query=query,
                purpose=_plan_for_route(route),
            )
        ],
        plan_type="single_step",
        rationale=";".join(rationale[:6]),
        fallback_used=True,
    )


def _task_type_for_route(route: str) -> str:
    if route == "figure_index":
        return "figure_query"
    if route == "document_profile":
        return "document_summary"
    return "fact_qa"


def _route_plan_diagnostics(
    route_plan: RoutePlan | None,
    *,
    current_step_index: int,
    completed_steps: list[dict[str, object]],
) -> dict[str, object]:
    if route_plan is None:
        return {
            "plan_type": "single_step",
            "used_llm": False,
            "fallback_used": True,
            "current_step_index": current_step_index,
            "rationale": "",
            "steps": [],
            "completed_steps": completed_steps,
        }
    return {
        "plan_type": route_plan.plan_type,
        "used_llm": route_plan.used_llm,
        "fallback_used": route_plan.fallback_used,
        "current_step_index": current_step_index,
        "rationale": route_plan.rationale,
        "steps": [
            {
                "step_id": step.step_id,
                "route": step.route,
                "task_type": step.task_type,
                "query": step.query,
                "purpose": step.purpose,
                "max_rounds": step.max_rounds,
                "requires_assessment": step.requires_assessment,
                "requires_answer_verification": step.requires_answer_verification,
            }
            for step in route_plan.steps
        ],
        "completed_steps": completed_steps,
    }


def _current_route_step(state: RagGraphState) -> RouteStep | None:
    route_plan = state.get("route_plan")
    if route_plan is None:
        return None
    index = state.get("current_step_index", 0)
    if index < 0 or index >= len(route_plan.steps):
        return None
    return route_plan.steps[index]


def _first_new_query(plan: QueryRefinementPlan, attempted_queries: set[str]) -> str | None:
    for query in plan.queries:
        if normalized_query(query) not in attempted_queries:
            return query
    return None


def _stop_reason_from_strategy(decision: RetrievalStrategyDecision) -> str:
    if decision.reason in {"assessment_fallback", "sufficient_evidence", "max_rounds_reached", "no_new_evidence"}:
        return decision.reason
    return "no_search_direction"


def _merge_verified_evidence(existing: list[Evidence], current: list[Evidence]) -> list[Evidence]:
    merged = list(existing)
    known = {(item.document_id, item.chunk_id) for item in merged}
    for item in current:
        key = (item.document_id, item.chunk_id)
        if key in known:
            continue
        known.add(key)
        merged.append(item)
    return merged


def _answer_from_route_plan(
    route_plan: RoutePlan,
    completed_steps: list[dict[str, object]],
    evidence: list[Evidence],
) -> str:
    lines = ["根据多步骤检索计划，当前可支撑的结果如下："]
    for step in completed_steps:
        route = step.get("route", "")
        query = step.get("query", "")
        verified_count = step.get("verified_evidence_count", 0)
        lines.append(f"- {route}: `{query}`，获得 {verified_count} 条可引用证据。")
    for index, item in enumerate(evidence[:8], start=1):
        prefix = "图像证据" if item.images else "文本证据"
        snippet = " ".join(item.text.split())[:240]
        lines.append(f"{index}. {prefix} p. {item.page_start}: {snippet}")
    return "\n".join(lines)


def _build_contextual_question(question: str, conversation_context: str | None) -> str:
    context = (conversation_context or "").strip()
    if not context:
        return question
    return (
        "对话上下文仅用于解析追问、指代和限定条件；回答仍必须依据本轮检索证据。\n"
        f"{context}\n\n"
        f"当前问题：{question}"
    )


def _answer_from_figure_index(evidence: list[Evidence]) -> str:
    if not evidence:
        return "没有在 figure 索引中找到足够相关的图片或图题。"
    lines = ["根据 figure 索引，最相关的图包括："]
    for index, item in enumerate(evidence[:8], start=1):
        image = item.images[0] if item.images else None
        label = image.label if image and image.label else ""
        caption = image.caption if image and image.caption else item.text
        lines.append(f"{index}. p. {item.page_start}: {label} {caption}".strip())
    return "\n".join(lines)


def _answer_from_document_profiles(question: str, evidence: list[Evidence]) -> str:
    if not evidence:
        return f"没有检索到足够的文档画像证据来回答：{question}"
    lines = ["根据文档画像，相关论文要点如下："]
    for index, item in enumerate(evidence[:5], start=1):
        title = item.section_title or "Document Profile"
        snippet = " ".join(item.text.split())[:700]
        lines.append(f"{index}. {title} (p. {item.page_start}): {snippet}")
    return "\n".join(lines)


def _with_selected_image_count(
    diagnostics: dict[str, object] | None,
    selected_images: dict[str, list[EvidenceImage]],
) -> dict[str, object]:
    payload = dict(diagnostics or {})
    llm = dict(payload.get("llm", {}))
    if not llm:
        return payload
    llm["selected_image_count"] = sum(len(images) for images in selected_images.values())
    payload["llm"] = llm
    return payload


def _limitations_from_assessment(assessment: EvidenceAssessment) -> list[str]:
    limitations = [f"资料缺口：{item}" for item in assessment.missing_aspects]
    limitations.extend(f"证据冲突：{item}" for item in assessment.conflicts)
    if not assessment.sufficient and not limitations:
        limitations.append(f"证据不足：{assessment.reason or '现有资料无法充分支撑回答。'}")
    if assessment.fallback_used:
        limitations.append(f"证据评估已回退：{assessment.reason}")
    return list(dict.fromkeys(limitations))


def _merge_evidence(existing: list[Evidence], current: list[Evidence]) -> tuple[list[Evidence], int]:
    # 去重后累积证据，确保多轮检索不会把相同 chunk 反复计入“新增”结果。
    accumulated = list(existing)
    known = {(item.document_id, item.chunk_id) for item in accumulated}
    new_count = 0
    for item in current:
        key = (item.document_id, item.chunk_id)
        if key in known:
            continue
        known.add(key)
        accumulated.append(item)
        new_count += 1
    return accumulated, new_count


def _build_refined_query(question: str, assessment: EvidenceAssessment) -> str:
    # 当前实现是“原问题 + 缺失方面 + 下一轮目标”的可解释拼接策略。
    # 如果未来要引入 LLM 改写，只需要替换这里，不影响外围控制流。
    components = [question, assessment.next_search_goal, *assessment.missing_aspects, *assessment.conflicts]
    query = " ".join(component.strip() for component in components if component and component.strip())
    return " ".join(query.split())[:1000]


def _normalized_query(query: str) -> str:
    return " ".join(query.lower().split())


def _final_warnings(state: RagGraphState) -> list[str]:
    warnings = list(state.get("warnings", []))
    assessment = state["evidence_assessment"]
    candidates = state.get("accumulated_evidence", [])
    if not candidates:
        warnings.append("没有检索到可用于回答的证据。")
    elif not assessment.sufficient:
        warnings.append(f"检索证据不足以回答问题：{assessment.reason}")
    elif assessment.fallback_used:
        warnings.append(f"证据评估已回退：{assessment.reason}")
    return list(dict.fromkeys(warnings))
