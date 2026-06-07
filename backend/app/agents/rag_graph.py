import re
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
from backend.app.retrieval.outline_resolution import (
    OutlineReferenceResolver,
    outline_resolution_diagnostics,
)
from backend.app.retrieval.profile_guided import ProfileGuidedRetrievalPlanner
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
        profile_guided_planner: ProfileGuidedRetrievalPlanner | None = None,
        outline_reference_resolver: OutlineReferenceResolver | None = None,
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
        self.profile_guided_planner = profile_guided_planner or ProfileGuidedRetrievalPlanner()
        self.outline_reference_resolver = outline_reference_resolver or OutlineReferenceResolver()
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
        owner_id: str | None = "demo",
        conversation_context: str | None = None,
        answer_delta_callback=None,
        stage_callback=None,
    ) -> RagGraphState:
        # invoke 负责初始化首轮状态，并把执行总耗时回填到 diagnostics。
        started_at = perf_counter()
        _emit_stage_callback(stage_callback, "planning", "正在规划检索路径")
        current_query = _build_contextual_question(question, conversation_context)
        state = self.graph.invoke(
            {
                "question": question,
                "original_question": question,
                "session_id": session_id,
                "owner_id": owner_id,
                "conversation_context": conversation_context,
                "document_ids": document_ids,
                "scope": scope,
                "outline_resolution": None,
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
                "outline_context_evidence": [],
                "accumulated_evidence": [],
                "retrieval_history": [],
                "assessment_history": [],
                "llm_call_counts": {"evidence_assessment": 0, "answer_generation": 0, "image_selection": 0},
                "limitations": [],
                "warnings": [],
                "answer_revision_count": 0,
                "answer_delta_callback": answer_delta_callback,
                "stage_callback": stage_callback,
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
        graph.add_node("outline_resolver", self._outline_resolver)
        graph.add_node("outline_retriever", self._outline_retriever)
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
        graph.add_edge("outline_resolver", "outline_retriever")
        graph.add_edge("outline_retriever", "planner")
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
        graph.set_entry_point("outline_resolver")
        return graph

    def _outline_resolver(self, state: RagGraphState) -> RagGraphState:
        # 问题重写层只负责消解指代、输出 basic_facts 和初步文档范围，不做路由规划。
        resolution = self.outline_reference_resolver.resolve(
            question=state["question"],
            conversation_context=state.get("conversation_context"),
            document_ids=state.get("document_ids"),
            scope=state.get("scope"),
        )
        effective_document_ids = resolution.target_document_ids if resolution.target_document_ids is not None else state.get("document_ids")
        effective_scope = "current_document" if resolution.decision == "single_document_outline" else (
            state.get("scope") or ("current_document" if effective_document_ids else "all_documents")
        )
        diagnostics = {
            **state.get("diagnostics", {}),
            "outline_resolution": outline_resolution_diagnostics(resolution),
        }
        return {
            **state,
            "outline_resolution": resolution,
            "document_ids": effective_document_ids,
            "scope": effective_scope,
            "current_query": resolution.rewritten_question,
            "diagnostics": diagnostics,
        }

    def _outline_retriever(self, state: RagGraphState) -> RagGraphState:
        # 大纲检索层发生在 planner 之前：单文档取该文档大纲，多文档先匹配候选论文。
        resolution = state.get("outline_resolution")
        if resolution is None or not resolution.needs_outline_retrieval:
            diagnostics = {
                **state.get("diagnostics", {}),
                "outline_context": _outline_context_diagnostics(
                    injected=False,
                    evidence=[],
                    document_ids=state.get("document_ids"),
                    reason="问题不需要 planner 前置大纲检索。",
                ),
            }
            return {**state, "outline_context_evidence": [], "diagnostics": diagnostics}

        _emit_stage(state, "retrieval", "正在定位相关论文")
        query = _outline_retrieval_query(resolution)
        top_k = 1 if resolution.decision == "single_document_outline" else self.profile_top_k
        started_at = perf_counter()
        evidence = self.retrieval_service.retrieve_document_profiles(
            query,
            top_k=top_k,
            document_ids=state.get("document_ids"),
            owner_id=state.get("owner_id"),
        )
        latency_ms = (perf_counter() - started_at) * 1000
        matched_document_ids = _document_ids_from_evidence(evidence)
        effective_document_ids = matched_document_ids or state.get("document_ids")
        diagnostics = {
            **state.get("diagnostics", {}),
            "outline_context": _outline_context_diagnostics(
                injected=bool(evidence),
                evidence=evidence,
                document_ids=effective_document_ids,
                latency_ms=latency_ms,
                query=query,
                reason="planner 前置大纲/画像检索已完成。",
            ),
        }
        return {
            **state,
            "document_ids": effective_document_ids,
            "outline_context_evidence": evidence,
            "accumulated_evidence": _merge_verified_evidence(state.get("accumulated_evidence", []), evidence),
            "diagnostics": diagnostics,
        }

    def _planner(self, state: RagGraphState) -> RagGraphState:
        # planner 消费问题重写层和大纲检索层产物，输出后续 RAG 执行步骤。
        started_at = perf_counter()
        _emit_stage(state, "planning", "正在选择检索策略")
        resolution = state.get("outline_resolution")
        rewritten_question = resolution.rewritten_question if resolution else state["question"]
        basic_facts = resolution.basic_facts if resolution else []
        outline_context_evidence = state.get("outline_context_evidence", [])
        effective_document_ids = state.get("document_ids")
        effective_scope = state.get("scope") or ("current_document" if effective_document_ids else "all_documents")
        context = RouteContext(
            scope=effective_scope,
            document_ids=effective_document_ids,
        )
        contextual_question = _build_contextual_question(rewritten_question, state.get("conversation_context"))
        decision = self.query_router.route(contextual_question, context)
        route_plan = (
            _call_plan_route(
                self.llm_provider,
                rewritten_question,
                conversation_context=state.get("conversation_context"),
                scope=context.scope,
                document_ids=effective_document_ids,
                basic_facts=basic_facts,
                outline_context=outline_context_evidence,
            )
            if _should_use_llm_route_plan(rewritten_question, decision)
            else None
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
        _emit_stage(state, "retrieval", _retrieval_stage_label(route))
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
                owner_id=state.get("owner_id"),
                exclude_keys=exclude_keys,
            )
        elif route == "document_profile" and state.get("outline_context_evidence") and state.get("retrieval_round", 1) == 1:
            evidence = list(state.get("outline_context_evidence", []))
            self.retrieval_service.last_recall_diagnostics = {
                "mode": "document_profile_prefetched",
                "latency_ms": 0.0,
                "embedding_latency_ms": 0.0,
                "search_latency_ms": 0.0,
                "candidate_count": len(evidence),
                "returned_count": len(evidence),
                "hit": bool(evidence),
            }
            self.retrieval_service.last_rerank_diagnostics = {
                "enabled": False,
                "candidate_count": len(evidence),
                "returned_count": len(evidence),
                "new_candidate_count": 0,
                "excluded_existing_count": 0,
                "incremental_only": False,
                "skip_reason": "document_profile_prefetched_before_planner",
            }
        elif route == "document_profile":
            evidence = self.retrieval_service.retrieve_document_profiles(
                query,
                top_k=self.profile_top_k,
                document_ids=state.get("document_ids"),
                owner_id=state.get("owner_id"),
                exclude_keys=exclude_keys,
            )
        else:
            evidence = self.retrieval_service.retrieve(
                query,
                top_k=self.retrieval_top_k,
                document_ids=state.get("document_ids"),
                owner_id=state.get("owner_id"),
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
        _emit_stage(state, "verification", "正在核对证据是否足够")
        candidates = [
            item
            for item in _merge_verified_evidence(
                state.get("plan_verified_evidence", []),
                state.get("accumulated_evidence", []),
            )
            if item.text.strip()
        ]
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
        lexical_verified = []
        if not verified:
            lexical_verified = _strong_lexical_verified_evidence(state["original_question"], candidates)
            if lexical_verified:
                assessment = EvidenceAssessment(
                    sufficient=True,
                    covered_aspects=_strong_query_terms(state["original_question"])[:5],
                    missing_aspects=[],
                    conflicts=[],
                    supporting_chunk_ids=[item.chunk_id for item in lexical_verified],
                    next_search_goal=None,
                    confidence=max(float(assessment.confidence or 0.0), 0.65),
                    reason=f"强词法召回命中，保留候选证据；原评估：{assessment.reason}",
                    used_llm=assessment.used_llm,
                    fallback_used=False,
                )
                assessment_allows_answer = True
                verified = lexical_verified
        if verified:
            verified = _augment_verified_with_visual_evidence(state["original_question"], verified, candidates)
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
        if lexical_verified:
            assessment_history[-1]["lexical_fallback_used"] = True
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
        followup = self.profile_guided_planner.plan(
            question=_build_contextual_question(state["original_question"], state.get("conversation_context")),
            task_decision=state.get("task_decision"),
            profile_evidence=step_verified or state.get("accumulated_evidence", []) or state.get("evidence", []),
        ) if state.get("route") == "document_profile" else None
        if route_plan and followup:
            next_step = RouteStep(
                step_id=f"profile-followup-{next_index + 1}",
                route=followup.route,
                task_type="profile_guided_detail",
                query=followup.query,
                purpose=followup.reason,
                max_rounds=1,
            )
            guided_plan = RoutePlan(
                steps=[*route_plan.steps, next_step],
                plan_type="profile_guided",
                rationale=route_plan.rationale or "profile 命中文档后补充文档内细节证据。",
                used_llm=route_plan.used_llm,
                fallback_used=route_plan.fallback_used,
            )
            guided_completed_steps = completed_steps
            guided_diagnostics = {
                **diagnostics,
                "route_plan": _route_plan_diagnostics(
                    guided_plan,
                    current_step_index=next_index,
                    completed_steps=guided_completed_steps,
                ),
                "profile_guided_followup": {
                    "route": followup.route,
                    "query": followup.query,
                    "document_ids": followup.document_ids,
                    "reason": followup.reason,
                },
            }
            return {
                **state,
                "route": next_step.route,
                "plan": _plan_for_route(next_step.route),
                "route_plan": guided_plan,
                "current_step_index": next_index,
                "current_query": next_step.query,
                "document_ids": followup.document_ids,
                "query_plans": [
                    *state.get("query_plans", []),
                    QueryPlan(
                        original_question=state["original_question"],
                        resolved_question=next_step.query,
                        queries=[next_step.query],
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
                "completed_route_steps": guided_completed_steps,
                "plan_verified_evidence": plan_verified,
                "is_multi_step": True,
                "diagnostics": guided_diagnostics,
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
        _emit_stage(state, "answering", "正在生成回答")
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
        used_llm = self.llm_provider.uses_llm_for_answer()
        contextual_question = _build_contextual_question(state["question"], state.get("conversation_context"))
        if state.get("answer_delta_callback") is None:
            answer = self.llm_provider.answer_with_evidence(
                contextual_question,
                state.get("verified_evidence", []),
            )
        else:
            chunks: list[str] = []
            for chunk in self.llm_provider.answer_with_evidence_stream(
                contextual_question,
                state.get("verified_evidence", []),
            ):
                if not chunk:
                    continue
                chunks.append(chunk)
                _emit_answer_delta(state, chunk)
            answer = "".join(chunks)
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
        _emit_stage(state, "verification", "正在检查回答依据")
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
        _emit_stage(state, "finalizing", "正在整理引用和图片")
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


def _emit_answer_delta(state: RagGraphState, chunk: str) -> None:
    callback = state.get("answer_delta_callback")
    if callback is None:
        return
    callback(chunk)


def _emit_stage(state: RagGraphState, stage: str, label: str) -> None:
    _emit_stage_callback(state.get("stage_callback"), stage, label)


def _emit_stage_callback(callback, stage: str, label: str) -> None:
    if callback is None:
        return
    callback(stage, label)


def _retrieval_stage_label(route: str) -> str:
    if route == "figure_index":
        return "正在检索图表线索"
    if route == "document_profile":
        return "正在检索论文概览"
    return "正在检索相关片段"


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


def _call_plan_route(
    provider: LlmProvider,
    question: str,
    *,
    conversation_context: str | None,
    scope: str,
    document_ids: list[str] | None,
    basic_facts: list[str],
    outline_context: list[Evidence],
) -> RoutePlan | None:
    # 新 planner 输入包含前置大纲检索结果；兼容旧测试 provider 的窄签名。
    try:
        return provider.plan_route(
            question,
            conversation_context=conversation_context,
            scope=scope,
            document_ids=document_ids,
            basic_facts=basic_facts,
            outline_context=outline_context,
        )
    except TypeError as exc:
        if "unexpected keyword" not in str(exc):
            raise
        return provider.plan_route(
            question,
            conversation_context=conversation_context,
            scope=scope,
            document_ids=document_ids,
        )


def _should_use_llm_route_plan(question: str, decision) -> bool:
    if _has_reference_detail_signal(question):
        return False
    if decision.needs_llm_router:
        return True
    text = question.lower()
    return bool(
        re.search(r"\b(?:and which|and what|then|also|plus)\b", text)
        or re.search(r"\bwhich\b.+\band\b.+\b(?:figure|spectrum|spectra|plot|evidence)\b", text)
        or re.search(r"顺便|并给出|并找|并总结|以及|同时", question)
    )


def _has_reference_detail_signal(question: str) -> bool:
    text = _normalize_query_aliases(question)
    return bool(
        re.search(r"\b(?:reference|citation|author|co-author|affiliation|department|doi)\b", text)
        or re.search(r"参考文献|作者|单位|机构|编号", question)
    )


def _outline_retrieval_query(resolution) -> str:
    components = [
        resolution.rewritten_question,
        " ".join(resolution.basic_facts),
        "论文大纲 文档画像 摘要 主题 章节 结构 关键词",
    ]
    return " ".join(component.strip() for component in components if component and component.strip())[:1000]


def _outline_context_diagnostics(
    *,
    injected: bool,
    evidence: list[Evidence],
    document_ids: list[str] | None,
    reason: str,
    latency_ms: float = 0.0,
    query: str = "",
) -> dict[str, object]:
    return {
        "injected": injected,
        "evidence_count": len(evidence),
        "document_ids": document_ids or [],
        "latency_ms": latency_ms,
        "query": query,
        "reason": reason,
        "evidence": [
            {
                "chunk_id": item.chunk_id,
                "document_id": item.document_id,
                "section_title": item.section_title,
                "score": item.score,
                "evidence_type": item.evidence_type,
                "source_scope": item.source_scope,
                "text": item.text[:500],
            }
            for item in evidence
        ],
    }


def _document_ids_from_evidence(evidence: list[Evidence]) -> list[str]:
    return list(dict.fromkeys(item.document_id for item in evidence if item.document_id))


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


def _build_contextual_question(question: str, conversation_context: str | None) -> str:
    context = (conversation_context or "").strip()
    if not context:
        return question
    return (
        "对话上下文仅用于解析追问、指代和限定条件；回答仍必须依据本轮检索证据。\n"
        f"{context}\n\n"
        f"当前问题：{question}"
    )


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


def _strong_lexical_verified_evidence(question: str, candidates: list[Evidence], limit: int = 5) -> list[Evidence]:
    terms = set(_strong_query_terms(question))
    if not terms:
        return []
    selected: list[Evidence] = []
    for item in sorted(candidates, key=_lexical_evidence_score, reverse=True):
        sources = item.metadata.get("retrieval_sources", [])
        lexical_source = item.evidence_type == "lexical_chunk" or (
            isinstance(sources, list) and "lexical" in sources
        )
        lexical_score = _lexical_evidence_score(item)
        if not lexical_source or lexical_score < 0.9:
            continue
        text = _normalize_query_aliases(item.text)
        if any(term in text for term in terms) or lexical_score >= 1.3:
            selected.append(item)
        if len(selected) >= limit:
            break
    if not selected:
        return []
    top = selected[0]
    narrowed = [
        item
        for item in selected
        if item.document_id == top.document_id and _lexical_evidence_score(item) >= _lexical_evidence_score(top) - 0.8
    ][:limit]
    if _has_visual_detail_signal(question):
        figure_items = [
            item
            for item in candidates
            if item.document_id == top.document_id
            and item.evidence_type == "figure"
            and _visual_term_overlap(question, item.text)
        ]
        narrowed = _merge_verified_evidence(narrowed, figure_items[:2])
    return narrowed


def _lexical_evidence_score(item: Evidence) -> float:
    lexical_score = item.metadata.get("lexical_score")
    if isinstance(lexical_score, (int, float)):
        return float(lexical_score)
    return float(item.score)


def _augment_verified_with_visual_evidence(
    question: str,
    verified: list[Evidence],
    candidates: list[Evidence],
) -> list[Evidence]:
    if not _has_visual_detail_signal(question):
        return verified
    document_ids = {item.document_id for item in verified}
    additions = [
        item
        for item in candidates
        if item.document_id in document_ids
        and item.evidence_type == "figure"
        and _visual_term_overlap(question, item.text)
    ]
    return _merge_verified_evidence(verified, additions[:2])


def _strong_query_terms(question: str) -> list[str]:
    normalized = _normalize_query_aliases(question)
    raw_terms = re.findall(r"[a-z][a-z0-9/-]{1,}|[\u4e00-\u9fff]{2,}", normalized)
    stopwords = {
        "what",
        "which",
        "where",
        "when",
        "how",
        "does",
        "with",
        "from",
        "into",
        "that",
        "this",
        "paper",
        "article",
        "shown",
        "work",
        "material",
        "system",
        "studied",
        "study",
    }
    terms: list[str] = []
    for raw in raw_terms:
        term = raw.strip(".-")
        if len(term) < 2 or term in stopwords:
            continue
        if term not in terms and (
            "-" in term
            or "/" in term
            or len(term) >= 6
            or term in {"reference", "number", "author", "affiliation", "department"}
        ):
            terms.append(term)
    return terms


def _normalize_query_aliases(text: str) -> str:
    normalized = " ".join(text.lower().split())
    aliases = {
        "referense": "reference",
        "numbir": "number",
        "materiel": "material",
        "resonans": "resonance",
        "spektra": "spectra",
        "potenshal": "potential",
        "soler": "solar",
        "energjy": "energy",
        "renewabel": "renewable",
        "discribed": "described",
    }
    for typo, correction in aliases.items():
        normalized = normalized.replace(typo, correction)
    return normalized


def _has_visual_detail_signal(question: str) -> bool:
    text = _normalize_query_aliases(question)
    return bool(
        re.search(r"\b(?:fig(?:ure)?|plot|spectrum|spectra|resonance|chart|image|scheme)\b", text)
        or re.search(r"图|谱图|曲线|共振|光谱", question)
    )


def _visual_term_overlap(question: str, text: str) -> bool:
    question_terms = set(re.findall(r"[a-z][a-z0-9/-]{2,}", _normalize_query_aliases(question)))
    text_terms = set(re.findall(r"[a-z][a-z0-9/-]{2,}", _normalize_query_aliases(text)))
    return bool(question_terms & text_terms)


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
