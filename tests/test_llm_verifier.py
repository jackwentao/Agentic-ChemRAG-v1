from backend.app.domain.documents import Evidence
from backend.app.domain.rag import TaskDecision
from backend.app.providers.llm import ComposedLlmProvider, DeepSeekLlmProvider, FakeLlmProvider, _sanitize_evidence_assessment


class ImmediateFuture:
    def __init__(self, value) -> None:
        self.value = value

    def result(self, timeout=None):
        return self.value


class ImmediateExecutor:
    def submit(self, function, **kwargs):
        return ImmediateFuture(function(**kwargs))


class RaisingFuture:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def result(self, timeout=None):
        raise self.error


class RaisingExecutor:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def submit(self, function, **kwargs):
        return RaisingFuture(self.error)


class VerificationCompletions:
    def __init__(self, content: str) -> None:
        self.content = content
        self.messages = None

    def create(self, model, messages, response_format=None):
        self.messages = messages

        class Message:
            content = self.content

        class Choice:
            message = Message()

        class Response:
            choices = [Choice()]

        return Response()


class VerificationClient:
    def __init__(self, content: str) -> None:
        self.chat = type("Chat", (), {"completions": VerificationCompletions(content)})()


def _evidence(chunk_id: str = "c1") -> Evidence:
    return Evidence(
        chunk_id=chunk_id,
        document_id="doc1",
        text="The ZnO sample exhibits a photoluminescence peak at 520 nm.",
        page_start=3,
        page_end=3,
        section_title="Results",
        score=0.9,
    )


def test_deepseek_assessor_requests_structured_sufficiency_decision():
    provider = DeepSeekLlmProvider.__new__(DeepSeekLlmProvider)
    provider.model = "test-model"
    provider.timeout_seconds = 2.0
    provider.executor = ImmediateExecutor()
    provider.client = VerificationClient(
        '{"sufficient":true,"covered_aspects":["发光峰值"],"missing_aspects":[],"conflicts":[],'
        '"supporting_chunk_ids":["c1"],"next_search_goal":null,"confidence":0.97,"reason":"片段直接包含峰值。"}'
    )

    result = provider.assess_evidence("What is the photoluminescence peak?", [_evidence()])

    assert result.sufficient is True
    assert result.supporting_chunk_ids == ["c1"]
    assert result.covered_aspects == ["发光峰值"]
    assert result.confidence == 0.97
    assert result.used_llm is True
    user_message = provider.client.chat.completions.messages[1]["content"]
    assert "What is the photoluminescence peak?" in user_message
    assert "520 nm" in user_message


def test_evidence_assessor_rejects_sufficient_decision_without_valid_supporting_chunk():
    result = _sanitize_evidence_assessment(
        {"sufficient": True, "supporting_chunk_ids": ["not-retrieved"], "reason": "Enough."},
        [_evidence()],
    )

    assert result.sufficient is False
    assert result.supporting_chunk_ids == []
    assert result.missing_aspects == ["没有返回可引用的有效支撑证据片段"]
    assert result.used_llm is True


def test_deepseek_assessor_fails_closed_on_invalid_json():
    provider = DeepSeekLlmProvider.__new__(DeepSeekLlmProvider)
    provider.model = "test-model"
    provider.timeout_seconds = 2.0
    provider.executor = ImmediateExecutor()
    provider.client = VerificationClient("not-json")

    result = provider.assess_evidence("What is the photoluminescence peak?", [_evidence()])

    assert result.sufficient is False
    assert result.supporting_chunk_ids == []
    assert result.missing_aspects == ["证据充分性评估未完成，不能据此生成可靠结论"]
    assert result.fallback_used is True


def test_deepseek_assessor_fails_closed_when_service_connection_is_unavailable():
    provider = DeepSeekLlmProvider.__new__(DeepSeekLlmProvider)
    provider.model = "test-model"
    provider.timeout_seconds = 2.0
    provider.executor = RaisingExecutor(ConnectionError("Connection error."))
    provider.client = VerificationClient("{}")

    result = provider.assess_evidence("What is the photoluminescence peak?", [_evidence()])

    assert result.sufficient is False
    assert result.supporting_chunk_ids == []
    assert result.fallback_used is True
    assert "服务暂时不可用" in result.reason


def test_deepseek_answer_falls_back_to_verified_evidence_when_connection_is_unavailable():
    provider = DeepSeekLlmProvider.__new__(DeepSeekLlmProvider)
    provider.model = "test-model"
    provider.timeout_seconds = 2.0
    provider.executor = RaisingExecutor(ConnectionError("Connection error."))
    provider.client = VerificationClient("{}")

    answer = provider.answer_with_evidence("What is the photoluminescence peak?", [_evidence()])

    assert "LLM 服务暂时不可用" in answer
    assert "520 nm" in answer


def test_composed_provider_uses_fake_answer_but_real_assessment_delegate():
    assessment_provider = DeepSeekLlmProvider.__new__(DeepSeekLlmProvider)
    assessment_provider.model = "test-model"
    assessment_provider.timeout_seconds = 2.0
    assessment_provider.executor = ImmediateExecutor()
    assessment_provider.client = VerificationClient(
        '{"sufficient":false,"covered_aspects":[],"missing_aspects":["高温稳定性"],'
        '"conflicts":[],"supporting_chunk_ids":[],"next_search_goal":"检索高温稳定性",'
        '"confidence":0.6,"reason":"现有证据缺少高温条件。"}'
    )
    provider = ComposedLlmProvider(FakeLlmProvider(), assessment_provider)

    assessment = provider.assess_evidence("Is it stable at high temperature?", [_evidence()])
    answer = provider.answer_with_evidence("What is reported?", [_evidence()])

    assert assessment.sufficient is False
    assert assessment.used_llm is True
    assert assessment.missing_aspects == ["高温稳定性"]
    assert provider.uses_llm_for_answer() is False
    assert "grounded draft answer" in answer


def test_deepseek_assessor_receives_task_specific_coverage_boundary():
    provider = DeepSeekLlmProvider.__new__(DeepSeekLlmProvider)
    provider.model = "test-model"
    provider.timeout_seconds = 2.0
    provider.executor = ImmediateExecutor()
    provider.client = VerificationClient(
        '{"sufficient":true,"covered_aspects":["PL 图"],"missing_aspects":[],"conflicts":[],'
        '"supporting_chunk_ids":["c1"],"next_search_goal":null,"confidence":0.9,"reason":"已找到相关图。"}'
    )

    provider.assess_evidence_for_task(
        "Which PL spectrum figures are shown?",
        [_evidence()],
        task_decision=TaskDecision(task_type="figure_query", requires_retrieval=True, risk_level="low"),
        requirements=["找到相关图即可，不要求穷尽全部图片。"],
        retrieval_round=1,
        max_retrieval_rounds=2,
    )

    user_message = provider.client.chat.completions.messages[1]["content"]
    assert '"task_type": "figure_query"' in user_message
    assert "不要求穷尽全部图片" in user_message
    assert '"max_retrieval_rounds": 2' in user_message
