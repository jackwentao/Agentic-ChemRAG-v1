import pytest

from backend.app.evaluation.judge import LlmJudge, parse_judge_payload


class FakeMessage:
    content = '{"faithfulness":5,"completeness":4,"relevance":3,"rationale":"grounded"}'


class FakeChoice:
    message = FakeMessage()


class FakeResponse:
    choices = [FakeChoice()]


class FakeClient:
    def __init__(self) -> None:
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeResponse()


class FakeSettings:
    deepseek_model = "judge-model"
    deepseek_api_key = "key"
    deepseek_base_url = "https://example.test"
    deepseek_timeout_seconds = 5


def test_parse_judge_payload_requires_scores_from_one_to_five():
    assert parse_judge_payload('{"faithfulness":5,"completeness":4,"relevance":3}') == {
        "faithfulness": 5,
        "completeness": 4,
        "relevance": 3,
        "rationale": "",
    }

    with pytest.raises(ValueError, match="faithfulness"):
        parse_judge_payload('{"faithfulness":6,"completeness":4,"relevance":3}')


def test_llm_judge_sends_grounded_prompt_to_client():
    client = FakeClient()
    judge = LlmJudge(FakeSettings(), client=client)

    result = judge.score(
        question="What is shown?",
        gold_hints={"expected_pages": [4]},
        answer="PL spectrum is shown.",
        citations=[{"quote": "Fig. 2 PL spectrum."}],
    )

    assert result["faithfulness"] == 5
    call = client.calls[0]
    assert call["model"] == "judge-model"
    assert call["response_format"] == {"type": "json_object"}
    assert "Do not use external knowledge" in call["messages"][0]["content"]
    assert "PL spectrum is shown" in call["messages"][1]["content"]
