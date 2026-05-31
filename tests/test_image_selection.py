from backend.app.domain.documents import Evidence, EvidenceImage
from backend.app.providers.llm import DeepSeekLlmProvider, _fallback_selected_images, _sanitize_selected_images


def test_sanitize_selected_images_filters_invalid_candidates_and_limits_results():
    by_chunk = {
        "c1": {
            "img1": EvidenceImage(image_id="img1", page=1),
            "img2": EvidenceImage(image_id="img2", page=1),
        },
        "c2": {
            "img3": EvidenceImage(image_id="img3", page=2),
            "img4": EvidenceImage(image_id="img4", page=2),
        },
    }

    selected = _sanitize_selected_images(
        [
            {"chunk_id": "c1", "image_id": "img1", "reason": "Used in the answer."},
            {"chunk_id": "c1", "image_id": "not-a-candidate", "reason": "Invalid."},
            {"chunk_id": "c2", "image_id": "img3", "reason": "Also useful."},
            {"chunk_id": "c2", "image_id": "img4", "reason": "Should be truncated."},
        ],
        by_chunk,
        max_images=2,
    )

    assert [image.image_id for image in selected["c1"]] == ["img1"]
    assert [image.image_id for image in selected["c2"]] == ["img3"]


def test_sanitize_selected_images_preserves_candidate_caption_metadata():
    by_chunk = {
        "c1": {
            "img1": EvidenceImage(
                image_id="img1",
                page=1,
                caption="Figure 1. Photoluminescence spectra.",
                nearby_text="The figure compares emission peaks.",
            )
        }
    }

    selected = _sanitize_selected_images(
        [{"chunk_id": "c1", "image_id": "img1", "reason": "Matches spectra discussion."}],
        by_chunk,
        max_images=3,
    )

    image = selected["c1"][0]
    assert image.caption == "Figure 1. Photoluminescence spectra."
    assert image.nearby_text == "The figure compares emission peaks."
    assert image.reason == "Matches spectra discussion."


class FakeChatCompletions:
    def __init__(self) -> None:
        self.messages = None

    def create(self, model, messages, response_format=None):
        self.messages = messages

        class Message:
            content = '{"images":[{"chunk_id":"c1","image_id":"img1","reason":"Caption matches."}]}'

        class Choice:
            message = Message()

        class Response:
            choices = [Choice()]

        return Response()


class FakeClient:
    def __init__(self) -> None:
        self.chat = type("Chat", (), {"completions": FakeChatCompletions()})()


class FailingChatCompletions:
    def create(self, model, messages, response_format=None):
        raise ConnectionError("Connection error.")


class FailingClient:
    def __init__(self) -> None:
        self.chat = type("Chat", (), {"completions": FailingChatCompletions()})()


def test_deepseek_image_selector_prompt_includes_caption_and_nearby_text():
    provider = DeepSeekLlmProvider.__new__(DeepSeekLlmProvider)
    provider.model = "test-model"
    provider.client = FakeClient()
    evidence = [
        Evidence(
            chunk_id="c1",
            document_id="doc1",
            text="The PL figure compares emission intensity.",
            page_start=3,
            page_end=3,
            section_title="Results",
            score=0.9,
            images=[
                EvidenceImage(
                    image_id="img1",
                    page=3,
                    caption="Figure 1. Photoluminescence spectra.",
                    nearby_text="Emission peaks increase after doping.",
                )
            ],
        )
    ]

    result = provider.select_images_for_answer("What changed in PL?", "PL intensity increased.", evidence)

    user_message = provider.client.chat.completions.messages[1]["content"]
    assert "Figure 1. Photoluminescence spectra." in user_message
    assert "Emission peaks increase after doping." in user_message
    assert result.used_llm is True
    assert result.selected_count == 1


def test_fallback_selected_images_uses_figure_references_when_llm_selects_none():
    evidence = [
        Evidence(
            chunk_id="c1",
            document_id="doc1",
            text="Figure 3. Photoluminescence spectra show green emission.",
            page_start=2,
            page_end=3,
            section_title="Results",
            score=0.9,
            images=[EvidenceImage(image_id="img1", page=2, width=640, height=480)],
        )
    ]

    selected = _fallback_selected_images("What do the PL spectra show?", "Figure 3 shows green emission.", evidence, 3)

    assert selected["c1"][0].image_id == "img1"
    assert selected["c1"][0].reason == "Selected from candidate images linked to figure-related evidence."


def test_deepseek_image_selector_falls_back_when_service_connection_is_unavailable():
    provider = DeepSeekLlmProvider.__new__(DeepSeekLlmProvider)
    provider.model = "test-model"
    provider.client = FailingClient()
    evidence = [
        Evidence(
            chunk_id="c1",
            document_id="doc1",
            text="Figure 3. Photoluminescence spectra show green emission.",
            page_start=2,
            page_end=3,
            section_title="Results",
            score=0.9,
            images=[EvidenceImage(image_id="img1", page=2, width=640, height=480)],
        )
    ]

    result = provider.select_images_for_answer("What do the PL spectra show?", "Figure 3 shows green emission.", evidence)

    assert result.images["c1"][0].image_id == "img1"
    assert result.used_llm is False
    assert result.fallback_used is True


def test_fallback_selected_images_stays_empty_without_visual_signal():
    evidence = [
        Evidence(
            chunk_id="c1",
            document_id="doc1",
            text="The synthesis temperature was 80 C.",
            page_start=2,
            page_end=2,
            section_title="Methods",
            score=0.9,
            images=[EvidenceImage(image_id="img1", page=2)],
        )
    ]

    selected = _fallback_selected_images("What was the synthesis temperature?", "It was 80 C.", evidence, 3)

    assert selected == {}
