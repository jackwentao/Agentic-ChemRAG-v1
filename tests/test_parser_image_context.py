from backend.app.parsing.parser import PyMuPDFParser


def test_parser_image_context_recognizes_english_caption():
    parser = PyMuPDFParser()

    contexts = parser._extract_image_contexts(
        "Results\nFig. 1. SEM image of ZnO nanoparticles after Ag doping.\nThe particles are uniform.",
        image_count=1,
    )

    assert contexts == [
        {
            "label": "Fig. 1",
            "caption": "Fig. 1. SEM image of ZnO nanoparticles after Ag doping. The particles are uniform.",
            "nearby_text": None,
        }
    ]


def test_parser_image_context_recognizes_chinese_caption():
    parser = PyMuPDFParser()

    contexts = parser._extract_image_contexts(
        "结果\n图1 ZnO纳米颗粒的荧光光谱。\n发射峰随掺杂变化。",
        image_count=1,
    )

    assert contexts[0]["caption"] == "图1 ZnO纳米颗粒的荧光光谱。 发射峰随掺杂变化。"
    assert contexts[0]["nearby_text"] is None


def test_parser_image_context_binds_multiple_images_to_captions_by_order():
    parser = PyMuPDFParser()

    contexts = parser._extract_image_contexts(
        "\n".join(
            [
                "Figure 1. XRD patterns of the samples.",
                "Figure 2. Photoluminescence spectra under 325 nm excitation.",
            ]
        ),
        image_count=2,
    )

    assert contexts[0]["caption"] == "Figure 1. XRD patterns of the samples."
    assert contexts[1]["caption"] == "Figure 2. Photoluminescence spectra under 325 nm excitation."


def test_parser_image_context_uses_nearby_text_when_caption_is_absent():
    parser = PyMuPDFParser()

    contexts = parser._extract_image_contexts("Results\nZnO particles show strong green emission.", image_count=1)

    assert contexts == [
        {
            "label": None,
            "caption": None,
            "nearby_text": "Results ZnO particles show strong green emission.",
        }
    ]
