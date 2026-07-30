from types import SimpleNamespace

from factloop import language, llm


def _fake_detector(answers):
    """Model stub that answers with a canned code, keyed by the text it is asked about."""

    def build(_role):
        def invoke(messages):
            text = messages[-1][1]
            return SimpleNamespace(content=answers[text])

        return SimpleNamespace(invoke=invoke)

    return build


def _forbid_llm(_role):
    raise AssertionError("statistical detection should not call the model here")


def test_latin_defers_to_the_model(monkeypatch):
    # Latin script maps to many languages, so it can no longer be assumed English;
    # the model is consulted even for a clearly English phrase.
    consulted = []

    def build(_role):
        consulted.append(True)
        return SimpleNamespace(invoke=lambda _messages: SimpleNamespace(content="en"))

    monkeypatch.setattr(llm, "build_llm", build)
    assert language.detect_language("coastal tidal energy") == "en"
    assert consulted, "Latin input must reach the model rather than short-circuit to English"


def test_unaccented_latin_is_not_mislabelled_english(monkeypatch):
    # The real boundary case: short European phrases with no diacritics. Script
    # frequency cannot tell these from English, so each must reach the model.
    answers = {"Le chat mange": "fr", "Los libros": "es", "Das Auto": "de"}
    monkeypatch.setattr(llm, "build_llm", _fake_detector(answers))

    for text, code in answers.items():
        assert language.detect_language(text) == code


def test_kana_is_japanese_without_a_model_call(monkeypatch):
    monkeypatch.setattr(llm, "build_llm", _forbid_llm)
    assert language.detect_language("気候変動サミット") == "ja"


def test_long_han_is_chinese_by_statistics(monkeypatch):
    monkeypatch.setattr(llm, "build_llm", _forbid_llm)
    assert language.detect_language("氣候變遷高峰會談") == "zh"


def test_short_han_falls_back_to_the_model(monkeypatch):
    monkeypatch.setattr(llm, "build_llm", _fake_detector({"地震": "ja"}))
    assert language.detect_language("地震") == "ja"


def test_low_confidence_mixed_script_defers_to_the_model(monkeypatch):
    # Han is present but far from dominant, so the confidence bar is not met and
    # the model settles it. A sentinel code proves statistics did not decide.
    monkeypatch.setattr(
        llm, "build_llm", lambda _role: SimpleNamespace(
            invoke=lambda _messages: SimpleNamespace(content="zz")
        )
    )
    assert language.detect_language("AI 大") == "zz"
