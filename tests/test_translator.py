from conftest import FakeLLM
from factloop import llm
from factloop.agents import translator
from factloop.schemas import FactFile, FactItem, TranslatedClaims


def _fact_file():
    return FactFile(
        language="en",
        items=[
            FactItem(claim="A", source_title="t1", source_url="https://a.example"),
            FactItem(claim="B", source_title="t2", source_url="https://b.example"),
        ],
    )


def test_translator_translates_claims_and_preserves_sources(monkeypatch):
    fake = FakeLLM(structured=TranslatedClaims(claims=["甲", "乙"]))
    monkeypatch.setattr(llm, "build_llm", lambda _role: fake)

    result = translator.translator_node({"fact_file": _fact_file(), "language": "zh"})

    translated = result["translated_fact_file"]
    assert translated.language == "zh"
    assert [item.claim for item in translated.items] == ["甲", "乙"]
    assert [item.source_url for item in translated.items] == [
        "https://a.example",
        "https://b.example",
    ]


def test_translator_falls_back_when_count_mismatches(monkeypatch):
    fake = FakeLLM(structured=TranslatedClaims(claims=["only one"]))
    monkeypatch.setattr(llm, "build_llm", lambda _role: fake)

    result = translator.translator_node({"fact_file": _fact_file(), "language": "zh"})

    assert [item.claim for item in result["translated_fact_file"].items] == ["A", "B"]


def _one_item_fact_file(claim):
    return FactFile(
        language="en",
        items=[FactItem(claim=claim, source_title="t", source_url="https://a.example")],
    )


def test_annotation_kept_when_original_name_is_in_source(monkeypatch):
    # The source names the person, so the appended original is grounded and stays.
    fake = FakeLLM(structured=TranslatedClaims(claims=["威善高(Scott Wiener) 提出了該法案。"]))
    monkeypatch.setattr(llm, "build_llm", lambda _role: fake)
    fact_file = _one_item_fact_file("State Senator Scott Wiener proposed the bill.")

    result = translator.translator_node({"fact_file": fact_file, "language": "zh"})

    assert result["translated_fact_file"].items[0].claim == "威善高(Scott Wiener) 提出了該法案。"


def test_fabricated_annotation_is_stripped(monkeypatch):
    # The source only says "the senator" (no name), so a parenthesized name the
    # model produced is ungrounded and must be removed, leaving no invented English.
    fake = FakeLLM(structured=TranslatedClaims(claims=["參議員(Scott Wiener) 提出了該法案。"]))
    monkeypatch.setattr(llm, "build_llm", lambda _role: fake)
    fact_file = _one_item_fact_file("The senator proposed the bill.")

    claim = translator.translator_node({"fact_file": fact_file, "language": "zh"})[
        "translated_fact_file"
    ].items[0].claim

    assert claim == "參議員 提出了該法案。"
    assert "Scott Wiener" not in claim
    assert "(" not in claim and "（" not in claim


def test_latin_target_never_triggers_annotation_logic(monkeypatch):
    # For a Latin-script target the whole feature is off: even an ungrounded Latin
    # parenthetical is left untouched, proving the verification pass never runs.
    fake = FakeLLM(structured=TranslatedClaims(claims=["El proyecto de (Ghost Person) avanza."]))
    monkeypatch.setattr(llm, "build_llm", lambda _role: fake)
    fact_file = _one_item_fact_file("The senator proposed the bill.")

    result = translator.translator_node({"fact_file": fact_file, "language": "es"})

    assert result["translated_fact_file"].items[0].claim == "El proyecto de (Ghost Person) avanza."
