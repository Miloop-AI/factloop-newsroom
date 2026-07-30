from conftest import FakeLLM
from factloop import graph, llm
from factloop.agents import researcher
from factloop.schemas import FactCheckResult, GeoArticle, TranslatedClaims

_GEO = GeoArticle(
    title="Tidal power moves ahead",
    subheadings=["What was approved", "Why it matters"],
    body="Body text.",
    seo_keywords=["tidal", "energy"],
    meta_description="A coastal city advances tidal power.",
)


def _search(days, results):
    return lambda _keyword: {"days": days, "results": results}


def _dispatcher(verdict):
    def build(role):
        if role == "detector":
            return FakeLLM(content="en")
        if role == "translator":
            return FakeLLM(structured=TranslatedClaims(claims=["翻一", "翻二"]))
        if role == "journalist":
            return FakeLLM(content="A short article.")
        if role == "factchecker":
            return FakeLLM(structured=verdict)
        if role == "geo":
            return FakeLLM(structured=_GEO)
        raise AssertionError(f"unexpected role {role}")

    return build


def test_happy_path_reaches_geo(monkeypatch, tavily_results):
    monkeypatch.setattr(researcher, "_fetch", _search(2, tavily_results))
    monkeypatch.setattr(llm, "build_llm", _dispatcher(FactCheckResult(passed=True)))

    result = graph.run_newsroom("coastal tidal energy")

    assert result["qc_blocked"] is False
    assert result["geo"].title == "Tidal power moves ahead"
    assert result["coverage_days"] == 2
    # Provenance survives to the end via the untranslated fact file.
    assert result["fact_file"].items[0].source_url == "https://news.example/tidal-array"


def test_persistent_hallucination_is_blocked_after_max_revisions(monkeypatch, tavily_results):
    monkeypatch.setattr(researcher, "_fetch", _search(7, tavily_results))
    failing = FactCheckResult(passed=False, unverified_claims=["ghost"], correction_notes="drop it")
    monkeypatch.setattr(llm, "build_llm", _dispatcher(failing))

    # Chinese keyword also exercises the Translator branch on the way in.
    result = graph.run_newsroom("氣候變遷高峰會談")

    assert result["qc_blocked"] is True
    assert result.get("geo") is None
    assert result["revisions"] == graph.MAX_REVISIONS
    assert result["fact_check"].unverified_claims == ["ghost"]


def test_no_results_stops_before_writing(monkeypatch):
    monkeypatch.setattr(researcher, "_fetch", _search(7, []))
    monkeypatch.setattr(llm, "build_llm", _dispatcher(FactCheckResult(passed=True)))

    result = graph.run_newsroom("nonexistent obscure topic")

    assert result["no_results"] is True
    assert result["qc_blocked"] is False
    assert result.get("geo") is None
    assert result.get("article") is None
