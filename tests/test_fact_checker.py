from conftest import FakeLLM
from factloop import llm
from factloop.agents import fact_checker
from factloop.schemas import FactCheckResult, FactFile, FactItem


def _state(article, revisions=0):
    fact_file = FactFile(
        language="en",
        items=[FactItem(claim="Tidal array approved", source_title="t", source_url="https://a")],
    )
    return {"fact_file": fact_file, "article": article, "revisions": revisions}


def test_fact_checker_flags_and_counts_a_failure(monkeypatch):
    verdict = FactCheckResult(
        passed=False, unverified_claims=["a named mayor"], correction_notes="remove the mayor"
    )
    monkeypatch.setattr(llm, "build_llm", lambda _role: FakeLLM(structured=verdict))

    result = fact_checker.fact_checker_node(_state("Mayor Reyes opened the array.", revisions=0))

    assert result["fact_check"].passed is False
    assert result["revisions"] == 1


def test_fact_checker_pass_does_not_increment(monkeypatch):
    monkeypatch.setattr(
        llm, "build_llm", lambda _role: FakeLLM(structured=FactCheckResult(passed=True))
    )

    result = fact_checker.fact_checker_node(_state("A tidal array was approved.", revisions=1))

    assert result["fact_check"].passed is True
    assert result["revisions"] == 1
