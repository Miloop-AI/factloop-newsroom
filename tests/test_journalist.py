from types import SimpleNamespace

from factloop import llm
from factloop.agents import journalist
from factloop.schemas import FactCheckResult, FactFile, FactItem


class _RecordingLLM:
    """Captures the human prompt so tests can assert what the writer was told."""

    def __init__(self):
        self.human = ""

    def invoke(self, messages):
        self.human = dict(messages)["human"]
        return SimpleNamespace(content="an article")


def _state(**extra):
    fact_file = FactFile(
        language="en",
        items=[FactItem(claim="Tidal array approved", source_title="t", source_url="https://a")],
    )
    return {"language": "en", "translated_fact_file": fact_file, **extra}


def test_journalist_writes_from_facts(monkeypatch):
    recorder = _RecordingLLM()
    monkeypatch.setattr(llm, "build_llm", lambda _role: recorder)

    result = journalist.journalist_node(_state())

    assert result["article"] == "an article"
    assert "Tidal array approved" in recorder.human


def test_journalist_incorporates_correction_notes(monkeypatch):
    recorder = _RecordingLLM()
    monkeypatch.setattr(llm, "build_llm", lambda _role: recorder)
    failed = FactCheckResult(
        passed=False, unverified_claims=["ghost"], correction_notes="drop ghost"
    )

    journalist.journalist_node(_state(fact_check=failed))

    assert "drop ghost" in recorder.human
