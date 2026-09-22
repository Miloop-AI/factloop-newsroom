from .. import llm
from ..schemas import FactCheckResult, FactFile
from ..state import NewsroomState

_SYSTEM = (
    "You are a rigorous fact-checker. The fact file is the only source of truth. It "
    "may be written in a different language than the article, so compare meaning, not "
    "wording. Flag every person, number, quote, place or event in the article that the "
    "fact file does not support. Pass only when nothing is unsupported."
)


_EMPTY_DRAFT_NOTE = "The draft came back empty. Write the article again from the facts."


def _facts_block(fact_file: FactFile) -> str:
    return "\n".join(f"- {item.claim}" for item in fact_file.items)


def fact_checker_node(state: NewsroomState) -> dict:
    revisions = state.get("revisions", 0)

    # A draft can come back empty, most often from a reasoning model that spends
    # its whole output budget thinking and returns no content. An empty draft
    # contradicts nothing, so asking the checker would earn it a pass and the
    # optimizer downstream would then invent an article out of its own prompt.
    # Failing it here keeps that out of the trust chain and costs no call.
    article = (state.get("article") or "").strip()
    if not article:
        result = FactCheckResult(passed=False, correction_notes=_EMPTY_DRAFT_NOTE)
        return {"fact_check": result, "revisions": revisions + 1}

    # Judged against the untranslated ground truth so a translation slip can never
    # be mistaken for a fabrication (or excuse one).
    fact_file = state["fact_file"]
    prompt = (
        f"Fact file:\n{_facts_block(fact_file)}\n\n"
        f"Article:\n{article}\n\n"
        "Return the structured verdict. Write correction_notes in the article's "
        "language, naming exactly what to change."
    )

    model = llm.build_llm("factchecker").with_structured_output(FactCheckResult)
    result = model.invoke([("system", _SYSTEM), ("human", prompt)])

    return {"fact_check": result, "revisions": revisions + (0 if result.passed else 1)}
