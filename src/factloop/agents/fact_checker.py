from .. import llm
from ..schemas import FactCheckResult, FactFile
from ..state import NewsroomState

_SYSTEM = (
    "You are a rigorous fact-checker. The fact file is the only source of truth. It "
    "may be written in a different language than the article, so compare meaning, not "
    "wording. Flag every person, number, quote, place or event in the article that the "
    "fact file does not support. Pass only when nothing is unsupported."
)


def _facts_block(fact_file: FactFile) -> str:
    return "\n".join(f"- {item.claim}" for item in fact_file.items)


def fact_checker_node(state: NewsroomState) -> dict:
    # Judged against the untranslated ground truth so a translation slip can never
    # be mistaken for a fabrication (or excuse one).
    fact_file = state["fact_file"]
    prompt = (
        f"Fact file:\n{_facts_block(fact_file)}\n\n"
        f"Article:\n{state['article']}\n\n"
        "Return the structured verdict. Write correction_notes in the article's "
        "language, naming exactly what to change."
    )

    model = llm.build_llm("factchecker").with_structured_output(FactCheckResult)
    result = model.invoke([("system", _SYSTEM), ("human", prompt)])

    revisions = state.get("revisions", 0) + (0 if result.passed else 1)
    return {"fact_check": result, "revisions": revisions}
