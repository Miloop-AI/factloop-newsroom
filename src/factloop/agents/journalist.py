from .. import llm
from ..schemas import FactFile
from ..state import NewsroomState

_SYSTEM = (
    "You are a newswire journalist. Write a clear, coherent article using ONLY the "
    "facts provided. Never introduce a person, number, place, quote or event that is "
    "not among them. Write entirely in the requested language."
)


def _facts_block(fact_file: FactFile) -> str:
    return "\n".join(f"- {item.claim} (source: {item.source_title})" for item in fact_file.items)


def journalist_node(state: NewsroomState) -> dict:
    # English input skips translation, so fall back to the original fact file.
    fact_file = state.get("translated_fact_file") or state["fact_file"]
    parts = [
        f"Language (ISO 639-1 code): {state['language']}",
        "",
        "Facts:",
        _facts_block(fact_file),
    ]

    previous = state.get("fact_check")
    if previous is not None and not previous.passed:
        parts += [
            "",
            "A previous draft failed fact-checking. Rewrite it, removing anything not "
            "supported by the facts above, and address these notes:",
            previous.correction_notes,
        ]

    response = llm.build_llm("journalist").invoke(
        [("system", _SYSTEM), ("human", "\n".join(parts))]
    )
    return {"article": response.content}
