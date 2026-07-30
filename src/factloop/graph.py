from langgraph.graph import END, START, StateGraph

from .agents.fact_checker import fact_checker_node
from .agents.geo_optimizer import geo_optimizer_node
from .agents.journalist import journalist_node
from .agents.researcher import researcher_node
from .agents.translator import translator_node
from .config import MAX_KEYWORD_CHARS
from .language import detect_language
from .state import NewsroomState

# How many times the Journalist may be sent back to rewrite before the run is
# stopped and the reader is told the piece could not be verified.
MAX_REVISIONS = 2


def _after_research(state: NewsroomState) -> str:
    if state.get("no_results"):
        return END
    # English needs no translation, so the Journalist reads the original facts.
    if state["language"] == "en":
        return "journalist"
    return "translator"


def _after_fact_check(state: NewsroomState) -> str:
    if state["fact_check"].passed:
        return "geo_optimizer"
    if state.get("revisions", 0) >= MAX_REVISIONS:
        return END
    return "journalist"


def build_graph():
    graph = StateGraph(NewsroomState)
    graph.add_node("researcher", researcher_node)
    graph.add_node("translator", translator_node)
    graph.add_node("journalist", journalist_node)
    graph.add_node("fact_checker", fact_checker_node)
    graph.add_node("geo_optimizer", geo_optimizer_node)

    graph.add_edge(START, "researcher")
    graph.add_conditional_edges("researcher", _after_research, ["translator", "journalist", END])
    graph.add_edge("translator", "journalist")
    graph.add_edge("journalist", "fact_checker")
    graph.add_conditional_edges(
        "fact_checker", _after_fact_check, ["journalist", "geo_optimizer", END]
    )
    graph.add_edge("geo_optimizer", END)
    return graph.compile()


NEWSROOM = build_graph()


def run_newsroom(keyword: str) -> NewsroomState:
    """Detect the keyword's language and run it through the newsroom pipeline.

    The length ceiling is enforced here, not just in the UI, so every entry point
    into the graph is bounded.
    """
    keyword = keyword.strip()
    if not keyword:
        raise ValueError("Keyword must not be empty.")
    if len(keyword) > MAX_KEYWORD_CHARS:
        raise ValueError(f"Keyword must be at most {MAX_KEYWORD_CHARS} characters.")

    language = detect_language(keyword)
    result = NEWSROOM.invoke({"keyword": keyword, "language": language, "revisions": 0})

    # The graph can stop for two distinct reasons; surface a QC block explicitly so
    # the UI can celebrate the fact-checker rather than look like a failure.
    result["qc_blocked"] = (
        result.get("geo") is None
        and not result.get("no_results")
        and result.get("fact_check") is not None
        and not result["fact_check"].passed
    )
    return result
