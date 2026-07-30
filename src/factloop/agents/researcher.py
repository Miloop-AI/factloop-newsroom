import asyncio

from fastmcp import Client

from ..mcp_server import mcp
from ..schemas import FactFile, FactItem
from ..state import NewsroomState

# Snippets shorter than this carry no verifiable substance; letting them into the
# fact file would only give the writer room to embellish, so they are dropped.
MIN_SNIPPET_CHARS = 40

_MAX_RESULTS = 5


def build_fact_file(results: list[dict], language: str) -> FactFile:
    """Map search hits one-to-one onto sourced facts, dropping thin snippets.

    Kept free of any LLM on purpose: this fact file is the ground truth the
    Fact-Checker judges against, so no model may sit between the sources and it.
    """
    items = [
        FactItem(
            claim=hit["snippet"].strip(),
            source_title=hit["title"].strip(),
            source_url=hit["url"].strip(),
        )
        for hit in results
        if len(hit.get("snippet", "").strip()) >= MIN_SNIPPET_CHARS and hit.get("url", "").strip()
    ]
    return FactFile(language=language, items=items)


def _fetch(keyword: str) -> dict:
    async def _run() -> dict:
        async with Client(mcp) as client:
            result = await client.call_tool(
                "search_news", {"query": keyword, "max_results": _MAX_RESULTS}
            )
            return result.data

    # The graph and Streamlit both run synchronously, so we own the event loop
    # here rather than forcing async plumbing through every downstream node.
    return asyncio.run(_run())


def researcher_node(state: NewsroomState) -> dict:
    search = _fetch(state["keyword"])
    fact_file = build_fact_file(search["results"], state["language"])
    return {
        "fact_file": fact_file,
        "coverage_days": search["days"],
        "no_results": not fact_file.items,
    }
