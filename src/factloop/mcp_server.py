from fastmcp import FastMCP

from . import tavily_client

mcp = FastMCP("factloop-newsroom")

# Recency is part of *this* tool's contract, not a knob the agents may relax, so
# the news topic and the search windows are wired in here rather than exposed.
# The windows widen only as far as needed: prefer the freshest reporting, but fall
# back to older days when a tight window returns too little to write from. Seven
# days stays the hard ceiling, so nothing downstream can cite a stale source.
_NEWS_TOPIC = "news"
_RECENCY_WINDOWS = (2, 5, 7)
_MIN_RESULTS = 3


@mcp.tool
async def search_news(query: str, max_results: int = 5) -> dict:
    """Recent news hits for a query, with the window that produced them.

    Returns ``{"days": int, "results": [{title, snippet, url}, ...]}``. The window
    starts at two days and widens to five then seven only when the current window
    yields fewer than three hits, so callers can report how far back the sources
    reach.
    """
    for days in _RECENCY_WINDOWS:
        results = await tavily_client.search(
            query, max_results=max_results, days=days, topic=_NEWS_TOPIC
        )
        if len(results) >= _MIN_RESULTS or days == _RECENCY_WINDOWS[-1]:
            return {"days": days, "results": results}


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
