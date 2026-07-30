import httpx

from .config import get_settings

_SEARCH_PATH = "/search"


async def search(
    query: str,
    *,
    max_results: int = 5,
    days: int = 7,
    topic: str = "news",
) -> list[dict]:
    """Query Tavily and normalise each hit to ``{title, snippet, url}``.

    ``content`` is Tavily's field for the passage that matched; we surface it as
    ``snippet`` so callers stay decoupled from the vendor's response shape.
    """
    settings = get_settings()
    if not settings.tavily_api_key:
        raise RuntimeError("TAVILY_API_KEY is not set; copy .env.example to .env and fill it in.")

    payload = {
        "query": query,
        "topic": topic,
        "days": days,
        "max_results": max_results,
        "search_depth": "basic",
    }
    headers = {"Authorization": f"Bearer {settings.tavily_api_key}"}
    async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
        response = await client.post(
            f"{settings.tavily_base_url}{_SEARCH_PATH}", json=payload, headers=headers
        )
        response.raise_for_status()
        data = response.json()

    return [
        {
            "title": hit.get("title", ""),
            "snippet": hit.get("content", ""),
            "url": hit.get("url", ""),
        }
        for hit in data.get("results", [])
    ]
