from fastmcp import Client

from factloop import mcp_server, tavily_client


def _hits(n):
    return [
        {"title": f"T{i}", "snippet": f"a matched passage {i}", "url": f"https://x{i}.example"}
        for i in range(n)
    ]


def _stub_search(monkeypatch, counts_by_day):
    """Make ``tavily_client.search`` return a fixed hit count per requested window."""
    calls = []

    async def fake_search(query, *, max_results, days, topic):
        calls.append(days)
        assert query == "tidal energy"
        assert topic == "news"
        return _hits(counts_by_day[days])

    monkeypatch.setattr(tavily_client, "search", fake_search)
    return calls


async def _run_search():
    async with Client(mcp_server.mcp) as client:
        result = await client.call_tool("search_news", {"query": "tidal energy", "max_results": 5})
    return result.data


async def test_tight_window_used_when_it_yields_enough(monkeypatch):
    calls = _stub_search(monkeypatch, {2: 3, 5: 9, 7: 9})

    data = await _run_search()

    assert calls == [2]
    assert data["days"] == 2
    assert len(data["results"]) == 3


async def test_window_widens_to_five_when_two_days_is_thin(monkeypatch):
    calls = _stub_search(monkeypatch, {2: 1, 5: 4, 7: 9})

    data = await _run_search()

    assert calls == [2, 5]
    assert data["days"] == 5
    assert len(data["results"]) == 4


async def test_falls_back_to_seven_days_even_when_still_thin(monkeypatch):
    calls = _stub_search(monkeypatch, {2: 0, 5: 1, 7: 2})

    data = await _run_search()

    assert calls == [2, 5, 7]
    assert data["days"] == 7
    assert len(data["results"]) == 2
