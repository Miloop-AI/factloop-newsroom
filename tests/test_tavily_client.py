import httpx

from factloop import tavily_client


class _FakeResponse:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self._data


class _FakeAsyncClient:
    """Records the outgoing request and replays a canned Tavily payload."""

    last_request: dict = {}

    def __init__(self, *_args, **_kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def post(self, url, json, headers):
        _FakeAsyncClient.last_request = {"url": url, "json": json, "headers": headers}
        return _FakeResponse(
            {"results": [{"title": "T", "content": "a matched passage", "url": "https://x.example"}]}
        )


async def test_search_normalises_hits_and_sends_filters(monkeypatch):
    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)

    results = await tavily_client.search("tidal energy", max_results=3, days=7, topic="news")

    assert results == [
        {"title": "T", "snippet": "a matched passage", "url": "https://x.example"}
    ]
    sent = _FakeAsyncClient.last_request["json"]
    assert sent["topic"] == "news"
    assert sent["days"] == 7
    assert sent["max_results"] == 3


async def test_search_requires_api_key(monkeypatch):
    from factloop import config

    monkeypatch.setattr(config.get_settings(), "tavily_api_key", "", raising=False)
    try:
        await tavily_client.search("anything")
    except RuntimeError as exc:
        assert "TAVILY_API_KEY" in str(exc)
    else:
        raise AssertionError("expected RuntimeError when the key is missing")
