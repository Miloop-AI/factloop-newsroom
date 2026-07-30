import api
import pytest
from fastapi.testclient import TestClient

from factloop.schemas import FactCheckResult, FactFile, FactItem, GeoArticle

_GEO = GeoArticle(
    title="Tidal power moves ahead",
    subheadings=["What was approved", "Why it matters"],
    body="## What was approved\nBody text.",
    seo_keywords=["tidal", "energy"],
    meta_description="A coastal city advances tidal power.",
)
_FACT_FILE = FactFile(
    language="en",
    items=[
        FactItem(
            claim="A 40 megawatt tidal power array was approved off the northern pier.",
            source_title="Coastal city trials tidal power",
            source_url="https://news.example/tidal-array",
        )
    ],
)


@pytest.fixture
def client():
    # slowapi keeps its counters in module-level storage, so clear them between
    # tests to stop one test's requests from spending another's budget.
    api.limiter.reset()
    return TestClient(api.app)


def _ok_state():
    return {
        "language": "en",
        "coverage_days": 2,
        "geo": _GEO,
        "fact_file": _FACT_FILE,
    }


def _stub_run(monkeypatch, result):
    def fake_run(keyword):
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(api, "run_newsroom", fake_run)


def test_generate_returns_article_and_sources(client, monkeypatch):
    _stub_run(monkeypatch, _ok_state())

    response = client.post("/generate", json={"keyword": "coastal tidal energy"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["language"] == "en"
    assert body["coverage_days"] == 2
    assert body["article"]["title"] == "Tidal power moves ahead"
    assert body["sources"] == [
        {"title": "Coastal city trials tidal power", "url": "https://news.example/tidal-array"}
    ]


def test_generate_reports_no_results(client, monkeypatch):
    _stub_run(monkeypatch, {"language": "en", "no_results": True})

    body = client.post("/generate", json={"keyword": "obscure"}).json()

    assert body["status"] == "no_results"
    assert body["article"] is None


def test_generate_reports_qc_block(client, monkeypatch):
    _stub_run(
        monkeypatch,
        {
            "language": "en",
            "qc_blocked": True,
            "fact_check": FactCheckResult(passed=False, unverified_claims=["ghost fact"]),
        },
    )

    body = client.post("/generate", json={"keyword": "coastal tidal energy"}).json()

    assert body["status"] == "qc_blocked"
    assert body["unverified_claims"] == ["ghost fact"]
    assert body["article"] is None


def test_invalid_keyword_is_rejected(client, monkeypatch):
    _stub_run(monkeypatch, ValueError("Keyword must not be empty."))

    response = client.post("/generate", json={"keyword": "   "})

    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_health_is_open(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def _preflight(client, origin):
    return client.options(
        "/generate",
        headers={"Origin": origin, "Access-Control-Request-Method": "POST"},
    )


def test_cors_allows_each_configured_origin(client):
    for origin in ("https://miloop.ai", "http://localhost:8000"):
        response = _preflight(client, origin)
        assert response.headers.get("access-control-allow-origin") == origin


def test_cors_blocks_unlisted_origin(client):
    response = _preflight(client, "https://evil.example.com")

    assert response.headers.get("access-control-allow-origin") != "https://evil.example.com"


def test_rate_limit_blocks_sixth_request_with_retry_after(client, monkeypatch):
    _stub_run(monkeypatch, _ok_state())
    headers = {"X-Forwarded-For": "203.0.113.7"}
    payload = {"keyword": "coastal tidal energy"}

    responses = [client.post("/generate", json=payload, headers=headers) for _ in range(6)]
    statuses = [response.status_code for response in responses]

    assert statuses[:5] == [200] * 5
    assert statuses[5] == 429
    assert "retry-after" in {key.lower() for key in responses[5].headers}
