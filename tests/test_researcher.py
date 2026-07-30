from factloop.agents import researcher


def test_build_fact_file_maps_hits_one_to_one(tavily_results):
    fact_file = researcher.build_fact_file(tavily_results, "en")

    assert fact_file.language == "en"
    assert len(fact_file.items) == 2
    first = fact_file.items[0]
    assert first.claim.startswith("The council approved")
    assert first.source_title == "Coastal city trials tidal power"
    assert first.source_url == "https://news.example/tidal-array"


def test_build_fact_file_drops_thin_or_sourceless_hits():
    hits = [
        {"title": "ok", "snippet": "x" * researcher.MIN_SNIPPET_CHARS, "url": "https://a.example"},
        {"title": "too short", "snippet": "brief", "url": "https://b.example"},
        {"title": "no url", "snippet": "y" * 80, "url": ""},
    ]

    fact_file = researcher.build_fact_file(hits, "en")

    assert [item.source_url for item in fact_file.items] == ["https://a.example"]


def test_researcher_node_flags_empty_results(monkeypatch):
    monkeypatch.setattr(researcher, "_fetch", lambda _keyword: {"days": 7, "results": []})

    state = researcher.researcher_node({"keyword": "obscure", "language": "en"})

    assert state["no_results"] is True
    assert state["fact_file"].items == []


def test_researcher_node_surfaces_coverage_days(monkeypatch, tavily_results):
    monkeypatch.setattr(
        researcher, "_fetch", lambda _keyword: {"days": 5, "results": tavily_results}
    )

    state = researcher.researcher_node({"keyword": "tidal", "language": "en"})

    assert state["coverage_days"] == 5
    assert state["no_results"] is False
    assert len(state["fact_file"].items) == 2
