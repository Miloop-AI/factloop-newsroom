import os
from types import SimpleNamespace

import pytest

# Give config real-looking credentials before anything imports the settings, so
# the modules under test never trip their "missing key" guards during unit runs.
os.environ.setdefault("TAVILY_API_KEY", "test-tavily-key")
os.environ.setdefault("OPENROUTER_API_KEY", "test-openrouter-key")


class _StructuredModel:
    """Stand-in for ``llm.with_structured_output(Schema)``; returns a fixed value."""

    def __init__(self, value):
        self._value = value

    def invoke(self, _messages):
        return self._value


class FakeLLM:
    """Minimal drop-in for a chat model: returns canned text or a canned schema."""

    def __init__(self, *, content="", structured=None):
        self._content = content
        self._structured = structured

    def invoke(self, _messages):
        return SimpleNamespace(content=self._content)

    def with_structured_output(self, _schema):
        return _StructuredModel(self._structured)


@pytest.fixture
def tavily_results():
    return [
        {
            "title": "Coastal city trials tidal power",
            "snippet": "The council approved a 40 megawatt tidal array off the northern pier.",
            "url": "https://news.example/tidal-array",
        },
        {
            "title": "Grid operator reports record demand",
            "snippet": "Peak demand reached 8.2 gigawatts during the heatwave on Tuesday.",
            "url": "https://news.example/record-demand",
        },
    ]
