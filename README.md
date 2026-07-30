# factloop-newsroom

A multi-agent newsroom that turns a keyword (in any language) into a fact-checked,
source-backed news article written in that same language. It searches recent news, widening its
window up to a week only when a tighter one returns too little, drafts an article strictly from what
it found, verifies its own draft against the sources, and only then formats the result for search
and generative-engine visibility.

```
keyword ──► Researcher ──► Translator ──► Journalist ──► Fact-Checker ──► GEO Optimizer ──► article
            (pure code)    (skips for EN)                     │ ▲                            + sources
                                                     rejected │ │ rewrite (max 2)
                                                              ▼ │
                                                          Journalist
```

## Quick start

```bash
python -m pip install -e ".[dev]"
cp .env.example .env          # then fill in your keys
streamlit run app.py
```

Run the tests (no network or API keys required, the LLM and search are stubbed):

```bash
pytest
```

## Design decisions

### Why Tavily for search?
The pipeline only ever wants *recent, rankable news with clean source URLs*, not a general web
crawl. Tavily returns exactly that shape (title, matched passage, and a canonical URL per hit) and
exposes a `topic="news"` mode plus a `days` window, so recency is a hard API filter rather than
something we post-process and hope for. One dependency covers retrieval and recency; everything
vendor-specific is isolated in `tavily_client.py`.

### Why wrap search in our own MCP tool instead of mounting Tavily's remote MCP?
Recency here is a *contract*, not a preference. By owning the tool (`search_news` in
`mcp_server.py`) we fix `topic="news"` and cap the lookback at seven days, so no agent can drift
off-topic or reach for stale reporting. Within that ceiling the tool widens its window
progressively: it first asks for the last two days, and only when that comes back with too few hits
(fewer than three) does it fall back to five days, then seven, stopping at the first window that
returns enough to work with. That balances two needs the reader cares about, freshness and having
enough sources to write from, without exposing either as a knob the model can turn. The tool also
reports how many days it actually used, so the rest of the pipeline (and the reader) can see the
real coverage. Mounting a general third-party MCP would re-expose those knobs and hand the model the
ability to relax the very constraint the product depends on. Writing our own tool also keeps the
vendor swappable behind a stable interface.

### Why LangGraph rather than a plain agent loop?
The interesting behaviour is control flow: skip translation for English, and bounce a failed draft
back to the writer up to twice before giving up. LangGraph makes those transitions explicit,
inspectable edges instead of nested conditionals buried in one long function. The retry loop and
its ceiling live in `graph.py` as routing functions you can read at a glance.

### Why is the Researcher pure code with no LLM?
The fact file is the single source of truth that everything downstream is judged against. If a
model sat between the search results and that file, it could quietly summarise, embellish, or drop
detail, and then the fact-checker would be verifying the article against an already-corrupted
baseline. So the Researcher just maps each search hit one-to-one onto a sourced fact and drops
snippets too thin to verify. No model, no drift.

### Why does the Translator sit *outside* the fact-check?
For non-English topics, sources are often in another language than the reader wants. The Translator
renders the facts into the reader's language so the Journalist can write natively, but the
Fact-Checker still compares the finished article against the **original, untranslated** fact file.
That keeps translation out of the trust chain: a translation slip can't launder a fabrication, and
it can't be mistaken for one either. The state carries both `fact_file` (ground truth) and
`translated_fact_file` (writing source) precisely so these never get conflated. English input skips
this step entirely.

### How does it stay language-agnostic?
The language is detected once from the keyword and threaded through the graph; every agent is told
to answer in that code, and nothing about a specific language is hardcoded. Detection uses one rule
for every writing system: Unicode script frequency only names a language when the script is
near-unique to it (kana → Japanese, hangul → Korean, Han → Chinese), and anything that clears
neither the confidence nor the length bar falls back to a cheap detector model. Crucially, Latin
script maps to many languages, so it never clears that bar on its own. Spanish, French and German
are resolved by the model rather than silently assumed to be English. That model is deliberately the
cheapest tier, because after this rule every non-CJK keyword (including the common English case) hits
it.

### Why a different model per agent, all via OpenRouter?
The jobs have different failure modes. The Fact-Checker is reliability-critical and leans on
structured output, so it gets the strongest model; the GEO formatter is a cheap, mechanical pass.
OpenRouter lets each role point at the best model for its job through one API, and each is a
separate environment variable so a role can be upgraded or swapped without touching the others. The
Researcher takes no model at all. Defaults live in `.env.example`; any role left blank falls back to
`OPENROUTER_MODEL`.

| Role | Default model | Environment variable |
| --- | --- | --- |
| Detector | `google/gemini-2.5-flash-lite` | `OPENROUTER_MODEL_DETECTOR` |
| Translator | `anthropic/claude-haiku-4.5` | `OPENROUTER_MODEL_TRANSLATOR` |
| Journalist | `google/gemini-2.5-pro` | `OPENROUTER_MODEL_JOURNALIST` |
| Fact-Checker | `openai/gpt-5` | `OPENROUTER_MODEL_FACTCHECKER` |
| GEO Optimizer | `google/gemini-2.5-flash` | `OPENROUTER_MODEL_GEO` |
| Researcher | None (pure code, no model) | n/a |

The Detector is the language classifier from the pipeline's front door; it runs the cheapest,
fastest model because every non-CJK keyword (the common English case included) reaches it.

### What happens when a hallucination can't be fixed?
If the Journalist can't produce a draft the Fact-Checker will pass within two rewrites, the run
stops without publishing and the UI says so, listing the specific claims that couldn't be
verified. That is the system working as intended: catching an unsupported claim and refusing to
ship it is the whole point, and it doubles as a live demonstration of the quality gate.

## Running as a service

`app.py` (Streamlit) is the local-development surface. For public deployment there is a FastAPI
entry point, `api.py`, that drives the **same** `run_newsroom` pipeline, so no logic is duplicated.

```bash
uvicorn api:app --port 8000        # local
curl -X POST localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"keyword": "coastal tidal energy"}'
```

```jsonc
// 200 response
{
  "status": "ok",                  // or "no_results" / "qc_blocked"
  "language": "en",
  "coverage_days": 2,              // how far back the sources actually reach
  "article": { "title": "…", "subheadings": ["…"], "body": "…",
               "seo_keywords": ["…"], "meta_description": "…" },
  "sources": [ { "title": "…", "url": "https://…" } ],
  "unverified_claims": []          // populated only when status is "qc_blocked"
}
```

`GET /health` returns `{"status": "ok"}` for platform health checks.

### Why is the API request synchronous?
The endpoint is a plain `def`, not `async def`, because `run_newsroom` is synchronous and calls
`asyncio.run` internally, and invoking it from a running event loop would raise. FastAPI runs sync
handlers in a threadpool, so there is no loop to clash with. A full run makes several sequential LLM
calls, so expect a long-held request; that is fine for a demo, and a job-queue would be the next step
if it needed to scale.

### How is abuse contained?
Keys live only in the server environment, and callers never send them. `/generate` is rate-limited
per client IP (default `5/hour`) via slowapi, keyed off the real caller from `X-Forwarded-For` rather
than the proxy. CORS is scoped to an explicit allowlist of origins. The rate limiter uses in-memory
storage, so it is
exact only with a single worker (which the Render config pins); the counter resets on restart and is
not shared across instances, and a Redis backend would be the swap-in to scale out.

### Deploying to Render
`render.yaml` is a Blueprint: it builds with `pip install .`, starts
`uvicorn api:app --host 0.0.0.0 --port $PORT --workers 1` (one worker keeps the rate-limit counter
exact), and health-checks `/health`. Create the Blueprint from the repo, then set the two secrets
(`OPENROUTER_API_KEY` and `TAVILY_API_KEY`) in the dashboard. Everything else has a code default.

## Configuration

All secrets and model choices come from the environment (see `.env.example`); nothing is hardcoded.

| Variable | Purpose |
| --- | --- |
| `TAVILY_API_KEY` | News search |
| `OPENROUTER_API_KEY` | LLM gateway |
| `OPENROUTER_MODEL` | Fallback model for any agent left unset |
| `OPENROUTER_MODEL_DETECTOR` / `_TRANSLATOR` / `_JOURNALIST` / `_FACTCHECKER` / `_GEO` | Per-agent models |
| `ALLOWED_ORIGIN` | Comma-separated CORS origins the API accepts (default `https://miloop.ai,http://localhost:8000`) |
| `RATE_LIMIT` | Per-IP limit on `/generate` (default `5/hour`) |

## Layout

```
src/factloop/
  mcp_server.py      FastMCP server exposing search_news (topic=news, progressive 2/5/7-day window)
  tavily_client.py   vendor-isolated search call
  agents/            researcher · translator · journalist · fact_checker · geo_optimizer
  language.py        script-frequency detection with a model fallback
  graph.py           the LangGraph state machine and run_newsroom entry point
  schemas.py         Pydantic contracts shared across the pipeline
app.py               Streamlit interface (local development)
api.py               FastAPI service: POST /generate, GET /health
render.yaml          Render deployment blueprint
tests/               unit tests plus an end-to-end graph run, all stubbed
```
