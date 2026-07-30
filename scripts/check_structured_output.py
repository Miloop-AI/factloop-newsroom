"""Live check that each structured-output agent parses real model responses.

Run after filling .env with a real OPENROUTER_API_KEY:

    .venv\\Scripts\\python scripts\\check_structured_output.py

It makes one live OpenRouter call per structured-output agent (Translator,
Fact-Checker, GEO Optimizer) with tiny hand-written inputs and prints the parsed
Pydantic object. The Researcher is excluded on purpose: it uses no model and no
structured output. Exits non-zero if any agent fails, so the result is unambiguous.
"""

import sys

from factloop.agents.fact_checker import fact_checker_node
from factloop.agents.geo_optimizer import geo_optimizer_node
from factloop.agents.translator import translator_node
from factloop.config import get_settings
from factloop.schemas import FactFile, FactItem

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
_ARTICLE = "The council approved a 40 megawatt tidal power array off the northern pier."

# Same facts rendered in Chinese, but with a planted detail the fact file never
# states (a named mayor and a press conference). A working cross-lingual checker
# must flag these even though the article and the facts are in different languages.
_ARTICLE_CROSS_LINGUAL = (
    "北码头外一座 40 兆瓦的潮汐能阵列已获批准,"
    "该项目由市长陈志明在周二的记者会上正式宣布。"
)


def _translator():
    return translator_node({"fact_file": _FACT_FILE, "language": "zh"})["translated_fact_file"]


def _fact_checker_same_language():
    state = {"fact_file": _FACT_FILE, "article": _ARTICLE, "revisions": 0}
    return fact_checker_node(state)["fact_check"]


def _fact_checker_cross_lingual():
    state = {"fact_file": _FACT_FILE, "article": _ARTICLE_CROSS_LINGUAL, "revisions": 0}
    result = fact_checker_node(state)["fact_check"]
    # A parse alone is not enough here: the point is that the checker sees through
    # the language gap, so treat a missed hallucination as a failure.
    if result.passed or not result.unverified_claims:
        raise AssertionError(
            "cross-lingual hallucination was not caught: "
            f"passed={result.passed}, unverified_claims={result.unverified_claims}"
        )
    return result


def _geo():
    return geo_optimizer_node({"article": _ARTICLE, "language": "en"})["geo"]


def _check(role, label, run):
    model = get_settings().model_for(role)
    print(f"\n=== {label} (model: {model}) ===")
    try:
        result = run()
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}")
        return False
    print(result.model_dump_json(indent=2))
    print("PASS")
    return True


def main():
    # Translations are non-ASCII by design; force UTF-8 so a Windows console
    # (cp1252 by default) prints CJK output instead of raising UnicodeEncodeError.
    sys.stdout.reconfigure(encoding="utf-8")

    if not get_settings().openrouter_api_key:
        print("OPENROUTER_API_KEY is not set; fill .env before running this check.")
        return 1

    checks = [
        ("translator", "Translator -> TranslatedClaims", _translator),
        (
            "factchecker",
            "Fact-Checker, same-language (EN article vs EN facts)",
            _fact_checker_same_language,
        ),
        (
            "factchecker",
            "Fact-Checker, cross-lingual (ZH article vs EN facts, planted hallucination)",
            _fact_checker_cross_lingual,
        ),
        ("geo", "GEO Optimizer -> GeoArticle", _geo),
    ]
    results = [_check(role, label, run) for role, label, run in checks]

    print(f"\n{sum(results)}/{len(results)} agents passed.")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
