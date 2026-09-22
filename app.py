import random

import streamlit as st

from factloop.config import MAX_KEYWORD_CHARS
from factloop.graph import run_newsroom

# Varied so a caught hallucination reads as the system working, not an error.
_BLOCKED_MESSAGES = [
    "Fact-check stopped this one: the draft kept asserting things the sources never "
    "said, so it was held back rather than published. That guardrail just did its job. "
    "Try a different or more specific keyword.",
    "Quality control blocked publication. The writer couldn't stay within what the "
    "sources actually support, and the checker refused to let it through. Give it "
    "another angle or a sharper keyword.",
    "No verified article this time. The fact-checker flagged claims with no source "
    "backing and stopped the piece. That's the safety net working; try another keyword.",
]


def _render_article(state):
    geo = state["geo"]
    st.subheader(geo.title)
    st.caption(geo.meta_description)
    # The body already carries its subheadings as Markdown '## ' sections.
    st.markdown(geo.body)
    if geo.seo_keywords:
        st.markdown("**SEO keywords:** " + ", ".join(geo.seo_keywords))


def _render_block_reason(fact_check):
    """Say what the checker objected to, whichever way it objected.

    A blocked run does not always carry claims: a draft that came back empty has
    none, and only the correction notes explain why it was stopped.
    """
    if fact_check.unverified_claims:
        st.markdown("**Claims the fact-checker could not verify:**")
        for claim in fact_check.unverified_claims:
            st.markdown(f"- {claim}")
    elif fact_check.correction_notes:
        st.markdown("**What the fact-checker objected to:**")
        st.markdown(fact_check.correction_notes)


def _render_sources(state):
    st.markdown("#### Sources")
    days = state.get("coverage_days")
    if days:
        st.caption(f"Sources cover the last {days} days.")
    for item in state["fact_file"].items:
        st.markdown(f"- [{item.source_title}]({item.source_url})")


st.set_page_config(page_title="factloop-newsroom", page_icon="📰")
st.title("factloop-newsroom")
st.write(
    "Enter a topic in any language. The newsroom searches the last week of news, "
    "fact-checks its own writing, and returns a source-backed article in your language."
)

keyword = st.text_input(
    "Topic",
    max_chars=MAX_KEYWORD_CHARS,
    placeholder="e.g. coastal tidal energy",
    help="Be specific: a focused phrase finds better sources than a single broad term.",
)

if st.button("Run newsroom", type="primary") and keyword.strip():
    with st.spinner("Researching, writing and fact-checking…"):
        state = run_newsroom(keyword)

    if state.get("no_results"):
        st.warning("No recent news matched that topic. Try a broader or more current keyword.")
    elif state.get("qc_blocked"):
        st.info(random.choice(_BLOCKED_MESSAGES))
        _render_block_reason(state["fact_check"])
    else:
        _render_article(state)
        _render_sources(state)
