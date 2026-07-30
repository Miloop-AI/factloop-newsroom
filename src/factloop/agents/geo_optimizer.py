from .. import llm
from ..schemas import GeoArticle
from ..state import NewsroomState

_SYSTEM = (
    "You are a GEO/SEO editor. Reshape the article for generative-engine and search "
    "visibility. Return a strong title, a concise meta description, relevant keywords, "
    "and a body written in Markdown whose two or three sections each open with a '## ' "
    "subheading. List those same subheadings in the subheadings field. Keep the "
    "article's language and add no new facts."
)


def geo_optimizer_node(state: NewsroomState) -> dict:
    prompt = (
        f"Language (ISO 639-1 code): {state['language']}\n\n"
        f"Article:\n{state['article']}\n\n"
        "Return the structured, optimized article."
    )
    model = llm.build_llm("geo").with_structured_output(GeoArticle)
    return {"geo": model.invoke([("system", _SYSTEM), ("human", prompt)])}
