import re

from .. import llm
from ..schemas import FactFile, FactItem, TranslatedClaims
from ..state import NewsroomState

# Languages whose script is non-Latin, where transliterating a name loses its
# original spelling — so pairing the transliteration with the source name helps
# the reader. Latin-script languages keep names verbatim and need no annotation.
_NON_LATIN_LANGUAGES = frozenset(
    {"zh", "ja", "ko", "ar", "ru", "he", "el", "th", "hi", "fa", "uk"}
)

_SYSTEM = (
    "You translate newsroom facts. Render each fact in the target language, keeping "
    "names, numbers, dates and quotes exact. Do not add, drop, merge or reorder facts."
)

_SYSTEM_ANNOTATED = (
    _SYSTEM + " The target language uses a non-Latin script. When you transliterate a "
    "person or organization name, append the original Latin-script name in parentheses "
    "right after it, e.g. 威善高(Scott Wiener). Only put a name in parentheses when that "
    "exact Latin name appears in the source fact; never invent or guess one."
)

# A parenthetical (half- or full-width) whose content is a Latin-script name: the
# annotation appended above. Parentheses holding CJK, numbers or other text are
# left untouched, so only the appended originals are ever verified or removed.
_LATIN_ANNOTATION = re.compile(r"[（(]\s*([A-Za-z][A-Za-z.\-'’\s]*?)\s*[）)]")


def _is_non_latin(language: str) -> bool:
    return language.lower() in _NON_LATIN_LANGUAGES


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _verify_annotations(claim: str, source_text: str) -> str:
    """Drop any appended Latin name that is not grounded in the source text.

    The model is asked to annotate transliterated names with their originals, but a
    model can hallucinate a plausible name, so this is enforced in code rather than
    trusted: each parenthesized Latin name must appear (loosely — case- and
    whitespace-insensitively) in the untranslated fact file, or the parenthetical is
    stripped entirely. An unverified attribution is worse than none.
    """

    def keep_or_drop(match: re.Match) -> str:
        if _normalize(match.group(1)) in source_text:
            return match.group(0)
        return ""

    cleaned = _LATIN_ANNOTATION.sub(keep_or_drop, claim)
    return re.sub(r"\s{2,}", " ", cleaned).strip()


def _prompt(claims: list[str], language: str) -> str:
    numbered = "\n".join(f"{index}. {claim}" for index, claim in enumerate(claims))
    return (
        f"Target language (ISO 639-1 code): {language}\n\n"
        f"Translate these {len(claims)} facts and return them in the same order:\n\n{numbered}"
    )


def translator_node(state: NewsroomState) -> dict:
    fact_file = state["fact_file"]
    language = state["language"]
    claims = [item.claim for item in fact_file.items]
    annotate = _is_non_latin(language)

    model = llm.build_llm("translator").with_structured_output(TranslatedClaims)
    system = _SYSTEM_ANNOTATED if annotate else _SYSTEM
    result = model.invoke([("system", system), ("human", _prompt(claims, language))])

    # If the model returns the wrong count we keep the original claim for that slot
    # rather than risk pairing a translation with the wrong source.
    translated = result.claims if len(result.claims) == len(claims) else claims

    if annotate:
        source_text = _normalize(" ".join(item.claim for item in fact_file.items))
        translated = [_verify_annotations(claim, source_text) for claim in translated]

    items = [
        FactItem(claim=claim, source_title=item.source_title, source_url=item.source_url)
        for claim, item in zip(translated, fact_file.items, strict=True)
    ]
    return {"translated_fact_file": FactFile(language=language, items=items)}
