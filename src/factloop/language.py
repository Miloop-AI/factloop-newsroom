import unicodedata

from . import llm

# Too few characters for script frequency to mean anything, whatever the writing
# system, so short input always defers to the model.
_MIN_CONFIDENT_CHARS = 4

# The identifying script must cover at least this share of the letters before the
# statistical guess is trusted over a model call.
_MIN_CONFIDENCE = 0.5


def _script_counts(text: str) -> dict[str, int]:
    counts = {"latin": 0, "han": 0, "kana": 0, "hangul": 0}
    for char in text:
        if not char.isalpha():
            continue
        name = unicodedata.name(char, "")
        if "CJK UNIFIED" in name:
            counts["han"] += 1
        elif "HIRAGANA" in name or "KATAKANA" in name:
            counts["kana"] += 1
        elif "HANGUL" in name:
            counts["hangul"] += 1
        elif "LATIN" in name:
            counts["latin"] += 1
    return counts


def _statistical_guess(text: str) -> tuple[str | None, float]:
    """Guess a language from script frequency, with a confidence in ``[0, 1]``.

    Only a script that belongs to essentially one language carries a signal: kana
    is unique to Japanese, hangul to Korean, and Han with no kana is almost always
    Chinese. Latin (and every other script shared across many languages) yields
    no guess and zero confidence, so the caller's single rule sends it to the model
    just as it would any other low-confidence input.
    """
    counts = _script_counts(text)
    total = sum(counts.values())
    if not total:
        return None, 0.0
    if counts["kana"]:
        return "ja", counts["kana"] / total
    if counts["hangul"]:
        return "ko", counts["hangul"] / total
    if counts["han"]:
        return "zh", counts["han"] / total
    return None, 0.0


def detect_language(text: str) -> str:
    """Best-effort ISO 639-1 code for a short keyword.

    One rule for every writing system: trust the statistical guess only when the
    text is long enough and its identifying script is dominant; otherwise ask the
    model. Latin-script input never clears the confidence bar on its own, so
    Spanish, French and German reach the model instead of being mislabelled English.
    """
    text = text.strip()
    if not text:
        return "en"

    guess, confidence = _statistical_guess(text)
    if len(text) < _MIN_CONFIDENT_CHARS or confidence < _MIN_CONFIDENCE:
        return _llm_detect(text)
    return guess


def _llm_detect(text: str) -> str:
    system = "Identify the language of the text. Reply with only its ISO 639-1 code."
    response = llm.build_llm("detector").invoke([("system", system), ("human", text)])
    code = response.content.strip().lower()[:2]
    return code or "en"
