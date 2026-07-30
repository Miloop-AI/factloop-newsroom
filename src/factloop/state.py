from typing import TypedDict

from .schemas import FactCheckResult, FactFile, GeoArticle


class NewsroomState(TypedDict, total=False):
    """Shared state threaded through the LangGraph pipeline.

    ``fact_file`` is the untranslated ground truth the Fact-Checker judges
    against; ``translated_fact_file`` is only what the Journalist writes from, so
    translation never enters the trust chain.
    """

    keyword: str
    language: str
    fact_file: FactFile
    coverage_days: int
    translated_fact_file: FactFile
    article: str
    fact_check: FactCheckResult
    revisions: int
    geo: GeoArticle
    no_results: bool
    qc_blocked: bool
