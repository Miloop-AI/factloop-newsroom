from pydantic import BaseModel, Field


class FactItem(BaseModel):
    """A single sourced fact lifted from one news result.

    Every fact stays bound to its origin so provenance survives translation and
    rewriting, and the final article can cite real, clickable links.
    """

    claim: str
    source_title: str
    source_url: str


class FactFile(BaseModel):
    language: str
    items: list[FactItem] = Field(default_factory=list)


class TranslatedClaims(BaseModel):
    """Carrier for translated claim text, kept separate from source metadata so a
    translation can never rewrite a URL or fabricate a new source."""

    claims: list[str]


class FactCheckResult(BaseModel):
    passed: bool
    unverified_claims: list[str] = Field(default_factory=list)
    correction_notes: str = ""


class GeoArticle(BaseModel):
    title: str
    subheadings: list[str]
    body: str
    seo_keywords: list[str]
    meta_description: str
