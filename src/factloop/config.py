from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# A hard ceiling on the free-text keyword, enforced in code (not only in the UI)
# so a pathological input can never fan out into an unbounded run cost.
MAX_KEYWORD_CHARS = 200

_ROLE_FIELDS = {
    "detector": "detector_model",
    "translator": "translator_model",
    "journalist": "journalist_model",
    "factchecker": "factchecker_model",
    "geo": "geo_model",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    tavily_api_key: str = Field(default="", alias="TAVILY_API_KEY")
    tavily_base_url: str = Field(default="https://api.tavily.com", alias="TAVILY_BASE_URL")
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1", alias="OPENROUTER_BASE_URL"
    )
    request_timeout_seconds: float = Field(default=30.0, alias="REQUEST_TIMEOUT_SECONDS")

    # Public-service knobs, read from the environment so the deployment can retune
    # them without a code change. ALLOWED_ORIGIN is a comma-separated list so the
    # production site and a local dev server can both be permitted; it defaults to
    # Miloop AI's own site plus localhost.
    allowed_origin: str = Field(
        default="https://miloop.ai,http://localhost:8000", alias="ALLOWED_ORIGIN"
    )
    rate_limit: str = Field(default="5/hour", alias="RATE_LIMIT")

    default_model: str = Field(default="", alias="OPENROUTER_MODEL")
    # Language detection is a one-token classification on the hottest path (every
    # non-CJK query hits it), so it runs on the cheapest, fastest tier by default.
    detector_model: str = Field(
        default="google/gemini-2.5-flash-lite", alias="OPENROUTER_MODEL_DETECTOR"
    )
    translator_model: str = Field(
        default="anthropic/claude-haiku-4.5", alias="OPENROUTER_MODEL_TRANSLATOR"
    )
    journalist_model: str = Field(
        default="google/gemini-2.5-pro", alias="OPENROUTER_MODEL_JOURNALIST"
    )
    factchecker_model: str = Field(default="openai/gpt-5", alias="OPENROUTER_MODEL_FACTCHECKER")
    geo_model: str = Field(default="google/gemini-2.5-flash", alias="OPENROUTER_MODEL_GEO")

    @property
    def allowed_origins(self) -> list[str]:
        """The configured CORS origins, split from the comma-separated setting."""
        return [origin.strip() for origin in self.allowed_origin.split(",") if origin.strip()]

    def model_for(self, role: str) -> str:
        """Model slug for an agent role, falling back to the shared default."""
        return getattr(self, _ROLE_FIELDS[role]) or self.default_model


@lru_cache
def get_settings() -> Settings:
    return Settings()
