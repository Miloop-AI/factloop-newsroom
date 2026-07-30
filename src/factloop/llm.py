from .config import get_settings


def build_llm(role: str):
    """Construct the OpenRouter-backed chat model bound to an agent role.

    The LangChain client is imported lazily so that importing an agent never drags
    in the heavy LLM SDK: nodes are exercised in tests with a stub, and the real
    client is only needed when a model is actually built.

    Temperature is intentionally left at the provider default: some models routed
    through OpenRouter reject a custom temperature, and the agents rely on
    structured output rather than sampling tricks for correctness.
    """
    from langchain_openai import ChatOpenAI

    settings = get_settings()
    if not settings.openrouter_api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set; copy .env.example to .env and fill it in."
        )
    return ChatOpenAI(
        model=settings.model_for(role),
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
    )
