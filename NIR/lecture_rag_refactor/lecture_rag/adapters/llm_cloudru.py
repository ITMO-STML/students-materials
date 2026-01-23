from __future__ import annotations

from langchain_openai import ChatOpenAI


def make_lector_llm(*, api_key: str, base_url: str, model: str, temperature: float = 0.4, max_tokens: int = 2500):
    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def make_enricher_llm(*, api_key: str, base_url: str, model: str, temperature: float = 0.5, max_tokens: int = 2500):
    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )
