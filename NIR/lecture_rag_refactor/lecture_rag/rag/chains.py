from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from .prompts import enrich_prompt, lecture_prompt


def build_lecture_chain(lector_llm, retriever):
    """LCEL chain: topic -> retrieve context -> generate lecture text."""
    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | lecture_prompt
        | lector_llm
        | StrOutputParser()
    )


def build_enrich_chain(enricher_llm):
    """LCEL chain: lecture -> enriched lecture."""
    return enrich_prompt | enricher_llm | StrOutputParser()
