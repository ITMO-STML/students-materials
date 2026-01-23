from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter


def make_splitter(chunk_size: int = 800, chunk_overlap: int = 100) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
