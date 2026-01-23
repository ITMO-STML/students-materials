from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from .splitters import make_splitter


def build_retriever(
    documents: List[Document],
    *,
    api_key: str,
    base_url: str,
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    top_k: int = 5,
):
    """Build a FAISS retriever from raw documents."""
    splitter = make_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        base_url=base_url,
        api_key=api_key,
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": top_k})
