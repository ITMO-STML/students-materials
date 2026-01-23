from __future__ import annotations

import os
from typing import Iterable, List

from langchain_core.documents import Document

# PDF
from langchain_community.document_loaders import PyPDFLoader, TextLoader


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".md",
    ".docx",
    ".pptx",
}


class LangChainMultiFormatLoader:
    """Loads a directory of files into LangChain Documents.

    Notes:
      - PDF support works out of the box (PyPDFLoader).
      - DOCX/PPTX use Unstructured loaders (requires `unstructured`).
    """

    def __init__(self, recursive: bool = True):
        self.recursive = recursive

    def load_dir(self, path: str) -> List[Document]:
        docs: List[Document] = []
        for file_path in self._iter_files(path):
            ext = os.path.splitext(file_path)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue
            docs.extend(self._load_file(file_path, ext))
        return docs

    def _iter_files(self, path: str) -> Iterable[str]:
        if os.path.isfile(path):
            yield path
            return

        for root, _, files in os.walk(path):
            for name in files:
                yield os.path.join(root, name)
            if not self.recursive:
                break

    def _load_file(self, file_path: str, ext: str) -> List[Document]:
        if ext == ".pdf":
            return PyPDFLoader(file_path).load()

        if ext in {".txt", ".md"}:
            return TextLoader(file_path, encoding="utf-8").load()

        if ext == ".docx":
            try:
                from langchain_community.document_loaders import UnstructuredWordDocumentLoader
            except Exception as e:
                raise RuntimeError(
                    "DOCX loading requires 'unstructured'. Install: pip install unstructured"
                ) from e
            return UnstructuredWordDocumentLoader(file_path).load()

        if ext == ".pptx":
            try:
                from langchain_community.document_loaders import UnstructuredPowerPointLoader
            except Exception as e:
                raise RuntimeError(
                    "PPTX loading requires 'unstructured'. Install: pip install unstructured"
                ) from e
            return UnstructuredPowerPointLoader(file_path).load()

        return []
