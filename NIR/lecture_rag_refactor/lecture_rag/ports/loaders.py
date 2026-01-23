from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from langchain_core.documents import Document


class DocumentLoaderPort(ABC):
    @abstractmethod
    def load_dir(self, path: str) -> List[Document]:
        """Load all supported files from a directory and return LangChain Documents."""
        raise NotImplementedError
