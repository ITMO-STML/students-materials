import abc
from pathlib import Path
from typing import Any

class ModelABC(abc.ABC):
    @abc.abstractmethod
    def run(self, file: str | Path, question: dict | str) -> Any:
        """Accepts file path and question as an input, run the model and return the response"""
        pass


    @abc.abstractmethod
    def name(self) -> str:
        """Returns the name of the model"""
        pass
