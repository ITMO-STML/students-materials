import abc
from typing import Iterable
from model.model_abc import ModelABC
from pathlib import Path

class BenchmarkABC(abc.ABC):
    @abc.abstractmethod
    def eval(self, data: Iterable, model: ModelABC, metric_path: str | Path) -> None:
        """Eval model with dataset. Save metrics on metric_path"""
        pass