"""Lazy adapters implementing the shared (configuration, output) contract."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from pathlib import Path

    from omegaconf import DictConfig


class Evaluator(Protocol):
    """Callable contract for an evaluator or its optional preparation hook."""

    def __call__(self, cfg: DictConfig, output: Path) -> None:
        """Execute with validated configuration and an owned output directory.

        Args:
            cfg: Composed evaluation configuration.
            output: Directory selected by the common lifecycle.
        """
        ...


@dataclass(frozen=True)
class Backend:
    """Describe an evaluator without importing its model dependencies.

    Args:
        target: Module and function separated by a colon.
        prepare: Optional preparation function, run before device selection.
        lock_output: Whether this process owns the run-wide lock.
    """

    target: str
    prepare: str | None = None
    lock_output: bool = True


BACKENDS = {
    "checkpoint": Backend("src.evaluation.checkpoint:run"),
    "retention": Backend("src.retention_adaptation.evaluation:run"),
    "asvspoof5_dev": Backend("src.dev_eval.run:run"),
    "asvspoof5": Backend("src.evaluation.backends:asvspoof5"),
    "external": Backend("src.steering.external_eval:run"),
    "speaker": Backend("src.speaker_steering.evaluation:evaluate"),
    "voxenes": Backend("src.voxenes_eval.run:run", prepare="src.evaluation.backends:prepare_voxenes"),
    "mlaad": Backend("src.mlaad_eval.run:run"),
    "asvspoof5_parallel": Backend("src.evaluation.parallel:coordinate"),
    "asvspoof5_worker": Backend("src.evaluation.parallel:worker", lock_output=False),
}


def get_backend(name: str) -> Backend:
    """Look up a backend before loading models or creating output files.

    Args:
        name: Backend selected by the evaluation preset.

    Returns:
        Registered evaluator description.
    """
    if name not in BACKENDS:
        message = f"Unknown evaluation backend {name!r}; choose from {', '.join(BACKENDS)}"
        raise ValueError(message)
    return BACKENDS[name]


def load_evaluator(target: str) -> Evaluator:
    """Load one backend on demand.

    Args:
        target: Module:function reference from the registry.

    Returns:
        Callable satisfying the common evaluation contract.
    """
    module, name = target.split(":", maxsplit=1)
    function = getattr(import_module(module), name)
    if not callable(function):
        message = f"Evaluator is not callable: {target}"
        raise TypeError(message)
    return cast("Evaluator", function)
