"""Compatibility adapters for unchanged, provenance-bound evaluation engines."""

from __future__ import annotations

import inspect
import logging
import time
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from omegaconf import DictConfig

log = logging.getLogger(__name__)


def asvspoof5(cfg: DictConfig, output: Path) -> None:
    """Call the full-protocol engine without invoking another Hydra CLI.

    Args:
        cfg: Frozen ASVspoof evaluation settings.
        output: Locked run directory.
    """
    from src.steering.full_eval import main

    cfg.steering.resume_dir = str(output)
    engine = cast("Callable[[DictConfig], None]", inspect.unwrap(main))
    engine(cfg)


def prepare_voxenes(cfg: DictConfig, output: Path) -> None:
    """Optionally verify/repair the full archive before selecting an eval device.

    Args:
        cfg: VoxENES settings; evaluation.extract explicitly enables extraction.
        output: Run-owned extraction progress destination.
    """
    from pathlib import Path

    from src.voxenes_eval.archive import finish_extraction, write_status
    from src.voxenes_eval.pipeline import extractors

    if not cfg.evaluation.extract:
        return
    archive = Path(cfg.steering.archive_path)
    while processes := extractors(archive):
        write_status(output / "extraction.json", {"status": "waiting_for_existing_unzip", "pids": processes})
        log.info("Waiting for existing unzip processes: %s", processes)
        time.sleep(30)
    finish_extraction(archive, Path(cfg.steering.external_root), output / "extraction.json")
