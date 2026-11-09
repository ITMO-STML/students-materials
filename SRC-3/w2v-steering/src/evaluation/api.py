"""Common configuration, locking and runtime lifecycle for detector evaluation."""

import fcntl
import hashlib
import json
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from .registry import get_backend, load_evaluator


@dataclass(frozen=True)
class EvaluationResult:
    """Describe the completed invocation, including preparation-only requests.

    Args:
        backend: Selected backend identifier.
        output_dir: Directory containing backend artifacts.
        status: Complete, prepared or smoke_passed.
    """

    backend: str
    output_dir: Path
    status: str


def output_directory(cfg: DictConfig, output: Path | None) -> Path:
    """Resolve explicit output, resume and Hydra destinations without ambiguity.

    Args:
        cfg: Evaluation configuration.
        output: Optional Python API destination.

    Returns:
        Absolute run directory.
    """
    choices = [output, cfg.evaluation.output_dir, OmegaConf.select(cfg, "steering.resume_dir")]
    if str(cfg.evaluation.backend) in {"asvspoof5_parallel", "asvspoof5_worker"}:
        choices.append(cfg.evaluation.run_directory)
    explicit = [Path(value).resolve() for value in choices if value is not None]
    if len(set(explicit)) > 1:
        message = "Conflicting output/resume directories; supply one destination or matching paths"
        raise ValueError(message)
    return explicit[0] if explicit else Path(cfg.paths.output_dir).resolve()


def validate(cfg: DictConfig) -> None:
    """Reject accidental fitting and unsupported options before any execution.

    Args:
        cfg: Composed config, including an evaluation preset.
    """
    name = str(cfg.evaluation.backend)
    get_backend(name)
    if int(cfg.evaluation.cpu_threads) < 1:
        message = "evaluation.cpu_threads must be positive"
        raise ValueError(message)
    if name.startswith("asvspoof5") and int(cfg.evaluation.cpu_threads) != 4:
        message = "Existing ASVspoof engines require evaluation.cpu_threads=4"
        raise ValueError(message)
    if name == "speaker" and cfg.steering.stage != "evaluate":
        message = "eval.py only evaluates frozen speaker adapters; stage must be evaluate"
        raise ValueError(message)
    if name in {"asvspoof5", "checkpoint"} and OmegaConf.select(cfg, "steering.prepare_only", default=False):
        message = f"prepare_only is unsupported by {name}"
        raise ValueError(message)
    if name == "checkpoint" and cfg.evaluation.device not in (None, "cpu"):
        message = "Trainer checkpoint evaluation uses training_arguments; evaluation.device supports only cpu here"
        raise ValueError(message)
    if name in {"asvspoof5_parallel", "asvspoof5_worker"}:
        directory = cfg.evaluation.run_directory
        if not directory or not (Path(directory) / "provenance.json").is_file():
            message = "Parallel evaluation requires evaluation.run_directory with existing provenance.json"
            raise ValueError(message)
        if name == "asvspoof5_parallel":
            devices = list(cfg.evaluation.devices)
            if len(devices) < 2 or len(set(devices)) != len(devices):
                message = "Parallel evaluation requires at least two distinct evaluation.devices"
                raise ValueError(message)


def configure_runtime(cfg: DictConfig, output: Path) -> None:
    """Seed the evaluator, set CPU threads and resolve optional automatic device.

    Args:
        cfg: Private configuration copy for this invocation.
        output: Run directory for device-selection metadata.
    """
    import torch
    from transformers import set_seed

    set_seed(int(cfg.seed))
    torch.set_num_threads(int(cfg.evaluation.cpu_threads))
    requested = cfg.evaluation.device
    if requested is None:
        return
    if cfg.evaluation.backend == "checkpoint":
        OmegaConf.update(cfg, "training_arguments.use_cpu", True, force_add=True)
        return
    if requested == "auto":
        from src.voxenes_eval.pipeline import select_device

        requested = select_device("auto", output / "device.json")
    cfg.steering.device = str(requested)


def write_launch(output: Path, cfg: DictConfig, status: str) -> None:
    """Record launcher provenance separately from immutable scoring provenance.

    Args:
        output: Run directory owned by this invocation.
        cfg: Effective configuration including evaluation controls.
        status: Current lifecycle status; errors still propagate to the caller.
    """
    sources = [Path("eval.py"), Path("src/eval.py"), *sorted(Path("src/evaluation").glob("*.py"))]
    record = {
        "status": status,
        "updated_at_utc": datetime.now(UTC).isoformat(),
        "backend": str(cfg.evaluation.backend),
        "output_dir": str(output),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "launcher_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
    }
    target = output / "evaluation_launch.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(target)


def evaluate(cfg: DictConfig, output: Path | None = None) -> EvaluationResult:
    """Execute a configured wav2vec2-AASIST evaluator through one public API.

    Args:
        cfg: Config composed from configs/eval.yaml and evaluation=<preset>.
        output: Optional destination for callers without Hydra runtime state.

    Returns:
        Invocation outcome and artifact directory; backend errors propagate.
    """
    validate(cfg)
    backend = get_backend(str(cfg.evaluation.backend))
    directory = output_directory(cfg, output)
    # Resolve runtime paths for Python API use, without mutating the caller's config.
    copied = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    assert isinstance(copied, DictConfig)
    copied.paths.output_dir = str(directory)
    copied.paths.work_dir = str(Path.cwd())
    copied.training_arguments.output_dir = str(directory)
    if copied.evaluation.backend != "checkpoint":
        copied.steering.resume_dir = str(directory)
    directory.mkdir(parents=True, exist_ok=True)
    context = (directory / ".lock").open("a") if backend.lock_output else nullcontext()
    with context as lock:
        if lock is not None:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = "failed_or_interrupted"
        try:
            if backend.lock_output:
                write_launch(directory, copied, "initializing")
            if backend.prepare is not None:
                load_evaluator(backend.prepare)(copied, directory)
            configure_runtime(copied, directory)
            if backend.lock_output:
                write_launch(directory, copied, "running")
            load_evaluator(backend.target)(copied, directory)
            status = "complete"
            if OmegaConf.select(copied, "steering.prepare_only", default=False):
                status = "prepared"
            elif OmegaConf.select(copied, "steering.smoke_only", default=False):
                status = "smoke_passed"
        finally:
            if backend.lock_output:
                write_launch(directory, copied, status)
    return EvaluationResult(str(copied.evaluation.backend), directory, status)
