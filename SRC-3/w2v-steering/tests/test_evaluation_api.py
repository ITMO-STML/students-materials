"""Shared entrypoint composition, dispatch, locking and artifact compatibility."""

from __future__ import annotations

import fcntl
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from src.evaluation import api
from src.evaluation.parallel import worker_command
from src.evaluation.registry import BACKENDS, Backend, get_backend

ROOT = Path(__file__).resolve().parents[1]
PRESETS = {
    "checkpoint": "checkpoint",
    "asvspoof5": "asvspoof5",
    "elevenlabs": "external",
    "wildspoof": "external",
    "voxenes": "voxenes",
    "mlaad": "mlaad",
    "speaker_asvspoof5": "speaker",
    "speaker_elevenlabs": "speaker",
    "speaker_wildspoof": "speaker",
    "asvspoof5_parallel": "asvspoof5_parallel",
    "asvspoof5_worker": "asvspoof5_worker",
}


def configuration(preset: str, overrides: list[str] | None = None) -> DictConfig:
    """Compose a real preset without requiring a Hydra runtime.

    Args:
        preset: Public evaluation config name.
        overrides: Additional test-specific overrides.

    Returns:
        Composed configuration with lazy runtime path interpolations.
    """
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        return compose(config_name="eval", overrides=[f"evaluation={preset}", *(overrides or [])])


@pytest.mark.parametrize("preset", PRESETS)
def test_presets_compose(preset: str) -> None:
    """Every advertised evaluator is reachable without importing its engine.

    Args:
        preset: Public configuration preset.
    """
    cfg = configuration(preset)
    assert cfg.evaluation.backend == PRESETS[preset]
    assert get_backend(str(cfg.evaluation.backend)).target
    if preset.startswith("speaker_"):
        assert cfg.steering.stage == "evaluate"
    if preset == "mlaad":
        assert not cfg.steering.include_oy
        assert len(cfg.steering.methods) == 9


@pytest.mark.parametrize(
    "preset,steering",
    [
        ("asvspoof5", "full_eval"),
        ("elevenlabs", "external_eval"),
        ("voxenes", "voxenes"),
        ("mlaad", "mlaad"),
    ],
)
def test_frozen_settings_unchanged(preset: str, steering: str) -> None:
    """New launch configs preserve existing scoring and resume identity inputs.

    Args:
        preset: Unified preset.
        steering: Original experiment configuration.
    """
    actual = configuration(preset)
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        original = compose(config_name="main", overrides=[f"steering={steering}"])
    for key in ("seed", "steering", "model", "data"):
        assert actual[key] == original[key]


def test_api_calls_one_backend_and_preserves_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Expose one callable API with preparation, device setup and owned output.

    Args:
        tmp_path: Run directory.
        monkeypatch: Isolated backends and runtime initialization.
    """
    cfg = configuration("mlaad", ["steering.prepare_only=true"])
    before = OmegaConf.to_container(cfg, resolve=False)
    calls = Mock()
    monkeypatch.setitem(BACKENDS, "mlaad", Backend("fake:run", prepare="fake:prepare"))
    dispatch = Mock(side_effect=[calls.prepare, calls.run])
    monkeypatch.setattr(api, "load_evaluator", dispatch)
    monkeypatch.setattr(api, "configure_runtime", calls.runtime)
    result = api.evaluate(cfg, tmp_path)
    assert [row[0] for row in calls.mock_calls] == ["prepare", "runtime", "run"]
    effective, output = calls.run.call_args.args
    assert output == tmp_path
    assert effective.steering.resume_dir == str(tmp_path)
    assert result.status == "prepared"
    assert json.loads((tmp_path / "evaluation_launch.json").read_text())["status"] == "prepared"
    assert OmegaConf.to_container(cfg, resolve=False) == before


def test_backend_errors_are_not_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Propagate scoring failures and record failure instead of completion.

    Args:
        tmp_path: Run directory.
        monkeypatch: Failing test backend.
    """
    error = Mock(side_effect=ValueError("bad audio"))
    monkeypatch.setattr(api, "load_evaluator", Mock(return_value=error))
    monkeypatch.setattr(api, "configure_runtime", Mock())
    with pytest.raises(ValueError, match="bad audio"):
        api.evaluate(configuration("mlaad"), tmp_path)
    assert json.loads((tmp_path / "evaluation_launch.json").read_text())["status"] == "failed_or_interrupted"


def test_active_run_lock_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the same lock as old evaluators, protecting active jobs.

    Args:
        tmp_path: Existing locked run.
        monkeypatch: Backend import spy.
    """
    loader = Mock()
    monkeypatch.setattr(api, "load_evaluator", loader)
    with (tmp_path / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            api.evaluate(configuration("mlaad"), tmp_path)
    loader.assert_not_called()
    assert not (tmp_path / "evaluation_launch.json").exists()


def test_invalid_configuration_fails_early(tmp_path: Path) -> None:
    """Reject unknown engines, contradictory output paths and accidental fit.

    Args:
        tmp_path: Would-be output root.
    """
    cfg = configuration("speaker_elevenlabs", ["steering.stage=fit"])
    with pytest.raises(ValueError, match="only evaluates"):
        api.evaluate(cfg, tmp_path / "unused")
    cfg = configuration("mlaad", [f"steering.resume_dir={tmp_path / 'other'}"])
    with pytest.raises(ValueError, match="Conflicting"):
        api.evaluate(cfg, tmp_path / "unused")
    cfg.evaluation.backend = "typo"
    with pytest.raises(ValueError, match="Unknown evaluation"):
        api.evaluate(cfg, tmp_path / "unused")
    assert not (tmp_path / "unused").exists()


def test_worker_has_no_coordinator_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Allow a configured worker under its coordinator's existing run lock.

    Args:
        tmp_path: Original run containing provenance.
        monkeypatch: Worker implementation stub.
    """
    (tmp_path / "provenance.json").write_text("{}")
    cfg = configuration("asvspoof5_worker", [f"evaluation.run_directory={tmp_path}"])
    worker = Mock()
    monkeypatch.setattr(api, "load_evaluator", Mock(return_value=worker))
    monkeypatch.setattr(api, "configure_runtime", Mock())
    with (tmp_path / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        api.evaluate(cfg)
    worker.assert_called_once()
    assert not (tmp_path / "evaluation_launch.json").exists()
    command = worker_command(tmp_path, 2)
    assert command[:4] == [sys.executable, "-m", "src.eval", "evaluation=asvspoof5_worker"]
    assert "evaluation.worker_index=2" in command


@pytest.mark.parametrize("entrypoint", [["eval.py"], ["-m", "src.eval"]])
def test_both_cli_forms_are_lazy(entrypoint: list[str]) -> None:
    """Resolve an evaluation config without loading checkpoints or GPU models.

    Args:
        entrypoint: Root script or module CLI syntax.
    """
    response = subprocess.run(  # noqa: S603 - fixed Python command and repository root
        [sys.executable, *entrypoint, "evaluation=mlaad", "--cfg", "job"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    cfg = OmegaConf.create(response.stdout)
    assert cfg.evaluation.backend == "mlaad"
    assert not cfg.steering.include_oy


@pytest.mark.parametrize("launch_failure", [False, True])
def test_parallel_coordinator_uses_common_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    launch_failure: bool,
) -> None:
    """Keep frozen scoring config while launching children through the new CLI.

    Args:
        tmp_path: Original run directory.
        monkeypatch: Replace model/worker execution with deterministic stubs.
        launch_failure: Whether the second child fails to start.
    """
    from src.evaluation import parallel
    from src.steering import full_eval, parallel_eval, protocol

    cfg = configuration("asvspoof5_parallel", ["evaluation.devices=[cuda:0,cuda:1]"])
    original = OmegaConf.create({"steering": {"methods": ["baseline"]}, "data": {"root": str(tmp_path)}})
    monkeypatch.setattr(parallel_eval, "original_config", Mock(return_value=(original, {"n_recordings": 5})))
    monkeypatch.setattr(parallel_eval, "make_plan", Mock(return_value={"total": 5}))
    monkeypatch.setattr(parallel_eval, "monitor", Mock())
    monkeypatch.setattr(full_eval, "load_chunks", Mock(side_effect=[0, 5]))
    report = Mock()
    monkeypatch.setattr(full_eval, "final_report", report)
    monkeypatch.setattr(protocol, "read_trials", Mock(return_value=[]))
    child = Mock()
    child.poll.return_value = None if launch_failure else 0
    spawn = Mock(side_effect=[child, OSError("cannot spawn")] if launch_failure else [child, child])
    monkeypatch.setattr(parallel.subprocess, "Popen", spawn)
    if launch_failure:
        with pytest.raises(OSError, match="cannot spawn"):
            parallel.coordinate(cfg, tmp_path)
        child.send_signal.assert_called_once()
        child.wait.assert_called_once()
        report.assert_not_called()
        return
    parallel.coordinate(cfg, tmp_path)
    assert spawn.call_count == 2
    for index, call in enumerate(spawn.call_args_list):
        assert call.args[0] == worker_command(tmp_path, index)
    assert report.call_args.args[2] is original
    assert report.call_args.args[3] == tmp_path


def test_checkpoint_backend_honors_api_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep baseline prediction behavior with a caller-owned output and no Hydra runtime.

    Args:
        tmp_path: Explicit API destination.
        monkeypatch: Lightweight checkpoint/model/data/trainer substitutes.
    """
    from src import trainer
    from src.evaluation import checkpoint

    cfg = configuration("checkpoint", ["ckpt_path=/fake/checkpoint"])
    cfg.paths.output_dir = str(tmp_path)
    cfg.training_arguments.output_dir = str(tmp_path)
    model, data = Mock(), Mock()
    monkeypatch.setattr(checkpoint.hydra.utils, "instantiate", Mock(side_effect=[model, data]))
    loaded = Mock()
    monkeypatch.setattr(checkpoint, "load_checkpoint", loaded)
    arguments = Mock()
    monkeypatch.setattr(checkpoint, "TrainingArguments", arguments)
    runner = Mock()
    monkeypatch.setattr(trainer, "build_trainer", Mock(return_value=runner))
    monkeypatch.setenv("TRACKIO_SUBDIR", "test_tracking")
    monkeypatch.setenv("TRACKIO_DIR", "initial")
    checkpoint.run(cfg, tmp_path)
    loaded.assert_called_once_with(model, "/fake/checkpoint")
    runner.predict.assert_called_once_with(data.test_dataset)
    assert arguments.call_args.kwargs["output_dir"] == str(tmp_path)
    assert (tmp_path / "test_tracking").is_dir()
