"""CPU checks for retention gradients, leakage, fitting and evaluation resumes."""

from collections import Counter
from pathlib import Path
from typing import override

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn

from src.retention_adaptation.evaluation import score
from src.retention_adaptation.gradients import combine_gradients, dot, retention_loss
from src.retention_adaptation.protocol import replay_pool
from src.retention_adaptation.training import gradients, predict_logits, train_epoch
from src.steering.protocol import Trial, check_disjoint


class Detector(nn.Module):
    """Tiny CPU detector with dropout and mutable normalization buffers."""

    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(2)
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(2, 2)

    @override
    def forward(self, input_values: Tensor, attention_mask: Tensor) -> dict[str, Tensor]:
        """Compute production-shaped logits.

        Args:
            input_values: Two input features.
            attention_mask: Valid input mask.

        Returns:
            Dictionary containing binary logits.
        """
        return {"logits": self.head(self.dropout(self.norm(input_values * attention_mask)))}


@pytest.mark.parametrize("mode,expected", [("target", [-2.0, 3.0]), ("replay", [-1.0, 3.0]), ("projected", [1.0, 3.0])])
def test_conflicting_gradient_modes(mode: str, expected: list[float]) -> None:
    """Projection removes only the conflicting target component before replay."""
    target, retain = [torch.tensor([-2.0, 3.0])], [torch.tensor([1.0, 0.0])]
    result, diagnostics = combine_gradients(target, retain, mode, 1.0)
    torch.testing.assert_close(result[0], torch.tensor(expected))
    assert diagnostics["conflict"] == 1
    torch.testing.assert_close(target[0], torch.tensor([-2.0, 3.0]))


@pytest.mark.parametrize("retain", [[0.0, 0.0], [2.0, 1.0], [-3.0, -2.0]])
def test_projection_first_order_constraint(retain: list[float]) -> None:
    """The actual projected SGD direction cannot increase sampled retain loss to first order."""
    target, reference = [torch.tensor([1.0, 2.0])], [torch.tensor(retain)]
    direction, _ = combine_gradients(target, reference, "projected", 0.0)
    assert float(dot(direction, reference)) >= -1e-6


def test_global_projection_and_nonfinite_rejection() -> None:
    """Projection uses one global coefficient across parameter blocks."""
    target = [torch.tensor([-2.0]), torch.tensor([3.0])]
    retain = [torch.tensor([1.0]), torch.tensor([2.0])]
    result, _ = combine_gradients(target, retain, "projected", 0.0)
    assert all(torch.equal(a, b) for a, b in zip(target, result, strict=True))
    with pytest.raises(ValueError, match="Nonfinite"):
        combine_gradients([torch.tensor([float("nan")])], [torch.ones(1)], "projected", 1.0)


def test_teacher_is_detached_and_kl_direction() -> None:
    """Forward KL matches its direct formula and does not update teacher logits."""
    logits = torch.tensor([[2.0, -1.0]], requires_grad=True)
    teacher = torch.tensor([[-1.0, 1.0]], requires_grad=True)
    labels = torch.tensor([1])
    loss = retention_loss(logits, labels, teacher, 0.7, 2.0)
    p = (teacher.detach() / 2).softmax(-1)
    expected = (
        nn.functional.cross_entropy(logits, labels) + 0.7 * 4 * (p * (p.log() - (logits / 2).log_softmax(-1))).sum()
    )
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert teacher.grad is None
    assert logits.grad is not None


@pytest.mark.parametrize("use_teacher", [False, True])
def test_accumulation_and_frozen_buffers(use_teacher: bool) -> None:
    """Partial microbatches preserve exact gradients, with unchanged BN buffers."""
    torch.manual_seed(4)
    torch.set_num_threads(2)
    model = Detector()
    waves, labels = torch.randn(7, 2), torch.tensor([0, 1, 1, 0, 1, 0, 1])
    buffers = {name: tensor.clone() for name, tensor in model.named_buffers()}
    teacher = torch.randn(7, 2) if use_teacher else None
    large, loss_large = gradients(model, waves, labels, 7, teacher)
    small, loss_small = gradients(model, waves, labels, 2, teacher)
    assert loss_small == pytest.approx(loss_large, abs=1e-6)
    for first, second in zip(large, small, strict=True):
        torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-5)
    for name, tensor in model.named_buffers():
        torch.testing.assert_close(tensor, buffers[name])
    assert model.training is False


@pytest.mark.parametrize("mode", ["target", "replay", "projected"])
def test_cpu_fitting_smoke(mode: str) -> None:
    """Bounded synthetic fitting reduces target loss without modifying teacher state."""
    torch.manual_seed(7)
    model = Detector()
    waves = torch.tensor([[-2.0, -1.0], [2.0, 1.0]]).repeat(8, 1)
    labels = torch.tensor([0, 1]).repeat(8)
    teacher = predict_logits(model, waves, 4)
    cache = teacher.clone()
    before = nn.functional.cross_entropy(teacher, labels)
    for epoch in range(5):
        history = train_epoch(
            model,
            waves,
            labels,
            waves,
            labels,
            teacher,
            mode=mode,
            learning_rate=0.1,
            replay_weight=1.0,
            beta=0.2,
            temperature=2.0,
            batch_size=2,
            effective_batch_size=8,
            seed=epoch,
            max_updates=2,
        )
        assert len(history) == 2
    after = nn.functional.cross_entropy(predict_logits(model, waves, 4), labels)
    assert after < before
    torch.testing.assert_close(teacher, cache)


def test_replay_balance_and_leakage() -> None:
    """Replay balances classes/generators, excludes A21, and rejects shared speakers."""
    rows = [
        Trial(f"s-{attack}-{i}", f"u-{attack}-{i}", attack, int(attack != "bonafide"), "-", "-", "/unused/audio")
        for attack in ("bonafide", "A01", "A02", "A21")
        for i in range(8)
    ]
    selected = replay_pool(rows, 2, 42)
    assert Counter(row.attack for row in selected) == {"bonafide": 4, "A01": 2, "A02": 2}
    assert replay_pool(list(reversed(rows)), 2, 42) == selected
    with pytest.raises(ValueError, match="Leakage"):
        check_disjoint({"retain": selected, "tune": [selected[0]]})


def test_hydra_presets() -> None:
    """Preparation is default and frozen evaluation uses the common launcher."""
    with initialize_config_dir(version_base="1.3", config_dir=str(Path("configs").resolve())):
        training = compose(config_name="main", overrides=["steering=retention_a21"])
        evaluation = compose(
            config_name="eval",
            overrides=["evaluation=retention_a21", "evaluation.adaptation_run=/tmp/fitted"],
        )
    assert training.steering.stage == "prepare"
    assert evaluation.evaluation.backend == "retention"
    assert evaluation.evaluation.selection == "guarded"


def test_resume_rejects_noncontiguous_scores(tmp_path: Path) -> None:
    """Scoring refuses corrupted resume chunks before attempting audio decoding."""
    directory = tmp_path / "chunks/baseline"
    directory.mkdir(parents=True)
    np.savez(directory / "000000001.npz", indices=np.array([1]), scores=np.array([0.5]))
    cfg = OmegaConf.create({"data": {"num_samples": 2}, "steering": {"batch_size": 2}})
    with pytest.raises(ValueError, match="Noncontiguous"):
        score(Detector(), [], cfg, tmp_path, "baseline")


def test_frozen_checkpoint_evaluation_and_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Frozen checkpoint scoring, report generation and complete resume agree on CPU."""
    import json
    from dataclasses import asdict

    from safetensors.torch import save_file

    from src.retention_adaptation import evaluation
    from src.retention_adaptation.protocol import digest
    from src.steering.external_protocol import ExternalRecording

    torch.manual_seed(9)
    model = Detector().eval()
    fitting = tmp_path / "fit"
    output = tmp_path / "evaluation"
    fitting.mkdir()
    output.mkdir()
    baseline, adapted = fitting / "baseline.safetensors", fitting / "adapted.safetensors"
    save_file(model.state_dict(), baseline)
    with torch.no_grad():
        model.head.bias.add_(torch.tensor([0.1, -0.1]))
    save_file(model.state_dict(), adapted)
    trials = [
        Trial(f"s{i}", f"u{i}", "bonafide" if i < 3 else "A21", int(i >= 3), "-", "-", str(tmp_path)) for i in range(6)
    ]
    for trial in trials:
        (tmp_path / f"{trial.utterance}.flac").write_bytes(b"metadata-only fixture")
    (fitting / "manifest.json").write_text(json.dumps({"test": [asdict(t) for t in trials]}))
    cfg = OmegaConf.create(
        {
            "data": {"num_samples": 2},
            "steering": {"device": "cpu", "batch_size": 2, "prepare_only": False},
            "evaluation": {"adaptation_run": str(fitting), "selection": "guarded", "cohort": "test"},
        },
    )
    (fitting / "plan.json").write_text(json.dumps({"config": OmegaConf.to_container(cfg)}))
    (fitting / "selection.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "threshold": 0.5,
                "baseline_checkpoint": str(baseline),
                "guarded_checkpoint": str(adapted),
                "checkpoint_sha256": {"baseline": digest(baseline), "guarded": digest(adapted)},
            },
        ),
    )

    def fake_audio(rows: list[ExternalRecording], num_samples: int, start: int = 0) -> list[tuple[int, Tensor]]:
        """Supply deterministic CPU features in place of audio decoding.

        Args:
            rows: Evaluation inventory.
            num_samples: Expected fixture feature count.
            start: Resume offset.

        Returns:
            Indexed CPU feature vectors.
        """
        assert num_samples == 2
        return [(i, torch.tensor([float(i - 2), 1.0])) for i in range(start, len(rows))]

    def fake_model(settings: DictConfig) -> Detector:
        """Return the CPU fixture detector without loading real model weights.

        Args:
            settings: Effective detector configuration.

        Returns:
            Small detector used by the integration test.
        """
        assert settings.steering.device == "cpu"
        return model

    monkeypatch.setattr(evaluation, "build_model", fake_model)
    monkeypatch.setattr(evaluation, "ExternalAudio", fake_audio)
    evaluation.run(cfg, output)
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "complete" and report["n"] == 6
    assert set(report["results"]) == {"baseline", "adapted"}
    paths = sorted((output / "chunks").rglob("*.npz"))
    assert len(paths) == 6
    saved = {str(path): path.read_bytes() for path in paths}
    evaluation.run(cfg, output)
    assert json.loads((output / "report.json").read_text()) == report
    assert {str(path): path.read_bytes() for path in paths} == saved
    cfg.steering.batch_size = 3
    with pytest.raises(ValueError, match="identity changed"):
        evaluation.run(cfg, output)
