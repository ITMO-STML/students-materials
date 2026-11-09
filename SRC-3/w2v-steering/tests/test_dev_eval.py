"""CPU checks for complete development cohorts and frozen CLEAR artifacts."""

import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.dev_eval import run
from src.evaluation.api import output_directory, validate
from src.steering.external_eval import digest
from src.steering.protocol import Trial


def test_development_preserves_full_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    """All attacks and genuine recordings are kept in protocol order."""
    rows = [Trial("s0", "u0", "bonafide", 0, "-", "-", "/audio")]
    rows += [Trial(f"s{i}", f"u{i}", f"A{i:02d}", 1, "-", "-", "/audio") for i in range(9, 17)]

    def fake_trials(root: Path, split: str) -> list[Trial]:
        """Return a protocol fixture for the requested development split.

        Args:
            root: Fixture dataset root.
            split: Requested official split.

        Returns:
            Ordered fixture trials.
        """
        assert root == Path("/audio") and split == "dev"
        return rows

    monkeypatch.setattr(run, "read_trials", fake_trials)
    cfg = OmegaConf.create({"data": {"root": "/audio"}, "steering": {"expected_bonafide": 1, "expected_per_attack": 1}})
    actual = run.development_rows(cfg)
    assert [r.key for r in actual] == [r.utterance for r in rows]
    assert [r.group for r in actual] == ["bonafide", *[f"A{i:02d}" for i in range(9, 17)]]
    rows.append(rows[-1])
    with pytest.raises(ValueError, match="unexpected counts"):
        run.development_rows(cfg)


def test_full_dev_preset_is_not_parallel(tmp_path: Path) -> None:
    """The full dev backend uses the ordinary launcher lifecycle, not shard coordination."""
    with initialize_config_dir(version_base="1.3", config_dir=str(Path("configs").resolve())):
        cfg = compose(config_name="eval", overrides=["evaluation=asvspoof5_dev"])
    validate(cfg)
    assert output_directory(cfg, tmp_path) == tmp_path
    assert cfg.steering.expected_bonafide + 8 * cfg.steering.expected_per_attack == 140950
    assert cfg.steering.include_oy


def test_clear_selection_identity(tmp_path: Path) -> None:
    """Only complete, baseline-compatible projected checkpoints can enter comparison."""
    weight = tmp_path / "weights.safetensors"
    weight.write_bytes(b"hash-only fixture; never loaded")
    checksum = digest(weight)
    selected = {
        "status": "complete",
        "mode": "projected",
        "threshold": 0.8,
        "best_a21_checkpoint": str(weight),
        "guarded_checkpoint": str(weight),
        "checkpoint_sha256": {"baseline": "baseline-sha", "best_a21": checksum, "guarded": checksum},
    }
    for name, payload in (
        ("selection.json", selected),
        ("plan.json", {"fixture": True}),
        ("manifest.json", {"fixture": True}),
    ):
        (tmp_path / name).write_text(json.dumps(payload))
    cfg = OmegaConf.create({"steering": {"clear_run": str(tmp_path)}})
    paths, hashes = run.clear_artifacts(cfg, "baseline-sha", 0.8)
    assert set(paths) == set(run.CLEAR_METHODS)
    assert hashes[str(weight)] == checksum
    with pytest.raises(ValueError, match="different baselines"):
        run.clear_artifacts(cfg, "other", 0.8)
    weight.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed after selection"):
        run.clear_artifacts(cfg, "baseline-sha", 0.8)


def test_report_refresh_is_idempotent() -> None:
    """Datasource and dev columns are inserted once without changing unrelated text."""
    from src.dev_eval.reporting import replace_table

    original = (
        "prefix\n### 4.1. Wav2vec2\n\n"
        "| Method | A21 | WildSpoof 2026 |\n| --- | ---: | ---: |\n"
        "| Baseline | 9.12 | 20.48 |\n\nsuffix\n"
    )
    values = {"Baseline": {"WildSpoof / SpoofCeleb": "29.19", "Dev A09": "pending"}}
    once = replace_table(original, values)
    assert once == replace_table(once, values)
    assert once.startswith("prefix") and once.endswith("suffix\n")
    assert "9.12" in once and "29.19" in once
    assert once.count("Dev A09") == 1
