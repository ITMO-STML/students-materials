"""Source leakage, frozen selection controls and complete comparator coverage."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.speaker_steering import evaluation
from src.speaker_steering.adaptation import FAMILIES, additive, transform
from src.speaker_steering.evaluation import cached_scores, speaker_cohorts
from src.speaker_steering.protocol import make_plan, records
from src.steering.external_protocol import ExternalRecording
from src.steering.full_eval import save_chunk


def example_rows() -> list[ExternalRecording]:
    """Create sibling segments with two genuine target sources.

    Returns:
        Target and other-speaker audio metadata with correlated source segments.
    """
    rows = []
    for speaker, label, group, sources in (
        ("OY", 0, "bonafide", 2),
        ("OY", 1, "elevenlabs", 8),
        ("DS", 0, "bonafide", 2),
        ("DS", 1, "elevenlabs", 2),
        ("OY", 1, "finevoice", 2),
    ):
        for source in range(sources):
            parent = f"{speaker}/{group}/{source}"
            rows.extend(
                ExternalRecording(f"{parent}_{segment}", "unused", label, group, speaker=speaker, source=parent)
                for segment in range(3)
            )
    return rows


def test_parent_sources_are_disjoint_and_all_siblings_excluded() -> None:
    """Unused sibling segments cannot leak back into the held-out cohort."""
    rows = example_rows()
    plan = make_plan(rows, "OY", 42, 3, 2, 2)
    assert plan == make_plan(list(reversed(rows)), "OY", 42, 3, 2, 2)
    support, tune = records(plan["support"]), records(plan["tune"])
    assert len(support) == 5 and len(tune) == 4
    assert not {row.source for row in support} & {row.source for row in tune}
    assert len({row.source for row in support if row.label == 1}) == 3
    assert all(row.group == "elevenlabs" for row in support + tune if row.label == 1)
    masks = speaker_cohorts(rows, plan)
    heldout = [rows[i] for i in masks["source_heldout"]]
    assert not {row.source for row in heldout} & set(plan["excluded_sources"])
    assert len(heldout) == len(rows) - 7 * 3
    assert plan["counts"]["target_heldout_bonafide"] == 0
    assert plan["counts"]["target_heldout_elevenlabs"] == 9
    assert all(rows[i].speaker != "OY" for i in masks["other_speakers"])
    target = [rows[i] for i in masks["target_voice_and_heldout_bonafide"]]
    assert sum(row.label == 0 for row in target) == 6
    assert sum(row.label == 1 for row in target) == 9
    assert len(masks["inclusive_diagnostic"]) == len(rows)


@pytest.mark.parametrize("speaker", ["Trump", "Biden", "Byden", "Putin"])
def test_excluded_speakers_are_rejected(speaker: str) -> None:
    """User-excluded identities cannot enter adaptation.

    Args:
        speaker: Excluded identity with mixed capitalization.
    """
    with pytest.raises(ValueError, match="excluded"):
        make_plan(example_rows(), speaker, 42, 3, 2, 2)


def test_insufficient_independent_sources_are_rejected() -> None:
    """Require unused target spoof sources and separate genuine sources."""
    rows = example_rows()
    with pytest.raises(ValueError, match="unused"):
        make_plan(rows, "OY", 42, 4, 4, 2)
    with pytest.raises(ValueError, match="two genuine"):
        make_plan([row for row in rows if row.source != "OY/bonafide/1"], "OY", 42, 3, 2, 2)
    with pytest.raises(ValueError, match="Insufficient genuine"):
        make_plan(rows, "OY", 42, 3, 2, 4)


def test_zero_strength_and_signed_dense_intervention(tmp_path: Path) -> None:
    """Zero is an exact identity and does not load unavailable fitted artifacts.

    Args:
        tmp_path: Empty artifact directory.
    """
    hidden = torch.randn(2, 3, 4)
    vector = torch.arange(4, dtype=torch.float32)
    assert additive(hidden, vector, 0) is hidden
    torch.testing.assert_close(additive(hidden, vector, 0.5), hidden + 0.5 * vector)
    assert transform(OmegaConf.create({}), tmp_path, "speaker_sae", 6, 0) is None


def test_cached_comparators_require_complete_coverage(tmp_path: Path) -> None:
    """Do not write final comparisons against partial comparator scores.

    Args:
        tmp_path: Comparator run directory.
    """
    (tmp_path / "provenance.json").write_text(json.dumps({"identity": {"methods": ["baseline"]}}))
    chunks = tmp_path / "chunks"
    chunks.mkdir()
    save_chunk(chunks, 0, {"baseline": np.array([0.8, 0.2], dtype=np.float32)})
    with pytest.raises(ValueError, match="incomplete"):
        cached_scores(tmp_path, 3)
    save_chunk(chunks, 2, {"baseline": np.array([0.1], dtype=np.float32)})
    np.testing.assert_allclose(cached_scores(tmp_path, 3)["baseline"], [0.8, 0.2, 0.1])


def test_nonzero_controls_remain_active_when_guarded_selection_is_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Diagnostic and reversed controls must not inherit a zero fallback.

    Args:
        tmp_path: Unused fitted artifact directory.
        monkeypatch: Replace weight restoration with a recording stub.
    """
    calls: list[tuple[str, int, float]] = []

    def restore(cfg: object, root: Path, method: str, layer: int, strength: float) -> None:
        """Record transform selection without loading detector weights.

        Args:
            cfg: Unused configuration.
            root: Unused artifact directory.
            method: Actual transform implementation name.
            layer: Selected layer.
            strength: Selected positive diagnostic strength.
        """
        del cfg, root
        calls.append((method, layer, strength))

    monkeypatch.setattr(evaluation, "transform", restore)
    selected = {name: {"layer": 6, "strength": 0.0} for name in FAMILIES}
    diagnostics = {name: {"layer": 12, "strength": 0.25} for name in FAMILIES}
    jobs, aliases = evaluation.inference_jobs(
        OmegaConf.create({"steering": {"layers": [6]}}),
        tmp_path,
        {"selected": selected, "nonzero_diagnostics": diagnostics},
        None,
    )
    assert aliases == dict.fromkeys(FAMILIES, "baseline")
    assert set(jobs) == {
        "baseline",
        *(f"{name}_nonzero" for name in FAMILIES),
        "speaker_sae_random",
        "speaker_sae_reversed",
    }
    assert calls == [(name, 12, 0.25) for name in (*FAMILIES, "speaker_sae_random", "speaker_sae_reversed")]
