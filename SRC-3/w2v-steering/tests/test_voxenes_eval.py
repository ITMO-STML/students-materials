"""Archive completeness, English selection and pooled/generator metric semantics."""

import argparse
import json
import stat
from pathlib import Path
from unittest.mock import Mock
from zipfile import ZipFile, ZipInfo

import numpy as np
import pytest

from src.steering.evaluation import attack_report
from src.steering.external_protocol import ExternalRecording
from src.voxenes_eval import pipeline
from src.voxenes_eval.archive import entries, finish_extraction
from src.voxenes_eval.protocol import build_plan
from src.voxenes_eval.run import metrics, save_report


def test_inventory_and_repair(tmp_path: Path) -> None:
    """Keep missing audio in plans, reject partial evaluation and repair CRC damage.

    Args:
        tmp_path: Disposable archive and extraction directory.
    """
    root = tmp_path / "voxenes-2026"
    archive_path = tmp_path / "data.zip"
    names = [
        "bonafide/english/real.flac",
        "tts/original/a/english/fixed/clip.wav",
        "tts/original/a/english/random/clip.wav",
        "vc/original/b/english/clip.wav",
        "bonafide/spanish/real.flac",
        "tts/original/a/spanish/fixed/clip.wav",
        "tts/augmented/a/english/fixed/clip.wav",
        "vc/augmented/b/english/clip.wav",
        "dataset-metadata.json",
    ]
    with ZipFile(archive_path, "w") as archive:
        for name in names:
            archive.writestr(f"{root.name}/{name}", b"original")
    rows, plan = build_plan(root, archive_path, verify_audio=False)
    assert len(rows) == 4
    assert len({row.key for row in rows}) == 4
    assert plan["groups"] == {"bonafide": 1, "tts/a": 2, "vc/b": 1}
    assert plan["missing_or_mismatched"] == 4
    with pytest.raises(ValueError, match="incomplete/corrupt"):
        build_plan(root, archive_path, verify_audio=True)
    result = finish_extraction(archive_path, root, tmp_path / "extraction.json")
    assert result["checked"] == len(names)
    assert result["repaired"] == len(names)
    valid = root / names[0]
    original_mtime = valid.stat().st_mtime_ns
    corrupt = root / names[1]
    corrupt.write_bytes(b"modified")
    with pytest.raises(ValueError, match="incomplete/corrupt"):
        build_plan(root, archive_path, verify_audio=True)
    result = finish_extraction(archive_path, root, tmp_path / "extraction.json")
    assert result["repaired"] == 1
    assert corrupt.read_bytes() == b"original"
    assert valid.stat().st_mtime_ns == original_mtime
    _, plan = build_plan(root, archive_path, verify_audio=True)
    assert plan["selected_audio_crc_verified"]
    assert plan["missing_or_mismatched"] == 0


@pytest.mark.parametrize("name", ["voxenes-2026/../escape.wav", "/absolute.wav", "wrong/clip.wav"])
def test_reject_unsafe_archive(tmp_path: Path, name: str) -> None:
    """Refuse archive traversal before any extraction.

    Args:
        tmp_path: Disposable archive directory.
        name: Unsafe archive member path.
    """
    path = tmp_path / "bad.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr(name, b"audio")
    with ZipFile(path) as archive, pytest.raises(ValueError, match="Unsafe"):
        entries(archive, tmp_path / "voxenes-2026")


def test_reject_archive_symlink(tmp_path: Path) -> None:
    """Disallow symlink members even when their names look safe.

    Args:
        tmp_path: Disposable archive directory.
    """
    path = tmp_path / "bad.zip"
    member = ZipInfo("voxenes-2026/bonafide/english/link.wav")
    member.external_attr = (stat.S_IFLNK | 0o777) << 16
    with ZipFile(path, "w") as archive:
        archive.writestr(member, "../../outside")
    with ZipFile(path) as archive, pytest.raises(ValueError, match="Unsafe"):
        entries(archive, tmp_path / "voxenes-2026")


def test_pooled_and_generator_metrics(tmp_path: Path) -> None:
    """Pool raw scores once and reuse all genuine scores for each generator.

    Args:
        tmp_path: Temporary report directory.
    """
    rows = [
        ExternalRecording(str(i), str(i), label, group)
        for i, (label, group) in enumerate(
            [(0, "bonafide"), (0, "bonafide"), (1, "tts/a"), (1, "tts/a"), (1, "vc/b")],
        )
    ]
    scores = {"baseline": np.array([0.9, 0.6, 0.2, 0.8, 0.7]), "steered": np.array([0.9, 0.6, 0.1, 0.3, 0.4])}
    result = metrics(rows, scores, 0.65)
    expected = attack_report([row.trial() for row in rows], scores["baseline"], 0.65)
    for group, values in expected.items():
        for name, value in values.items():
            assert result["baseline"][group][name] == value
    assert result["baseline"]["pooled"]["n_spoof"] == 3
    assert result["baseline"]["tts/a"]["n_bonafide"] == 2
    assert result["baseline"]["vc/b"]["n_spoof"] == 1
    assert result["baseline"]["pooled"]["classification"]["accuracy"] == pytest.approx(2 / 5)
    assert result["steered"]["pooled"]["delta_eer_pp"] < 0
    save_report(tmp_path, rows, scores, 0.65)
    assert json.loads((tmp_path / "report.json").read_text())["status"] == "complete"
    assert len((tmp_path / "comparison.csv").read_text().splitlines()) == 7
    with pytest.raises(ValueError, match="aligned"):
        metrics(rows, {"baseline": np.array([np.nan] * len(rows))}, 0.65)


def test_gpu_selection() -> None:
    """Choose maximum free VRAM without an idle/utilization requirement."""
    assert pipeline.most_free_device("0, 6000\n1, 15000\n2, 14000\n3, 48000\n") == "cuda:3"
    assert pipeline.most_free_device("2, 20000\n1, 20000\n") == "cuda:1"
    with pytest.raises(ValueError, match="No GPUs"):
        pipeline.most_free_device("")


def test_pipeline_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Do not run inference before existing unzip, repair and device selection.

    Args:
        tmp_path: Temporary run directory.
        monkeypatch: Controlled process and extraction substitutes.
    """
    calls = Mock()
    observations: list[list[int]] = [[42], []]
    calls.extractors.side_effect = observations
    calls.select_device.return_value = "cpu"
    for name in ("extractors", "finish_extraction", "select_device"):
        monkeypatch.setattr(pipeline, name, getattr(calls, name))
    monkeypatch.setattr(pipeline.time, "sleep", calls.sleep)
    monkeypatch.setattr(pipeline.subprocess, "run", calls.evaluate)
    pipeline.execute(
        argparse.Namespace(output=tmp_path, root=tmp_path / "data", archive=tmp_path / "data.zip", device="auto"),
    )
    assert [entry[0] for entry in calls.mock_calls] == [
        "extractors",
        "sleep",
        "extractors",
        "finish_extraction",
        "select_device",
        "evaluate",
    ]
    assert json.loads((tmp_path / "pipeline.json").read_text())["status"] == "complete"
