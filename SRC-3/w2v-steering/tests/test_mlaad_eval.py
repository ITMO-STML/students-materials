"""Validate metadata completeness, English selection and genuine deduplication."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from src.mlaad_eval.protocol import build_plan, safe_path
from src.mlaad_eval.run import save_report

FIELDS = ["path", "original_file", "language", "is_original_language", "model_name"]


def create_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Create two English generators sharing two original recordings.

    Args:
        tmp_path: Disposable root.

    Returns:
        Synthetic and genuine dataset roots.
    """
    fake, real = tmp_path / "MLAAD", tmp_path / "M_AILABS"
    for generator in ("one", "two"):
        directory = fake / "fake" / "en" / generator
        directory.mkdir(parents=True)
        with (directory / "meta.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="|")
            writer.writeheader()
            for index in range(2):
                source = f"en_US/by_book/female/speaker/book/wavs/{index}.wav"
                original = real / source
                original.parent.mkdir(parents=True, exist_ok=True)
                original.write_bytes(b"audio")
                (directory / f"{index}.wav").write_bytes(b"audio")
                writer.writerow(
                    {
                        "path": f"./fake/en/{generator}/{index}.wav",
                        "original_file": source,
                        "language": "en",
                        "is_original_language": "True",
                        "model_name": generator,
                    },
                )
    spanish = fake / "fake" / "es" / "other"
    spanish.mkdir(parents=True)
    (spanish / "ignored.wav").write_bytes(b"audio")
    return fake, real


def test_english_inventory_and_metrics(tmp_path: Path) -> None:
    """Reuse genuine files once while preserving repeated source generation.

    Args:
        tmp_path: Disposable datasets and output directory.
    """
    fake, real = create_fixture(tmp_path)
    rows, plan = build_plan(fake, real)
    assert len(rows) == 6
    assert plan["n_bonafide"] == 2 and plan["n_spoof"] == 4
    assert plan["n_generators"] == 2
    assert len({row.key for row in rows}) == 6
    assert {row.language for row in rows} == {"en"}
    assert len({row.source for row in rows}) == 2
    values = np.array([0.9 if row.label == 0 else 0.2 for row in rows])
    save_report(tmp_path, rows, {"baseline": values}, 0.8)
    report = json.loads((tmp_path / "report.json").read_text())
    assert "MLAAD" in report["dataset"] and "VoxENES" not in report["dataset"]
    result = report["results"]["english"]["metrics"]["baseline"]
    assert result["pooled"]["n_spoof"] == 4
    assert result["one"]["n_bonafide"] == 2
    assert result["two"]["n_spoof"] == 2
    assert result["pooled"]["eer"] == 0


@pytest.mark.parametrize("kind", ["real", "fake"])
def test_missing_audio_fails(tmp_path: Path, kind: str) -> None:
    """Never silently exclude a missing genuine or spoof recording.

    Args:
        tmp_path: Disposable datasets.
        kind: Class from which to remove one file.
    """
    fake, real = create_fixture(tmp_path)
    path = next((real if kind == "real" else fake / "fake/en/one").rglob("*.wav"))
    path.unlink()
    with pytest.raises(FileNotFoundError, match="Missing or empty"):
        build_plan(fake, real)


def test_unlisted_audio_and_missing_metadata_fail(tmp_path: Path) -> None:
    """Require a complete metadata inventory for every English generator.

    Args:
        tmp_path: Disposable datasets.
    """
    fake, real = create_fixture(tmp_path)
    unexpected = fake / "fake/en/one/extra.wav"
    unexpected.write_bytes(b"audio")
    with pytest.raises(ValueError, match="Incomplete"):
        build_plan(fake, real)
    unexpected.unlink()
    (fake / "fake/en/one/meta.csv").unlink()
    with pytest.raises(FileNotFoundError):
        build_plan(fake, real)


@pytest.mark.parametrize(
    "field,value",
    [
        ("language", "es"),
        ("is_original_language", "False"),
        ("original_file", "es_ES/original.wav"),
        ("path", "../outside.wav"),
    ],
)
def test_metadata_constraints(tmp_path: Path, field: str, value: str) -> None:
    """Reject inconsistent English metadata and escaping paths.

    Args:
        tmp_path: Disposable datasets.
        field: Metadata field to invalidate.
        value: Invalid value.
    """
    fake, real = create_fixture(tmp_path)
    meta = fake / "fake/en/one/meta.csv"
    with meta.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="|"))
    rows[0][field] = value
    with meta.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="|")
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError):
        build_plan(fake, real)


def test_duplicate_row_fails(tmp_path: Path) -> None:
    """Refuse duplicate metadata rows instead of reweighting their scores.

    Args:
        tmp_path: Disposable datasets.
    """
    fake, real = create_fixture(tmp_path)
    meta = fake / "fake/en/one/meta.csv"
    text = meta.read_text()
    meta.write_text(text + text.splitlines()[1] + "\n")
    with pytest.raises(ValueError, match="Duplicate"):
        build_plan(fake, real)


def test_symlink_escape_fails(tmp_path: Path) -> None:
    """Do not follow metadata paths outside their declared dataset root.

    Args:
        tmp_path: Disposable files.
    """
    root = tmp_path / "data"
    root.mkdir()
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"audio")
    (root / "escape.wav").symlink_to(outside)
    with pytest.raises(ValueError, match="Invalid audio"):
        safe_path(root, "escape.wav")
