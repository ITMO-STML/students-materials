"""Strict metadata-based English MLAAD protocol with unique real references."""

import csv
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from src.steering.external_eval import digest
from src.steering.external_protocol import ExternalRecording


def safe_path(root: Path, relative: str) -> Path:
    """Resolve a metadata path strictly within its declared dataset root.

    Args:
        root: Dataset directory.
        relative: POSIX relative audio path from metadata.

    Returns:
        Existing nonempty waveform path.
    """
    parts = PurePosixPath(relative)
    if parts.is_absolute() or ".." in parts.parts or "\\" in relative:
        message = f"Unsafe metadata path: {relative}"
        raise ValueError(message)
    path = root.joinpath(*parts.parts)
    if not path.resolve().is_relative_to(root.resolve()) or path.suffix.lower() != ".wav":
        message = f"Invalid audio path: {relative}"
        raise ValueError(message)
    if not path.is_file() or path.stat().st_size == 0:
        message = f"Missing or empty audio: {path}"
        raise FileNotFoundError(message)
    return path


def generator_rows(root: Path, real_root: Path, meta: Path) -> tuple[list[ExternalRecording], set[str], dict[str, Any]]:
    """Read one complete English generator and validate its source references.

    Args:
        root: MLAAD root containing fake/en.
        real_root: M-AILABS directory containing en_US and en_UK.
        meta: Generator metadata file, with pipe-separated fields.

    Returns:
        Spoof records, unique original references and generator metadata.
    """
    checksum = digest(meta)
    with meta.open(newline="") as handle:
        records = list(csv.DictReader(handle, delimiter="|"))
    rows = []
    sources: set[str] = set()
    names: set[str] = set()
    seen: set[Path] = set()
    for record in records:
        if record["language"] != "en" or record["is_original_language"] != "True":
            message = f"Unexpected language/original-language flag: {meta}"
            raise ValueError(message)
        audio = safe_path(root, record["path"])
        original = PurePosixPath(record["original_file"])
        if not original.parts or original.parts[0] not in {"en_US", "en_UK"}:
            message = f"Non-English genuine reference: {original}"
            raise ValueError(message)
        safe_path(real_root, str(original))
        if audio.parent != meta.parent or audio in seen:
            message = f"Duplicate or wrong-generator audio: {audio}"
            raise ValueError(message)
        seen.add(audio)
        sources.add(str(original))
        names.add(record["model_name"])
        rows.append(
            ExternalRecording(
                "spoof/" + audio.relative_to(root).as_posix(),
                str(audio),
                1,
                meta.parent.name,
                language="en",
                source=str(original),
            ),
        )
    if not rows or len(names) != 1 or seen != set(meta.parent.glob("*.wav")) or digest(meta) != checksum:
        message = f"Incomplete, mixed-model or changing generator metadata: {meta}"
        raise ValueError(message)
    return rows, sources, {"model_name": next(iter(names)), "n_spoof": len(rows), "n_originals": len(sources)}


def build_plan(root: Path, real_root: Path) -> tuple[list[ExternalRecording], dict[str, Any]]:
    """Enumerate all English spoof metadata and its deduplicated genuine sources.

    Args:
        root: Local MLAAD snapshot, not assumed to be the latest public release.
        real_root: Matching M-AILABS extraction.

    Returns:
        Deterministically ordered records and an auditable population summary.
    """
    english = root / "fake" / "en"
    directories = sorted(path for path in english.iterdir() if path.is_dir())
    if not directories:
        message = "No English MLAAD generator directories"
        raise ValueError(message)
    rows: list[ExternalRecording] = []
    sources: set[str] = set()
    groups = {}
    metadata = {}
    for directory in directories:
        meta = directory / "meta.csv"
        checksum = digest(meta)
        fake, originals, group = generator_rows(root, real_root, meta)
        if digest(meta) != checksum:
            message = f"Metadata changed during inventory: {meta}"
            raise ValueError(message)
        metadata[str(meta)] = checksum
        groups[directory.name] = group
        rows.extend(fake)
        sources.update(originals)
    n_spoof = len(rows)
    for source in sorted(sources):
        path = safe_path(real_root, source)
        rows.append(
            ExternalRecording(
                "bonafide/" + source,
                str(path),
                0,
                "bonafide",
                language="en",
                source=source,
            ),
        )
    rows.sort(key=lambda row: row.key)
    return rows, {
        "dataset": "MLAAD local English snapshot + unique corresponding M-AILABS originals",
        "n_recordings": len(rows),
        "n_spoof": n_spoof,
        "n_bonafide": len(sources),
        "n_generators": len(groups),
        "generators": groups,
        "metadata_sha256": metadata,
        "real_locales": dict(Counter(PurePosixPath(source).parts[0] for source in sources)),
        "policy": "All fake/en rows; unique referenced English originals once; each generator versus all real.",
        "source_correlation": "Real/spoof sources overlap by design; no fitting or source-disjoint claim.",
    }
