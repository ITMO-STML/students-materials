"""Explicit query-level protocols for frozen cross-dataset transfer evaluation."""

import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, override

import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset

from src.data.dataset import SAMPLE_RATE, prepare_waveform
from src.steering.protocol import Trial


@dataclass(frozen=True)
class ExternalRecording:
    """Describe an audio file independently of enrollment trial multiplicity.

    Args:
        key: Dataset-relative unique query or segment identifier.
        path: Audio file path.
        label: Zero for bonafide, one for spoof.
        group: Spoof generator or WildSpoof datasource.
        speaker: Directory speaker when available; never an enrollment identity.
        language: Directory language when available.
        source: Parent recording identifier for correlated prepared segments.
        multiplicity: Number of original protocol rows containing this query.
    """

    key: str
    path: str
    label: int
    group: str
    speaker: str = "unknown"
    language: str = "unknown"
    source: str = "unknown"
    multiplicity: int = 1

    def trial(self) -> Trial:
        """Convert metadata for existing, unchanged detection metrics.

        Returns:
            Trial carrying the group as attack and zero for bonafide.
        """
        return Trial(
            self.speaker,
            self.key,
            self.group if self.label else "bonafide",
            self.label,
            Path(self.path).suffix,
            self.source,
            str(Path(self.path).parent),
        )


def read_wildspoof(root: Path) -> list[ExternalRecording]:
    """Collapse enrollment trials to unique queries with consistent binary labels.

    Args:
        root: WildSpoof 2026 directory containing its labeled test protocol.

    Returns:
        Queries in first-occurrence order, with original trial multiplicities.

    Raises:
        ValueError: Unknown labels, conflicting metadata, or malformed rows.
    """
    unique: dict[str, tuple[int, str]] = {}
    counts: Counter[str] = Counter()
    with (root / "protocol_test_label.tsv").open() as handle:
        for line in handle:
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 4 or fields[2] not in {"target", "nontarget", "spoof"}:
                message = f"Invalid WildSpoof protocol row: {fields}"
                raise ValueError(message)
            _, key, label, group = fields
            if Path(key).name != key:
                message = "WildSpoof query identifiers must be plain filenames"
                raise ValueError(message)
            metadata = (int(label == "spoof"), group)
            if key in unique and unique[key] != metadata:
                message = f"Conflicting query metadata: {key}"
                raise ValueError(message)
            unique[key] = metadata
            counts[key] += 1
    return [
        ExternalRecording(
            key,
            str(root / "data_v2.0" / f"{key}.flac"),
            label,
            group,
            source=key,
            multiplicity=counts[key],
        )
        for key, (label, group) in unique.items()
    ]


def read_elevenlabs(root: Path) -> list[ExternalRecording]:
    """Read the user-selected prepared segments, retaining every generator.

    Args:
        root: On-disk data_airi directory, displayed as the ElevenLabs set.

    Returns:
        All prepared WAV segments in stable relative-path order.

    Raises:
        ValueError: Unrecognized class or generator directory structure.
    """
    prepared = root / "processed_audio"
    rows = []
    for path in sorted(prepared.rglob("*.wav")):
        relative = path.relative_to(prepared)
        parts = relative.parts
        if len(parts) < 3 or parts[1] not in {"bonafide", "spoof"}:
            message = f"Unrecognized ElevenLabs class layout: {relative}"
            raise ValueError(message)
        label = int(parts[1] == "spoof")
        if label and len(parts) < 4:
            message = f"Missing generator directory: {relative}"
            raise ValueError(message)
        group = parts[2] if label else "bonafide"
        language_index = 3 if label else 2
        language = parts[language_index] if len(parts) > language_index + 1 else "unknown"
        parent = relative.with_name(re.sub(r"_\d+$", "", path.stem)).as_posix()
        rows.append(ExternalRecording(relative.as_posix(), str(path), label, group, parts[0], language, parent))
    return rows


def inventory(rows: list[ExternalRecording]) -> dict[str, Any]:
    """Check complete local coverage and record file identities before inference.

    Args:
        rows: Full unique-query or segment manifest.

    Returns:
        Counts and serializable rows including audio size and modification time.

    Raises:
        ValueError: Empty, single-class, duplicate, or invalid records.
        FileNotFoundError: Any listed audio file is absent.
    """
    if not rows or {row.label for row in rows} != {0, 1} or len({row.key for row in rows}) != len(rows):
        message = "Require unique nonempty records containing both classes"
        raise ValueError(message)
    manifest = []
    for row in rows:
        if row.multiplicity < 1:
            message = "Trial multiplicity must be positive"
            raise ValueError(message)
        path = Path(row.path)
        stat = path.stat()
        if not path.is_file() or stat.st_size == 0:
            message = f"Empty or non-file audio: {path}"
            raise ValueError(message)
        manifest.append({**asdict(row), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return {
        "n_recordings": len(rows),
        "n_protocol_rows": sum(row.multiplicity for row in rows),
        "n_bonafide": sum(row.label == 0 for row in rows),
        "n_spoof": sum(row.label == 1 for row in rows),
        "groups": dict(Counter(row.group for row in rows)),
        "audio_identity_policy": "Relative IDs, paths, sizes and mtime_ns; not a content hash of every waveform.",
        "recordings": manifest,
    }


class ExternalAudio(Dataset[tuple[int, Tensor]]):
    """Load arbitrary audio formats using unchanged detector preprocessing.

    Args:
        rows: Ordered recording metadata.
        num_samples: Center-crop or zero-pad length after mono resampling.
        start: First unscored index.
    """

    def __init__(self, rows: list[ExternalRecording], num_samples: int, start: int = 0) -> None:
        self.rows, self.num_samples, self.start = rows, num_samples, start

    def __len__(self) -> int:
        """Return the remaining number of recordings."""
        return len(self.rows) - self.start

    @override
    def __getitem__(self, index: int) -> tuple[int, Tensor]:
        """Decode and preprocess a recording without substituting failures.

        Args:
            index: Relative suffix index.

        Returns:
            Absolute manifest index and deterministic waveform.
        """
        absolute = self.start + index
        waveform, rate = torchaudio.load(self.rows[absolute].path)
        waveform = waveform.mean(dim=0)
        if waveform.numel() == 0 or not torch.isfinite(waveform).all():
            message = f"Empty or nonfinite audio: {self.rows[absolute].key}"
            raise ValueError(message)
        if rate != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, rate, SAMPLE_RATE)
        return absolute, prepare_waveform(waveform, self.num_samples, random_crop=False, augmentation=None)
