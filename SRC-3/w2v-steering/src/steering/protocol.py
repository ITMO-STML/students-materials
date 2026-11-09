"""Auditable speaker-disjoint ASVspoof 5 adaptation subsets."""

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Trial:
    """Recording metadata retained throughout scoring.

    Args:
        speaker: Protocol speaker ID.
        utterance: Recording ID.
        attack: Attack ID or bonafide.
        label: Zero for bonafide, one for spoof.
        codec: Codec annotation.
        source: Original recording for processed variants, or a dash.
        audio_dir: Directory containing the FLAC file.
    """

    speaker: str
    utterance: str
    attack: str
    label: int
    codec: str
    source: str
    audio_dir: str


def read_trials(root: Path, split: str) -> list[Trial]:
    """Parse official metadata, preserving attack and source annotations.

    Args:
        root: Dataset root.
        split: train, dev, or eval.

    Returns:
        Protocol records in file order.
    """
    if split not in {"train", "dev", "eval"}:
        message = f"Unknown split: {split}"
        raise ValueError(message)
    is_eval = split == "eval"
    metadata = root / "last_eval/ASVspoof5.eval.track_1.tsv" if is_eval else root / f"ASVspoof5.{split}.metadata.txt"
    directory = "last_eval/flac_E_eval" if is_eval else {"train": "flac_T", "dev": "flac_D"}[split]
    trials = []
    with metadata.open() as handle:
        for line in handle:
            fields = line.split()
            if not fields:
                continue
            expected = 10 if is_eval else 6
            if len(fields) != expected:
                message = f"Expected {expected} fields in {metadata}, got {len(fields)}"
                raise ValueError(message)
            label = {"bonafide": 0, "spoof": 1}[fields[-2] if is_eval else fields[-1]]
            trials.append(
                Trial(
                    fields[0],
                    fields[1],
                    fields[7] if is_eval else fields[4],
                    label,
                    fields[3],
                    fields[5] if is_eval else "-",
                    str(root / directory),
                ),
            )
    return trials


def speaker_partition(speaker: str, seed: int) -> str:
    """Hash speakers into 20/20/60 percent support/tune/test pools.

    Args:
        speaker: Protocol speaker ID.
        seed: Partition seed.

    Returns:
        Partition independent of recording order and attack label.
    """
    bucket = int(hashlib.sha256(f"{seed}:{speaker}".encode()).hexdigest(), 16) % 10
    return "support" if bucket < 2 else "tune" if bucket < 4 else "test"


def select_trials(trials: list[Trial], groups: list[str], count: int, seed: int) -> list[Trial]:
    """Select seeded samples, deduplicating source recordings within each group.

    Args:
        trials: Candidate records in a single partition.
        groups: Attack IDs, bonafide, or spoof for all spoof attacks.
        count: Positive number of recordings per group.
        seed: Sampling seed.

    Returns:
        Selected records in group then seeded-hash order.

    Raises:
        ValueError: A group has too few unique recordings.
    """
    if count < 1:
        message = "count must be positive"
        raise ValueError(message)
    selected: list[Trial] = []
    for group in groups:
        candidates = sorted(
            (t for t in trials if t.attack == group or (group == "spoof" and t.label == 1)),
            key=lambda t: hashlib.sha256(f"{seed}:{t.utterance}".encode()).digest(),
        )
        seen = set()
        chosen = []
        for trial in candidates:
            source = trial.utterance if trial.source == "-" else trial.source
            if source not in seen:
                seen.add(source)
                chosen.append(trial)
            if len(chosen) == count:
                break
        if len(chosen) != count:
            message = f"{group}: requested {count} independent recordings, found {len(chosen)}"
            raise ValueError(message)
        selected.extend(chosen)
    return selected


def check_disjoint(partitions: dict[str, list[Trial]]) -> None:
    """Reject speaker, recording, or source overlap across named subsets.

    Args:
        partitions: Selected subsets to audit before loading audio.

    Raises:
        ValueError: Any provenance key occurs in multiple subsets.
    """
    owners: dict[str, str] = {}
    for name, trials in partitions.items():
        for trial in trials:
            keys = [f"speaker:{trial.speaker}", f"audio:{trial.utterance}"]
            if trial.source != "-":
                keys.append(f"audio:{trial.source}")
            for key in keys:
                if key in owners and owners[key] != name:
                    message = f"Leakage between {owners[key]} and {name}: {key}"
                    raise ValueError(message)
                owners[key] = name
