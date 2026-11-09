"""Native SASV pairs with two-sided speaker-disjoint support and tuning."""

import hashlib
import json
from pathlib import Path
from typing import Any

from src.steering.protocol import read_trials, speaker_partition

LABELS = {"verified": 0, "rejected": 1, "spoof": 2}


def enrollment_speakers(path: Path) -> dict[str, str]:
    """Map every official enrollment recording to its speaker.

    Args:
        path: Official enrollment metadata.

    Returns:
        Enrollment recording ID to speaker ID.
    """
    result = {}
    for line in path.read_text().splitlines():
        speaker, recordings = line.split()
        for recording in recordings.split(","):
            result[recording] = speaker
    return result


def pair_partition(reference_speaker: str, query_speaker: str, seed: int = 42) -> str:
    """Reserve both enrollment and query speakers before choosing adaptation data.

    Args:
        reference_speaker: Enrollment speaker.
        query_speaker: Official query speaker.
        seed: Prespecified partition seed.

    Returns:
        Shared support/tune/test pool, or mixed when the speakers span pools.
    """
    reference = speaker_partition(reference_speaker, seed)
    query = speaker_partition(query_speaker, seed)
    return reference if reference == query else "mixed"


def read_pairs(path: Path, dataset_root: Path, split: str) -> list[dict[str, Any]]:
    """Join native pair metadata with official attack and speaker annotations.

    Args:
        path: Existing native pair JSON.
        dataset_root: ASVspoof 5 root.
        split: dev or eval.

    Returns:
        Exact ordered pairs with validated binary ground truth and stable keys.
    """
    query_metadata = {trial.utterance: trial for trial in read_trials(dataset_root, split)}
    enroll_file = "last_eval/ASVspoof5.eval.track_2.enroll.tsv" if split == "eval" else "ASVspoof5.dev.enroll.txt"
    reference_metadata = enrollment_speakers(dataset_root / enroll_file)
    original = json.loads(path.read_text())
    result = []
    for index, row in enumerate(original):
        if len(row["reference_audios"]) != 1 or len(row["query_audios"]) != 1:
            message = "This experiment requires one native enrollment and one query"
            raise ValueError(message)
        reference = Path(row["reference_audios"][0]["original_path"])
        query = Path(row["query_audios"][0]["original_path"])
        metadata = query_metadata[query.stem]
        label = LABELS[row["gt"]]
        if int(label == 2) != metadata.label:
            message = f"Native pair and official spoof labels disagree: {query.stem}"
            raise ValueError(message)
        reference_speaker = reference_metadata[reference.stem]
        result.append(
            {
                "id": f"{path.stem}:{index}",
                "key": f"{reference.stem}:{query.stem}:{label}",
                "reference": str(reference),
                "query": str(query),
                "label": label,
                "attack": metadata.attack,
                "query_speaker": metadata.speaker,
                "reference_speaker": reference_speaker,
                "query_source": metadata.utterance if metadata.source == "-" else metadata.source,
                "pool": pair_partition(reference_speaker, metadata.speaker),
            },
        )
    return result


def choose(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    """Choose a fixed-size balanced bonafide/A21 support or tuning group.

    Args:
        rows: Pairs already restricted to a single speaker pool.
        count: Number of spoof and total bonafide pairs; must be even.
        seed: Deterministic sampling salt.

    Returns:
        Half yes, half no, and equally many A21 spoof pairs, source-deduplicated.
    """
    chosen = []
    seen = set()
    for label, target in ((0, count // 2), (1, count // 2), (2, count)):
        candidates = sorted(
            (r for r in rows if r["label"] == label and (label != 2 or r["attack"] == "A21")),
            key=lambda r: hashlib.sha256(f"{seed}:{r['key']}".encode()).digest(),
        )
        selected = []
        for row in candidates:
            if row["query_source"] in seen:
                continue
            selected.append(row)
            seen.add(row["query_source"])
            if len(selected) == target:
                break
        if len(selected) != target:
            message = f"Insufficient disjoint native pairs for class {label}: {len(selected)}/{target}"
            raise ValueError(message)
        chosen.extend(selected)
    return chosen


def prepare(cfg: dict[str, Any]) -> dict[str, Any]:
    """Construct the native pilot without selecting on transfer labels.

    Args:
        cfg: Dataset paths and prespecified sample counts.

    Returns:
        Ordered development, adaptation, and paired evaluation cohorts.
    """
    root, pairs = Path(cfg["dataset_root"]), Path(cfg["pairs_root"])
    dev = read_pairs(pairs / "sasv_asv5_dev_pairs_balanced.json", root, "dev")
    full = read_pairs(pairs / "sasv_asv5_eval_pairs.json", root, "eval")
    support = choose([r for r in full if r["pool"] == "support"], cfg["support_per_class"], cfg["seed"])
    tune = choose([r for r in full if r["pool"] == "tune"], cfg["tune_per_class"], cfg["seed"])
    selected = {
        "support": support,
        "tune": tune,
        "full_native_pair_count": len(full),
        "dev_baseline": dev[: cfg["dev_baseline_records"]],
        "dev_guard": dev[: cfg["dev_guard_records"]],
        "known": [r for r in dev[cfg["dev_baseline_records"] :] if r["label"] == 2][: cfg["known_spoof_records"]],
    }
    del full
    pilot = read_pairs(pairs / "sasv_asv5_eval_pairs_20K.json", root, "eval")
    selected["pilot_full"] = pilot
    selected["pilot_held_out"] = [r for r in pilot if r["pool"] == "test"]
    owners = {}
    for group in ("support", "tune", "pilot_held_out"):
        for row in selected[group]:
            for speaker in (row["reference_speaker"], row["query_speaker"]):
                if speaker in owners and owners[speaker] != group:
                    message = "Speaker leakage across native support/tune/test"
                    raise ValueError(message)
                owners[speaker] = group
    return selected
