"""Source-isolated support and tuning for one ElevenLabs voice."""

import hashlib
from collections import defaultdict
from dataclasses import asdict, fields
from typing import Any

from src.steering.external_protocol import ExternalRecording

EXCLUDED_SPEAKERS = {"trump", "biden", "byden", "putin"}


def records(raw: list[dict[str, Any]]) -> list[ExternalRecording]:
    """Restore recording metadata, ignoring inventory-only file stat fields.

    Args:
        raw: Serialized recording rows.

    Returns:
        Typed records preserving manifest order.
    """
    names = {field.name for field in fields(ExternalRecording)}
    return [ExternalRecording(**{key: value for key, value in row.items() if key in names}) for row in raw]


def order_key(value: str, seed: int) -> str:
    """Compute a reproducible ordering without inspecting model scores.

    Args:
        value: Source or segment identifier.
        seed: Fixed partition seed.

    Returns:
        Stable hash ordering key.
    """
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def cohort_counts(rows: list[ExternalRecording], heldout: list[ExternalRecording], speaker: str) -> dict[str, int]:
    """Count evaluation records separately from adaptation split selection.

    Args:
        rows: Complete prepared audio inventory.
        heldout: Source-disjoint evaluation subset.
        speaker: Adaptation speaker ID.

    Returns:
        Recording counts and the genuine target-speaker coverage limitation.
    """
    return {
        "inclusive": len(rows),
        "source_heldout": len(heldout),
        "other_speakers": sum(row.speaker != speaker for row in rows),
        "target_heldout_elevenlabs": sum(
            row.label == 1 and row.speaker == speaker and row.group == "elevenlabs" for row in heldout
        ),
        "target_heldout_bonafide": sum(row.label == 0 and row.speaker == speaker for row in heldout),
    }


def speaker_sources(
    rows: list[ExternalRecording],
    speaker: str,
) -> tuple[dict[str, list[ExternalRecording]], dict[str, list[ExternalRecording]]]:
    """Group the selected speaker by genuine and ElevenLabs parent recordings.

    Args:
        rows: Prepared-segment inventory.
        speaker: Exact speaker directory identifier.

    Returns:
        Genuine and ElevenLabs source-to-segment mappings.
    """
    bona: dict[str, list[ExternalRecording]] = defaultdict(list)
    spoof: dict[str, list[ExternalRecording]] = defaultdict(list)
    for row in rows:
        if row.speaker != speaker:
            continue
        if row.label == 0:
            bona[row.source].append(row)
        elif row.group == "elevenlabs":
            spoof[row.source].append(row)
    return bona, spoof


def make_plan(
    rows: list[ExternalRecording],
    speaker: str,
    seed: int,
    support_spoof: int,
    tune_spoof: int,
    bonafide_per_split: int,
) -> dict[str, Any]:
    """Separate entire source recordings before selecting fitting segments.

    Args:
        rows: Complete prepared ElevenLabs-set inventory.
        speaker: Permitted directory speaker, selected from counts only.
        seed: Deterministic source and segment selection seed.
        support_spoof: Distinct ElevenLabs source recordings for fitting.
        tune_spoof: Distinct ElevenLabs source recordings for tuning.
        bonafide_per_split: Segments from one genuine source per split.

    Returns:
        Support/tune rows, excluded parent sources and evaluation cohort sizes.

    Raises:
        ValueError: Excluded speaker, insufficient sources or inconsistent inputs.
    """
    if speaker.casefold() in EXCLUDED_SPEAKERS:
        message = "Trump, Biden/Byden and Putin are excluded adaptation speakers"
        raise ValueError(message)
    if min(support_spoof, tune_spoof, bonafide_per_split) < 1:
        message = "Adaptation budgets must be positive"
        raise ValueError(message)
    bona, spoof = speaker_sources(rows, speaker)
    genuine_sources = sorted(bona, key=lambda source: (-len(bona[source]), source))
    fake_sources = sorted(spoof, key=lambda source: order_key(source, seed))
    if len(genuine_sources) < 2 or len(fake_sources) <= support_spoof + tune_spoof:
        message = "Require two genuine sources and unused ElevenLabs source recordings for testing"
        raise ValueError(message)
    partitions = {}
    source_splits = {}
    for name, sources, bona_source in (
        ("support", fake_sources[:support_spoof], genuine_sources[0]),
        ("tune", fake_sources[support_spoof : support_spoof + tune_spoof], genuine_sources[1]),
    ):
        if len(bona[bona_source]) < bonafide_per_split:
            message = "Insufficient genuine segments in a source-disjoint split"
            raise ValueError(message)
        genuine = sorted(bona[bona_source], key=lambda row: order_key(row.key, seed))[:bonafide_per_split]
        generated = [min(spoof[source], key=lambda row: order_key(row.key, seed)) for source in sources]
        partitions[name] = [asdict(row) for row in genuine + generated]
        source_splits[name] = [bona_source, *sources]
    if set(source_splits["support"]) & set(source_splits["tune"]):
        message = "Support and tuning source overlap"
        raise ValueError(message)
    blocked = set(source_splits["support"]) | set(source_splits["tune"])
    heldout = [row for row in rows if row.source not in blocked]
    if not any(row.label == 0 for row in heldout) or not any(row.label == 1 for row in heldout):
        message = "Held-out evaluation must retain both classes"
        raise ValueError(message)
    return {
        "speaker": speaker,
        "seed": seed,
        **partitions,
        "source_splits": source_splits,
        "excluded_sources": sorted(blocked),
        "counts": cohort_counts(rows, heldout, speaker),
        "interpretation": (
            "Source-held-out excludes every segment of support/tuning parent recordings. "
            "Genuine support/tune each use one source; inspect target_heldout_bonafide "
            "before claiming unseen genuine target-speaker coverage. Other-speaker transfer is separate."
        ),
    }
