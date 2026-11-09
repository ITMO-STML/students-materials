"""Select original English audio from the archive's authoritative inventory."""

from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any
from zipfile import ZipFile

from src.steering.external_protocol import ExternalRecording
from src.voxenes_eval.archive import archive_identity, entries, matches


def english_record(path: Path, root: Path) -> ExternalRecording | None:
    """Accept English genuine audio and original English TTS/VC only.

    Args:
        path: Archive-listed audio path.
        root: Dataset root.

    Returns:
        Selected recording, or None for Spanish, augmented and metadata files.
    """
    relative = path.relative_to(root)
    parts = relative.parts
    if len(parts) == 3 and parts[:2] == ("bonafide", "english"):
        label, group = 0, "bonafide"
    elif len(parts) >= 5 and parts[0] in {"tts", "vc"} and parts[1] == "original" and parts[3] == "english":
        if (parts[0] == "tts" and (len(parts) != 6 or parts[4] not in {"fixed", "random"})) or (
            parts[0] == "vc" and len(parts) != 5
        ):
            message = f"Unknown original-English audio layout: {relative}"
            raise ValueError(message)
        label, group = 1, f"{parts[0]}/{parts[2]}"
    else:
        return None
    if path.suffix.lower() not in {".flac", ".wav"}:
        message = f"Unexpected selected audio format: {relative}"
        raise ValueError(message)
    return ExternalRecording(relative.as_posix(), str(path), label, group, language="english")


def build_plan(root: Path, archive_path: Path, *, verify_audio: bool) -> tuple[list[ExternalRecording], dict[str, Any]]:
    """Use the ZIP inventory so incomplete extraction cannot silently shrink eval.

    Args:
        root: Dataset extraction directory.
        archive_path: Complete local dataset archive.
        verify_audio: Require every selected file to match size and ZIP CRC.

    Returns:
        Stable sorted records and an expected inventory with extraction status.
    """
    rows = []
    expected = []
    missing = []
    with ZipFile(archive_path) as archive:
        members = entries(archive, root)
        identity = archive_identity(archive_path, members)
        for member, path in sorted(members, key=lambda entry: str(entry[1])):
            row = english_record(path, root)
            if row is None:
                continue
            valid = matches(path, member, crc=verify_audio)
            if not valid:
                missing.append(row.key)
            rows.append(row)
            expected.append({**asdict(row), "expected_bytes": member.file_size, "crc32": member.CRC})
    if verify_audio and missing:
        message = f"Original-English audio is incomplete/corrupt: {len(missing)} files; first: {missing[0]}"
        raise ValueError(message)
    if {row.label for row in rows} != {0, 1}:
        message = "Original-English protocol requires both genuine and spoof audio"
        raise ValueError(message)
    return rows, {
        "archive": identity,
        "n_recordings": len(rows),
        "n_bonafide": sum(row.label == 0 for row in rows),
        "n_spoof": sum(row.label == 1 for row in rows),
        "groups": dict(Counter(row.group for row in rows)),
        "missing_or_mismatched": len(missing),
        "missing_examples": missing[:10],
        "selected_audio_crc_verified": verify_audio,
        "recordings": expected,
        "selection": "bonafide/english; tts/original/*/english/{fixed,random}; vc/original/*/english",
    }
