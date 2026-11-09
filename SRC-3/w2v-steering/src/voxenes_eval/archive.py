"""Inspect and finish the provided VoxENES ZIP without trusting partial trees."""

import hashlib
import json
import stat
import tempfile
import zlib
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import ZipFile, ZipInfo


def write_status(path: Path, value: dict[str, Any]) -> None:
    """Atomically publish a progress or completion record.

    Args:
        path: Run-owned status file.
        value: Serializable status fields.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def entries(archive: ZipFile, root: Path) -> list[tuple[ZipInfo, Path]]:
    """Validate every member before allowing any extraction.

    Args:
        archive: Open dataset ZIP.
        root: Intended dataset directory, matching the archive's top directory.

    Returns:
        File members and safe destination paths in archive order.

    Raises:
        ValueError: Unsafe path, nonregular entry, or duplicate destination.
    """
    result = []
    seen = set()
    for member in archive.infolist():
        relative = PurePosixPath(member.filename)
        mode = member.external_attr >> 16
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or "\\" in member.filename
            or not relative.parts
            or relative.parts[0] != root.name
            or stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)
        ):
            message = f"Unsafe archive member: {member.filename}"
            raise ValueError(message)
        if member.is_dir():
            continue
        destination = root.joinpath(*relative.parts[1:])
        if not destination.resolve().is_relative_to(root.resolve()) or destination in seen:
            message = f"Escaping or duplicate archive path: {member.filename}"
            raise ValueError(message)
        seen.add(destination)
        result.append((member, destination))
    if not result:
        message = "Empty dataset archive"
        raise ValueError(message)
    return result


def file_crc(path: Path) -> int:
    """Compute ZIP-compatible CRC32 without loading a waveform into memory.

    Args:
        path: Existing regular file.

    Returns:
        Unsigned CRC32.
    """
    checksum = 0
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            checksum = zlib.crc32(block, checksum)
    return checksum & 0xFFFFFFFF


def matches(path: Path, member: ZipInfo, *, crc: bool) -> bool:
    """Check size and optionally content integrity against a ZIP member.

    Args:
        path: Intended extracted file.
        member: Expected archive metadata.
        crc: Whether to verify content, not just extraction coverage.

    Returns:
        Whether the extracted file matches the expected member.
    """
    return path.is_file() and path.stat().st_size == member.file_size and (not crc or file_crc(path) == member.CRC)


def archive_identity(path: Path, members: list[tuple[ZipInfo, Path]]) -> dict[str, Any]:
    """Record archive stats and a digest of the central-directory inventory.

    Args:
        path: Dataset ZIP.
        members: Validated file entries.

    Returns:
        Archive identity; CRC32 is an integrity check, not an authenticity claim.
    """
    catalog = [(entry.filename, entry.file_size, entry.CRC, entry.compress_size) for entry, _ in members]
    info = path.stat()
    return {
        "path": str(path),
        "size_bytes": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "catalog_sha256": hashlib.sha256(json.dumps(catalog, sort_keys=True).encode()).hexdigest(),
        "files": len(members),
        "uncompressed_bytes": sum(entry.file_size for entry, _ in members),
    }


def restore_member(archive: ZipFile, member: ZipInfo, destination: Path) -> None:
    """Publish one verified missing/corrupt member using an atomic replacement.

    Args:
        archive: Open ZIP; its reader also validates compressed data CRC.
        member: Expected member metadata.
        destination: Validated path inside the dataset root.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".voxenes-", delete=False) as handle:
            temporary = Path(handle.name)
            with archive.open(member) as source:
                while block := source.read(1024 * 1024):
                    handle.write(block)
        if not matches(temporary, member, crc=True):
            message = f"Extracted file failed CRC: {member.filename}"
            raise ValueError(message)
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def finish_extraction(archive_path: Path, root: Path, status: Path) -> dict[str, Any]:
    """Verify the full archive tree and repair only missing/corrupt members.

    Args:
        archive_path: Provided local ZIP.
        root: Extraction root; existing valid files remain untouched.
        status: Run-owned progress file.

    Returns:
        Complete extraction verification, including the number of repairs.
    """
    repaired = 0
    with ZipFile(archive_path) as archive:
        members = entries(archive, root)
        identity = archive_identity(archive_path, members)
        for index, (member, path) in enumerate(members, start=1):
            if not matches(path, member, crc=True):
                restore_member(archive, member, path)
                repaired += 1
            if index % 250 == 0 or index == len(members):
                write_status(
                    status,
                    {
                        "status": "verifying_and_extracting",
                        "checked": index,
                        "total": len(members),
                        "repaired": repaired,
                    },
                )
        if archive_identity(archive_path, members) != identity:
            message = "Archive changed during extraction"
            raise ValueError(message)
    result = {
        "status": "complete",
        "checked": len(members),
        "repaired": repaired,
        "crc32_verified": True,
        "archive": identity,
    }
    write_status(status, result)
    return result
