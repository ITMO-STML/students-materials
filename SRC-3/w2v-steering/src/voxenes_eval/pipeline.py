"""Sequential archive recovery and frozen evaluation, suitable for tmux."""

import argparse
import fcntl
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

from src.voxenes_eval.archive import finish_extraction, write_status

log = logging.getLogger(__name__)
DATA = Path("/mnt/shared_ru.ml.SZ-2_000049/antispoofing_datasets/voxenes/voxenes-2026")


def extractors(archive: Path) -> list[int]:
    """Find existing unzip processes handling this exact archive.

    Args:
        archive: ZIP archive to wait for.

    Returns:
        Matching process IDs, excluding already terminated processes.
    """
    result = []
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            args = (process / "cmdline").read_bytes().decode().split("\0")
            if not args or Path(args[0]).name != "unzip":
                continue
            cwd = (process / "cwd").resolve(strict=True)
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any((cwd / arg).resolve() == archive.resolve() for arg in args[1:] if arg):
            result.append(int(process.name))
    return result


def most_free_device(csv_text: str) -> str:
    """Select the GPU with the most free VRAM, regardless of utilization.

    Args:
        csv_text: nvidia-smi index and memory.free CSV, in MiB.

    Returns:
        CUDA device with maximum free memory; lower index breaks ties.
    """
    devices = [tuple(map(int, line.split(","))) for line in csv_text.strip().splitlines()]
    if not devices:
        message = "No GPUs reported by nvidia-smi"
        raise ValueError(message)
    index, _ = max(devices, key=lambda row: (row[1], -row[0]))
    return f"cuda:{index}"


def select_device(requested: str, state: Path) -> str:
    """Choose the device immediately before evaluation without waiting for idle.

    Args:
        requested: Explicit torch device, or auto for maximum free VRAM.
        state: Pipeline status path.

    Returns:
        Selected torch device; the memory observation is not a reservation.
    """
    if requested != "auto":
        return requested
    executable = shutil.which("nvidia-smi")
    if executable is None:
        message = "nvidia-smi is required for automatic GPU selection"
        raise FileNotFoundError(message)
    response = subprocess.run(  # noqa: S603 - fixed read-only command
        [executable, "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    device = most_free_device(response.stdout)
    write_status(state, {"status": "gpu_selected", "device": device, "policy": "maximum_free_vram"})
    log.info("Selected %s with the most free VRAM; no idle wait", device)
    return device


def execute(args: argparse.Namespace) -> None:
    """Wait for existing extraction, repair and verify all members, then evaluate.

    Args:
        args: Paths and device from the command line.
    """
    state = args.output / "pipeline.json"
    succeeded = False
    try:
        while processes := extractors(args.archive):
            write_status(state, {"status": "waiting_for_existing_unzip", "pids": processes})
            log.info("Waiting for existing unzip processes: %s", processes)
            time.sleep(30)
        write_status(state, {"status": "verifying_and_repairing_full_extraction"})
        finish_extraction(args.archive, args.root, args.output / "extraction.json")
        device = select_device(args.device, state)
        command = [
            sys.executable,
            "-u",
            "-m",
            "src.voxenes_eval.run",
            "steering=voxenes",
            "exp_name=w2v_voxenes_original_english_v1",
            f"steering.external_root={args.root}",
            f"steering.archive_path={args.archive}",
            f"steering.device={device}",
            f"steering.resume_dir={args.output}",
            f"hydra.run.dir={args.output}",
        ]
        write_status(state, {"status": "evaluating", "device": device, "command": command})
        subprocess.run(command, check=True)  # noqa: S603 - explicit local Python module
        write_status(state, {"status": "complete", "report": str(args.output / "report.json")})
        succeeded = True
    finally:
        if not succeeded:
            write_status(state, {"status": "failed_or_interrupted", "details": "See pipeline.log and extraction.json"})


def main() -> None:
    """Run one exclusively locked sequential pipeline from the repository root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DATA)
    parser.add_argument("--archive", type=Path, default=DATA.with_suffix(".zip"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with (args.output / ".pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(args)


if __name__ == "__main__":
    main()
