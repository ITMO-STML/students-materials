"""Engineering-only strict-load and generation smoke for native SASV."""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from src.alm_reasoning.artifacts import write_json
from src.alm_sasv.metrics import parse_answer
from src.alm_sasv.runtime import SASVALM
from src.alm_sasv.steering import prefill_steer
from src.alm_steering.interventions import Intervention

LOGGER = logging.getLogger(__name__)


def smoke(model: SASVALM, rows: list[dict[str, Any]], output: Path) -> None:
    """Require strict loading and cached/uncached token agreement on dev audio.

    Args:
        model: Frozen native SASV checkpoint.
        rows: Development-only engineering examples.
        output: Experiment artifact directory.
    """
    started = time.perf_counter()
    audio = model.encode_pairs(rows[:2])
    cached = model.decode(audio, 8)
    uncached = model.decode(audio, 8, use_cache=False)
    if [r["token_ids"] for r in cached] != [r["token_ids"] for r in uncached]:
        message = "Cached/uncached native SASV tokens differ"
        raise ValueError(message)
    prefix, span = model.prefix(audio)
    intervention = Intervention(torch.ones(prefix.shape[-1]), 0.0, "audio", prefix.shape[1], span)
    with prefill_steer(model.layers[15], intervention):
        zero = model.decode(audio, 8)
    if [r["token_ids"] for r in zero] != [r["token_ids"] for r in cached]:
        message = "Zero-steering SASV smoke changed the output"
        raise ValueError(message)
    # Check the requested batch size without consuming new evaluation examples.
    expanded = audio[:1].expand(model.cfg["batch_size"], -1, -1).contiguous()
    model.decode(expanded, 16)
    rows_generated = model.decode(audio)
    write_json(
        output / "smoke.json",
        {
            "status": "passed",
            "strict_load": model.audit,
            "cache_token_parity": True,
            "zero_steering_token_parity": True,
            "configured_batch_size_smoke": model.cfg["batch_size"],
            "wall_seconds": time.perf_counter() - started,
            "audio_shape": list(audio.shape),
            "prefix_length": model.prefix(audio)[0].shape[1],
            "examples": [
                {"id": row["id"], "prediction": parse_answer(gen["text"]), **gen}
                for row, gen in zip(rows, rows_generated, strict=False)
            ],
        },
    )
    LOGGER.info("Native SASV smoke passed; saved full generations")


def main() -> None:
    """Run a bounded detached compatibility smoke using supplied dev pairs."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/alm_sasv/default.yaml"))
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    output = Path(cfg["output"])
    output.mkdir(parents=True, exist_ok=True)
    raw = json.loads((Path(cfg["pairs_root"]) / "sasv_asv5_dev_pairs_balanced.json").read_text())[:2]
    rows = [
        {
            "id": r["task_id"],
            "reference": r["reference_audios"][0]["original_path"],
            "query": r["query_audios"][0]["original_path"],
        }
        for r in raw
    ]
    torch.set_num_threads(4)
    torch.manual_seed(cfg["seed"])
    model = SASVALM(cfg)
    write_json(output / "load.json", model.audit)
    smoke(model, rows, output)


if __name__ == "__main__":
    main()
