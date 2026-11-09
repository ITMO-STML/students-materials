"""Strict native SASV answer parsing and classification-only reporting."""

import re
from collections.abc import Sequence
from typing import Any

from src.alm_reasoning.metrics import summarize

LABELS = ("yes", "no", "gen")


def parse_answer(text: str) -> int:
    """Parse a single native SASV answer tag without searching the reasoning.

    Args:
        text: Generated continuation.

    Returns:
        Index in yes/no/gen, or -1 for malformed or ambiguous answers.
    """
    tags = re.findall(r"<answer\s*>\s*(yes|no|gen)\s*</answer\s*>", text, flags=re.IGNORECASE)
    if (
        len(tags) != 1
        or len(re.findall(r"<answer\s*>", text, flags=re.IGNORECASE)) != 1
        or len(re.findall(r"</answer\s*>", text, flags=re.IGNORECASE)) != 1
    ):
        return -1
    return LABELS.index(tags[0].lower())


def classification(labels: Sequence[int], predictions: Sequence[int]) -> dict[str, Any]:
    """Report three-class macro metrics and binary spoof-positive metrics.

    Args:
        labels: True yes/no/gen indices.
        predictions: Generated class indices, or -1 for invalid answers.

    Returns:
        Native three-class and collapsed antispoofing metrics with denominators.
    """
    if not labels or len(labels) != len(predictions):
        message = "Require aligned, nonempty labels and predictions"
        raise ValueError(message)
    counts = [[0] * 4 for _ in LABELS]
    binary = [[0] * 3 for _ in range(2)]
    for label, pred in zip(labels, predictions, strict=True):
        if label not in range(3) or pred not in (-1, 0, 1, 2):
            message = "Invalid SASV class index"
            raise ValueError(message)
        counts[label][3 if pred == -1 else pred] += 1
        binary[int(label == 2)][2 if pred == -1 else int(pred == 2)] += 1
    per_class = {}
    for index, name in enumerate(LABELS):
        tp = counts[index][index]
        predicted = sum(row[index] for row in counts)
        actual = sum(counts[index])
        precision = tp / predicted if predicted else 0.0
        recall = tp / actual if actual else 0.0
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "n": actual,
        }
    return {
        "n": len(labels),
        "accuracy": sum(counts[i][i] for i in range(3)) / len(labels),
        "macro_precision": sum(v["precision"] for v in per_class.values()) / 3,
        "macro_recall": sum(v["recall"] for v in per_class.values()) / 3,
        "macro_f1": sum(v["f1"] for v in per_class.values()) / 3,
        "invalid": sum(row[3] for row in counts),
        "per_class": per_class,
        "confusion": counts,
        "confusion_rows": list(LABELS),
        "confusion_columns": [*LABELS, "invalid"],
        "binary": summarize(binary),
    }


def report(rows: list[dict[str, Any]], predictions: list[int]) -> dict[str, Any]:
    """Aggregate complete supplied pair cohorts, with all bonafide pairs per attack.

    Args:
        rows: Pair metadata, including labels and official query attack IDs.
        predictions: Aligned parsed predictions.

    Returns:
        Pooled three-class and per-attack binary classification reports.
    """
    labels = [row["label"] for row in rows]
    result = {"pooled": classification(labels, predictions), "by_attack": {}}
    for attack in sorted({row["attack"] for row in rows if row["label"] == 2}):
        indices = [i for i, row in enumerate(rows) if row["label"] != 2 or row["attack"] == attack]
        result["by_attack"][attack] = classification(
            [labels[i] for i in indices],
            [predictions[i] for i in indices],
        )["binary"]
    return result
