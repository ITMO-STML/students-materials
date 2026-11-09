"""Generated-answer classification with explicit abstentions and pooled counts."""

import re
from collections import defaultdict
from typing import Any

from src.steering.protocol import Trial, speaker_partition


def parse_answer(text: str) -> int:
    """Parse exactly one complete answer, returning -1 for malformed output.

    Args:
        text: Generated continuation, without any reference answer.

    Returns:
        Zero for Real, one for Fake, or minus one for invalid/ambiguous output.
    """
    opening = re.findall(r"<answer\s*>", text, flags=re.IGNORECASE)
    closing = re.findall(r"</answer\s*>", text, flags=re.IGNORECASE)
    matches = re.findall(r"<answer\s*>\s*(Real|Fake)\s*</answer\s*>", text, flags=re.IGNORECASE)
    if len(opening) != 1 or len(closing) != 1 or len(matches) != 1:
        return -1
    return int(matches[0].lower() == "fake")


def summarize(counts: list[list[int]]) -> dict[str, Any]:
    """Summarize a two-by-three true-class/prediction confusion matrix.

    Args:
        counts: Rows bonafide/spoof, columns bonafide/spoof/invalid.

    Returns:
        Classification metrics on all records, including invalid errors.
    """
    tn, fp, invalid_bona = counts[0]
    fn_valid, tp, invalid_spoof = counts[1]
    n_bona, n_spoof = sum(counts[0]), sum(counts[1])
    total = n_bona + n_spoof
    fn = fn_valid + invalid_spoof
    return {
        "n": total,
        "n_bonafide": n_bona,
        "n_spoof": n_spoof,
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / n_spoof if n_spoof else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "bonafide_recall": tn / n_bona if n_bona else 0.0,
        "invalid": invalid_bona + invalid_spoof,
        "invalid_bonafide": invalid_bona,
        "invalid_spoof": invalid_spoof,
        "confusion_columns": ["bonafide", "spoof", "invalid"],
        "confusion_rows": ["bonafide", "spoof"],
        "confusion": counts,
    }


class Cohorts:
    """Accumulate full and speaker-held-out counts without storing waveforms."""

    def __init__(self) -> None:
        self.counts: dict[str, dict[str, list[int]]] = {
            cohort: defaultdict(lambda: [0, 0, 0]) for cohort in ("full_evaluation", "speaker_held_out")
        }

    def add(self, trial: Trial, prediction: int) -> None:
        """Add one prediction to its applicable cohorts.

        Args:
            trial: Ground truth used only after generation.
            prediction: Parsed answer, with -1 denoting invalid.
        """
        if prediction not in (-1, 0, 1):
            message = "Prediction must be -1, 0, or 1"
            raise ValueError(message)
        group = "bonafide" if trial.label == 0 else trial.attack
        column = 2 if prediction == -1 else prediction
        self.counts["full_evaluation"][group][column] += 1
        if speaker_partition(trial.speaker, 42) == "test":
            self.counts["speaker_held_out"][group][column] += 1

    def report(self) -> dict[str, Any]:
        """Pair each observed attack with every bonafide record in its cohort.

        Returns:
            Pooled and per-attack metrics, never means of batch metrics.
        """
        result = {}
        for cohort, groups in self.counts.items():
            bona = groups.get("bonafide", [0, 0, 0])
            attacks = sorted(key for key in groups if key != "bonafide")
            spoof = [sum(groups[key][column] for key in attacks) for column in range(3)]
            result[cohort] = {"pooled": summarize([bona, spoof])}
            result[cohort].update({attack: summarize([bona, groups[attack]]) for attack in attacks})
        return result
