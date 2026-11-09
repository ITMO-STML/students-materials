"""Hugging Face Trainer for antispoofing batches."""

from functools import partial

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from transformers import EvalPrediction, Trainer, TrainingArguments

from .data.datamodule import AudioData
from .data.dataset import collate_audio
from .metrics.detection import equal_error_rate, min_a_dcf


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=-1, keepdims=True)


def compute_metrics(eval_prediction: EvalPrediction) -> dict[str, float]:
    """Score classification and ASVspoof 5 detection metrics.

    Precision, recall, F1, and ROC-AUC treat spoof (`1`) as the positive class. ROC-AUC uses the
    spoof softmax probability. EER and min a-DCF use the bonafide softmax probability.

    Args:
        eval_prediction: Logits and integer labels gathered by the trainer.

    Returns:
        Accuracy, precision, recall, F1, ROC-AUC, EER, and minimum a-DCF.

    """
    logits = eval_prediction.predictions
    labels = eval_prediction.label_ids
    if not isinstance(logits, np.ndarray) or labels is None:
        message = "Evaluation did not return logits and labels"
        raise TypeError(message)
    label_array = np.asarray(labels)
    predictions = np.argmax(logits, axis=-1)
    probabilities = _softmax(logits)
    bonafide_probability = probabilities[:, 0]
    spoof_probability = probabilities[:, 1]
    bonafide_mask = label_array == 0
    spoof_mask = label_array == 1
    return {
        "accuracy": float(accuracy_score(label_array, predictions)),
        "precision": float(precision_score(label_array, predictions, pos_label=1, zero_division=0)),
        "recall": float(recall_score(label_array, predictions, pos_label=1, zero_division=0)),
        "f1": float(f1_score(label_array, predictions, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(label_array, spoof_probability)),
        "eer": equal_error_rate(bonafide_probability[bonafide_mask], bonafide_probability[spoof_mask]),
        "min_dcf": min_a_dcf(bonafide_probability[bonafide_mask], bonafide_probability[spoof_mask]),
    }


def build_trainer(model: nn.Module, data: AudioData, args: TrainingArguments) -> Trainer:
    """Build a Trainer over waveform batches.

    Args:
        model: Antispoofing module. Its forward accepts `input_values`, `attention_mask`, and `labels`.
        data: Datasets and the Wav2Vec2 feature extractor used to collate them.
        args: Training arguments, including the Hydra run directory.

    Returns:
        Trainer ready to call `train`.

    """
    return Trainer(
        model=model,
        args=args,
        train_dataset=data.train_dataset,
        eval_dataset=data.val_dataset,
        data_collator=partial(collate_audio, feature_extractor=data.feature_extractor),
        compute_metrics=compute_metrics,
    )
