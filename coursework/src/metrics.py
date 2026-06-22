"""
Метрики качества для baseline CMU-MOSEI.

Sentiment (3 класса, classification head): macro-F1, accuracy (3-class),
Acc-2 (negative vs non-negative — то же разбиение raw>=0/<0, что и в EDA).

Emotion (multilabel, 6 меток): macro-F1 и weighted-F1 при threshold=0.5,
ROC-AUC по каждой эмоции отдельно + усреднённый. Колонки без обоих классов
в батче (например, редкая эмоция отсутствует целиком) пропускаются в AUC
с NaN, а не вызывают падение — типичный edge case на маленьких батчах.
"""

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

EMOTION_NAMES = ["happy", "sad", "anger", "surprise", "disgust", "fear"]


def sentiment_metrics(y_true, y_pred):
    """y_true, y_pred: (N,) int classes {0:negative, 1:neutral, 2:positive}"""
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    acc3 = accuracy_score(y_true, y_pred)

    true_bin = (y_true != 0).astype(int)   # negative=0 vs {neutral,positive}=1, совпадает с raw>=0
    pred_bin = (y_pred != 0).astype(int)
    acc2 = accuracy_score(true_bin, pred_bin)
    f1_2 = f1_score(true_bin, pred_bin, average="macro", zero_division=0)

    return {"macro_f1_3class": macro_f1, "acc3": acc3, "acc2": acc2, "macro_f1_2class": f1_2}


def emotion_metrics(y_true, y_pred_proba, threshold=0.5):
    """y_true: (N,6) binary {0,1}. y_pred_proba: (N,6) вероятности после sigmoid."""
    y_pred_bin = (y_pred_proba >= threshold).astype(int)

    macro_f1 = f1_score(y_true, y_pred_bin, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred_bin, average="weighted", zero_division=0)

    per_emotion_auc = {}
    aucs = []
    for i, name in enumerate(EMOTION_NAMES):
        if len(np.unique(y_true[:, i])) < 2:
            per_emotion_auc[name] = np.nan
            continue
        auc = roc_auc_score(y_true[:, i], y_pred_proba[:, i])
        per_emotion_auc[name] = auc
        aucs.append(auc)

    mean_auc = float(np.mean(aucs)) if aucs else float("nan")
    return {"macro_f1": macro_f1, "weighted_f1": weighted_f1,
            "mean_auc": mean_auc, "per_emotion_auc": per_emotion_auc}


def print_report(split_name, sent_metrics, emo_metrics):
    print(f"=== {split_name} ===")
    print(f"  Sentiment: macro-F1(3cls)={sent_metrics['macro_f1_3class']:.3f} | "
          f"Acc3={sent_metrics['acc3']:.3f} | Acc2={sent_metrics['acc2']:.3f}")
    print(f"  Emotion:   macro-F1={emo_metrics['macro_f1']:.3f} | "
          f"weighted-F1={emo_metrics['weighted_f1']:.3f} | mean-AUC={emo_metrics['mean_auc']:.3f}")
    for name, auc in emo_metrics["per_emotion_auc"].items():
        auc_str = f"{auc:.3f}" if not np.isnan(auc) else "n/a"
        print(f"    {name:10s} AUC = {auc_str}")