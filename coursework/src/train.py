"""
Тренировочный цикл для multitask baseline CMU-MOSEI.

Multitask loss = CrossEntropy(sentiment) + lambda * BCEWithLogits(emotion),
где вклад emotion считается только по сэмплам с mask=True (см. data_loader.py,
раздел EDA 3.2 — несматченные сэмплы остаются валидны для sentiment, но не
имеют надёжной эмоциональной метки).

Веса классов sentiment вычисляются из train-распределения (инверсия частоты)
для частичной компенсации дисбаланса 49/29/22%, обнаруженного в EDA.
pos_weight для emotion аналогично компенсирует дисбаланс по каждой из 6 эмоций
(от 53.1% у happy до 8.4% у fear) — без этого BCE почти не учится на редких
эмоциях, предсказывая константный 0.
"""

import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.metrics import sentiment_metrics, emotion_metrics


class MOSEIFeatureDataset(Dataset):
    """X: (N, D) признаки одной модальности/типа. y_sent: (N,) int. y_emo: (N,6) float (может содержать NaN). mask: (N,) bool."""

    def __init__(self, X, y_sent, y_emo, mask):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_sent = torch.tensor(y_sent, dtype=torch.long)
        y_emo_filled = np.nan_to_num(y_emo, nan=0.0)  # NaN -> 0, реальный вклад исключается через mask в лоссе
        self.y_emo = torch.tensor(y_emo_filled, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.bool)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_sent[idx], self.y_emo[idx], self.mask[idx]


def compute_sentiment_class_weights(y_sent_train, n_classes=3):
    counts = np.bincount(y_sent_train, minlength=n_classes).astype(np.float32)
    weights = counts.sum() / (n_classes * np.clip(counts, 1, None))
    return torch.tensor(weights, dtype=torch.float32)


def compute_emotion_pos_weights(y_emo_train, mask_train):
    """pos_weight[i] = (# негативных) / (# позитивных) для BCEWithLogitsLoss, по валидным (mask=True) сэмплам."""
    valid = y_emo_train[mask_train]
    pos = valid.sum(axis=0)
    neg = valid.shape[0] - pos
    pos_weight = neg / np.clip(pos, 1, None)
    return torch.tensor(pos_weight, dtype=torch.float32)


def masked_multitask_loss(sent_logits, emo_logits, y_sent, y_emo, mask,
                           sent_criterion, emo_criterion, lam=1.0):
    loss_sent = sent_criterion(sent_logits, y_sent)

    if mask.sum() > 0:
        loss_emo_raw = emo_criterion(emo_logits[mask], y_emo[mask])
    else:
        loss_emo_raw = torch.tensor(0.0, device=sent_logits.device)

    return loss_sent + lam * loss_emo_raw, loss_sent.item(), loss_emo_raw.item()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_sent_true, all_sent_pred = [], []
    all_emo_true, all_emo_proba, all_emo_mask = [], [], []

    for X, y_sent, y_emo, mask in loader:
        X = X.to(device)
        sent_logits, emo_logits = model(X)

        all_sent_true.append(y_sent.numpy())
        all_sent_pred.append(sent_logits.argmax(dim=1).cpu().numpy())

        all_emo_true.append(y_emo.numpy())
        all_emo_proba.append(torch.sigmoid(emo_logits).cpu().numpy())
        all_emo_mask.append(mask.numpy())

    sent_true = np.concatenate(all_sent_true)
    sent_pred = np.concatenate(all_sent_pred)
    emo_true = np.concatenate(all_emo_true)
    emo_proba = np.concatenate(all_emo_proba)
    emo_mask = np.concatenate(all_emo_mask)

    sm = sentiment_metrics(sent_true, sent_pred)
    em = emotion_metrics(emo_true[emo_mask], emo_proba[emo_mask])  # метрики emotion только по валидным сэмплам
    return sm, em


def fit(model, train_loader, valid_loader, device,
        sentiment_class_weights, emotion_pos_weights,
        epochs=30, lr=1e-3, lam=1.0, patience=5, verbose=True):

    model = model.to(device)
    sentiment_class_weights = sentiment_class_weights.to(device)
    emotion_pos_weights = emotion_pos_weights.to(device)

    sent_criterion = nn.CrossEntropyLoss(weight=sentiment_class_weights)
    emo_criterion = nn.BCEWithLogitsLoss(pos_weight=emotion_pos_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    best_f1 = -1.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss_sent, epoch_loss_emo, n_batches = 0.0, 0.0, 0

        for X, y_sent, y_emo, mask in train_loader:
            X, y_sent, y_emo, mask = X.to(device), y_sent.to(device), y_emo.to(device), mask.to(device)

            optimizer.zero_grad()
            sent_logits, emo_logits = model(X)
            loss, l_sent, l_emo = masked_multitask_loss(
                sent_logits, emo_logits, y_sent, y_emo, mask,
                sent_criterion, emo_criterion, lam=lam,
            )
            loss.backward()
            optimizer.step()

            epoch_loss_sent += l_sent
            epoch_loss_emo += l_emo
            n_batches += 1

        sm_val, em_val = evaluate(model, valid_loader, device)
        val_f1 = sm_val["macro_f1_3class"]

        history.append({
            "epoch": epoch,
            "train_loss_sent": epoch_loss_sent / n_batches,
            "train_loss_emo": epoch_loss_emo / n_batches,
            "val_sentiment_macro_f1": val_f1,
            "val_emotion_macro_f1": em_val["macro_f1"],
            "val_emotion_mean_auc": em_val["mean_auc"],
        })

        if verbose:
            print(f"[epoch {epoch:2d}] train_loss_sent={epoch_loss_sent/n_batches:.3f} "
                  f"train_loss_emo={epoch_loss_emo/n_batches:.3f} | "
                  f"val sentiment macro-F1={val_f1:.3f} emotion macro-F1={em_val['macro_f1']:.3f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"Early stopping на эпохе {epoch} (нет улучшения {patience} эпох подряд)")
                break

    model.load_state_dict(best_state)
    return model, history

class MOSEISequenceDataset(Dataset):
    """
    X: (N, T, D) — полная последовательность признаков (без усреднения).
    seq_mask: (N, T) bool — маска паддинга, нужна МОДЕЛИ (CNN/LSTM/Transformer).
    y_sent, y_emo: как раньше.
    label_mask: (N,) bool — валидность эмоциональной метки (та же, что в baseline),
                нужна только ЛОССУ, не модели. НЕ путать с seq_mask.
    """
    def __init__(self, X, seq_mask, y_sent, y_emo, label_mask):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.seq_mask = torch.tensor(seq_mask, dtype=torch.bool)
        self.y_sent = torch.tensor(y_sent, dtype=torch.long)
        y_emo_filled = np.nan_to_num(y_emo, nan=0.0)
        self.y_emo = torch.tensor(y_emo_filled, dtype=torch.float32)
        self.label_mask = torch.tensor(label_mask, dtype=torch.bool)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.seq_mask[idx], self.y_sent[idx], self.y_emo[idx], self.label_mask[idx]


@torch.no_grad()
def evaluate_sequence(model, loader, device):
    model.eval()
    all_sent_true, all_sent_pred = [], []
    all_emo_true, all_emo_proba, all_label_mask = [], [], []

    for X, seq_mask, y_sent, y_emo, label_mask in loader:
        X, seq_mask = X.to(device), seq_mask.to(device)
        sent_logits, emo_logits = model(X, seq_mask)

        all_sent_true.append(y_sent.numpy())
        all_sent_pred.append(sent_logits.argmax(dim=1).cpu().numpy())
        all_emo_true.append(y_emo.numpy())
        all_emo_proba.append(torch.sigmoid(emo_logits).cpu().numpy())
        all_label_mask.append(label_mask.numpy())

    sent_true = np.concatenate(all_sent_true)
    sent_pred = np.concatenate(all_sent_pred)
    emo_true = np.concatenate(all_emo_true)
    emo_proba = np.concatenate(all_emo_proba)
    label_mask = np.concatenate(all_label_mask)

    sm = sentiment_metrics(sent_true, sent_pred)
    em = emotion_metrics(emo_true[label_mask], emo_proba[label_mask])
    return sm, em


def fit_sequence(model, train_loader, valid_loader, device,
                  sentiment_class_weights, emotion_pos_weights,
                  epochs=30, lr=1e-3, lam=1.0, patience=5, verbose=True):
    model = model.to(device)
    sentiment_class_weights = sentiment_class_weights.to(device)
    emotion_pos_weights = emotion_pos_weights.to(device)

    sent_criterion = nn.CrossEntropyLoss(weight=sentiment_class_weights)
    emo_criterion = nn.BCEWithLogitsLoss(pos_weight=emotion_pos_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history, best_f1, best_state, epochs_no_improve = [], -1.0, None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss_sent, epoch_loss_emo, n_batches = 0.0, 0.0, 0

        for X, seq_mask, y_sent, y_emo, label_mask in train_loader:
            X, seq_mask = X.to(device), seq_mask.to(device)
            y_sent, y_emo, label_mask = y_sent.to(device), y_emo.to(device), label_mask.to(device)

            optimizer.zero_grad()
            sent_logits, emo_logits = model(X, seq_mask)
            loss, l_sent, l_emo = masked_multitask_loss(
                sent_logits, emo_logits, y_sent, y_emo, label_mask,
                sent_criterion, emo_criterion, lam=lam,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss_sent += l_sent
            epoch_loss_emo += l_emo
            n_batches += 1

        sm_val, em_val = evaluate_sequence(model, valid_loader, device)
        val_f1 = sm_val["macro_f1_3class"]

        history.append({
            "epoch": epoch,
            "train_loss_sent": epoch_loss_sent / n_batches,
            "train_loss_emo": epoch_loss_emo / n_batches,
            "val_sentiment_macro_f1": val_f1,
            "val_emotion_macro_f1": em_val["macro_f1"],
            "val_emotion_mean_auc": em_val["mean_auc"],
        })

        if verbose:
            print(f"[epoch {epoch:2d}] train_loss_sent={epoch_loss_sent/n_batches:.3f} "
                  f"train_loss_emo={epoch_loss_emo/n_batches:.3f} | "
                  f"val sentiment macro-F1={val_f1:.3f} emotion macro-F1={em_val['macro_f1']:.3f}")

        if val_f1 > best_f1:
            best_f1, best_state, epochs_no_improve = val_f1, copy.deepcopy(model.state_dict()), 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"Early stopping на эпохе {epoch}")
                break

    model.load_state_dict(best_state)
    return model, history

class MOSEIBimodalDataset(Dataset):
    """Два X и две маски на сэмпл — для late/early/cross-modal fusion."""

    def __init__(self, text_X, text_mask, audio_X, audio_mask,
                 y_sent, y_emo, label_mask):
        self.text_X = torch.tensor(text_X, dtype=torch.float32)
        self.text_mask = torch.tensor(text_mask, dtype=torch.bool)
        self.audio_X = torch.tensor(audio_X, dtype=torch.float32)
        self.audio_mask = torch.tensor(audio_mask, dtype=torch.bool)
        self.y_sent = torch.tensor(y_sent, dtype=torch.long)
        y_emo_filled = np.nan_to_num(y_emo, nan=0.0)
        self.y_emo = torch.tensor(y_emo_filled, dtype=torch.float32)
        self.label_mask = torch.tensor(label_mask, dtype=torch.bool)

    def __len__(self):
        return len(self.text_X)

    def __getitem__(self, idx):
        return (self.text_X[idx], self.text_mask[idx],
                self.audio_X[idx], self.audio_mask[idx],
                self.y_sent[idx], self.y_emo[idx], self.label_mask[idx])


@torch.no_grad()
def evaluate_bimodal(model, loader, device):
    model.eval()
    sent_true_l, sent_pred_l, emo_true_l, emo_proba_l, lmask_l = [], [], [], [], []
    for tx, tm, ax_, am, y_sent, y_emo, label_mask in loader:
        tx, tm, ax_, am = tx.to(device), tm.to(device), ax_.to(device), am.to(device)
        sent_logits, emo_logits = model((tx, ax_), (tm, am))
        sent_true_l.append(y_sent.numpy())
        sent_pred_l.append(sent_logits.argmax(dim=1).cpu().numpy())
        emo_true_l.append(y_emo.numpy())
        emo_proba_l.append(torch.sigmoid(emo_logits).cpu().numpy())
        lmask_l.append(label_mask.numpy())

    sent_true = np.concatenate(sent_true_l)
    sent_pred = np.concatenate(sent_pred_l)
    emo_true = np.concatenate(emo_true_l)
    emo_proba = np.concatenate(emo_proba_l)
    label_mask = np.concatenate(lmask_l)
    sm = sentiment_metrics(sent_true, sent_pred)
    em = emotion_metrics(emo_true[label_mask], emo_proba[label_mask])
    return sm, em


def fit_bimodal(model, train_loader, valid_loader, device,
                sentiment_class_weights, emotion_pos_weights,
                epochs=30, lr=5e-4, lam=1.0, patience=5, verbose=True):
    model = model.to(device)
    sentiment_class_weights = sentiment_class_weights.to(device)
    emotion_pos_weights = emotion_pos_weights.to(device)

    sent_criterion = nn.CrossEntropyLoss(weight=sentiment_class_weights)
    emo_criterion = nn.BCEWithLogitsLoss(pos_weight=emotion_pos_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history, best_f1, best_state, epochs_no_improve = [], -1.0, None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sent_sum, loss_emo_sum, n_batches = 0.0, 0.0, 0

        for tx, tm, ax_, am, y_sent, y_emo, label_mask in train_loader:
            tx, tm, ax_, am = tx.to(device), tm.to(device), ax_.to(device), am.to(device)
            y_sent, y_emo, label_mask = y_sent.to(device), y_emo.to(device), label_mask.to(device)

            optimizer.zero_grad()
            sent_logits, emo_logits = model((tx, ax_), (tm, am))
            loss, l_sent, l_emo = masked_multitask_loss(
                sent_logits, emo_logits, y_sent, y_emo, label_mask,
                sent_criterion, emo_criterion, lam=lam,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_sent_sum += l_sent
            loss_emo_sum += l_emo
            n_batches += 1

        sm_val, em_val = evaluate_bimodal(model, valid_loader, device)
        val_f1 = sm_val["macro_f1_3class"]
        history.append({
            "epoch": epoch,
            "train_loss_sent": loss_sent_sum / n_batches,
            "train_loss_emo": loss_emo_sum / n_batches,
            "val_sentiment_macro_f1": val_f1,
            "val_emotion_macro_f1": em_val["macro_f1"],
            "val_emotion_mean_auc": em_val["mean_auc"],
        })
        if verbose:
            print(f"[epoch {epoch:2d}] train_loss_sent={loss_sent_sum/n_batches:.3f} "
                  f"train_loss_emo={loss_emo_sum/n_batches:.3f} | "
                  f"val sentiment macro-F1={val_f1:.3f} emotion macro-F1={em_val['macro_f1']:.3f}")

        if val_f1 > best_f1:
            best_f1, best_state, epochs_no_improve = val_f1, copy.deepcopy(model.state_dict()), 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"Early stopping на эпохе {epoch}")
                break

    model.load_state_dict(best_state)
    return model, history