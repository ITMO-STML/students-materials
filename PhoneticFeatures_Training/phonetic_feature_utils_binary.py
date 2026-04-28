import math
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset


FEATURE_NAMES = [
    "syl",
    "son",
    "cons",
    "cont",
    "delrel",
    "lat",
    "nas",
    "strid",
    "voi",
    "sg",
    "cg",
    "ant",
    "cor",
    "distr",
    "lab",
    "hi",
    "lo",
    "back",
    "round",
    "velaric",
    "tense",
    "long",
    "add",
    "other",
]


class PhonemeEmbeddingDataset(Dataset):
    def __init__(
        self,
        data,
        phoneme_list,
        corpres_to_ipa_symbol=None,
        panphon_features=None,
        triplet_labels=False,
        target_position=1,
    ):
        super().__init__()
        self.data = data
        self.triplet_labels = triplet_labels
        self.target_position = target_position
        self.phoneme_list = phoneme_list
        self.phoneme2idx = {ph: i for i, ph in enumerate(self.phoneme_list)}
        self.corpres_to_ipa_symbol = corpres_to_ipa_symbol
        self.panphon_features = panphon_features
        self.phoneme_feature_cache = {}

        if corpres_to_ipa_symbol is not None and panphon_features is not None:
            self._precompute_features()

    @staticmethod
    def _encode_features(vector):
        mapping = {-1: 0, 1: 1, 0: 2}
        return [mapping[v] for v in vector]

    def _precompute_features(self):
        for phoneme in self.phoneme_list:
            ipa_symbol = self.corpres_to_ipa_symbol(phoneme)
            panphon_vector = self.panphon_features(ipa_symbol)

            if panphon_vector:
                features = self._encode_features(panphon_vector[0].numeric())
            else:
                features = None

            self.phoneme_feature_cache[phoneme] = {
                "ipa": ipa_symbol,
                "features": features,
            }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        embedding = sample["embedding"]
        phonemes = sample["label"] if self.triplet_labels else [sample["label"][self.target_position]]

        phoneme_indices = []
        ipa_symbols = []
        phonetic_features = []

        for phoneme in phonemes:
            phoneme_indices.append(self.phoneme2idx.get(phoneme, -1))
            cached = self.phoneme_feature_cache.get(phoneme)

            if cached is None or cached["features"] is None:
                ipa_symbols.append(phoneme)
                phonetic_features.append(None)
            else:
                ipa_symbols.append(cached["ipa"])
                phonetic_features.append(cached["features"])

        phoneme_labels = torch.tensor(phoneme_indices, dtype=torch.long)
        feature_tensor = None

        if phonetic_features and phonetic_features[0] is not None:
            feature_tensor = torch.tensor(phonetic_features, dtype=torch.long)
            if not self.triplet_labels:
                feature_tensor = feature_tensor[0]

        return embedding, phoneme_labels, phonemes, ipa_symbols, feature_tensor


def compute_class_weights(dataset, num_features):
    all_targets = [features for _, _, _, _, features in dataset if features is not None]
    all_targets = torch.stack(all_targets)
    weights_by_feature = {}

    for feature_idx in range(num_features):
        values = all_targets[:, feature_idx]
        counts = torch.bincount(values, minlength=int(values.max().item()) + 1).float()
        nonzero_mask = counts > 0
        present_classes = int(nonzero_mask.sum().item())
        total = counts.sum().item()

        weights = torch.ones_like(counts)
        weights[nonzero_mask] = total / (present_classes * counts[nonzero_mask])
        weights_by_feature[feature_idx] = weights

    return weights_by_feature


def infer_num_classes(dataset):
    all_targets = [features for _, _, _, _, features in dataset if features is not None]
    all_targets = torch.stack(all_targets)
    return [int(all_targets[:, idx].max().item()) + 1 for idx in range(all_targets.shape[1])]


class PhonemeFeaturePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes_per_feature, active_heads=None):
        super().__init__()
        self.num_classes_per_feature = list(num_classes_per_feature)
        self.num_features = len(self.num_classes_per_feature)
        self.active_heads = sorted(active_heads or list(range(self.num_features)))

        invalid_heads = [idx for idx in self.active_heads if idx < 0 or idx >= self.num_features]
        if invalid_heads:
            raise ValueError(f"Active heads out of range: {invalid_heads}")

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_dim, self.num_classes_per_feature[idx]) for idx in range(self.num_features)]
        )

    def forward(self, x):
        shared = self.backbone(x)
        return [head(shared) if idx in self.active_heads else None for idx, head in enumerate(self.heads)]


def compute_loss(outputs, targets, active_heads, criterion_dict):
    total_loss = 0
    losses_per_feature = {}

    for feature_idx in active_heads:
        logits = outputs[feature_idx]
        target = targets[:, feature_idx]

        loss = criterion_dict[feature_idx](logits, target)
        total_loss += loss
        losses_per_feature[feature_idx] = loss.item()

    return total_loss, losses_per_feature


def compute_accuracy(outputs, targets, active_heads):
    accuracies = {}

    for feature_idx in active_heads:
        logits = outputs[feature_idx]
        predictions = torch.argmax(logits, dim=1)
        accuracies[feature_idx] = (predictions == targets[:, feature_idx]).float().mean().item()

    return sum(accuracies.values()) / len(accuracies), accuracies

def compute_balanced_accuracy(outputs, targets, active_heads, ignore_index=2):
    per_feature_balanced_acc = {}

    for feature_idx in active_heads:
        logits = outputs[feature_idx]
        preds = torch.argmax(logits, dim=1)
        true = targets[:, feature_idx]

        valid_mask = true != ignore_index
        if valid_mask.sum().item() == 0:
            per_feature_balanced_acc[feature_idx] = float("nan")
            continue

        preds = preds[valid_mask]
        true = true[valid_mask]

        recalls = []
        for cls in torch.unique(true):
            cls_mask = true == cls
            recall = (preds[cls_mask] == true[cls_mask]).float().mean().item()
            recalls.append(recall)

        per_feature_balanced_acc[feature_idx] = float(sum(recalls) / len(recalls))

    valid_values = [v for v in per_feature_balanced_acc.values() if not math.isnan(v)]
    overall_balanced_acc = float(sum(valid_values) / len(valid_values)) if valid_values else float("nan")

    return overall_balanced_acc, per_feature_balanced_acc



@torch.no_grad()
# def evaluate(model, loader, device, active_heads, criterion_dict):
#     model.eval()
#     total_loss = 0.0
#     total_acc = 0.0
#     per_feature_acc = {feature_idx: 0.0 for feature_idx in active_heads}
#
#     for embeddings, _, _, _, features in loader:
#         embeddings = embeddings.to(device)
#         targets = features.to(device)
#         outputs = model(embeddings)
#
#         loss, _ = compute_loss(outputs, targets, active_heads, criterion_dict)
#         overall_acc, acc_dict = compute_balanced_accuracy(outputs, targets, active_heads)
#
#         total_loss += loss.item()
#         total_acc += overall_acc
#         for feature_idx in active_heads:
#             per_feature_acc[feature_idx] += acc_dict[feature_idx]
#
#     batches = len(loader)
#     return (
#         total_loss / batches,
#         total_acc / batches,
#         {feature_idx: per_feature_acc[feature_idx] / batches for feature_idx in active_heads},
#     )

@torch.no_grad()
def evaluate(model, loader, device, active_heads, criterion_dict, ignore_index=2):
    model.eval()

    total_loss = 0.0
    all_preds = {i: [] for i in active_heads}
    all_targets = {i: [] for i in active_heads}

    for embeddings, _, _, _, features in loader:
        embeddings = embeddings.to(device)
        targets = features.to(device)

        outputs = model(embeddings)
        loss, _ = compute_loss(outputs, targets, active_heads, criterion_dict)
        total_loss += loss.item()

        for i in active_heads:
            logits = outputs[i]
            preds = torch.argmax(logits, dim=1)

            valid_mask = targets[:, i] != ignore_index
            if valid_mask.sum().item() == 0:
                continue

            all_preds[i].append(preds[valid_mask].cpu())
            all_targets[i].append(targets[valid_mask, i].cpu())

    per_feature_acc = {}
    per_feature_balanced_acc = {}

    for i in active_heads:
        if not all_targets[i]:
            per_feature_acc[i] = float("nan")
            per_feature_balanced_acc[i] = float("nan")
            continue

        preds = torch.cat(all_preds[i])
        true = torch.cat(all_targets[i])

        per_feature_acc[i] = (preds == true).float().mean().item()

        recalls = []
        for cls in torch.unique(true):
            cls_mask = true == cls
            recall = (preds[cls_mask] == true[cls_mask]).float().mean().item()
            recalls.append(recall)

        per_feature_balanced_acc[i] = sum(recalls) / len(recalls)

    valid_acc = [v for v in per_feature_acc.values() if not math.isnan(v)]
    valid_bal_acc = [v for v in per_feature_balanced_acc.values() if not math.isnan(v)]

    overall_acc = sum(valid_acc) / len(valid_acc) if valid_acc else float("nan")
    overall_balanced_acc = sum(valid_bal_acc) / len(valid_bal_acc) if valid_bal_acc else float("nan")

    return {
        "loss": total_loss / len(loader),
        "accuracy": overall_acc,
        "balanced_accuracy": overall_balanced_acc,
        "per_feature_acc": per_feature_acc,
        "per_feature_balanced_acc": per_feature_balanced_acc,
    }


def evaluate_before_training(model, loader, device, active_heads, criterion_dict):
    results = evaluate(model, loader, device, active_heads, criterion_dict)
    loss, acc, balanced_acc, feature_acc, feature_balanced_acc = results['loss'], results['accuracy'], results['balanced_accuracy'], results['per_feature_acc'], results['per_feature_balanced_acc'],
    print("\nBaseline before training:")
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    for feature_idx, value in feature_acc.items():
        print(f"Feature {feature_idx} acc: {value:.4f} balanced acc {balanced_acc:.4f}")
    return loss, acc, feature_acc


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    active_heads,
    criterion_dict,
    epochs=10,
    eval_steps_per_epoch=4,
    save_model_path=None,
):
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_feature_acc": {feature_idx: [] for feature_idx in active_heads},
        "val_feature_acc": {feature_idx: [] for feature_idx in active_heads},
        "steps": [],
    }
    best_val_loss = float("inf")
    step = 0
    eval_interval = max(1, len(train_loader) // max(1, eval_steps_per_epoch))

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        per_feature_acc = {feature_idx: 0.0 for feature_idx in active_heads}
        last_val_loss = None

        for batch_idx, (embeddings, _, _, _, features) in enumerate(train_loader):
            embeddings = embeddings.to(device)
            targets = features.to(device)

            optimizer.zero_grad()
            outputs = model(embeddings)
            loss, _ = compute_loss(outputs, targets, active_heads, criterion_dict)
            loss.backward()
            optimizer.step()

            overall_acc, acc_dict = compute_balanced_accuracy(outputs, targets, active_heads)
            running_loss += loss.item()
            running_acc += overall_acc

            for feature_idx in active_heads:
                per_feature_acc[feature_idx] += acc_dict[feature_idx]

            step += 1
            should_eval = ((batch_idx + 1) % eval_interval == 0) or (batch_idx + 1 == len(train_loader))
            if not should_eval:
                continue

            batches_seen = batch_idx + 1
            train_loss = running_loss / batches_seen
            train_acc = running_acc / batches_seen
            train_feature_acc = {
                feature_idx: per_feature_acc[feature_idx] / batches_seen for feature_idx in active_heads
            }

            val_loss, val_acc, val_feature_acc = evaluate(
                model,
                val_loader,
                device,
                active_heads,
                criterion_dict,
            )
            last_val_loss = val_loss

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            for feature_idx in active_heads:
                history["train_feature_acc"][feature_idx].append(train_feature_acc[feature_idx])
                history["val_feature_acc"][feature_idx].append(val_feature_acc[feature_idx])
            history["steps"].append(step)

            print(
                f"Epoch {epoch + 1} | step {batch_idx + 1}/{len(train_loader)} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"train_acc={train_acc:.4f} | val_acc={val_acc:.4f}"
            )

        if scheduler is not None:
            scheduler.step()

        if save_model_path is not None and last_val_loss is not None and last_val_loss < best_val_loss:
            best_val_loss = last_val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                    "best_val_loss": best_val_loss,
                    "model_config": {
                        "input_dim": model.backbone[0].in_features,
                        "hidden_dim": model.backbone[0].out_features,
                        "num_classes_per_feature": model.num_classes_per_feature,
                        "active_heads": model.active_heads,
                    },
                },
                save_model_path,
            )
            print(f"  -> Best model saved at epoch {epoch + 1} with val_loss={best_val_loss:.4f}")

    return history


def plot_training(history, baseline_loss, baseline_acc):
    steps = history["steps"]
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(0, baseline_loss, label="baseline")
    plt.plot(steps, history["train_loss"], label="train")
    plt.plot(steps, history["val_loss"], label="val")
    plt.title("Loss")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(0, baseline_acc, label="baseline")
    plt.plot(steps, history["train_acc"], label="train")
    plt.plot(steps, history["val_acc"], label="val")
    plt.title("Accuracy")
    plt.xlabel("step")
    plt.ylabel("accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_feature_accuracy(history, baseline_feature_acc, feature_names, active_heads):
    for feature_idx in active_heads:
        if feature_idx not in baseline_feature_acc:
            continue

        plt.figure()
        plt.scatter(0, baseline_feature_acc[feature_idx], label="baseline")
        plt.plot(history["steps"], history["train_feature_acc"][feature_idx], label="train")
        plt.plot(history["steps"], history["val_feature_acc"][feature_idx], label="val")
        plt.title(f"Feature '{feature_names[feature_idx]}' accuracy")
        plt.xlabel("step")
        plt.ylabel("accuracy")
        plt.legend()
        plt.show()


@torch.no_grad()
def compute_confusion_matrices(model, loader, device, active_heads):
    model.eval()
    all_preds = {feature_idx: [] for feature_idx in active_heads}
    all_targets = {feature_idx: [] for feature_idx in active_heads}

    for embeddings, _, _, _, features in loader:
        embeddings = embeddings.to(device)
        targets = features.to(device)
        outputs = model(embeddings)

        for feature_idx in active_heads:
            predictions = torch.argmax(outputs[feature_idx], dim=1)
            all_preds[feature_idx].extend(predictions.cpu().tolist())
            all_targets[feature_idx].extend(targets[:, feature_idx].cpu().tolist())

    result = {}
    for feature_idx in active_heads:
        labels = sorted(set(all_targets[feature_idx]) | set(all_preds[feature_idx]))
        result[feature_idx] = {
            "matrix": confusion_matrix(all_targets[feature_idx], all_preds[feature_idx], labels=labels),
            "labels": labels,
        }

    return result


def _init_heatmap_axes(num_plots):
    rows = int(math.ceil(num_plots / 3))
    fig, axes = plt.subplots(ncols=3, nrows=rows, figsize=(18, 5 * rows))
    return fig, np.atleast_1d(axes).flatten()


def plot_confusion_matrices(confusion_matrices, feature_names):
    fig, axes = _init_heatmap_axes(len(confusion_matrices))

    for plot_idx, (feature_idx, payload) in enumerate(confusion_matrices.items()):
        sns.heatmap(
            payload["matrix"],
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes[plot_idx],
            cbar_kws={"shrink": 0.8},
            xticklabels=payload["labels"],
            yticklabels=payload["labels"],
        )
        axes[plot_idx].set_title(f"Confusion Matrix: {feature_names[feature_idx]}")
        axes[plot_idx].set_xlabel("Predicted")
        axes[plot_idx].set_ylabel("True")

    for plot_idx in range(len(confusion_matrices), len(axes)):
        fig.delaxes(axes[plot_idx])

    plt.tight_layout()
    plt.show()


def plot_confusion_matrices_norm(confusion_matrices, feature_names):
    fig, axes = _init_heatmap_axes(len(confusion_matrices))

    for plot_idx, (feature_idx, payload) in enumerate(confusion_matrices.items()):
        matrix = payload["matrix"]
        row_sums = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums != 0) * 100

        sns.heatmap(
            normalized,
            annot=True,
            fmt=".1f",
            cmap="Blues",
            ax=axes[plot_idx],
            cbar_kws={"label": "%"},
            xticklabels=payload["labels"],
            yticklabels=payload["labels"],
            vmin=0,
            vmax=100,
        )
        axes[plot_idx].set_title(f"Confusion Matrix: {feature_names[feature_idx]}\n(Row %)")
        axes[plot_idx].set_xlabel("Predicted")
        axes[plot_idx].set_ylabel("True")

    for plot_idx in range(len(confusion_matrices), len(axes)):
        fig.delaxes(axes[plot_idx])

    plt.tight_layout()
    plt.show()


def inspect_single_prediction(model, dataset, device, active_heads, feature_names, skip_symbols=None):
    skip_symbols = set(skip_symbols or [])
    model.eval()

    with torch.no_grad():
        for embeddings, _, _, ipa_symbols, features in dataset:
            symbol = ipa_symbols[0]
            if symbol in skip_symbols:
                continue

            outputs = model(embeddings.to(device))
            print(f"Symbol: {symbol}")

            for feature_idx in active_heads:
                logits = outputs[feature_idx]
                pred = torch.argmax(logits, dim=-1).item() if logits.dim() == 1 else torch.argmax(logits, dim=1).item()
                print(feature_names[feature_idx], int(features[feature_idx].item()), pred)
            break


def collect_symbol_prediction_stats(model, dataset, device, active_heads):
    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    true_classes = {}

    model.eval()
    with torch.no_grad():
        for embeddings, _, _, ipa_symbols, features in dataset:
            symbol = ipa_symbols[0]
            outputs = model(embeddings.to(device))

            for feature_idx in active_heads:
                logits = outputs[feature_idx]
                pred = torch.argmax(logits, dim=-1).item() if logits.dim() == 1 else torch.argmax(logits, dim=1).item()
                stats[symbol][feature_idx][pred] += 1
                true_classes[(symbol, feature_idx)] = int(features[feature_idx].item())

    return stats, true_classes


def build_prediction_rows(stats, true_classes):
    rows = []
    for symbol, heads_dict in stats.items():
        for head_id, class_counts in heads_dict.items():
            for pred_class, count in class_counts.items():
                rows.append(
                    {
                        "symbol": symbol,
                        "head": head_id,
                        "class": pred_class,
                        "count": count,
                        "true_class": true_classes[(symbol, head_id)],
                    }
                )
    return rows


def plot_symbol_prediction_distributions(prediction_rows, feature_names):
    df = pd.DataFrame(prediction_rows)

    for symbol, subdf in df.groupby("symbol"):
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(data=subdf, x="head", y="count", hue="class")

        true_sub = subdf[["head", "true_class"]].drop_duplicates().sort_values("head")
        for plot_pos, (_, row) in enumerate(true_sub.iterrows()):
            ax.scatter(plot_pos, 0, color="red", s=80, zorder=10)

        ax.set_title(f'Prediction distribution for symbol "{symbol}"')
        ax.set_xlabel("Head")
        ax.set_ylabel("Count")
        ax.set_xticks(range(len(true_sub)))
        ax.set_xticklabels([feature_names[int(head)] for head in true_sub["head"]], rotation=90)
        ax.legend(title="Predicted class")
        plt.tight_layout()
        plt.show()
