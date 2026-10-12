import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer

from model import CLSCrossAttentionFusion
from data import ProductDataset, image_transform


DATA_DIR = Path("data")
CSV_PATH = DATA_DIR / "data.csv"
CKPT_PATH = Path("checkpoints/cls_xattn.pt")
CKPT_PATH.parent.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def prepare_data():
    df = pd.read_csv(CSV_PATH)
    top = df["category"].value_counts().head(10).index
    df = df[df["category"].isin(top)].copy()
    df = (df.groupby("category", group_keys=False)
            .apply(lambda x: x.sample(min(len(x), 500), random_state=42))
            .reset_index(drop=True))

    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["category"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["category"], random_state=42
    )

    le = LabelEncoder()
    train_df["label"] = le.fit_transform(train_df["category"])
    val_df["label"]   = le.transform(val_df["category"])
    test_df["label"]  = le.transform(test_df["category"])
    return train_df, val_df, test_df, le


def prepare_batch(batch, tokenizer, device):
    images = batch["image"].to(device)
    labels = batch["label"].to(device)
    text_inputs = tokenizer(
        batch["text"], padding=True, truncation=True,
        max_length=128, return_tensors="pt"
    )
    return (images,
            text_inputs["input_ids"].to(device),
            text_inputs["attention_mask"].to(device),
            labels)


@torch.no_grad()
def evaluate(model, loader, tokenizer, criterion, device):  # Возвращает loss и accuracy по всей выборке
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        images, ids, mask, labels = prepare_batch(batch, tokenizer, device)
        outputs = model(images, ids, mask)
        total_loss += criterion(outputs, labels).item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), correct / total


def main():
    set_seed(42)

    train_df, val_df, test_df, le = prepare_data()
    print("Train/Val/Test:", train_df.shape, val_df.shape, test_df.shape)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    train_loader = DataLoader(
        ProductDataset(train_df, DATA_DIR, image_transform),
        batch_size=32, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        ProductDataset(val_df, DATA_DIR, image_transform),
        batch_size=32, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        ProductDataset(test_df, DATA_DIR, image_transform),
        batch_size=32, shuffle=False, num_workers=0
    )

    model = CLSCrossAttentionFusion(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-4, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

    # Early stopping по сглаженному val_acc
    from collections import deque
    val_window = deque(maxlen=3)
    best_val, best_state, best_epoch = 0.0, None, -1
    patience, min_delta, no_improve = 5, 1e-4, 0

    epochs = 15
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for batch in train_loader:
            images, ids, mask, labels = prepare_batch(batch, tokenizer, device)
            optimizer.zero_grad()
            outputs = model(images, ids, mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        tr_loss, tr_acc = total_loss / len(train_loader), correct / total
        va_loss, va_acc = evaluate(model, val_loader, tokenizer, criterion, device)

        val_window.append(va_acc)
        va_smooth = sum(val_window) / len(val_window)

        improved = va_smooth > best_val + min_delta
        if improved:
            best_val, best_epoch = va_smooth, epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        flag = "*best*" if improved else f"(no improve {no_improve}/{patience})"
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train {tr_loss:.4f}/{tr_acc:.4f} | "
              f"Val {va_loss:.4f}/{va_acc:.4f} | "
              f"Smoothed {va_smooth:.4f} {flag}")

        scheduler.step()

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    print(f"Best smoothed val: {best_val:.4f} @ epoch {best_epoch}")

    te_loss, te_acc = evaluate(model, test_loader, tokenizer, criterion, device)
    print(f"Test: loss={te_loss:.4f}  acc={te_acc:.4f}")

    torch.save({
        "model_state": model.state_dict(),
        "label_classes": list(le.classes_),
        "best_val": best_val,
        "best_epoch": best_epoch,
    }, CKPT_PATH)
    print(f"Checkpoint saved to {CKPT_PATH}")


if __name__ == "__main__":
    main()