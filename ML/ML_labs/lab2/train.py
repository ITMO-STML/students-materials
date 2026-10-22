import os

import copy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from PIL import Image
from sklearn.metrics import accuracy_score

from model import CLSCrossAttentionFusion
from data import ProductDataset, image_transform, extract_gender, download_data_if_needed
from report import generate_training_report

RANDOM_STATE = 42

DATA_DIR: Path = Path("data")
CSV_PATH: Path = DATA_DIR / "data.csv"
CKPT_PATH: Path = Path("checkpoints/cls_xattn.pt")
CKPT_PATH.parent.mkdir(exist_ok=True)

device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


def set_seed(seed: int = RANDOM_STATE) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def prepare_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, LabelEncoder]:
    df: pd.DataFrame = pd.read_csv(CSV_PATH)

    df["gender_label"] = df["display name"].apply(extract_gender)
    df = df[df["gender_label"] != 3].copy()
    print(df["gender_label"].value_counts())

    top: pd.Index = df["category"].value_counts().head(10).index
    df = df[df["category"].isin(top)].copy()
    df = (df.groupby("category", group_keys=False)
            .apply(lambda x: x.sample(min(len(x), 500), random_state=RANDOM_STATE))
            .reset_index(drop=True))

    train_df: pd.DataFrame
    temp_df: pd.DataFrame
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["category"], random_state=RANDOM_STATE
    )
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["category"], random_state=RANDOM_STATE
    )

    le: LabelEncoder = LabelEncoder()
    train_df["label"] = le.fit_transform(train_df["category"])
    val_df["label"]   = le.transform(val_df["category"])
    test_df["label"]  = le.transform(test_df["category"])
    return train_df, val_df, test_df, le

def prepare_batch(batch: dict, tokenizer: PreTrainedTokenizerBase, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    images: torch.Tensor = batch["image"].to(device)
    labels: torch.Tensor = batch["label"].to(device)
    genders: torch.Tensor = batch["gender"].to(device)
    
    text_inputs: Any = tokenizer(
        batch["text"], padding=True, truncation=True,
        max_length=128, return_tensors="pt"
    )
    return (images, text_inputs["input_ids"].to(device),
            text_inputs["attention_mask"].to(device), labels, genders)

@torch.no_grad()
def evaluate(model: CLSCrossAttentionFusion, loader: DataLoader, tokenizer: PreTrainedTokenizerBase, criterion: nn.Module, device: torch.device) -> Tuple[float, float, float]:
    model.eval()
    total_loss: float = 0.0
    correct_cat: int = 0
    correct_gen: int = 0
    total: int = 0
    for batch in loader:
        images: torch.Tensor
        ids: torch.Tensor
        mask: torch.Tensor
        labels: torch.Tensor
        genders: torch.Tensor
        images, ids, mask, labels, genders = prepare_batch(batch, tokenizer, device)
        
        logits_cat: torch.Tensor
        logits_gen: torch.Tensor
        logits_cat, logits_gen = model(images, ids, mask)
        
        loss: torch.Tensor = criterion(logits_cat, labels) + criterion(logits_gen, genders)
        total_loss += loss.item()
        
        correct_cat += (logits_cat.argmax(1) == labels).sum().item()
        correct_gen += (logits_gen.argmax(1) == genders).sum().item()
        total += labels.size(0)
        
    return total_loss / len(loader), correct_cat / total, correct_gen / total


@torch.no_grad()
def predict(model: CLSCrossAttentionFusion, loader: DataLoader, tokenizer: PreTrainedTokenizerBase, device: torch.device) -> Tuple[List[int], List[int], List[int], List[int]]:
    model.eval()
    all_true_cat: List[int] = []
    all_pred_cat: List[int] = []
    all_true_gen: List[int] = []
    all_pred_gen: List[int] = []
    
    for batch in loader:
        images: torch.Tensor
        ids: torch.Tensor
        mask: torch.Tensor
        labels: torch.Tensor
        genders: torch.Tensor
        images, ids, mask, labels, genders = prepare_batch(batch, tokenizer, device)
        
        logits_cat: torch.Tensor
        logits_gen: torch.Tensor
        logits_cat, logits_gen = model(images, ids, mask)
        
        preds_cat: torch.Tensor = logits_cat.argmax(1)
        preds_gen: torch.Tensor = logits_gen.argmax(1)
        
        all_true_cat.extend(labels.cpu().numpy())
        all_pred_cat.extend(preds_cat.cpu().numpy())
        
        all_true_gen.extend(genders.cpu().numpy())
        all_pred_gen.extend(preds_gen.cpu().numpy())
        
    return all_true_cat, all_pred_cat, all_true_gen, all_pred_gen


def main() -> None:
    download_data_if_needed() 
    set_seed(42)

    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    le: LabelEncoder
    train_df, val_df, test_df, le = prepare_data()
    print("Train/Val/Test:", train_df.shape, val_df.shape, test_df.shape)

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    train_loader: DataLoader = DataLoader(
        ProductDataset(train_df, DATA_DIR, image_transform),
        batch_size=32, shuffle=True, num_workers=0
    )
    val_loader: DataLoader = DataLoader(
        ProductDataset(val_df, DATA_DIR, image_transform),
        batch_size=32, shuffle=False, num_workers=0
    )
    test_loader: DataLoader = DataLoader(
        ProductDataset(test_df, DATA_DIR, image_transform),
        batch_size=32, shuffle=False, num_workers=0
    )

    model: CLSCrossAttentionFusion = CLSCrossAttentionFusion(num_classes=10).to(device)

    criterion: nn.Module = nn.CrossEntropyLoss()
    optimizer: torch.optim.AdamW = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-4, weight_decay=1e-4
    )
    scheduler: torch.optim.lr_scheduler.CosineAnnealingLR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

    from collections import deque
    val_window: deque = deque(maxlen=3)
    best_val: float = 0.0
    best_state: Any = None
    best_epoch: int = -1
    patience: int = 5
    min_delta: float = 1e-4
    no_improve: int = 0

    epochs: int = 15

    history: Dict[str, List[float]] = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_acc_smooth": [],
    }

    for epoch in range(epochs):
        model.train()
        total_loss: float = 0.0
        correct_cat: int = 0
        correct_gen: int = 0
        total: int = 0

        for batch in train_loader:
            images: torch.Tensor
            ids: torch.Tensor
            mask: torch.Tensor
            labels: torch.Tensor
            genders: torch.Tensor
            images, ids, mask, labels, genders = prepare_batch(batch, tokenizer, device)
            
            optimizer.zero_grad()
            logits_cat: torch.Tensor
            logits_gen: torch.Tensor
            logits_cat, logits_gen = model(images, ids, mask)
            
            loss_cat: torch.Tensor = criterion(logits_cat, labels)
            loss_gen: torch.Tensor = criterion(logits_gen, genders)
            loss: torch.Tensor = loss_cat + 0.5 * loss_gen 
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            correct_cat += (logits_cat.argmax(1) == labels).sum().item()
            correct_gen += (logits_gen.argmax(1) == genders).sum().item()
            total += labels.size(0)    

        tr_loss: float = total_loss / len(train_loader)
        tr_acc: float = correct_cat / total
        va_loss: float
        va_acc: float
        va_acc_gen: float
        va_loss, va_acc, va_acc_gen = evaluate(model, val_loader, tokenizer, criterion, device)

        val_window.append(va_acc)
        va_smooth: float = sum(val_window) / len(val_window)

        improved: bool = va_smooth > best_val + min_delta
        if improved:
            best_val = va_smooth
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["val_acc_smooth"].append(va_smooth)

        flag: str = "*best*" if improved else f"(no improve {no_improve}/{patience})"
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
    history["best_val_acc"] = best_val
    history["best_epoch"] = best_epoch

    te_loss: float
    te_acc: float
    te_acc_gen: float
    te_loss, te_acc, te_acc_gen = evaluate(model, test_loader, tokenizer, criterion, device)
    print(f"Test: loss={te_loss:.4f}  acc_cat={te_acc:.4f}  acc_gen={te_acc_gen:.4f}")

    torch.save({
        "model_state": model.state_dict(),
        "label_classes": list(le.classes_),
        "best_val": best_val,
        "best_epoch": best_epoch,
    }, CKPT_PATH)
    print(f"Checkpoint saved to {CKPT_PATH}")

    y_true_cat: List[int]
    y_pred_cat: List[int]
    y_true_gen: List[int]
    y_pred_gen: List[int]
    y_true_cat, y_pred_cat, y_true_gen, y_pred_gen = predict(model, test_loader, tokenizer, device)

    config: Dict[str, Any] = {
        "Model": "CLSCrossAttentionFusion",
        "Image encoder": "ResNet18 (frozen)",
        "Text encoder": "DistilBERT (frozen)",
        "Attention": "Cross-attention, 8 heads, embed_dim=768",
        "Classifier": "MLP 768→256→10",
        "Epochs": epochs,
        "Batch size": 32,
        "Optimizer": "AdamW (lr=5e-4, weight_decay=1e-4)",
        "Scheduler": "CosineAnnealingLR (T_max=15)",
        "Early stopping": "patience=5, smooth_window=3",
        "Device": str(device),
        "Seed": 42,
    }

    generate_training_report(
        history=history,
        y_true_cat=y_true_cat,
        y_pred_cat=y_pred_cat,
        y_true_gen=y_true_gen,
        y_pred_gen=y_pred_gen,
        test_loss=te_loss,
        test_acc_cat=te_acc,
        test_acc_gen=te_acc_gen,
        config=config,
    )


if __name__ == "__main__":
    main()