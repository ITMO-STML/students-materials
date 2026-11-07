"""Оценка DDPM: обучение классификатора на MNIST, генерация, метрики."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.optim import Adam

from data import get_dataloaders
from diffusion import Diffusion
from model import UNet

OUTPUTS_DIR = Path("/app/outputs")
N_PER_CLASS = 100   # столько генерировать на каждый класс
CLASSIFIER_EPOCHS = 3
T_DIFF = 1000


class SimpleCNN(nn.Module):
    """CNN для классификации MNIST"""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def train_classifier(device: torch.device) -> nn.Module:
    ckpt = OUTPUTS_DIR / "classifier.pt"

    model = SimpleCNN().to(device)
    # if ckpt.exists():
    #     model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    #     return model

    train_loader, _ = get_dataloaders(batch_size=128, num_workers=0)
    optimizer = Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, CLASSIFIER_EPOCHS + 1):
        model.train()
        total, correct, running = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

        print(f"Classifier epoch {epoch}/{CLASSIFIER_EPOCHS} | "
              f"loss={running / len(train_loader):.4f} | acc={correct / total:.4f}")

    # Оценка на тесте
    _, test_loader = get_dataloaders(batch_size=128, num_workers=0)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    print(f"Classifier test accuracy: {correct / total:.4f}")    

    torch.save(model.state_dict(), ckpt)
    return model


def run_evaluation(device: torch.device) -> None:

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    classifier = train_classifier(device)
    classifier.eval()

    ddpm = UNet().to(device)
    ddpm.load_state_dict(torch.load(OUTPUTS_DIR / "checkpoint.pt", map_location=device, weights_only=True))
    ddpm.eval()
    diffusion = Diffusion(T=T_DIFF, device=device)

    # Генерация по классам
    all_images = []
    all_labels = []
    for cls in range(10):
        labels = torch.full((N_PER_CLASS,), cls, dtype=torch.long, device=device)
        samples = diffusion.sample(ddpm, N_PER_CLASS, labels)
        samples = (samples + 1.0) / 2.0 # [-1, 1] в [0, 1]
        all_images.append(samples.cpu())
        all_labels.extend([cls] * N_PER_CLASS)

    images = torch.cat(all_images, dim=0)
    labels_np = np.array(all_labels)

    # Классификация сгенерированных кртинок
    preds = []
    with torch.no_grad():
        for i in range(0, len(images), 256):
            batch = images[i:i + 256].to(device)
            batch = batch * 2.0 - 1.0 # [-1, 1]
            preds.append(classifier(batch).argmax(1).cpu().numpy())
    preds_np = np.concatenate(preds)

    report = classification_report(labels_np, preds_np, target_names=[str(i) for i in range(10)], digits=4)
    cm = confusion_matrix(labels_np, preds_np)

    print("\nClassification report:")
    print(report)

    (OUTPUTS_DIR / "classification_report.txt").write_text(report, encoding="utf-8")

    with (OUTPUTS_DIR / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + [str(i) for i in range(10)])
        for i, row in enumerate(cm):
            writer.writerow([str(i)] + row.tolist())

    torch.save({"images": images, "labels": labels_np}, OUTPUTS_DIR / "generated.pt")