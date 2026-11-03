"""Сборка HTML-отчёта: EDA + loss + forward process + метрики."""
from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from jinja2 import Template
from sklearn.metrics import confusion_matrix
from torchvision import datasets, transforms

from diffusion import Diffusion
from model import UNet

OUTPUTS_DIR = Path("/app/outputs")
DATA_DIR = Path("/app/data")
T_DIFF = 1000


def fig_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def make_eda() -> str:
    ds = datasets.MNIST(root=str(DATA_DIR), train=False, download=False,
                        transform=transforms.ToTensor())
    fig, axes = plt.subplots(2, 10, figsize=(12, 3))
    for i in range(20):
        img, label = ds[i]
        ax = axes[i // 10, i % 10]
        ax.imshow(img.squeeze(), cmap="gray")
        ax.set_title(str(label), fontsize=9)
        ax.axis("off")
    fig.suptitle("Примеры изображений MNIST", fontsize=12)
    return fig_to_base64(fig)


def make_class_distribution() -> str:
    ds = datasets.MNIST(root=str(DATA_DIR), train=True, download=False,
                        transform=transforms.ToTensor())
    labels = [int(y) for _, y in ds]
    counts = np.bincount(labels, minlength=10)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(10), counts, color="steelblue")
    ax.set_xticks(range(10))
    ax.set_xlabel("Класс (цифра)")
    ax.set_ylabel("Количество изображений")
    ax.set_title("Распределение классов в train-выборке MNIST")
    for i, c in enumerate(counts):
        ax.text(i, c, str(c), ha="center", va="bottom", fontsize=8)
    return fig_to_base64(fig)


def make_loss_curve() -> str:
    df = pd.read_csv(OUTPUTS_DIR / "loss.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["epoch"], df["loss"], marker="o", color="crimson")
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("MSE loss")
    ax.set_title("Кривая обучения DDPM")
    ax.grid(alpha=0.3)
    return fig_to_base64(fig)


def make_forward_process(device: torch.device) -> str:
    ds = datasets.MNIST(root=str(DATA_DIR), train=False, download=False,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,)),
                        ]))
    x0, _ = ds[0]
    x0 = x0.unsqueeze(0).to(device)

    diffusion = Diffusion(T=T_DIFF, device=device)
    steps = [0, 100, 300, 500, 700, 999]

    fig, axes = plt.subplots(1, len(steps), figsize=(12, 2.5))
    for ax, t_val in zip(axes, steps):
        t = torch.tensor([t_val], device=device)
        noise = torch.randn_like(x0)
        x_t = diffusion.q_sample(x0, t, noise)
        img = ((x_t.squeeze().cpu().numpy() + 1) / 2).clip(0, 1)
        ax.imshow(img, cmap="gray")
        ax.set_title(f"t={t_val}")
        ax.axis("off")
    fig.suptitle("Forward process: зашумление цифры", fontsize=12)
    return fig_to_base64(fig)


def make_generated_grid() -> str:
    data = torch.load(OUTPUTS_DIR / "generated.pt", map_location="cpu", weights_only=False)
    images = data["images"].numpy()
    labels = data["labels"]

    # по одному представителю каждого класса в строке
    fig, axes = plt.subplots(10, 10, figsize=(8, 8))
    for cls in range(10):
        cls_imgs = images[labels == cls][:10]
        for j in range(10):
            ax = axes[cls, j]
            ax.imshow(cls_imgs[j].squeeze(), cmap="gray")
            ax.axis("off")
            if j == 0:
                ax.set_ylabel(str(cls), fontsize=12, rotation=0, labelpad=15)
    fig.suptitle("Сгенерированные изображения (строки - целевой класс)", fontsize=12)
    return fig_to_base64(fig)


def make_confusion_matrix() -> str:
    df = pd.read_csv(OUTPUTS_DIR / "confusion_matrix.csv", index_col=0)
    cm = df.values

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Предсказанный класс (классификатор)")
    ax.set_ylabel("Целевой класс (генерация)")
    ax.set_title("Confusion matrix: сгенерированные изображения")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    fig.colorbar(im, ax=ax)
    return fig_to_base64(fig)


def build_report() -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eda_img = make_eda()
    dist_img = make_class_distribution()
    loss_img = make_loss_curve()
    fwd_img = make_forward_process(device)
    gen_img = make_generated_grid()
    cm_img = make_confusion_matrix()

    report_txt = (OUTPUTS_DIR / "classification_report.txt").read_text(encoding="utf-8")

    template = Template("""
<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>Lab 4 — DDPM на MNIST</title>
<style>
  body { font-family: -apple-system, Segoe UI, sans-serif; max-width: 1000px;
         margin: 2em auto; padding: 0 1em; color: #222; line-height: 1.5; }
  h1, h2 { color: #1a1a1a; }
  h2 { border-bottom: 2px solid #eee; padding-bottom: 0.3em; margin-top: 2em; }
  img { max-width: 100%; display: block; margin: 1em 0; }
  pre { background: #f4f4f4; padding: 1em; overflow-x: auto;
        border-radius: 6px; font-size: 0.85em; }
</style>
</head>
<body>

<h1>Lab 4 - DDPM на MNIST (class-conditional)</h1>

<h2>1. EDA: датасет MNIST</h2>
<img src="data:image/png;base64,{{ eda_img }}" alt="EDA samples">
<img src="data:image/png;base64,{{ dist_img }}" alt="Class distribution">

<h2>2. Кривая обучения</h2>
<p>MSE loss по эпохам. Модель предсказывает добавленный шум на случайном шаге t.</p>
<img src="data:image/png;base64,{{ loss_img }}" alt="Loss curve">

<h2>3. Forward process</h2>
<p>Постепенное зашумление одной и той же цифры от t=0 до t=999.</p>
<img src="data:image/png;base64,{{ fwd_img }}" alt="Forward process">

<h2>4. Сгенерированные изображения</h2>
<img src="data:image/png;base64,{{ gen_img }}" alt="Generated samples">

<h2>5. Метрики качества генерации</h2>
<p>Независимый CNN-классификатор (обучен на MNIST, ~99% accuracy)
проверяет, к какому классу относятся сгенерированные изображения.</p>
<p><b>Classification report:</b></p>
<pre>{{ report_txt }}</pre>
<p><b>Confusion matrix:</b></p>
<img src="data:image/png;base64,{{ cm_img }}" alt="Confusion matrix">

</body>
</html>
""")

    html = template.render(
        eda_img=eda_img,
        dist_img=dist_img,
        loss_img=loss_img,
        fwd_img=fwd_img,
        gen_img=gen_img,
        cm_img=cm_img,
        report_txt=report_txt,
    )

    out_path = OUTPUTS_DIR / "report.html"
    out_path.write_text(html, encoding="utf-8")