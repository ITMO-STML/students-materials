import os
import base64
from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PIL import Image


DATA_DIR: Path = Path("data")
CSV_PATH: Path = DATA_DIR / "data.csv"
REPORT_DIR: Path = Path("reports")
IMG_DIR: Path = REPORT_DIR / "img"


def save_fig(fig: Figure, name: str) -> Path:
    path: Path = IMG_DIR / f"{name}.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_to_base64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def main() -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    IMG_DIR.mkdir(exist_ok=True)

    df: pd.DataFrame = pd.read_csv(CSV_PATH)

    # 2. топ-15 категорий
    fig: Figure
    ax: plt.Axes
    fig, ax = plt.subplots(figsize=(10, 4))
    df["category"].value_counts().head(15).plot(
        kind="bar", ax=ax, title="Top-15 categories"
    )
    ax.set_xlabel("Category")
    ax.set_ylabel("Number of samples")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig_categories: Path = save_fig(fig, "top15_categories")

    # 3. распределение длины описаний
    df["description_length"] = df["description"].fillna("").str.len()

    # Общая информация
    shape: tuple[int, int] = df.shape
    missing: pd.Series = df.isna().sum()
    dtype_info: pd.Series = df.dtypes


    fig, ax = plt.subplots(figsize=(10, 5))
    df["description_length"].plot(kind="hist", bins=100, ax=ax)
    ax.set_title("Description length distribution")
    ax.set_xlabel("Number of characters")
    ax.set_ylabel("Number of samples")
    ax.set_xlim(0, 4000)
    fig.tight_layout()
    fig_description: Path = save_fig(fig, "description_length")

    # 4. Примеры
    samples: pd.DataFrame = df.sample(6, random_state=42)

    fig, axes = plt.subplots(2, 3, figsize=(20, 8))
    for ax, (_, row) in zip(axes.flat, samples.iterrows()):
        image: Image.Image = Image.open(DATA_DIR / row["image"])
        ax.imshow(image)
        ax.set_title(
            f'{row["category"]}\n{str(row["description"])[:80]}...',
            fontsize=9,
        )
        ax.axis("off")
    fig.tight_layout()
    fig_samples: Path = save_fig(fig, "samples")

    # 5. HTML-отчёт
    html: str = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>EDA Report — Product Dataset</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1100px; margin: 40px auto; padding: 0 20px; color: #222; }}
  h1 {{ border-bottom: 2px solid #333; padding-bottom: 10px; }}
  h2 {{ margin-top: 40px; color: #444; }}
  table {{ border-collapse: collapse; margin: 20px 0; }}
  th, td {{ border: 1px solid #ccc; padding: 8px 12px; text-align: left; }}
  th {{ background: #f0f0f0; }}
  img {{ max-width: 100%; height: auto; border: 1px solid #eee; border-radius: 4px; }}
  .meta {{ color: #666; font-size: 14px; }}
</style>
</head>
<body>

<h1>EDA Report — Product Dataset</h1>
<p class="meta">Generated from data/data.csv</p>

<h2>1. Общая информация</h2>
<table>
  <tr><th>Параметр</th><th>Значение</th></tr>
  <tr><td>Число строк</td><td>{shape[0]}</td></tr>
  <tr><td>Число колонок</td><td>{shape[1]}</td></tr>
  <tr><td>Колонки</td><td>{", ".join(df.columns)}</td></tr>
</table>

<h3>Пропущенные значения</h3>
<table>
  <tr><th>Колонка</th><th>Пропусков</th></tr>
  {"".join(f"<tr><td>{col}</td><td>{int(missing[col])}</td></tr>" for col in df.columns)}
</table>

<h3>Типы данных</h3>
<table>
  <tr><th>Колонка</th><th>Тип</th></tr>
  {"".join(f"<tr><td>{col}</td><td>{dtype_info[col]}</td></tr>" for col in df.columns)}
</table>

<h2>2. Топ-15 категорий</h2>
<img src="data:image/png;base64,{fig_to_base64(fig_categories)}" alt="Top-15 categories">

<h2>3. Распределение длины описаний</h2>
<img src="data:image/png;base64,{fig_to_base64(fig_description)}" alt="Description length">

<h2>4. Примеры из датасета</h2>
<img src="data:image/png;base64,{fig_to_base64(fig_samples)}" alt="Samples">

<h2>5. Первые 5 строк датасета</h2>
{df.head().to_html(index=False)}

</body>
</html>
"""

    report_path: Path = REPORT_DIR / "eda_report.html"
    report_path.write_text(html, encoding="utf-8")
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()