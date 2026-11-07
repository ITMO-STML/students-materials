"""
HTML-отчёт по итогам MTL (Multi-Task Learning) обучения.
"""
import base64
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.metrics import classification_report

REPORT_DIR: Path = Path("reports")
IMG_DIR: Path = REPORT_DIR / "img"

def _save_fig(fig: Figure, name: str) -> Path:
    path: Path = IMG_DIR / f"{name}.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path

def _fig_to_base64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def _plot_loss(history: Dict[str, Any]) -> Path:
    epochs: range = range(1, len(history["train_loss"]) + 1)
    fig: Figure
    ax: plt.Axes
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, history["train_loss"], label="Train loss", marker="o")
    ax.plot(epochs, history["val_loss"], label="Val loss", marker="s")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (Category + 0.5 * Gender)")
    ax.set_title("Training and validation loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save_fig(fig, "training_loss")

def _make_table(y_true: List[Any], y_pred: List[Any], title: str) -> str:
    report_dict: Dict[str, Any] = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    rows: List[Dict[str, Any]] = []
    for label, metrics in report_dict.items():
        if label in ("accuracy", "macro avg", "weighted avg"):
            continue
        rows.append({
            "class": label,
            "precision": f"{metrics['precision']:.3f}",
            "recall": f"{metrics['recall']:.3f}",
            "f1": f"{metrics['f1-score']:.3f}",
            "support": int(metrics["support"]),
        })
    
    macro: Dict[str, Any] = report_dict["macro avg"]
    weighted: Dict[str, Any] = report_dict["weighted avg"]
    
    class_rows: str = "".join(
        f"<tr><td>{r['class']}</td><td>{r['precision']}</td>"
        f"<td>{r['recall']}</td><td>{r['f1']}</td><td>{r['support']}</td></tr>"
        for r in rows
    )
    
    return f"""
    <h3>{title}</h3>
    <table>
      <tr><th>Класс</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr>
      {class_rows}
      <tr style="font-weight:bold; background:#f8f8f8;">
        <td>macro avg</td><td class="num">{macro['precision']:.3f}</td>
        <td class="num">{macro['recall']:.3f}</td><td class="num">{macro['f1-score']:.3f}</td>
        <td class="num">{int(macro['support'])}</td>
      </tr>
    </table>
    """

def generate_training_report(history: Dict[str, Any], y_true_cat: List[Any], y_pred_cat: List[Any], 
                              y_true_gen: List[Any], y_pred_gen: List[Any],
                              test_loss: float, test_acc_cat: float, test_acc_gen: float, 
                              config: Dict[str, Any]) -> Path:
    REPORT_DIR.mkdir(exist_ok=True)
    IMG_DIR.mkdir(exist_ok=True)

    report_cat: str = _make_table(y_true_cat, y_pred_cat, "Classification Report: Категории товаров")
    report_gen: str = _make_table(y_true_gen, y_pred_gen, "Classification Report: Пол (Gender)")

    config_html: str = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in config.items())

    fig_loss: Path = _plot_loss(history)
    img_loss: str = _fig_to_base64(fig_loss)

    html: str = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>MTL Training Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1000px; margin: 40px auto; padding: 0 20px; color: #222; }}
  h1, h2 {{ border-bottom: 2px solid #333; padding-bottom: 10px; }}
  table {{ border-collapse: collapse; margin: 20px 0; width: 100%; }}
  th, td {{ border: 1px solid #ccc; padding: 8px 12px; text-align: left; }}
  th {{ background: #f0f0f0; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  img {{ max-width: 100%; height: auto; border: 1px solid #eee; border-radius: 4px; }}
  .kpi {{ display: flex; gap: 15px; margin: 20px 0; flex-wrap: wrap; }}
  .kpi-card {{ background: #f8f8f8; border-left: 4px solid #4a7ab8;
              padding: 12px 20px; border-radius: 4px; min-width: 160px; }}
  .kpi-value {{ font-size: 24px; font-weight: bold; }}
  .kpi-label {{ font-size: 12px; color: #666; text-transform: uppercase; }}
</style>
</head>
<body>
<h1>MTL Training Report</h1>
<p style="color: #666;">Сгенерировано: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

<h2>1. Ключевые метрики</h2>
<div class="kpi">
  <div class="kpi-card">
    <div class="kpi-value">{test_acc_cat:.4f}</div>
    <div class="kpi-label">Test Acc (Категории)</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-value">{test_acc_gen:.4f}</div>
    <div class="kpi-label">Test Acc (Пол)</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-value">{test_loss:.4f}</div>
    <div class="kpi-label">Test Total Loss</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-value">{history.get('best_epoch', '-')}</div>
    <div class="kpi-label">Best Epoch</div>
  </div>
</div>

<h2>2. График лосса</h2>
<img src="data:image/png;base64,{img_loss}" alt="Loss">

<h2>3. Детальные отчёты (MTL)</h2>
{report_cat}
{report_gen}

<h2>4. Параметры обучения</h2>
<table>
  <tr><th>Параметр</th><th>Значение</th></tr>
  {config_html}
</table>
</body>
</html>
"""
    report_path: Path = REPORT_DIR / "training_report.html"
    report_path.write_text(html, encoding="utf-8")
    print(f"MTL Training report saved: {report_path}")
    return report_path