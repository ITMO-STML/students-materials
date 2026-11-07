from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jinja2 import Template

from env import ACTION_NAMES, state_to_rc

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>RL Lab. FrozenLake + Tabular DQN</title>
    <style>
        body { font-family: sans-serif; max-width: 950px; margin: 40px auto; color: #2c3e50; }
        h1, h2 { color: #2c3e50; }
        .metric { display: inline-block; padding: 12px 20px; margin: 8px;
                  background: #ecf0f1; border-radius: 8px; }
        .metric b { display: block; font-size: 1.3em; color: #2980b9; }
        img { max-width: 100%; margin: 12px 0; border: 1px solid #ddd; }
        table { border-collapse: collapse; margin-top: 12px; }
        td, th { border: 1px solid #bbb; padding: 6px 10px; text-align: center; font-family: monospace; }
        th { background: #ecf0f1; }
        .note { color: #7f8c8d; font-size: 0.9em; }
    </style>
</head>
<body>
    <h1>Лабораторная: Tabular DQN на FrozenLake 4x4</h1>

    <div class="metric">Успешных эпизодов (greedy)<b>{{ success }}</b></div>
    <div class="metric">Эпизодов обучения<b>{{ n_episodes }}</b></div>
    <div class="metric">Финальный epsilon<b>{{ final_eps }}</b></div>

    <h2>1. Выученная политика (стрелка = куда идёт агент)</h2>
    <img src="policy_map.png" alt="policy map">
    <p class="note">S - старт, F - лёд, H - лунка, G - цель. Стрелка показывает жадное действие.</p>

    <h2>2. Кривая обучения</h2>
    <img src="training_curve.png" alt="training curve">

    <h2>3. Epsilon (исследование и использование)</h2>
    <img src="epsilon_curve.png" alt="epsilon curve">

    <h2>4. Q-таблица (16 состояний x 4 действия)</h2>
    <table>
        <tr>
            <th>state</th><th>клетка</th>{{ action_headers }}
        </tr>
        {% for row in q_rows %}
        <tr>
            <td>{{ row.state }}</td>
            <td>{{ row.cell }}</td>
            {% for v in row.q_vals %}<td>{{ v }}</td>{% endfor %}
        </tr>
        {% endfor %}
    </table>
</body>
</html>
"""


def _draw_policy_map(policy: np.ndarray, board: list[str], path: Path) -> None:
    """Рисует сетку 4x4 и стрелку действия в каждой клетке."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")

    for state in range(16):
        r, c = state_to_rc(state)
        cell = board[r][c]
        # координаты клетки (ось Y инвертирована, чтобы S была сверху слева)
        x, y = c + 0.5, 4 - r - 0.5

        color = {"S": "#dff9fb", "F": "#ecf0f1", "H": "#e74c3c", "G": "#2ecc71"}[cell]
        ax.add_patch(plt.Rectangle((c, 4 - r - 1), 1, 1, facecolor=color, edgecolor="black"))

        if cell in ("H", "G"):
            ax.text(x, y, cell, ha="center", va="center", fontsize=18, fontweight="bold")
        else:
            ax.text(x, y, ACTION_NAMES[policy[state]], ha="center", va="center", fontsize=20)

    ax.set_title("Выученная политика")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _draw_training_curve(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["episode"], df["reward"], alpha=0.25, label="reward per episode")
    smoothed = df["reward"].rolling(50, min_periods=1).mean()
    ax.plot(df["episode"], smoothed, color="red", label="rolling mean (50)")
    ax.set_xlabel("Episode"); ax.set_ylabel("Reward")
    ax.set_title("Кривая обучения (reward)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _draw_epsilon_curve(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(df["episode"], df["epsilon"], color="purple")
    ax.set_xlabel("Episode"); ax.set_ylabel("epsilon")
    ax.set_title("Затухание epsilon")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def generate_html_report(
    df: pd.DataFrame,
    q_table: np.ndarray,
    greedy_policy: np.ndarray,
    success_rate: float,
    board: list[str],
    output_dir: Path,
) -> None:
    _draw_policy_map(greedy_policy, board, output_dir / "policy_map.png")
    _draw_training_curve(df, output_dir / "training_curve.png")
    _draw_epsilon_curve(df, output_dir / "epsilon_curve.png")

    q_rows = []
    for s in range(q_table.shape[0]):
        r, c = state_to_rc(s)
        q_rows.append({
            "state": s,
            "cell": board[r][c],
            "q_vals": [f"{v:.3f}" for v in q_table[s]],
        })
    action_headers = "".join(f"<th>{a}</th>" for a in ACTION_NAMES)

    html = Template(HTML_TEMPLATE).render(
        success=f"{success_rate:.0%}",
        n_episodes=len(df),
        final_eps=f"{df['epsilon'].iloc[-1]:.3f}",
        action_headers=action_headers,
        q_rows=q_rows,
    )

    (output_dir / "report.html").write_text(html, encoding="utf-8")