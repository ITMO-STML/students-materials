from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from agent import TabularDQNAgent
from env import make_env, get_map_from_env
from report import generate_html_report


def train(episodes: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    env = make_env()
    board = get_map_from_env(env)
    n_states = env.observation_space.n          # 16
    n_actions = env.action_space.n              # 4

    agent = TabularDQNAgent(n_states=n_states, n_actions=n_actions)

    log: List[dict] = []

    print(f"[INFO] Старт: {episodes} эпизодов, {n_states} состояний, {n_actions} действий")

    for episode in range(1, episodes + 1):
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        td_errors: List[float] = []

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            td = agent.update(state, action, float(reward), next_state, done)
            td_errors.append(td)

            state = next_state
            total_reward += float(reward)

        agent.decay_epsilon()

        log.append({
            "episode": episode,
            "reward": total_reward,
            "epsilon": agent.epsilon,
            "mean_abs_td": float(np.mean(np.abs(td_errors))) if td_errors else 0.0,
        })

        if episode % 200 == 0:
            recent = [r["reward"] for r in log[-200:]]
            print(
                f"[EP {episode:4d}] mean_reward(last200)={np.mean(recent):.3f} "
                f"epsilon={agent.epsilon:.3f} "
                f"mean|TD|={log[-1]['mean_abs_td']:.4f}"
            )

    # Сохраняем Q-таблицу и лог
    q_path = output_dir / "q_table.npy"
    np.save(q_path, agent.q_table)
    print(f"[INFO] Q-таблица сохранена: {q_path}")

    df = pd.DataFrame(log)
    csv_path = output_dir / "training_log.csv"
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Лог обучения сохранён: {csv_path}")

    # Оценка жадной политики
    success = evaluate_greedy(agent, episodes=100)
    print(f"[METRIC] Доля успешных эпизодов (greedy, 100 эпизодов): {success:.2%}")

    # HTML-отчёт
    generate_html_report(
        df=df,
        q_table=agent.q_table,
        greedy_policy=agent.greedy_policy(),
        success_rate=success,
        board=board,
        output_dir=output_dir,
    )
    print(f"[INFO] HTML-отчёт: {output_dir / 'report.html'}")


def evaluate_greedy(agent: TabularDQNAgent, episodes: int = 100) -> float:
    """Прогон жадной политики без исследования — сколько раз дошли до цели."""
    env = make_env()
    wins = 0
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = int(np.argmax(agent.q_table[state]))
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if done and reward > 0:
                wins += 1
    return wins / episodes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    train(episodes=args.episodes, output_dir=Path(args.output_dir))


if __name__ == "__main__":
    main()