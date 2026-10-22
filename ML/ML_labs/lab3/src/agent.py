from __future__ import annotations

import numpy as np


class TabularDQNAgent:
    """Табличный DQN: Q-таблица + epsilon-greedy + обновление по Беллману."""

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,      # скорость обучения
        gamma: float = 0.99,     # дисконт будущих наград
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
    ) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.q_table = np.ones((n_states, n_actions), dtype=np.float64)

    def select_action(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.n_actions))
        return int(np.argmax(self.q_table[state]))

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> float:
        """Один шаг обновления Q-значения. Возвращает TD-ошибку (для логов)."""
        # Если эпизод закончился, будущей награды нет
        best_next = 0.0 if done else float(np.max(self.q_table[next_state]))
        target = reward + self.gamma * best_next
        td_error = target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error
        return float(td_error)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def greedy_policy(self) -> np.ndarray:
        """Массив лучших действий для каждого состояния."""
        return np.argmax(self.q_table, axis=1)