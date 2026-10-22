from typing import Tuple

import gymnasium as gym
import numpy as np

ACTION_NAMES = ["←", "↓", "→", "↑"]


class ShapedFrozenLake(gym.Wrapper):

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if not terminated:
            reward = -0.01
        return obs, reward, terminated, truncated, info


def make_env() -> gym.Env:
    base = gym.make(
        "FrozenLake-v1",
        map_name="4x4",
        is_slippery=False,
    )
    return ShapedFrozenLake(base)

def get_map_from_env(env: gym.Env) -> list[str]:
    desc = env.unwrapped.desc
    return ["".join(row.decode() for row in line) for line in desc]

def state_to_rc(state: int, size: int = 4) -> Tuple[int, int]:
    return divmod(state, size)