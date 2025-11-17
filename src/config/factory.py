'''
Factory functions for creating
    - env
    - agent (?)
objects.
'''
from typing import Any
import gymnasium as gym
from .schemas import GameConfig

def create_env_from_config(config: GameConfig) -> Any:
    env = gym.make(config.name, max_episode_steps=config.max_timesteps, **config.kwargs)
    env = env.unwrapped
    return env