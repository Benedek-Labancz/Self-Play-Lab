'''
Factory functions for creating
    - env
    - agent (?)
objects.
'''
from typing import Any
from pathlib import Path
from yaml import safe_load
import gymnasium as gym

from .schemas import GameConfig, AgentConfig, GenerationConfig

import src.agents.random
import src.agents.minimax
import src.agents.alphabeta

CONFIG_SCHEMAS   = [GameConfig, AgentConfig, GenerationConfig]
AGENT_SUBMODULES = [src.agents.random, src.agents.minimax, src.agents.alphabeta]

def parse_config(path: str | Path, config_schema: Any) -> Any | Exception:
    if type(path) == str:
        path = Path(path)
    if path.is_file():
        with open(path, 'r') as f:
            config = safe_load(f)
        config = config_schema(**config)
        return config
    else:
        raise Exception(f"Invalid filepath provided: {path}")

def build_env(path: str | Path) -> Any:
    config = parse_config(path, GameConfig)
    env = gym.make(config.name, max_episode_steps=config.max_timesteps, **config.kwargs)
    env = env.unwrapped
    env.set_config(dict(config))
    return env

def build_agent(path: str | Path) -> Any | Exception:
    config = parse_config(path, AgentConfig)
    for submodule in AGENT_SUBMODULES:
            if hasattr(submodule, config.name):
                try:
                    agent = getattr(submodule, config.name)(**config.kwargs)
                    agent.set_config(dict(config))
                except:
                    raise Exception(f'Invalid keyword argument among {config.kwargs}')
                return agent
    raise Exception(f"Agent {config.name} does not exist.")