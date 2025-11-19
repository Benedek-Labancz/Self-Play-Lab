'''
TODO: we might make the API a bit cleaner here. Just make one function for parsing a config (avoid repeating this code),
and create separate functions for agent / other object creation where needed.

Defines convenience functions for parsing different types of config files.

Supported configs:
    - AgentConfig
    - GameConfig
    - GenerationConfig
'''
from typing import Any
from pathlib import Path
from yaml import safe_load
from .schemas import AgentConfig, GenerationConfig, GameConfig
import src.agents.random
import src.agents.minimax

AGENTS_SUBMODULES = [src.agents.random, src.agents.minimax]

def parse_agent_config(path: str | Path) -> tuple[Any, AgentConfig] | Exception:
    if type(path) == str:
        path = Path(path)
    if path.is_file():
        with open(path, 'r') as f:
            config = safe_load(f)
        config = AgentConfig(**config)
        for submodule in AGENTS_SUBMODULES:
            if hasattr(submodule, config.name):
                try:
                    agent = getattr(submodule, config.name)(**config.kwargs)
                except:
                    raise Exception(f'Invalid keyword argument among {config.kwargs}')
                return agent, config
        raise Exception(f"Agent {config.name} does not exist.")
    else:
        raise Exception(f"Invalid filepath provided: {path}")
    

def parse_game_config(path: str | Path) -> GameConfig | Exception:
    if type(path) == str:
        path = Path(path)
    if path.is_file():
        with open(path, 'r') as f:
            config = safe_load(f)
        config = GameConfig(**config)
        return config
    else:
        raise Exception(f"Invalid filepath provided: {path}")
    

def parse_generation_config(path: str | Path) -> GenerationConfig | Exception:
    if type(path) == str:
        path = Path(path)
    if path.is_file():
        with open(path, 'r') as f:
            config = safe_load(f)
        config = GenerationConfig(**config)
        return config
    else:
        raise Exception(f"Invalid filepath provided: {path}")