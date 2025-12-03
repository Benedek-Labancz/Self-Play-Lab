from pydantic import BaseModel
from pathlib import Path
from typing import Optional

class GenerationConfig(BaseModel):
    n: int
    player0: str | Path
    player1: str | Path
    game: str | Path
    log_dir: str | Path
    experiment_name: Optional[str] = None

class AgentConfig(BaseModel):
    name: str
    kwargs: dict

class GameConfig(BaseModel):
    name: str
    max_timesteps: Optional[int] = None
    kwargs: Optional[dict] = {}