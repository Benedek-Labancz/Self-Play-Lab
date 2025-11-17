import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from gymnasium.envs.registration import register
from .two_dims import TwoDims

def register_envs():
    register(
        id='4CE-TwoDims',
        entry_point='4CE-Reloaded.src.environments.two_dims:TwoDims',
        max_episode_steps=9
    )

register_envs()

__all__ = ['TwoDims']
