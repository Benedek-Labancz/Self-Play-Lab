import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from gymnasium.envs.registration import register
from .two_dims import TwoDims
from .three_dims import ThreeDims

def register_envs():
    register(
        id='4CE-TwoDims',
        entry_point='4CE-Reloaded.src.environments.two_dims:TwoDims',
        max_episode_steps=9
    )

    register(
        id='4CE-ThreeDims',
        entry_point='4CE-Reloaded.src.environments.three_dims:ThreeDims',
        max_episode_steps=27
    )

register_envs()

__all__ = ['TwoDims', 'ThreeDims']
