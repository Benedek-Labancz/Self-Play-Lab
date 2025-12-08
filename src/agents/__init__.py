'''
This module implements the various types of Agents available,
each having a unified interface to interact with the environment.

List of Agents:
    1. Random Agent
    2. Minimax Agent
    3. PPO Agent (?)
    4. AC Agent (?)
    5. ...
'''
from .random import RandomAgent
from .minimax import MinimaxAgent
from .alphabeta import AlphaBetaMinimaxAgent

__all__ = ['RandomAgent', 'MinimaxAgent', 'AlphaBetaMinimaxAgent']