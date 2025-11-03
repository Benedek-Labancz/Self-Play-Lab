'''
This file includes the various types of Agents available,
each having a unified interface to interact with the environment.

List of Agents:
    1. Random Agent
    2. Minimax Agent
    3. PPO Agent
    4. ...
'''


import numpy as np



class BaseAgent:
    def __init__(self, environment) -> None:
        # The agent needs to have an instance of the environment
        # in order to access game rules (to perform lookahead)
        self.environment = environment

class RandomAgent(BaseAgent):
    def __init__(self, random_seed: int = 42) -> None:
        super().__init__()
        self.random_seed = random_seed


    def choose_action(self):
        raise NotImplementedError