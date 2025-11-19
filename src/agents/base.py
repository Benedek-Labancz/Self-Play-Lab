'''
Base parent class for each agent.
'''
import numpy as np

class BaseAgent:
    def __init__(self, random_seed: int = 42) -> None:
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)