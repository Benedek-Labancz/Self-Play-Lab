import numpy as np
from .base import BaseAgent
from gymnasium import Env

class RandomAgent(BaseAgent):
    def __init__(self, random_seed: int = 42) -> None:
        super().__init__()
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)


    def choose_action(self, env: Env, history: list[dict]) -> np.array:
        '''
        Randomly choose an action from the valid actions.
        env is ignored, it's passed to maintain API consistency
        '''
        observation = history[-1]
        dim_indices = list(np.nonzero(observation["action_mask"])) # [rows, columns] in 2D, generalises for higher dimensions
        num_valid_actions = len(dim_indices[0])
        action_idx = self.rng.integers(0, num_valid_actions, size=1)
        action = np.array([dim_indices[dim][action_idx] for dim in range(len(dim_indices))]).reshape(-1)
        return action