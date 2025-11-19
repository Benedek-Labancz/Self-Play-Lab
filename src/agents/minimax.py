'''
Agent implementing an epsilon-greedy policy over minimax of depth d.
Epsilon can be 0 and the policy therefore greedy.
'''
from typing import Any
import numpy as np
from .base import BaseAgent

class MinimaxAgent(BaseAgent):
    def __init__(self, search_depth: int, epsilon: int = 0, random_seed: int = 42) -> None:
        super().__init__(random_seed=random_seed)
        self.search_depth = search_depth
        self.epsilon = epsilon

    def choose_action(self, env: Any, history: list[dict]) -> np.array:
        '''
        With probability epsilon:
            - choose a random action
        and probability (1 - epsilon):
            - choose the action that leads to the state with the greatest *minimax value*
        '''
        observation = history[-1]
        dim_indices = list(np.nonzero(observation["action_mask"])) # [rows, columns] in 2D, generalises for higher dimensions
        num_valid_actions = len(dim_indices[0])
        if self.rng.random() < self.epsilon:
            action_idx = self.rng.integers(0, num_valid_actions, size=1)
            action = np.array([dim_indices[dim][action_idx] for dim in range(len(dim_indices))]).reshape(-1)
            return action
        else:
            actions = np.stack(dim_indices).T # (num_valid_actions, num_dimensions)
            root_current_player = env.get_current_player() 
            root_next_player = env.get_next_player()

            next_players_observations = np.apply_along_axis(lambda a: env.simulate_step(observation["board"], root_current_player, a)[0], axis=1, arr=actions)
            # The next player will want to minimise the minimax value..
            minimax_values = np.array([self.get_minimax_value(env, o, current_player=root_next_player, next_player=root_current_player, current_role='min', next_role='max', depth=1) for o in next_players_observations]) # (num_actions)

            # And we take the action that leads to the maximum of these.
            action_idx = np.argmax(minimax_values) # (1)
            action = actions[action_idx]
            return action
        
    def get_minimax_value(self, env: Any, observation: dict, current_player: int, next_player: int, current_role: str, next_role: str, depth: int) -> float:
        '''
        Compute the minimax value of `observation`, where "board" represents the board position.
        Search is carried out until the agent's specified search depth is reached.
        The final evaluation is based on the score difference between the players.
        The root player always wants to maixmise their score.

        `env` is used to gain access to environment dynamics, i.e. the rules of the game.
        This is needed to perform search.
        '''
        if depth == self.search_depth or env.terminal_state(observation["board"]):
            root_current_player = env.get_current_player() # Note that this does not change during traversal as we do not step the env.
            root_next_player = env.get_next_player()
            leaf_value = self.evaluate_leaf(env, observation, root_current_player, root_next_player)
            return leaf_value
        else:
            dim_indices = list(np.nonzero(observation["action_mask"])) # [rows, columns] in 2D, generalises for higher dimensions
            actions = np.stack(dim_indices).T # (num_valid_actions, num_dimensions)
            new_observations = [env.simulate_step(observation["board"], current_player, a)[0] for a in actions]
            minimax_values = np.array([self.get_minimax_value(
                                                env, o, 
                                                current_player=next_player, next_player=current_player, 
                                                current_role=next_role, next_role=current_role,
                                                depth=depth + 1
                                            ) for o in new_observations])
            if current_role == 'max':
                return np.max(minimax_values)
            elif current_role == 'min':
                return np.min(minimax_values)
            
    def evaluate_leaf(self, env: Any, observation: dict, root_current_player: int, root_next_player: int) -> float:
        '''
        Evaluate an observation representing an environment state that is a leaf node of the search.
        This method is added for modularity and clarity.

        Implementing: score difference between players.
        '''
        return env.get_score(observation["board"], root_current_player) - env.get_score(observation["board"], root_next_player)


