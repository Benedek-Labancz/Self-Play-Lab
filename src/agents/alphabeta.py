'''
Minimax Agent with Alpha-Beta Pruning
'''

from typing import Any
import numpy as np
from .minimax import MinimaxAgent


class AlphaBetaMinimaxAgent(MinimaxAgent):
    def __init__(self, search_depth: int, epsilon: float = 0, random_seed: int = 42) -> None:
        super().__init__(random_seed=random_seed, search_depth=search_depth, epsilon=epsilon)

        self.nodes_searched = 0

    def choose_action(self, env: Any, history: list[dict]) -> np.ndarray:
        '''
        With probability epsilon:
            - choose a random action
        and probability (1 - epsilon):
            - choose the action that leads to the state with the greatest *minimax value*
        '''
        self.nodes_searched = 0
        observation = history[-1]
        dim_indices = list(np.nonzero(observation["action_mask"])) # [rows, columns] in 2D, generalises for higher dimensions
        num_valid_actions = len(dim_indices[0])
        if self.rng.random() < self.epsilon:
            action_idx = self.rng.integers(0, num_valid_actions, size=1)
            action = np.array([dim_indices[dim][action_idx] for dim in range(len(dim_indices))]).reshape(-1)
            return action
        else:
            actions = np.stack(dim_indices).T # (num_valid_actions, num_dimensions)
            root_current_player = observation["current_player"]
            root_next_player = observation["next_player"]

            next_players_observations = np.apply_along_axis(lambda a: env.simulate_step(observation["board"], root_current_player, a)[0], axis=1, arr=actions)
            minimax_values = []
            alpha, beta = -np.inf, np.inf
            for o in next_players_observations:
                # The next player will want to minimise the minimax value..
                mm_value = self.get_minimax_value(env, o, current_player=root_next_player, next_player=root_current_player, current_role='min', next_role='max', depth=1, alpha=alpha, beta=beta)
                minimax_values.append(mm_value)
                # Update alpha based on what we know to be the best option for MAX so far
                if mm_value > alpha:
                    alpha = mm_value
                
            minimax_values = np.array(minimax_values)

            # And we take the action that leads to the maximum of these.
            action_idx = np.argmax(minimax_values) # (1)
            action = actions[action_idx]
            return action
        
    def get_minimax_value(self, env: Any, 
                          observation: dict, 
                          current_player: int, next_player: int, 
                          current_role: str, next_role: str, 
                          depth: int,
                          alpha: float, beta: float) -> float:
        
        '''
        Identical to MinimaxAgent.get_minimax_value but with alpha-beta pruning.
        alpha and beta represent the best already explored option for MAX and MIN respectively.
        They are updated during the search and passed down the tree.
        '''

        self.nodes_searched += 1
        if depth == self.search_depth or env.terminal_state(observation["board"]):
            root_current_player = current_player if current_role == 'max' else next_player
            root_next_player = current_player if current_role == 'min' else next_player
            leaf_value = self.evaluate_leaf(env, observation, root_current_player, root_next_player)
            return leaf_value
        else:
            dim_indices = list(np.nonzero(observation["action_mask"])) # [rows, columns] in 2D, generalises for higher dimensions
            actions = np.stack(dim_indices).T # (num_valid_actions, num_dimensions)
            minimax_values = []
            for a in actions:
                new_observation = env.simulate_step(observation["board"], current_player, a)[0]
                # We directly update alpha and beta here so that information is "passed upwards" on the search tree
                mm_value = self.get_minimax_value(
                                                env, new_observation, 
                                                current_player=next_player, next_player=current_player, 
                                                current_role=next_role, next_role=current_role,
                                                depth=depth + 1,
                                                alpha=alpha, beta=beta
                                            )
                minimax_values.append(mm_value)

                if current_role == 'max': # you can only touch alpha
                    if mm_value > alpha:
                        alpha = mm_value 
                if current_role == 'min': # you can only touch beta
                    if mm_value < beta:
                        beta = mm_value

                # Prune the node if we know it is never going to be reached
                if alpha >= beta:
                    break

            if current_role == 'max':
                value = np.max(minimax_values)
                return value
            elif current_role == 'min':
                value = np.min(minimax_values)
                return value