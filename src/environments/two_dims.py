import numpy as np
from typing import Optional
from copy import deepcopy
from src.enums.game import RoleEnum, BoardEnum
from .base import BaseEnv
from src.environments.render.printing import (
    clear_terminal,
    print_board
)


class TwoDims(BaseEnv):

    dimensions = 2

    def __init__(self, render_mode: Optional[str] = None, **kwargs) -> None:
        super().__init__(render_mode, **kwargs)

        self._initial_state = BoardEnum.EMPTY.value * np.ones(self.dimensions * [self.size])
        self._board_state = deepcopy(self._initial_state)

        self._scoring_cases = self._get_scoring_cases()


    def _get_reward(self, state: np.array, player: int, 
                    action: np.array, new_state: np.array) -> float | Exception:
        '''
        Computes reward for the action of the current player.

        Currently supported reward types:
            - "dense": reward equals the immadiate score received due to the action
        '''

        # TODO: we might have a bit of a confusion here
        # with what state and new_state mean
        
        reward = 0 # Base case
        if self.reward_type == "dense":
            reward += self._get_dense_reward(state, player, action, new_state)
        else:
            raise Exception(f"Reward type {self.reward_type} is not supported in this environment.")
        if self.bonus:
            reward += self._get_bonus(new_state, player)
        return reward

    def _get_dense_reward(self, state: np.array, player: int, action: np.array, new_state: np.array):
        '''
        Reward equals the immediate score received due to the action.
        '''
        previous_score = self.get_score(state, player)
        new_score = self.get_score(new_state, player)
        return new_score - previous_score
    
    def _get_bonus(self, state: np.array, player: int) -> float:
        '''
        Bonus is awarded in the terminal state, if and only if there player won the game.
        '''
        if self._terminal_state(state):
            winner = self._determine_winner(state)
            if winner == player:
                return self.bonus_value
        return 0

    def _get_scoring_cases(self) -> np.array:
        '''
        Compute all the N cases of coordinate triplets
        that score a point. The resulting array will be
        of shape (N, 3, 2).
        '''

        u_base = np.array(self.size * [1]).reshape(self.size, 1)
        roll_base = np.arange(self.size).reshape(self.size, 1)

        horizontals = np.array([
            np.concatenate((i * u_base, roll_base), axis=1) for i in range(self.size)
        ])

        verticals = np.array([
            np.concatenate((roll_base, i * u_base), axis=1) for i in range(self.size)
        ])

        diagonals = np.array([
            np.concatenate((roll_base, roll_base), axis=1),
            np.concatenate((roll_base, np.flip(roll_base, axis=0)), axis=1)
        ])

        return np.concatenate((horizontals, verticals, diagonals), axis=0)
    
    def render(self) -> None:
        if self.render_mode in ["human", "ansi"]:
            clear_terminal()
            print_board(self._board_state)