from typing import Optional
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from copy import deepcopy
from src.enums.game import RoleEnum, BoardEnum
from abc import ABC, abstractmethod
from src.environments.render.printing import (
    clear_terminal,
    print_board
)

class BaseEnv(gym.Env, ABC):
    metadata = {"render_modes": ["human", "ansi"]}

    size = 3

    def __init__(self, render_mode: Optional[str] = None, 
                 max_timesteps: Optional[int] = None,
                 reward_type: Optional[str] = "dense",
                 bonus: Optional[bool] = False,
                 bonus_value: Optional[float] = 100, **kwargs) -> None:
        super().__init__()

        self.render_mode = render_mode
        self.max_timesteps = np.inf if max_timesteps is None else max_timesteps
        self.reward_type = reward_type
        self.bonus = bonus
        self.bonus_value = bonus_value

        # Each square can have four different values 
        # (0-empty, 1-X, 2-O, 3-invalid)
        self.observation_space = spaces.MultiDiscrete(4 * np.ones(self.dimensions * [self.size]))
        
        # Actions are represented by coordinates
        self.action_space = spaces.MultiDiscrete(self.dimensions * [self.size])

        # To be defined by subclass
        self._initial_state = None
        self._board_state = None

        self._players = [RoleEnum.X.value, RoleEnum.O.value]
        
        # Reset the env to reset timesteps, initialize scores,
        # current and next player
        self.reset()

    def _switch_player(self):
        self._current_player, self._next_player = self._next_player, self._current_player

    def _get_action_mask(self, state: np.array) -> np.array:
        '''
        Returns an array with 1s and 0s at the coordinates corresponding 
        to valid and invalid actions respectively.
        (Important note: we make the assumption here that an action is valid
        iff the corresponding square is empty. We do not use the internal method checking
        this property in order to gain efficiency.)
        '''
        return (state == BoardEnum.EMPTY.value)

    def _get_obs(self) -> dict:
        '''
        Returns the observation of the board state and the current player,
        along with the action mask describing valid actions/
        '''
        return {
            "current_player": self._current_player,
            "board": self._board_state,
            "action_mask": self._get_action_mask()
        }

    def _get_info(self) -> dict:
        '''
        Provides additional information for debugging
        and rendering purposes. This should not be used
        by the learning algorithm.
        '''
        return {
            "score": self._score
        }

    def reset(self, seed: Optional[int] = None, 
              options: Optional[dict] = None) -> tuple[dict, dict]:
        '''
        Sets the random seed, and resets the timestep,
        resets the board to its initial position,
        zeros out the scores and resets the current player.
        '''
        super().reset(seed=seed)

        self.timestep = 0

        self._board_state = deepcopy(self._initial_state)
        
        self._current_player = self._players[0]
        self._next_player = self._players[1]

        self._score = {
            RoleEnum.X: 0,
            RoleEnum.O: 0
        }

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def _simulate_step(self, state: np.array, player: int, 
                       action: np.array) -> tuple[np.array, float] | Exception:
        '''
        This method does not change the internal state of this class.
        The purpose of it is to provide knowledge of the game rules to the agents
        interacting with the environment.
        It takes a state and a player along with the performed action and
        simulates a step of the game, returning the new state of the board and the
        reward associated with it. 
        (Important note: this reward signal is computed in the way specified by the class instance,
        i.e. it might be dense, sparse, etc.)
        '''
        if not self._valid_action(action):
            raise Exception(f"Invalid action {tuple(action)} encountered.")
        
        # TODO: making this copy here is a costly operation, which might cause problems when searching through environments with large action-spaces
        new_state = deepcopy(state)
        new_state[tuple(action)] = player
        reward = self._get_reward(state, player, action, new_state)
        return new_state, reward


    def step(self, 
             action: np.array) -> tuple[dict, float, bool, bool, dict] | Exception:
        '''
        Main environment logic.

        Checks if action is valid, executes the action,
        determines the reward, updates the game score,
        asserts terminal state and truncation, switches players.
        '''
        if not self._valid_action(action):
            raise Exception(f"Invalid action {tuple(action)} encountered.")
        
        ground_state = deepcopy(self._board_state)
        self._board_state[tuple(action)] = self._current_player

        self.timestep += 1

        reward = self._get_reward(ground_state, self._current_player, action, self._board_state)

        self._score[self._current_player] = self._get_score(self._board_state, self._current_player)

        terminated = self._terminal_state(self._board_state)
        truncated = self.timestep >= self.max_timesteps

        self._switch_player()

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info
    
    
    def _valid_action(self, action: np.array) -> bool:
        '''
        An action is valid if and only if the corresponding
        square is empty. (Note that the mirror rule currently
        omitted from this implementation.)
        '''
        assert action in self.action_space
        if self._board_state[tuple(action)] == BoardEnum.EMPTY.value:
            return True
        else:
            return False
        
    def _terminal_state(self, state: np.array) -> bool:
        '''
        A state is terminal if and only if there is no empty
        square left on the board.
        This excludes the case when the game ends after the first point
        scored, which should be separately defined in the subclass.
        '''
        return not (BoardEnum.EMPTY.value in state)
    
    def _determine_winner(self, state: np.array) -> int | None:
        '''
        The player with more points wins. In case of a tie, there is no winner.
        '''
        scores = [self._get_score(state, player) for player in self._players]
        max_score, min_score = scores.max(), scores.min()
        if max_score == min_score:
            return None
        else:
            return self._players[scores.index(max_score)]

    @abstractmethod
    def _get_reward(self, state: np.array, player: int, action: np.array, new_state: np.array) -> float:
        '''
        Determines R(s, a, s'). Subclasses must implement this method.
        '''
        pass

    @abstractmethod
    def _get_score(self, state: np.array, player: int) -> int:
        '''
        Determines the total score of the given player
        according to the game rules. Subclasses must implement this.
        '''
        pass

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
        previous_score = self._get_score(state, player)
        new_score = self._get_score(new_state, player)
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

    def _get_score(self, state: np.array, player: int) -> float:
        '''
        Calculating total score of a player according to
        standard rules of Tic-Tac-Toe, 3-in-a-row scores a point.
        '''

        # We need to twist the coordinates a bit for our purpose
        scoring_positions = np.transpose(self._scoring_cases, axes=(2, 0, 1)) # (8, 3, 2) -> (2, 8, 3)
        rows, cols = scoring_positions[0], scoring_positions[1]

        # Board state simplified to True where the player's marks are,
        # and False everywhere else
        board_mask = (state == player)

        is_at_position = board_mask[rows, cols] # (8, 3)
        scores = np.all(is_at_position, axis=1).astype(int) # (8)
        total_score = scores.sum()
        return total_score

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
            np.concatenate((roll_base, np.flip(roll_base, axis=1)), axis=1)
        ])

        return np.concatenate((horizontals, verticals, diagonals), axis=0)
    
    def render(self, state: np.array) -> None:
        if self.render_mode in ["human", "ansi"]:
            clear_terminal()
            print_board(state)