from copy import deepcopy
from abc import ABC, abstractmethod
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional
from src.enums.game import RoleEnum, BoardEnum
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

        self.config = None

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


    def set_config(self, config: dict) -> None:
        self.config = config

    def _switch_player(self) -> None:
        self._current_player, self._next_player = self._next_player, self._current_player

    def get_current_player(self) -> int:
        return self._current_player

    def get_next_player(self) -> int:
        return self._next_player

    def get_action_mask(self, state: np.array) -> np.array:
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
        along with the action mask describing valid actions.
        '''
        return {
            "current_player": self._current_player,
            "next_player": self._next_player,
            "board": self._board_state,
            "action_mask": self.get_action_mask(self._board_state)
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
            RoleEnum.X.value: 0,
            RoleEnum.O.value: 0
        }

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def simulate_step(self, state: np.array, player: int, 
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
        player_idx = self._players.index(player)
        observation = {
            "current_player": self._players[1] if player_idx == 0 else self._players[0],
            "next_player": player,
            "board": new_state,
            "action_mask": self.get_action_mask(new_state)
        }
        return observation, reward


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

        self._score[self._current_player] = self.get_score(self._board_state, self._current_player)

        terminated = self.terminal_state(self._board_state)
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
        
    def terminal_state(self, state: np.array) -> bool:
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
        scores = [self.get_score(state, player) for player in self._players]
        max_score, min_score = scores.max(), scores.min()
        if max_score == min_score:
            return None
        else:
            return self._players[scores.index(max_score)]

    def get_score(self, state: np.array, player: int) -> float:
        '''
        Calculating total score of a player according to
        standard rules of Tic-Tac-Toe, 3-in-a-row scores a point.
        '''

        # Bring the last axis to the front. This is where we can index into the array
        scoring_positions = np.transpose(self._scoring_cases, axes=(2, 0, 1)) # (N, 3, size) -> (N, size, 3)

        # Board state simplified to True where the player's marks are,
        # and False everywhere else
        board_mask = (state == player)

        is_at_position = board_mask[*scoring_positions] # (N, 3)


        scores = np.all(is_at_position, axis=1).astype(int) # (N)

        total_score = scores.sum()
        return total_score
    
    def get_board_state(self) -> np.array:
        return self._board_state
    
    def render(self):
        if self.render_mode in ["ansi"]:
            clear_terminal()
            print(f"Score - X: {self._score[RoleEnum.X.value]} | O: {self._score[RoleEnum.O.value]}\n")
            for i in range(self.size):
                print_board(self._board_state[i])
                print("\n")
    
    @abstractmethod
    def _get_reward(self, state: np.array, player: int, action: np.array, new_state: np.array) -> float:
        '''
        Determines R(s, a, s'). Subclasses must implement this method.
        '''
        pass
