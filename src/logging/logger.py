'''
Main class for logging information about runs of environments.
'''
import os
from pathlib import Path
from typing import Optional
import uuid
import json
import h5py
import numpy as np

class Logger:
    def __init__(self, log_dir: str | Path, experiment_name: Optional[str]
                 ) -> None:
        self.log_dir = log_dir
        self.unique_id = str(uuid.uuid4())
        self.experiment_name = experiment_name + self.unique_id if experiment_name is not None else self.unique_id

        self.filepath = os.path.join(self.log_dir, f"{self.experiment_name}.h5")
        self.episode_count = 0
        
        with h5py.File(self.filepath, 'w') as f:
            f.create_group('configs')
            f.create_group('episodes')
            f.attrs['experiment_name'] = self.experiment_name

        # Internal storage for current episode data
        self.states = []
        self.players = []
        self.observations = []
        self.actions = []
        self.rewards = []

    def log_config(self, config: dict, name: str) -> None:
        with h5py.File(self.filepath, 'a') as f:
            f['configs'].attrs[name] = json.dumps(config)

    def log_player_ids(self, player_ids: list) -> None:
        with h5py.File(self.filepath, 'a') as f:
            f.attrs['player_ids'] = player_ids

    def log_step(self, state: np.ndarray, player: np.ndarray, 
                    observation: np.ndarray, action: np.ndarray, 
                    reward: np.ndarray) -> None:
        self.states.append(state)
        self.players.append(player)
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)

    def log_episode(self, states: np.ndarray, players: np.ndarray, 
                    observations: np.ndarray, actions: np.ndarray, 
                    rewards: np.ndarray) -> None:
        """
        Log a complete episode.
        
        Args:
            states: (num_steps, *state_shape)
            players: (num_steps,) - scalar per step
            observations: (num_steps, *obs_shape)
            actions: (num_steps, *action_shape)
            rewards: (num_steps,) - scalar per step
        """
        with h5py.File(self.filepath, 'a') as f:
            episode_group = f['episodes'].create_group(f'episode_{self.episode_count}')
            
            # Store all data without compression for perfect recovery
            episode_group.create_dataset('states', data=states, compression=None)
            episode_group.create_dataset('players', data=players, compression=None)
            episode_group.create_dataset('observations', data=observations, compression=None)
            episode_group.create_dataset('actions', data=actions, compression=None)
            episode_group.create_dataset('rewards', data=rewards, compression=None)
            
            # Store metadata as attributes
            episode_group.attrs['num_steps'] = len(states)
            episode_group.attrs['episode_id'] = self.episode_count
            
        self.episode_count += 1

    def end_episode(self) -> None:
        """
        Finalize logging for the current episode and reset internal storage.
        """
        if not self.states:
            return  # No data to log

        states_array = np.array(self.states)
        players_array = np.array(self.players)
        observations_array = np.array(self.observations)
        actions_array = np.array(self.actions)
        rewards_array = np.array(self.rewards)

        self.log_episode(states_array, players_array, observations_array, actions_array, rewards_array)

        # Reset internal storage for next episode
        self.states = []
        self.players = []
        self.observations = []
        self.actions = []
        self.rewards = []