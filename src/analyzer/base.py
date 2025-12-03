import h5py
from pathlib import Path

class BaseAnalyzer:
    def __init__(self, overwrite: bool = False) -> None:
        self.overwrite = overwrite

    def open_hdf5(self, file_path: str | Path) -> h5py.File:
        """Open an HDF5 file and return the file object."""
        return h5py.File(file_path, 'r')
    
    def make_analysis_group(self, hdf5_file: h5py.File) -> h5py.Group:
        """Create a new analysis group in the HDF5 file."""
        if hdf5_file.get("analysis") is not None:
            return hdf5_file["analysis"]
        return hdf5_file.create_group("analysis")
    
    def compute_mean_undiscounted_return_per_episode(self, experiment: h5py.File) -> dict:
        """
        Compute the mean undiscounted return per episode from the experiment data.
        """
        if experiment.get("analysis/mean_undiscounted_return_per_episode") is not None and not self.overwrite:
            print("Mean undiscounted return per episode already exists. Skipping computation.")
            return experiment['analysis']['mean_undiscounted_return_per_episode'][:]
        else:
            analysis_group = self.make_analysis_group(experiment)
            if analysis_group.get('mean_undiscounted_return_per_episode') is not None:
                del analysis_group['mean_undiscounted_return_per_episode']

            mean_reward_per_player_per_episode = {id: [] for id in experiment.attrs['player_ids']}
            for episode in experiment['episodes']:
                rewards_per_player = {id: [] for id in experiment.attrs['player_ids']}
                for reward, player in zip(experiment['episodes'][episode]['rewards'], 
                                        experiment['episodes'][episode]['players']):
                    rewards_per_player[player].append(reward)
                for id in mean_reward_per_player_per_episode:
                    mean_reward_per_player_per_episode[id].append(sum(rewards_per_player[id]) / len(rewards_per_player[id]))
            
            analysis_group.create_dataset('mean_undiscounted_return_per_episode', 
                                        data=[mean_reward_per_player_per_episode[id] for id in experiment.attrs['player_ids']],
                                        compression=None)
            return mean_reward_per_player_per_episode