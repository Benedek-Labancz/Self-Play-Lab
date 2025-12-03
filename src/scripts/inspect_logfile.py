import h5py
from argparse import ArgumentParser
from pathlib import Path


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    args = parser.parse_args()

    with h5py.File(args.file, 'r') as f:
        filepath = Path(args.file)
        size_bytes = filepath.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        print(f"Size: {size_mb:.2f} MB")
    
        print("Experiment Name:", f.attrs['experiment_name'])
        print("\nConfigs:")
        for config_name, config_json in f['configs'].attrs.items():
            print(f"  {config_name}: {config_json}")

        print("\nEpisodes:")
        for episode_key in f['episodes'].keys():
            episode_group = f['episodes'][episode_key]
            print(f"\nContents of {episode_key}:")
            for dataset_key in episode_group.keys():
                data = episode_group[dataset_key][:]
                print(f"  {dataset_key}: shape {data.shape}, dtype {data.dtype}")
            print("  Attributes:")
            for attr_key, attr_value in episode_group.attrs.items():
                print(f"    {attr_key}: {attr_value}")
