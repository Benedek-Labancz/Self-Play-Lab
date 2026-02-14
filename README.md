# Self-Play Lab

A Python framework for experimenting with reinforcement learning and game-playing agents in custom two-player environments.

## Overview

The Self-Play Lab provides extensible environments and agent implementations for studying multi-agent learning in Tic-Tac-Toe variants. The studied game variant introduces score accummulation, i.e. the game only terminates when no moves are available. The project currently implements both 2D and 3D boards, while supporting custom two-player environments that implement the Gym API.

## Features

- **Multiple Game Environments**
  - 2D Tic-Tac-Toe
  - 3D Tic-Tac-Toe
  - 4D variant (in development)
  - Dense reward structures with optional bonuses

- **Agent Implementations**
  - Random agent (baseline)
  - Minimax algorithm
  - Alpha-Beta pruning
  - Policy gradient methods (in development)

- **Experiment Management**
  - YAML-based configuration system
  - HDF5 logging for experiment data
  - Analysis tools for computing metrics (mean returns, etc.)
  - Visualization utilities for plotting results

- **Modular Architecture**
  - Base classes for environments and agents
  - Factory pattern for building components from configs
  - Extensible reward functions
  - Gymnasium-compatible environment interface

## Project Structure

```
Self-Play-Lab/
├── configs/
│   ├── agents/         # Agent configuration files
│   ├── games/          # Game environment configurations
│   └── generations/    # Experiment generation configs
├── scripts/
│   ├── lab/           # Experiment scripts
│   └── ui/            # User interface scripts
├── src/
│   ├── agents/        # Agent implementations
│   ├── analyzer/      # Analysis tools
│   ├── config/        # Configuration parsing and factory
│   ├── enums/         # Enumerations (roles, board states)
│   ├── environments/  # Game environments
│   ├── logging/       # Logging utilities
│   └── visualizer/    # Visualization tools
└── tests/             # Unit and algorithm tests
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Self-Play-Lab
```

2. Create and activate a virtual environment:
```bash
python -m venv .env
source .env/bin/activate  # On Windows: .env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running Experiments

Generate games between two agents using the configuration system:

```bash
python scripts/lab/generate_games.py --config configs/generations/test.yml
```

The generation config specifies:
- Number of games to run
- Player configurations (agents)
- Game environment settings
- Logging directory and experiment name

### Configuration Files

**Agent Configuration** (`configs/agents/`):
```yaml
name: 'RandomAgent'
kwargs:
  random_seed: 42
```

**Game Configuration** (`configs/games/`):
```yaml
name: '4CE-TwoDims'
max_timesteps: 9
kwargs:
  render_mode: 'ansi'
  reward_type: 'dense'
  bonus: false
```

**Generation Configuration** (`configs/generations/`):
```yaml
n: 10
player0: "configs/agents/random.yml"
player1: "configs/agents/alphabeta.yml"
game: "configs/games/threedims_default.yml"
log_dir: "logs/"
experiment_name: "alphabeta_test"
```

### Analyzing Results

```python
from src.analyzer import BaseAnalyzer
from src.visualizer import BaseVisualizer
import h5py

with h5py.File("logs/experiment.h5", 'a') as experiment_file:
    analyzer = BaseAnalyzer()
    mean_returns = analyzer.compute_mean_undiscounted_return_per_episode(experiment_file)

    visualizer = BaseVisualizer()
    visualizer.plot_timeseries(
        experiment_file['analysis']['mean_undiscounted_return_per_episode'][:],
        title="Mean Undiscounted Return Per Episode",
        xlabel="Episode",
        ylabel="Mean Return",
        legends=experiment_file.attrs['player_ids']
    )
```

## Development Status

This project is currently under active development. The following components are works in progress:

- Policy gradient algorithms
- Advanced training scripts
- Additional agent architectures
- Extended analysis capabilities

## Dependencies

Key dependencies include:
- `numpy` - Numerical computing
- `gymnasium` - RL environment framework
- `torch` - Deep learning framework
- `h5py` - Experiment data storage
- `matplotlib` - Visualization
- `pydantic` - Configuration validation
- `pytest` - Testing framework

See `requirements.txt` for the complete list.

## Testing

Run tests using pytest:
```bash
pytest tests/
```