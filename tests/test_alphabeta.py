import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


import numpy as np
from typing import Any
from tqdm import tqdm

from src.agents import MinimaxAgent, AlphaBetaMinimaxAgent
from src.enums.game import BoardEnum
from src.environments import TwoDims, ThreeDims
import pandas as pd

def generate_board_position(env: Any) -> np.ndarray:
    '''
    Generate a random position for any 4CE game.
    This position does not adhere the rules of the game,
    e.g there might be more Xs than Os.
    '''
    bs = env.get_board_state()
    allowed_values = [BoardEnum.EMPTY.value, BoardEnum.X.value, BoardEnum.O.value]
    # These values have to appear at least once in the generated position
    enforced_values = [BoardEnum.EMPTY.value] # make sure there is a valid move
    num_squares = np.prod(bs.shape)
    pos = np.concatenate([
        enforced_values,
        np.random.choice(allowed_values, size=num_squares - len(enforced_values))])
    np.random.shuffle(pos)
    pos = pos.reshape(bs.shape)
    return pos
    

def dummy_observation_from_position(env: Any, position: np.ndarray) -> dict:
    return {
            "current_player": 0,
            "next_player": 1,
            "board": position,
            "action_mask": env.get_action_mask(position)
        }

def test_identical_output():
    envs = [TwoDims(), ThreeDims()]
    depths = [1, 2, 3, 4, 5]
    num_positions = [100, 100, 75, 10, 10]
    num_nodes_explored = {'minimax': [], 'alphabeta': []}
    for i, depth in enumerate(depths):
        minimax = MinimaxAgent(search_depth=depth)
        alphabeta = AlphaBetaMinimaxAgent(search_depth=depth)
        for env in envs:
            positions = [generate_board_position(env) for _ in range(num_positions[i])]
            for position in tqdm(positions):
                history = [dummy_observation_from_position(env, position)]
                mm_action = minimax.choose_action(env, history)
                ab_action = alphabeta.choose_action(env, history)
                num_nodes_explored['minimax'].append(minimax.nodes_searched)
                num_nodes_explored['alphabeta'].append(alphabeta.nodes_searched)
                assert np.all(mm_action == ab_action)


    depth_nodes = {'minimax': {env.__class__.__name__: {d: [] for d in depths} for env in envs}, 'alphabeta': {env.__class__.__name__: {d: [] for d in depths} for env in envs}}

    idx = 0
    for i, depth in enumerate(depths):
        for _ in range(num_positions[i]):
            for env in envs:
                depth_nodes['minimax'][env.__class__.__name__][depth].append(num_nodes_explored['minimax'][idx])
                depth_nodes['alphabeta'][env.__class__.__name__][depth].append(num_nodes_explored['alphabeta'][idx])
                idx += 1

    for env in envs:
        table_data = {
            'Depth': depths,
            'Minimax': [np.mean(depth_nodes['minimax'][env.__class__.__name__][d]) for d in depths],
            'AlphaBeta': [np.mean(depth_nodes['alphabeta'][env.__class__.__name__][d]) for d in depths]
        }
        
        df = pd.DataFrame(table_data)
        print(f"\n{env.__class__.__name__}:")
        print(df.to_string(index=False))