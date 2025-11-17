'''
Linearly generate games based on two agent configs and a game config.
Config of the script:
    - n       : (int)    : number of games
    - player0 : (string) : path to agent 1's config
    - player1 : (string) : path to agent 2's config
    - game    : (string) : path to the game config
    - log_dir : (string) : path to the logging directory

Add path to config using the argument:
    --config "path/to/config"  
'''

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from argparse import ArgumentParser
from typing import Any
import gymnasium as gym
from src.agents.random import RandomAgent
from src.logging.json_encoder import NumpyArrayEncoder
from src.config.parser import parse_agent_config, parse_generation_config, parse_game_config
from src.config.factory import create_env_from_config
from src.config.schemas import GameConfig
from src.enums.game import RoleEnum
import src.environments

def generate_game(env: Any, p0: Any, p1: Any):
    # TODO: do we need player configs here for logging? And also log_dir?

    '''
    Simulate a game between two players.
    Convention is p0 = X (first player), p1 = O
    '''

    observation, _ = env.reset()
    done, truncated = False, False

    players = [p0, p1]
    histories = [[observation], []]

    while not (done or truncated):
        current_player = env.get_current_player()
        next_player = env.get_next_player()
        action = players[current_player].choose_action(env, histories[current_player])
        observation, reward, done, truncated, info = env.step(action)
        # We add the observation about the new state to the history of the next player because moves alternate
        histories[next_player].append(observation)

        env.render()
        _ = input()

        # perhaps some logging logic here ?
    env.close()

    return histories




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="", required=True)
    args = parser.parse_args()

    config = parse_generation_config(args.config)
    player0, player0_config = parse_agent_config(config.player0)
    player1, player1_config = parse_agent_config(config.player1)
    game_config = parse_game_config(config.game)
    game = create_env_from_config(game_config)

    # TODO: need to dump data in the log-dir here
    games = [generate_game(game, player0, player1) for i in range(config.n)]