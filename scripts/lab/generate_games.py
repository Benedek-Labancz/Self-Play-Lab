'''
Linearly generate games based on two agent configs and a game config.
Config of the script:
    - n       : (int)    : number of games
    - player0 : (string) : path to agent 1's config
    - player1 : (string) : path to agent 2's config
    - game    : (string) : path to the game config
    - log_dir : (string) : path to the logging directory
    - experiment_name : (string, optional) : name of the experiment.

Add path to config using the argument:
    --config "path/to/config"
'''

import sys
from pathlib import Path
from tqdm import tqdm

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from argparse import ArgumentParser
from typing import Any
from src.config.factory import parse_config, build_env, build_agent
from src.config.schemas import GenerationConfig
from src.enums.game import RoleEnum
from src.logging.logger import Logger
import src.environments

def generate_game(env: Any, p0: Any, p1: Any, logger: Logger) -> list:
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

        logger.log_step(env.get_board_state(), 
                        current_player, 
                        observation['board'], 
                        action,
                        reward)
    
    logger.end_episode()
    env.close()

    return histories




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="", required=True)
    args = parser.parse_args()

    config = parse_config(path=args.config, config_schema=GenerationConfig)
    player0 = build_agent(config.player0)
    player1 = build_agent(config.player1)
    game = build_env(config.game)

    logger = Logger(config.log_dir, config.experiment_name)
    logger.log_config(player0.config, "player0_config")
    logger.log_config(player1.config, "player1_config")
    logger.log_config(game.config, "game_config")
    logger.log_player_ids([RoleEnum.X.value, RoleEnum.O.value])

    games = [generate_game(game, player0, player1, logger) for _ in tqdm(range(config.n))]