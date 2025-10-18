import numpy as np
from src.environments import TwoDims

if __name__ == "__main__":
    env = TwoDims(render_mode="human")
    print(env._scoring_cases)
    # env.step(np.array([0, 0]))
    # env.step(np.array([0, 1]))
    # env.step(np.array([1, 0]))
    # env.step(np.array([0, 2]))
    # env.step(np.array([2, 0]))
    # env.render(env._board_state)

    # print(env._get_score(env._board_state, env._next_player))