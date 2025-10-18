import pytest
import numpy as np
from src.environments import TwoDims


env2 = TwoDims()

def test_scoring_cases():
    cases = env2._scoring_cases()
    assert cases.shape == (8, env2.size, 2)

    state1 = np.array([
        [1, 1, 1],
        [0, 0, 0],
        [0, 0, 0]
    ])

    state2 = np.array([
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0]
    ])

    state3 = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    state4 = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0]
    ])

    state5 = np.array([
        [1, 1, 1],
        [0, 1, 0],
        [0, 0, 1]
    ])

    state6 = np.array([
        [1, 1, 1],
        [0, 1, 0],
        [1, 0, 1]
    ])

    assert env2._get_score(state1, 1) == 1
    assert env2._get_score(state2, 1) == 1
    assert env2._get_score(state3, 1) == 1
    assert env2._get_score(state4, 1) == 1

    assert env2._get_score(state5, 1) == 2
    assert env2._get_score(state6, 1) == 3