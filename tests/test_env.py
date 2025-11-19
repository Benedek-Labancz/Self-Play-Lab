import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


import pytest
import numpy as np
from src.environments import TwoDims
from src.enums.game import BoardEnum


env2 = TwoDims()

def test_scoring_cases():
    cases = env2._scoring_cases
    assert cases.shape == (8, env2.size, 2)

    e = BoardEnum.EMPTY.value
    p = BoardEnum.O.value
    q = BoardEnum.X.value

    state1 = np.array([
        [p, p, p],
        [e, e, e],
        [e, e, e]
    ])

    state2 = np.array([
        [p, e, e],
        [p, e, e],
        [p, e, e]
    ])

    state3 = np.array([
        [p, e, e],
        [e, p, e],
        [e, e, p]
    ])

    state4 = np.array([
        [e, e, p],
        [e, p, e],
        [p, e, e]
    ])

    state5 = np.array([
        [p, p, p],
        [e, p, e],
        [e, e, p]
    ])

    state6 = np.array([
        [p, p, p],
        [e, p, e],
        [p, e, p]
    ])

    state7 = np.array([
        [q, p, p],
        [q, q, p],
        [e, e, q]
    ])

    assert env2.get_score(state1, p) == 1
    assert env2.get_score(state2, p) == 1
    assert env2.get_score(state3, p) == 1
    assert env2.get_score(state4, p) == 1

    assert env2.get_score(state5, p) == 2
    assert env2.get_score(state6, p) == 3

    assert env2.get_score(state7, p) == 0
    assert env2.get_score(state7, q) == 1