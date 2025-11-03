'''
Utilities for pretty printing.
'''

import numpy as np
import os
from src.enums.game import RoleEnum

def red_text(text: str) -> str:
    return f"\033[91m{text}\033[0m"

def blue_text(text: str) -> str:
    return f"\033[94m{text}\033[0m"

def clear_terminal():
    if os.name == 'nt':
        os.system("cls")
    else:
        os.system("clear")

def print_board(board: np.array):
    colored = [["", "", ""], ["", "", ""], ["", "", ""]]
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            # TODO: add actual coloring
            # TODO: solve mapping
            if board[i][j] == RoleEnum.X.value:
                colored[i][j] = "X"
            elif board[i][j] == RoleEnum.O.value:
                colored[i][j] = "O"
            else:
                colored[i][j] = "-"

    print("     " + colored[0][0] + " │ " + colored[0][1] + " │ " + colored[0][2])
    print("    ───┼───┼───")
    print("     " + colored[1][0] + " │ " + colored[1][1] + " │ " + colored[1][2])
    print("    ───┼───┼───")
    print("     " + colored[2][0] + " │ " + colored[2][1] + " │ " + colored[2][2])