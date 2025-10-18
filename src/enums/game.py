from enum import Enum

class RoleEnum(Enum):
    X = 1
    O = 2

class BoardEnum(Enum):
    EMPTY = 0
    X = RoleEnum.X
    O = RoleEnum.O
    INVALID = 3