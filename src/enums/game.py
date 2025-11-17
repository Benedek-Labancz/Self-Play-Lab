from enum import Enum

class RoleEnum(Enum):
    X = 0
    O = 1

class BoardEnum(Enum):
    X = RoleEnum.X
    O = RoleEnum.O
    EMPTY = 2
    INVALID = 3