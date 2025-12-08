from enum import Enum

class RoleEnum(Enum):
    X = 0
    O = 1

class BoardEnum(Enum):
    X = RoleEnum.X.value
    O = RoleEnum.O.value
    EMPTY = 2
    INVALID = 3