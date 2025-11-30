from typing import Optional
import numpy as np

from .two_dims import TwoDims

class ThreeDims(TwoDims):

    dimensions = 3

    def __init__(self, render_mode: Optional[str] = None, **kwargs) -> None:
        super().__init__(render_mode, **kwargs)

    
    def _get_scoring_cases(self) -> np.array:
        '''
        Compute coordinate triplets. 
        Resulting array will be of shape (N, 3, 3)
        '''

        u = np.array(self.size * [1]).reshape(self.size, 1)
        roll = np.arange(self.size).reshape(self.size, 1)

        # 27 cases
        columns = np.concatenate([
            np.array([np.concatenate((roll, i*u, j*u), axis=1) for i in range(self.size) for j in range(self.size)]),
            np.array([np.concatenate((i*u, roll, j*u), axis=1) for i in range(self.size) for j in range(self.size)]),
            np.array([np.concatenate((i*u, j*u, roll), axis=1) for i in range(self.size) for j in range(self.size)])
        ])

        diagonals = np.concatenate([
            # 18 cases (3 x 3 x 2)
            np.array([np.concatenate((i*u, roll, roll), axis=1) for i in range(self.size)]),
            np.array([np.concatenate((i*u, roll, np.flip(roll, axis=0)), axis=1) for i in range(self.size)]),
            np.array([np.concatenate((roll, i*u, roll), axis=1) for i in range(self.size)]),
            np.array([np.concatenate((roll, i*u, np.flip(roll, axis=0)), axis=1) for i in range(self.size)]),
            np.array([np.concatenate((roll, roll, i*u), axis=1) for i in range(self.size)]),
            np.array([np.concatenate((roll, np.flip(roll, axis=0), i*u), axis=1) for i in range(self.size)]),
            # 4 cases
            np.array([np.concatenate((roll, roll, roll), axis=1)]),
            np.array([np.concatenate((roll, roll, np.flip(roll, axis=0)), axis=1)]),
            np.array([np.concatenate((roll, np.flip(roll, axis=0), roll), axis=1)]),
            np.array([np.concatenate((roll, np.flip(roll, axis=0), np.flip(roll, axis=0)), axis=1)])
        ])

        return np.concatenate((columns, diagonals), axis=0)
    