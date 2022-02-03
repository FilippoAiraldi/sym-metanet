import numpy as np
from itertools import count


def create_profile(t, x, y):
    '''
    Creates a piece-wise linear profile along time vector t, passing through 
    all x and y points.

    Parameters
    ----------
        t : {list, array}
            1D array or list representing the (monotonically increasing) time 
            vector.
        x : {list, array}
            1D array or list of x-coordinates where the piecewise profile 
            passes through.
        y : {list, array}
            1D array or list of y-coordinates where the piecewise profile 
            passes through. Must have same length as x.

    Returns
    -------
        profile : array 
            the piecewise linear function, of the same size of t.

    Raises
    ------
        ValueError : length mismatch
            If x and y don't share the same length
    '''

    if len(x) != len(y):
        raise ValueError('length mismatch between \'x\' and \'y\' '
                         f'({len(x)} vs {len(y)})')

    # convert x from time to indices
    t = np.array(t)
    x = list(map(lambda p: np.argmin(np.abs(t - p)), x))

    # add first and last timestep
    if x[0] != 0:
        x = [0, *x, t.size - 1]
        y = [y[0], *y, y[-1]]
    else:
        x = [*x, t.size - 1]
        y = [*y, y[-1]]

    # create profile of piecewise affine functions (i.e., lines)
    profile = np.zeros_like(t)
    for i in range(len(x) - 1):
        m = (y[i + 1] - y[i]) / (t[x[i + 1]] - t[x[i]])
        q = y[i] - m * t[x[i]]
        profile[x[i]:x[i + 1]] = m * t[x[i]:x[i + 1]] + q

    # take care of last point
    profile[-1] = profile[-2]
    return profile


class NamedClass:
    '''Class with a (automatic) name'''

    __ids = {}

    def __init__(self, name: str = None) -> None:
        t = self.__class__
        id = self.__ids.get(t)
        if id is None:
            id = count(0)
            self.__ids[t] = id
        self.name = name or f'{t.__name__}_{next(id)}'

    def __repr__(self) -> str:
        return f'{self.__class__.__name__ } {self.name}'

    def __str__(self) -> str:
        return self.name


class SmartList(list):
    def __setitem__(self, idx, val):
        if not isinstance(idx, int) or idx != len(self):
            return super().__setitem__(idx, val)
        return super().append(val)
