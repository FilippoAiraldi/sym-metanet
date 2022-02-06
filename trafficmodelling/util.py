import numpy as np
import casadi as cs
import itertools


from typing import Union, Any


def create_profile(t: list[int], x: list[int], y: list[int]) -> np.ndarray:
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
            id = itertools.count(0)
            self.__ids[t] = id
        self.name = name or f'{t.__name__}_{next(id)}'

    def __repr__(self) -> str:
        return f'{self.__class__.__name__} {self.name}'

    def __str__(self) -> str:
        return self.name


class SmartList(list):
    def __setitem__(self, idx: Union[int, slice], val: Any) -> None:
        if not isinstance(idx, int) or idx != len(self):
            return super().__setitem__(idx, val)
        return super().append(val)

    def __str__(self) -> str:
        return f'SL #{len(self)}: {super().__str__()}'

    def __repr__(self) -> str:
        return f'SL #{len(self)}: {super().__repr__()}'

    @classmethod
    def from_list(cls, other: list[Any]) -> 'SmartList[Any]':
        o = cls()
        o.extend(other)
        return o


def repinterl(x: cs.SX, n: int, m: int = 1) -> cs.SX:
    '''
    Repeats element of the matrix n x m times in an interleave fashion.

    Params
    ------
        x : array
            Array to be repeated.

        n : int 
            Number of repetitions along the first axis.

        m : int , optional
            Number of repetitions along the second axis. Defaults to 1.

    Returns
    -------
        array
            Repetead array.
    '''

    d1, d2 = x.shape
    if n != 1:
        x = cs.vertcat(*[cs.repmat(x[i, :], n, 1) for i in range(d1)])
    if m != 1:
        x = cs.horzcat(*[cs.repmat(x[:, i], 1, m) for i in range(d2)])
    return x


def pad(x: cs.SX, pad1: tuple[int, int],
        pad2: tuple[int, int] = (0, 0),
        mode: str = 'constant', constant_value: Any = 0.0) -> cs.SX:
    '''
    Pads a casadi array.

    Parameters
    ----------
        pad1, pad2 : tuple[int,int]
            Number of elements to add before and after each of the two 
            dimensions.

        mode : str, {'constant', 'edge'}
            Padding mode: see numpy.pad.

        constant_value : float, Any        
            Constant value used in padding if mode is constant.

    Raises
    ------
        ValueError : invalid mode
            If the mode is not 'constant' or 'edge'.
    '''

    if mode not in ('constant', 'edge'):
        raise ValueError('Invalid padding mode; expected \'constant\' or '
                         f'\'edge\', got \'{mode}\' instead.')

    is_mode_const = mode == 'constant'

    if is_mode_const:
        before = np.full((max(0, pad1[0]), x.shape[1]), constant_value)
        after = np.full((max(0, pad1[1]), x.shape[1]), constant_value)
    else:  # i.e., edge
        before = cs.repmat(x[0, :].reshape((1, -1)), max(0, pad1[0]), 1)
        after = cs.repmat(x[-1, :].reshape((1, -1)), max(0, pad1[1]), 1)

    x = cs.vertcat(before, x, after)

    if is_mode_const:
        before = np.full((x.shape[0], max(0, pad2[0])), constant_value)
        after = np.full((x.shape[0], max(0, pad2[1])), constant_value)
    else:  # i.e., edge
        before = cs.repmat(x[:, 0].reshape((-1, 1)), 1, max(0, pad2[0]))
        after = cs.repmat(x[:, -1].reshape((-1, 1)), 1, max(0, pad2[1]))

    x = cs.horzcat(before, x, after)
    return x


def shift(x, n: int = 1, axis: int = 1):
    '''
    Shifts the array along the axis and pads with last value.
        [1,2,3,4]    --- shift n=2 --> [3,4,4,4]

    Parameters
    ----------
        n : int, optional
            Size of shift. Can also be negative. Defaults to 1.

        axis : int, {1, 2}, optional
            Axis along which to shift.

    Raises
    ------
        ValueError : invalid axis
            If the axis is invalid.
    '''

    if axis != 1 and axis != 2:
        raise ValueError(f'Invalid axis; expected 1 or 2, got {axis} instead.')

    if n == 0:
        return x

    if n > 0:
        if axis == 1:
            return cs.vertcat(x[n:, :], *([x[-1, :]] * n))
        return cs.horzcat(x[:, n:], *([x[:, -1]] * n))

    if axis == 1:
        return cs.vertcat(*([x[0, :]] * (-n)), x[:n, :])
    return cs.horzcat(*([x[:, 0]] * (-n)), x[:, :n])
