from typing import Sequence, Union
import numpy as np


def as1darray(
    x: Union[int, float, Sequence[Union[int, float]], np.ndarray],
    n: int
) -> np.ndarray:
    '''Converts a scalar or array_like to a 1D array with the given length.

    Parameters
    ----------
    x : int, float, array_like
        Value to be converted to a 1D array. Can be a scalar or an array_like
    n : int
        Expected length/number of elements of the 1D array. 

    Returns
    -------
    np.ndarray
        The value as an array of length `n`.

    Raises
    ------
    ValueError
        Raises if the sequence has more than 1 dimension, or if its length is 
        different from `n`.
    '''
    if isinstance(x, (int, float)):
        return np.array([x] * n)
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(
            'Expected input to be either a scalar or a sequence '
            f'compatible to a 1D array; got {x.ndim} dimensions instead.')
    if x.size != n:
        raise ValueError(
            f'Expected input to be of length 1 or {n}; got {x.size} instead.')
    return x
