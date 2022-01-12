import numpy as np
import casadi as cs
import functools


###############################################################################
################################## INTERNAL ###################################
###############################################################################


class _ConfigurableObj:
    '''A class holding a configuration object.'''

    def __init__(self, config) -> None:
        self._config = config


def _nonnegative(f):
    def _f(*args, **kwargs):
        return cs.fmax(0, f(*args, **kwargs))
    return _f


def _check_shapes(f, type):
    def _f(*args, **kwargs):
        x = args[1]  # 0 is self
        out = f(*args, **kwargs)
        if type == 'col':
            assert out.shape[0] == x.shape[0]
            if len(out.shape) > 1:
                assert out.shape[1] == 1
        elif type == 'cols_out':
            assert sum(o.shape[0] for o in out) == x.shape[0]
            if len(x.shape) > 1:
                assert all(o.shape[1] == x.shape[1] for o in out)
        elif type == 'row':
            assert out.shape == (1, x.shape[1])
        elif type == 'all':
            assert out.shape == x.shape
        return out
    return _f


_check_shapes_col = functools.partial(_check_shapes, type='col')
_check_shapes_cols_out = functools.partial(_check_shapes, type='cols_out')
_check_shapes_row = functools.partial(_check_shapes, type='row')
_check_shapes_all = functools.partial(_check_shapes, type='all')


###############################################################################
################################## EXTERNAL ###################################
###############################################################################


def force_2d(o):
    '''Forces the input to reshape to a 2D array, if 1D.'''
    ndim = len(o.shape)  # casadi does not have ndim property
    if ndim == 1:
        return o.reshape((-1, 1))  # casadi requires a tuple for shape
    if ndim == 2:
        return o
    raise ValueError(f'Unsupported shape {o.shape}')


def force_args_2d(f):
    '''Forces the arguments to the function to reshape to 2D arrays, if 1D.'''
    def _f(*args, **kwargs):
        args = [force_2d(a) if hasattr(a, 'shape') else a for a in args]
        return f(*args, **kwargs)
    return _f


def create_profile(t, x, y):
    '''
    Creates a piece-wise linear profile along time vector t, passing through 
    all x and y points.

    t is the time array (monotonically increasing). x and y are a list of the x
    y coordinates through which the profile must pass.
    '''
    x = list(map(lambda p: np.argmin(np.abs(t - p)), x))
    x = [0, *x, t.size - 1]
    y = [y[0], *y, y[-1]]
    profile = np.zeros_like(t)
    for i in range(len(x) - 1):
        m = (y[i + 1] - y[i]) / (t[x[i + 1]] - t[x[i]])
        q = y[i] - m * t[x[i]]
        profile[x[i]:x[i + 1]] = m * t[x[i]:x[i + 1]] + q
    profile[-1] = profile[-2]
    return profile


@force_args_2d
def TTS(w, rho, T, lanes, L, reduce='sum'):
    '''Total Time Spent cost.'''
    J = T * (cs.sum1(rho * lanes * L) + cs.sum1(w))
    if reduce == 'sum':
        return cs.sum2(J)
    return J
