import numpy as np
from dataclasses import dataclass, asdict

from .. import util as tm_util


@dataclass
class Config:
    '''Configurations for a METANET model.'''
    O: int
    I: int
    C0: float
    v_free: float
    rho_crit: float
    rho_max: float
    a: float
    delta: float
    eta: float
    kappa: float
    tau: float
    phi: float
    lanes: np.ndarray
    L: np.ndarray
    T: float

    def __post_init__(self):
        # check positivity
        for key, val in asdict(self).items():
            if not ((val >= 0).all()
                    if isinstance(val, np.ndarray) else
                    val >= 0):
                raise ValueError(f'parameter \'{key}\' cannot be negative.')

        # check array-like parameters
        for name, n, types in [('lanes', 'I', (int,)),
                               ('L', 'I', (float, int)),
                               ('C0', 'O', (float, int))]:
            arr = getattr(self, name)

            # check type
            is_scalar = isinstance(arr, types)
            is_array = isinstance(arr, np.ndarray)
            if not is_scalar and not is_array or (
                    is_array and self.lanes.dtype not in types):
                wrong_type = arr.dtype if is_array else type(arr)
                raise ValueError(f'parameter \'{name}\' must be a scalar '
                                 f'or an array of types \'{types}\'; '
                                 f'got {wrong_type} instead.')

            # check size
            sz = getattr(self, n)
            if is_scalar:
                if sz != 1:
                    raise ValueError(f'parameter \'{n}\' expected to be 1; '
                                     f'got {sz} instead.')
            else:
                if arr.ndim > 2 or (arr.ndim == 2 and arr.shape[1] != 1):
                    raise ValueError(
                        f'parameter \'{name}\' must be a 2D '
                        f'column vector; got {arr.shape} instead.')
                if arr.size != sz:
                    raise ValueError(
                        f'parameter \'{name}\' must have {sz} '
                        f'elements (like \'{n}\'); got {arr.size} instead.')
            setattr(self, name, arr)

        # make 1D arrays 2D
        for attr in ('lanes', 'L', 'C0'):
            v = getattr(self, attr)
            if isinstance(v, np.ndarray):
                setattr(self, attr, tm_util.force_2d(v))
