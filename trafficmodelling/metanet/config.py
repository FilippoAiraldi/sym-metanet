import numpy as np
from dataclasses import dataclass

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
        # TODO: check everythin for positivity
        if self.T < 0:
            raise ValueError('sampling time T cannot be negative.')

        # TODO: check I and O sizes

        # make 1D arrays 2D
        for attr in ('lanes', 'L', 'C0'):
            v = getattr(self, attr)
            if isinstance(v, np.ndarray):
                setattr(self, attr, tm_util.force_2d(v))
