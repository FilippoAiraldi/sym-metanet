import numpy as np
import casadi as cs

from .config import Config
from . import links
from . import ramps
from .. import util as tm_util


class Model(tm_util._ConfigurableObj):
    '''METANET model class'''

    def __init__(self, config: Config, version: int = 1) -> None:
        '''
        Initialize a new instance of a \'metanet.model\' with the given 
        \'metanet.Config\'.
        '''
        super().__init__(config)
        if version == 1:
            self.links = links.Links_v1(self._config)
            self.ramps = ramps.Ramps_v1(self._config)
        elif version == 2:
            self.links = links.Links_v2(self._config)
            self.ramps = ramps.Ramps_v2(self._config)
        else:
            raise ValueError(f'Unknown version {version}; 1 and 2 are valid.')

    def q2x(self, w, rho, v):
        '''Creates a vector state x from w, rho, v'''
        return cs.vertcat(w, rho, v)

    @tm_util._check_shapes_cols_out
    def x2q(self, x):
        '''Retuns w, rho, v from a vector state x'''
        O = self._config.O
        I = self._config.I
        return ((x[: O], x[O: O + I], x[O + I:])
                if len(x.shape) == 1 else
                (x[: O, :], x[O: O + I, :], x[O + I:, :]))
