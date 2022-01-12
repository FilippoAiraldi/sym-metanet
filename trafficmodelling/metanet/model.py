import numpy as np
import casadi as cs

from .config import Config
from . import links
from . import ramps
from .. import util as tm_util


class Model:
    '''METANET model class'''

    config: Config

    def __init__(self, config: Config, version: int = 1) -> None:
        '''
        Initialize a new instance of a \'metanet.model\' with the given 
        \'metanet.Config\'.
        '''
        self.config = config
        if version == 1:
            self.links = links.Links(self.config)
            self.ramps = ramps.Ramps(self.config)
        elif version == 2:
            pass
        else:
            raise ValueError(f'Unknown version {version}; 1 and 2 are valid.')

    def q2x(self, w, rho, v):
        '''Creates a vector state x from w, rho, v'''
        return cs.vertcat(w, rho, v)

    @tm_util._check_shapes_cols_out
    def x2q(self, x):
        '''Retuns w, rho, v from a vector state x'''
        O = self.config.O
        I = self.config.I
        return ((x[: O], x[O: O + I], x[O + I:])
                if len(x.shape) == 1 else
                (x[: O, :], x[O: O + I, :], x[O + I:, :]))
