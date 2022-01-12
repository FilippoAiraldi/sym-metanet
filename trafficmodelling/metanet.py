import numpy as np
import casadi as cs
from dataclasses import dataclass
from . import util as tm_util


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


class Model:
    '''METANET model class'''

    __config: Config

    @property
    def config(self) -> Config:
        return self.__config

    @config.setter
    def config(self, new_config: Config) -> None:
        self.__config = new_config
        for attr in ('lanes', 'L', 'C0'):
            v = getattr(self.__config, attr)
            if isinstance(v, np.ndarray):
                setattr(self.__config, attr, tm_util.force_2d(v))

    def __init__(self, config: Config) -> None:
        '''
        Initialize a new instance of a \'metanet.model\' with the given 
        \'metanet.Config\'.
        '''
        self.config = config
        self.links = Links(self.config)
        self.ramps = Ramps(self.config)

    def q2x(self, w, rho, v):
        '''Creates a vector state x from w, rho, v'''
        return cs.vertcat(w, rho, v)

    @tm_util._check_shapes_cols_out
    def x2q(self, x):
        '''Retuns w, rho, v from a vector state x'''
        O = self.__config.O
        I = self.__config.I
        return ((x[: O], x[O: O + I], x[O + I:])
                if len(x.shape) == 1 else
                (x[: O, :], x[O: O + I, :], x[O + I:, :]))


class Links:
    __config: Config

    def __init__(self, config: None) -> None:
        self.__config = config

    @tm_util._check_shapes_all
    def get_flow(self, rho, v):
        '''
        Computes the flow (veh/h).
        Rho is current density (veh/km/lane), v the current speed (km/h).
        '''
        return rho * v * self.__config.lanes

    @tm_util._check_shapes_all
    def get_Veq(self, rho):
        '''
        Computes the equivalent velocity (km/h).
        Rho is current density (veh/km/lane).
        '''
        a = self.__config.a
        v_free = self.__config.v_free
        rho_crit = self.__config.rho_crit
        return v_free * cs.exp((-1 / a) * cs.power(rho / rho_crit, a))

    @tm_util._nonnegative
    @tm_util._check_shapes_col
    def step_density(self, rho, q, q_up, r, s):
        '''
        Computes the density (veh/km/lane) at the next time instant.
        Rho is current density (veh/km/lane), q the current flow (veh/h),
        q_up the current upstream flow (i.e., the same flow but shifted up
        by one index), r and s the ramps' in- and out-flows.
        '''
        return rho + (self.__config.T * (q_up - q + r - s) /
                      (self.__config.lanes * self.__config.L))

    @tm_util._nonnegative
    @tm_util._check_shapes_col
    def step_speed(self, v, v_up, Veq, rho, rho_down, r):
        '''
        Computes the speed (km/h) at the next time instant.
        v is the current speed (km/h), v_up the current upstream speed
        (i.e., the same speed but shifted up  by one index), Veq the
        equivalent speed, rho the current density (veh/km/lane), rho_down
        the current downstream density (i.e., the same density but shifted
        down by one index), r the ramps' in-flows.
        '''
        lanes = self.__config.lanes
        L = self.__config.L
        T = self.__config.T
        tau = self.__config.tau
        kappa = self.__config.kappa

        lanes_down = np.array([*lanes[1:], lanes[-1]])
        # equilibrium speed error
        t1 = (T / tau) * (Veq - v)
        # on-ramp merging phenomenum
        t2 = (self.__config.delta * T) * (r * v) / (L * lanes * (rho + kappa))
        # speed difference error
        t3 = T * (v / L) * (v_up - v)
        # density difference error
        t4 = (self.__config.eta * T / tau) * (rho_down - rho) / (L * (rho +
                                                                      kappa))
        # lane drop phenomenum
        t5 = (self.__config.phi * T) * (((lanes - lanes_down)
                                         * rho * cs.power(v, 2))
                                        / (L * lanes * self.__config.rho_crit))
        return v + t1 - t2 + t3 - t4 - t5

    def get_downstream_density(self, rho):
        return cs.fmin(rho, self.__config.rho_crit)

    def get_upstream_speed(self, v):
        return v


class Ramps:
    __config: Config

    def __init__(self, config: None) -> None:
        self.__config = config

    @tm_util._nonnegative
    @tm_util._check_shapes_col
    def get_flow(self, u, w, d, rho_first):
        '''
        Computes the flow (veh/h).
        w is the current queue (veh), d the current demand (veh/h), 
        rho_first the current density (veh/km/lane) of the first link after
        the ramp.
        '''
        rho_max = self.__config.rho_max
        r_hat = self.__config.C0 * cs.fmin(1, (rho_max - rho_first) /
                                           (rho_max - self.__config.rho_crit))
        return u * cs.fmin(d + w / self.__config.T, r_hat)

    @tm_util._nonnegative
    @tm_util._check_shapes_col
    def step_queue(self, w, d, r):
        return w + self.__config.T * (d - r)


class util(tm_util._NonInstantiable):
    '''Collection of utility functions for the METANET framework'''

    @staticmethod
    def steady_state(mdl: Model, flow_in):
        '''
        Approximately computes the steady-state values for the given
        incoming flow.
        '''

        I = mdl.config.I
        lanes = mdl.config.lanes
        v_free = mdl.config.v_free

        # create x (rho + v)
        x = cs.SX.sym('x', 2 * I)
        rho, v = x[:I], x[I:]

        # next density should return same density, i.e., rho - rho_next = 0
        q = lanes * rho * v
        q_up = cs.vertcat(flow_in, q[:-1])
        G_rho = rho - mdl.links.step_density(rho, q, q_up, np.zeros((I, 1)),
                                             np.zeros((I, 1)))

        # next speed should return same speed, i.e., v - v_next = 0
        v_up = cs.vertcat(mdl.links.get_upstream_speed(v[0]), v[:-1])
        Veq = mdl.links.get_Veq(rho)
        rho_down = cs.vertcat(rho[1:],
                              mdl.links.get_downstream_density(rho[-1]))
        G_v = v - mdl.links.step_speed(v, v_up, Veq, rho, rho_down,
                                       np.zeros((I, 1)))

        # initial guess for x
        x0 = cs.vertcat(flow_in / lanes / v_free, v_free * np.ones((I, 1)))

        # find roots
        G = cs.vertcat(G_rho, G_v)
        opts = {'constraints': np.ones((2 * I,), dtype=int)}
        G = cs.rootfinder('G', 'newton', cs.Function('g', [x], [G]), opts)
        x_ss = G(x0).full()
        rho_ss, v_ss = x_ss[:I], x_ss[I:]
        return rho_ss, v_ss
