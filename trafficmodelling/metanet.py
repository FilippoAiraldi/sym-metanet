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


def init(config: Config) -> None:
    '''Sets the METANET model's configuration.'''
    F.cfg = config
    for attr in ('lanes', 'L', 'C0'):
        v = getattr(F.cfg, attr)
        if isinstance(v, np.ndarray):
            setattr(F.cfg, attr, tm_util.force_2d(v))


class F(tm_util._NonInstantiable):
    '''METANET functional namespace.'''
    cfg: Config

    class links(tm_util._NonInstantiable):
        '''Collection of functions for links.'''

        @staticmethod
        @tm_util._check_shapes_all
        def get_flow(rho, v):
            '''
            Computes the flow (veh/h).
            Rho is current density (veh/km/lane), v the current speed (km/h).
            '''
            return rho * v * F.cfg.lanes

        @staticmethod
        @tm_util._check_shapes_all
        def get_Veq(rho):
            '''
            Computes the equivalent velocity (km/h).
            Rho is current density (veh/km/lane).
            '''
            a, v_free, rho_crit = F.cfg.a, F.cfg.v_free, F.cfg.rho_crit
            return v_free * cs.exp((-1 / a) * cs.power(rho / rho_crit, a))

        @staticmethod
        @tm_util._nonnegative
        @tm_util._check_shapes_col
        def step_density(rho, q, q_up, r, s):
            '''
            Computes the density (veh/km/lane) at the next time instant.
            Rho is current density (veh/km/lane), q the current flow (veh/h),
            q_up the current upstream flow (i.e., the same flow but shifted up
            by one index), r and s the ramps' in- and out-flows.
            '''
            return rho + F.cfg.T * (q_up - q + r - s) / (F.cfg.lanes * F.cfg.L)

        @staticmethod
        @tm_util._nonnegative
        @tm_util._check_shapes_col
        def step_speed(v, v_up, Veq, rho, rho_down, r):
            '''
            Computes the speed (km/h) at the next time instant.
            v is the current speed (km/h), v_up the current upstream speed
            (i.e., the same speed but shifted up  by one index), Veq the
            equivalent speed, rho the current density (veh/km/lane), rho_down
            the current downstream density (i.e., the same density but shifted
            down by one index), r the ramps' in-flows.
            '''
            lanes, L, T, tau, kappa = (
                F.cfg.lanes, F.cfg.L, F.cfg.T, F.cfg.tau, F.cfg.kappa)
            lanes_down = np.array([*lanes[1:], lanes[-1]])
            # equilibrium speed error
            t1 = (T / tau) * (Veq - v)
            # on-ramp merging phenomenum
            t2 = (F.cfg.delta * T) * (r * v) / (L * lanes * (rho + kappa))
            # speed difference error
            t3 = T * (v / L) * (v_up - v)
            # density difference error
            t4 = (F.cfg.eta * T / tau) * (rho_down - rho) / (L * (rho + kappa))
            # lane drop phenomenum
            t5 = (F.cfg.phi * T) * (((lanes - lanes_down)
                                     * rho * cs.power(v, 2))
                                    / (L * lanes * F.cfg.rho_crit))
            return v + t1 - t2 + t3 - t4 - t5

        @staticmethod
        def get_downstream_density(rho):
            return cs.fmin(rho, F.cfg.rho_crit)

        @staticmethod
        def get_upstream_speed(v):
            return v

    class ramps(tm_util._NonInstantiable):
        '''Collection of functions for ramps.'''
        @staticmethod
        @tm_util._nonnegative
        @tm_util._check_shapes_col
        def get_flow(u, w, d, rho_first):
            '''
            Computes the flow (veh/h).
            w is the current queue (veh), d the current demand (veh/h), 
            rho_first the current density (veh/km/lane) of the first link after
            the ramp.
            '''
            rho_max = F.cfg.rho_max
            r_hat = F.cfg.C0 * cs.fmin(1, (rho_max - rho_first) /
                                       (rho_max - F.cfg.rho_crit))
            return u * cs.fmin(d + w / F.cfg.T, r_hat)

        @staticmethod
        @tm_util._nonnegative
        @tm_util._check_shapes_col
        def step_queue(w, d, r):
            return w + F.cfg.T * (d - r)

    class util(tm_util._NonInstantiable):
        '''Collection of utility functions for the METANET framework'''
        @ staticmethod
        def steady_state(flow_in):
            '''
            Approximately computes the steady-state values for the given
            incoming flow.
            '''
            I, lanes, v_free = F.cfg.I, F.cfg.lanes, F.cfg.v_free

            # create x (rho + v)
            x = cs.SX.sym('x', 2 * I)
            rho, v = x[:I], x[I:]

            # next density should return same density, i.e., rho - rho_next = 0
            q = lanes * rho * v
            q_up = cs.vertcat(flow_in, q[:-1])
            G_rho = rho - F.links.step_density(rho, q, q_up, np.zeros((I, 1)),
                                               np.zeros((I, 1)))

            # next speed should return same speed, i.e., v - v_next = 0
            v_up = cs.vertcat(F.links.get_upstream_speed(v[0]), v[:-1])
            Veq = F.links.get_Veq(rho)
            rho_down = cs.vertcat(rho[1:],
                                  F.links.get_downstream_density(rho[-1]))
            G_v = v - F.links.step_speed(v, v_up, Veq, rho, rho_down,
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

        @staticmethod
        @tm_util._check_shapes_cols_out
        def x2q(x):
            '''Retuns w, rho, v from a vector state x'''
            O = F.cfg.O
            I = F.cfg.I
            return ((x[: O], x[O: O + I], x[O + I:])
                    if len(x.shape) == 1 else
                    (x[: O, :], x[O: O + I, :], x[O + I:, :]))

        @staticmethod
        def q2x(w, rho, v):
            '''Creates a vector state x from w, rho, v'''
            return cs.vertcat(w, rho, v)
