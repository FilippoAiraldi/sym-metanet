import numpy as np
import casadi as cs

from .. import util as tm_util


class Links_v1(tm_util._ConfigurableObj):
    @tm_util._check_shapes_all
    def get_flow(self, rho, v):
        '''
        Computes the flow (veh/h).
        Rho is current density (veh/km/lane), v the current speed (km/h).
        '''
        return rho * v * self._config.lanes

    @tm_util._check_shapes_all
    def get_Veq(self, rho):
        '''
        Computes the equivalent velocity (km/h).
        Rho is current density (veh/km/lane).
        '''
        a = self._config.a
        v_free = self._config.v_free
        rho_crit = self._config.rho_crit
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
        return rho + (self._config.T * (q_up - q + r - s) /
                      (self._config.lanes * self._config.L))

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
        lanes = self._config.lanes
        L = self._config.L
        T = self._config.T
        tau = self._config.tau
        kappa = self._config.kappa

        lanes_down = np.array([*lanes[1:], lanes[-1]])
        # equilibrium speed error
        t1 = (T / tau) * (Veq - v)
        # on-ramp merging phenomenum
        t2 = (self._config.delta * T) * (r * v) / (L * lanes * (rho + kappa))
        # speed difference error
        t3 = T * (v / L) * (v_up - v)
        # density difference error
        t4 = (self._config.eta * T / tau) * (rho_down - rho) / (L * (rho +
                                                                      kappa))
        # lane drop phenomenum
        t5 = (self._config.phi * T) * (((lanes - lanes_down)
                                         * rho * cs.power(v, 2))
                                       / (L * lanes * self._config.rho_crit))
        return v + t1 - t2 + t3 - t4 - t5

    def get_downstream_density(self, rho):
        return cs.fmin(rho, self._config.rho_crit)

    def get_upstream_speed(self, v):
        return v


class Links_v2(Links_v1):
    pass