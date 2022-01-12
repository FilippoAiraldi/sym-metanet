import casadi as cs

from .. import util as tm_util


class Ramps_v1(tm_util._ConfigurableObj):
    @tm_util._nonnegative
    @tm_util._check_shapes_col
    def get_flow(self, u, w, d, rho_first):
        '''
        Computes the flow (veh/h).
        w is the current queue (veh), d the current demand (veh/h), 
        rho_first the current density (veh/km/lane) of the first link after
        the ramp.
        '''
        rho_max = self._config.rho_max
        r_hat = self._config.C0 * cs.fmin(1, (rho_max - rho_first) /
                                          (rho_max - self._config.rho_crit))
        return u * cs.fmin(d + w / self._config.T, r_hat)

    @tm_util._nonnegative
    @tm_util._check_shapes_col
    def step_queue(self, w, d, r):
        return w + self._config.T * (d - r)


class Ramps_v2(Ramps_v1):
    pass
