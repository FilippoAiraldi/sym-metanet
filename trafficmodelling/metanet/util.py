import numpy as np
import casadi as cs

from .model import Model


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
