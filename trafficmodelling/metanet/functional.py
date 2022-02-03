import casadi as cs


#################################### NODES ####################################


def get_upstream_flow(q_lasts, beta):
    '''Compute upstream flow for a node (pag. 43).'''
    return cs.sum1(q_lasts) * beta


def get_upstream_speed(v_lasts, q_lasts):
    '''Compute upstream speed for a node (eq. 3.10, pag. 43).'''
    return (v_lasts.T @ q_lasts) / cs.sum1(q_lasts)


def get_downstream_density(rho_firsts):
    '''Compute downstream density for a node (eq. 3.9, pag. 43).'''
    return (rho_firsts.T @ rho_firsts) / cs.sum1(rho_firsts)


#################################### LINKS ####################################


def Veq(rho, v_free, a, rho_crit):
    '''Compute link equivalent speed (eq. 3.4, pag. 39).'''
    return v_free * cs.exp((-1 / a) * cs.power(rho / rho_crit, a))


def Veq_ext(rho, v_free, a, rho_crit, v_ctrl, alpha):
    '''Compute extended link equivalent speed (eq. 3.11, pag. 49).'''
    return cs.fmin(Veq(rho, v_free, a, rho_crit), (1 + alpha) * v_ctrl)


def get_link_flow(rho, v, lanes):
    '''Compute link flow (eq. 3.1, pag. 38).'''
    return rho * v * lanes


def step_link_density(rho, q, q_up, lanes, L, T):
    '''Compute link density at the next timestep (eq. 3.2, pag. 39).'''
    return rho + (T / lanes / L) * (q_up - q)


def step_link_speed(
        v, v_up, rho, rho_down, V, lanes, L, tau, eta, kappa, q_r, delta, T):
    '''Compute link speed at the next timestep (eq. 3.3, pag. 39, and eq. 3.7, pag. 41).'''
    v_next = (v
              + T / tau * (V - v)
              + T / L * v * (v_up - v)
              - eta * T / tau / L * (rho_down - rho) / (rho + kappa))
    v_next[0] -= delta * T / L / lanes * q_r * v[0] / (rho[0] + kappa)
    return v_next

################################### ORIGINS ###################################


def get_onramp_flow(d, w, C, r, rho_max, rho_first, rho_crit, T):
    '''Compute ramp flow (eq. 3.5, pag. 40).'''
    return cs.fmin(d + w / T, C *
                   cs.fmin(r, (rho_max - rho_first) / (rho_max - rho_crit)))


def get_mainorigin_flow(
        d, w, v_ctrl, v_first, rho_crit, v_free, a, lanes, alpha, T):
    '''Compute mainstream origin flow (pag. 51).'''
    # v_ctrl is the control speed of the first link's segment after origin

    # NOTE: should be Veq or Veq_ext?
    V_rho_crit = Veq_ext(rho_crit, v_free, a, rho_crit, v_ctrl, alpha)
    v_lim = cs.fmin(v_ctrl, v_first)

    q_cap = lanes * V_rho_crit * rho_crit
    q_speed = (lanes * v_lim * rho_crit *
               cs.power(-a * cs.log(v_lim / v_free), 1 / a))

    q_lim = cs.if_else(v_lim >= V_rho_crit, q_cap, q_speed)
    return cs.fmin(d + w / T, q_lim)


def step_origin_queue(w, d, q, T):
    '''Compute origin queue at the next timestep (pag. 39).'''
    return w + T * (d - q)
