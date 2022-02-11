import casadi as cs

from typing import Dict

from ..sim.simulations import Simulation


# Any cost should have 2 mandatory arguments:
#   1. vars: Dict[str, cs.SX]
#   2. pars: Dict[str, cs.SX]
# which are dictionaries of variable names to variables. After, any argument
# is valid. 
# The cost is called automatically by the MPC with the two required args, so 
# the rest should be specified in a lambda. 


def TTS(sim: Simulation, vars: Dict[str, cs.SX], pars: Dict[str, cs.SX]
        ) -> cs.SX:
    # time spent in links and queues
    time_l = sum(cs.sum1(l.lengths * l.lanes * vars[f'rho_{l}'])
                 for l in sim.net.links)
    time_w = sum(vars[f'w_{o}'] for o in sim.net.origins)

    # total cost
    return sim.T * cs.sum2(time_l + time_w)


def TTS_with_input_penalty(
        sim: Simulation, vars: Dict[str, cs.SX], pars: Dict[str, cs.SX],
        weight_r: float = None, weigth_vms: float = None) -> cs.SX:
    # variability of rates
    var_r = 0
    if weight_r is None:
        weight_r = 0
    elif weight_r != 0:
        for o, _ in sim.net.onramps:
            r = cs.horzcat(pars[f'r_{o}_last'], vars[f'r_{o}'])
            var_r += cs.diff(r, 1, 1)**2

    # variability of vms
    var_vms = 0
    if weigth_vms is None:
        weigth_vms = 0
    elif weigth_vms != 0:
        for l, _ in sim.net.links_with_vms:
            v_ctrl = cs.horzcat(pars[f'v_ctrl_{l}_last'], vars[f'v_ctrl_{l}'])
            var_vms += cs.sum1((cs.diff(v_ctrl, 1, 1) / l.v_free)**2)

    # total cost
    return (TTS(sim, vars, pars)
            + weight_r * cs.sum2(var_r) + weigth_vms * cs.sum2(var_vms))
