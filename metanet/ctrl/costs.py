import casadi as cs

from typing import Dict, Union

from ..sim.simulations import Simulation


# Any cost should have 3 mandatory arguments:
#   0. sim: Simulation
#   1. vars: Dict[str, cs.SX]
#   2. pars: Dict[str, cs.SX]
# where 1 and 2 are dictionaries of variable names to variables. After, any
# argument is valid.
# The cost is called automatically by the MPC with the three required args, so
# the rest should be specified in a lambda.


def TTS(sim: Simulation,
        vars: Dict[str, cs.SX],
        pars: Dict[str, cs.SX]) -> cs.SX:
    '''Total-Time-Spent (in queues and in links) cost'''
    # time spent in links and queues
    time_l = sum(cs.sum1(l.lengths * l.lanes * vars[f'rho_{l}'])
                 for l in sim.net.links)
    time_w = sum(vars[f'w_{o}'] for o in sim.net.origins)
    # total cost
    return sim.T * cs.sum2(time_l + time_w)


def input_variability_penalty(sim: Simulation,
                              vars: Dict[str, cs.SX],
                              pars: Dict[str, cs.SX],
                              weight_r: float = None,
                              weigth_vms: float = None) -> cs.SX:
    '''
    Penalizes the input (onramps metering rates and VMS speeds) variability.
    Pass None to each weight to disable the penalization.
    '''
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
    # add everything
    return weight_r * cs.sum2(var_r) + weigth_vms * cs.sum2(var_vms)


def slacks_penalty(sim: Simulation,
                   vars: Dict[str, cs.SX],
                   pars: Dict[str, cs.SX],
                   *weights: Union[float, 'array']):
    '''
    Penalizes the slack variables with the given weights as dot product
                            w^T @ slack_var
    List of weights must have the same length as the number of slacks 
    variables, and will be used in the same order the slack variables where 
    created. Finally, weights must have compatible sizes for a dot product with
    the variable. 
    '''
    slacks = filter(lambda o: o[0].startswith('slack_'), vars.items())
    return sum(cs.dot(w, slack) for w, (_, slack) in zip(weights, slacks))
