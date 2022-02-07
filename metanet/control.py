import casadi as cs
from copy import deepcopy
import itertools

from typing import Union, Callable, Tuple, List, Dict, Any

from .origins import Origin, OnRamp
from .links import Link, LinkWithVms
from .networks import Network
from .simulations import Simulation
from .util import pad, repinterl


##################################### UTIL ####################################


def __create_args(in_states: bool,
                  in_inputs: bool,
                  in_disturbances: bool,
                  in_origin_params: bool,
                  in_link_params: bool,
                  in_sim_params: bool,
                  sim: Simulation,
                  origins: List[Origin],
                  onramps: List[OnRamp],
                  links: List[Link],
                  links_with_vms: List[LinkWithVms],
                  csXX: Union[cs.SX, cs.MX],
                  k: int) -> Dict[str, Union[cs.SX, cs.MX]]:
    args = {}

    def add_arg(name, *size):
        var = csXX.sym(name, *size)
        args[name] = var
        return var

    if in_states:
        for origin in origins:
            origin.queue[k] = add_arg(f'w_{origin.name}', 1, 1)
        for link in links:
            link.density[k] = add_arg(f'rho_{link.name}', link.nb_seg, 1)
            link.speed[k] = add_arg(f'v_{link.name}', link.nb_seg, 1)
    if in_inputs:
        for origin in onramps:
            origin.rate[k] = add_arg(f'r_{origin.name}', 1, 1)
        for link in links_with_vms:
            link.v_ctrl[k] = add_arg(f'v_ctrl_{link.name}', link.nb_vms, 1)
    if in_disturbances:
        for origin in origins:
            # if the demand is an array, assigning a SX to it will cause nan
            # and create problems
            if not isinstance(origin.demand, list):
                raise TypeError('unknown behaviour for demand not as list')
            origin.demand[k] = add_arg(f'd_{origin.name}', 1, 1)
    if in_origin_params:
        for origin in onramps:
            origin.capacity = add_arg(f'C_{origin.name}', 1, 1)
    if in_link_params:
        for link in links:
            for name in ['lanes', 'lengths', 'v_free', 'rho_crit', 'a']:
                setattr(link, name, add_arg(f'{name}_{link.name}', 1, 1))
    if in_sim_params:
        for name in ['rho_max', 'eta', 'tau', 'kappa', 'delta', 'alpha', 'T']:
            setattr(sim, name, add_arg(name, 1, 1))
    return args


def __create_outs(out_nonneg: bool,
                  out_nonstate: bool,
                  origins: List[Origin],
                  links: List[Link],
                  k: int) -> Dict[str, Union[cs.SX, cs.MX]]:
    outs = {}  # sourcery skip: dict-comprehension
    nonneg = (lambda o: cs.fmax(0, o)) if out_nonneg else (lambda o: o)
    for origin in origins:
        if out_nonstate:
            outs[f'q_{origin.name}'] = nonneg(origin.flow[k])
        outs[f'w+_{origin.name}'] = nonneg(origin.queue[k + 1])
    for link in links:
        if out_nonstate:
            outs[f'q_{link.name}'] = nonneg(link.flow[k])
        outs[f'rho+_{link.name}'] = nonneg(link.density[k + 1])
        outs[f'v+_{link.name}'] = nonneg(link.speed[k + 1])
    return outs


def sim2func(sim: Simulation,
             in_states: bool = True,
             in_inputs: bool = True,
             in_disturbances: bool = True,
             in_origin_params: bool = False,
             in_link_params: bool = False,
             in_sim_params: bool = False,
             out_nonstate: bool = False,
             out_nonneg: bool = False,
             k: int = 0,
             funcname: str = 'F',
             cs_type: str = 'SX',
             return_args: bool = False) -> cs.Function:
    '''
    Converts the simulation to a one-step lookahead casadi function 
                    x_next = f( x, u, d )
    where x and x_next are the current and next states, u the input (metering
    rates, speed controls, etc.), and d the disturbances (origin demands). 

    In this case the state x is a tuple of 1 queue for each origin and N 
    densities and velocities for each link (where N is the number of segments 
    in that link).

    Parameters
    ----------
        sim : metanet.Simulation
            Simulation from which to extract the casadi function. If not 
            symbolic, the values/parameters are taken from this simulation as 
            they are.

        in_states, in_inputs, in_disturbances, 
        in_link_params in_origin_params, in_sim_params : bool, optional
            Whether to make various inputs symbolic or not.

        out_nonstate : bool, optional
            Whether the function should output also other quantities apart 
            from the next states, namely origin and link flows (not in state).
            Note that these quantities belong to the current timestep k, while
            the next states are from time k + 1. Defaults to False.

        out_nonneg : bool, optional
            Whether to force outputs to be non negative. Defaults to False.

        k : int, optional
            Specify a specific timestep to evaluate the simulation. Defaults 
            to 0.

        cs_type : str, {'SX', 'MX'}, optional
            Instruct whether to use casadi symbolic SX or symbolic MX.

         return_args : bool 
            Whether also the symbolic arguments (in and out) of the function 
            should be returned. Defaults to False.

    Returns
    -------
        F : casadi.Function
            A function that computes the next states for the simulation.

        args : dict[str, SX | MX], optional
            A dictionary of names and symbolic arguments (inputs of F). Only 
            returned if return_args is True.

        out : dict[str, SX | MX], optional
            A dictionary of names and symbolic outputs (outputs of F). Only 
            returned if return_args is True.

    Raises
    ------
        ValueError : non-unique naming
            If the same name is used more than once for nodes or links.
    '''

    if cs_type in {'SX', 'sx'}:
        csXX = cs.SX
    elif cs_type in {'MX', 'mx'}:
        csXX = cs.MX
    else:
        raise ValueError('Invalid casadi type; must be either'
                         f'\'SX\' or \'MX\', got \'{cs_type}\'')

    # since we are using names to identify symbolic variables, better not to
    # have conflicts
    sim._check_unique_names()

    # create a copy of the simulation which will be symbolic
    sym_sim = deepcopy(sim)
    origins = sym_sim.net.origins
    onramps = list(map(lambda o: o[0], sym_sim.net.onramps))
    links = sym_sim.net.links
    links_with_vms = list(map(lambda o: o[0], sym_sim.net.links_with_vms))

    # create the function arguments
    args = __create_args(
        in_states, in_inputs, in_disturbances, in_origin_params,
        in_link_params, in_sim_params, sym_sim, origins, onramps, links,
        links_with_vms, csXX, k)

    # perform one step
    sym_sim.step(k)

    # gather outputs
    outs = __create_outs(out_nonneg, out_nonstate, origins, links, k)

    # create function from state k to state k+1
    F = cs.Function(funcname,
                    list(args.values()),
                    list(outs.values()),
                    list(args.keys()),
                    list(outs.keys()))
    return ((F, args, outs) if return_args else F)


##################################### COST ####################################


def TTS(
        sim: Simulation, vars: Dict[str, cs.SX],
        pars: Dict[str, cs.SX]) -> cs.SX:
    # time spent in links and queues
    time_l = sum(cs.sum1(l.lengths * l.lanes * vars[f'rho_{l}'])
                 for l in sim.net.links)
    time_w = sum(vars[f'w_{o}'] for o in sim.net.origins)

    # total cost
    return sim.T * cs.sum2(time_l + time_w)


def TTS_with_input_penalty(
        sim: Simulation, vars: Dict[str, cs.SX], pars: Dict[str, cs.SX],
        weight_r: float = 0.4, weigth_vms: float = 0.4) -> cs.SX:
    # variability of rates
    var_r = 0
    if weight_r != 0:
        for o, _ in sim.net.onramps:
            r = cs.horzcat(pars[f'r_{o}_last'], vars[f'r_{o}'])
            var_r += cs.diff(r, 1, 1)**2

    # variability of vms
    var_vms = 0
    if weigth_vms != 0:
        for l, _ in sim.net.links_with_vms:
            v_ctrl = cs.horzcat(pars[f'v_ctrl_{l}_last'], vars[f'v_ctrl_{l}'])
            var_vms += (cs.diff(v_ctrl, 1, 1) / l.v_free)**2

    # total cost
    return (TTS(sim, vars, pars)
            + weight_r * cs.sum2(var_r) + weigth_vms * cs.sum2(var_vms))


################################## OPTI/MPC ###################################


class MPC:
    def __init__(self, sim: Simulation, Np: int, Nc: int,
                 cost: Callable[[Simulation, Dict[str, cs.SX], Dict[str, cs.SX]], cs.SX],
                 M: int = 1,
                 disable_ramp_metering: bool = False,
                 disable_vms: bool = False,
                 solver: str = 'ipopt',
                 plugin_opts: Dict[str, Any] = None,
                 solver_opts: Dict[str, Any] = None) -> None:
        '''
        Instantiates an MPC for METANET control.

        Parameters
        ----------
            sim : metanet.Simulation
                The simulation to control via MPC.

            Np, Nc : int
                Prediction and control horizon. Must be positive and Np >= Nc.

            cost : Callable
                The cost function to be minimized.

            M : int, optional
                Control action multiplier, i.e., one action free every M steps,
                or, in other words, every action is kept constant for M steps.

            disable_ramp_metering, disable_vms : bool
                Whether to disable control of ramp metering rates and Variable
                Sign Message speed control. 
        '''
        opti = cs.Opti()
        vars, vars_ext, pars = self.__create_vars_and_pars(
            sim.net, opti, Np, Nc, M)
        self.__create_constraints(
            sim, opti, vars, vars_ext, pars, Np, M,
            disable_ramp_metering, disable_vms)
        self.__create_objective_and_others(
            sim, opti, vars, pars, cost, solver, plugin_opts, solver_opts)

        # save to self
        self.opti = opti
        self.vars, self.vars_ext, self.pars = vars, vars_ext, pars
        self.sim = sim
        self.Np, self.Nc, self.M = Np, Nc, M

    def __create_vars_and_pars(self, net: Network, opti: cs.Opti, Np: int,
                               Nc: int, M: int) -> Tuple[Dict[str, cs.SX], ...]:
        vars, vars_ext, pars = {}, {}, {}

        # create state
        for origin in net.origins:
            vars[f'w_{origin}'] = opti.variable(1, M * Np + 1)
        for link in net.links:
            vars[f'rho_{link}'] = opti.variable(link.nb_seg, M * Np + 1)
            vars[f'v_{link}'] = opti.variable(link.nb_seg, M * Np + 1)

        # create control and last values of control (ramp rates and vms speeds)
        for origin, _ in net.onramps:
            u = opti.variable(1, Nc)
            vars[f'r_{origin}'] = u
            vars_ext[f'r_{origin}'] = pad(repinterl(u, 1, M), (0, 0),
                                          (0, M * (Np - Nc)), mode='edge')
        for link, _ in net.links_with_vms:
            v_ctrl = opti.variable(link.nb_vms, Nc)
            vars[f'v_ctrl_{link}'] = v_ctrl
            vars_ext[f'v_ctrl_{link}'] = pad(repinterl(v_ctrl, 1, M), (0, 0),
                                             (0, M * (Np - Nc)), mode='edge')

        # create parameters (demands, initial conditions, last action)
        for origin in net.origins:
            pars[f'd_{origin}'] = opti.parameter(1, M * Np)
            pars[f'w0_{origin}'] = opti.parameter(1, 1)
        for link in net.links:
            pars[f'rho0_{link}'] = opti.parameter(link.nb_seg, 1)
            pars[f'v0_{link}'] = opti.parameter(link.nb_seg, 1)
        for origin, _ in net.onramps:
            pars[f'r_{origin}_last'] = opti.parameter(1, 1)
        for link, _ in net.links_with_vms:
            pars[f'v_ctrl_{link}_last'] = opti.parameter(link.nb_vms, 1)

        return vars, vars_ext, pars

    def __create_constraints(
            self, sim: Simulation, opti: cs.Opti, vars: Dict[str, cs.SX],
            vars_ext: Dict[str, cs.SX], pars: Dict[str, cs.SX],
            Np: int, M: int,
            disable_ramps: bool, disable_vms: bool) -> None:
        net = sim.net

        # set input constraints
        for origin, _ in net.onramps:
            u = cs.vec(vars[f'r_{origin}'])
            if disable_ramps:
                opti.subject_to(u == 1)
            else:
                opti.subject_to(u >= 0)
                opti.subject_to(u <= 1)
        for link, _ in net.links_with_vms:
            v_ctrl = cs.vec(vars[f'v_ctrl_{link}'])
            if disable_vms:
                opti.subject_to(v_ctrl == 9e3)  # or v_free
            else:
                opti.subject_to(v_ctrl >= 0)
                opti.subject_to(v_ctrl <= link.v_free)

        # set state positivity constraints
        for origin in net.origins:
            opti.subject_to(cs.vec(vars[f'w_{origin}']) >= 0)
        for link in net.links:
            opti.subject_to(cs.vec(vars[f'rho_{link}']) >= 0)
            opti.subject_to(cs.vec(vars[f'v_{link}']) >= 0)

        # set initial conditions constraint
        for origin in net.origins:
            opti.subject_to(vars[f'w_{origin}'][:, 0] == pars[f'w0_{origin}'])
        for link in net.links:
            opti.subject_to(vars[f'rho_{link}'][:, 0] == pars[f'rho0_{link}'])
            opti.subject_to(vars[f'v_{link}'][:, 0] == pars[f'v0_{link}'])

        # create x+ = F(x, u, d)
        # Example
        # (w_O1,w_O2,rho_L1[4],v_L1[4],rho_L2[2],v_L2[2],r_O2,d_O1,d_O2) ->
        # (w+_O1,w+_O2,rho+_L1[4],v+_L1[4],rho+_L2[2],v+_L2[2])
        F = sim2func(sim, out_nonneg=True)

        # set trajectory evolution constraint
        for k in range(M * Np):
            outs = F(
                *(vars[f'w_{o}'][:, k] for o in net.origins),
                *itertools.chain.from_iterable(
                    (vars[f'rho_{l}'][:, k], vars[f'v_{l}'][:, k])
                    for l in net.links),
                *(vars_ext[f'r_{o}'][:, k] for o, _ in net.onramps),
                *(vars_ext[f'v_ctrl_{l}'][:, k]
                  for l, _ in net.links_with_vms),
                *(pars[f'd_{o}'][:, k] for o in net.origins))
            i = 0
            for origin in net.origins:
                opti.subject_to(vars[f'w_{origin}'][:, k + 1] == outs[i])
                i += 1
            for link in net.links:
                opti.subject_to(vars[f'rho_{link}'][:, k + 1] == outs[i])
                opti.subject_to(vars[f'v_{link}'][:, k + 1] == outs[i + 1])
                i += 2

    def __create_objective_and_others(
            self, sim: Simulation, opti: cs.Opti, vars: Dict[str, cs.SX],
            pars: Dict[str, cs.SX],
            cost: Callable, solver: str,
            plugin_opts: Dict[str, Any], solver_opts: Dict[str, Any]) -> None:
        # optimization criterion
        opti.minimize(cost(sim, vars, pars))

        # set solver
        opti.solver(solver,
                    {} if plugin_opts is None else plugin_opts,
                    {} if solver_opts is None else solver_opts)

    def to_func(self) -> Callable[[int, Dict[str, float], Dict[str, float]],
                                  Tuple[Dict[str, float], Dict[str, str]]]:
        '''
        Returns a callable function to automatically run the MPC optimization.
        '''
        def _f(k: int, vars_init: Dict[str, float],
                pars_val: Dict[str, float]):
            for par in self.pars:
                self.opti.set_value(self.pars[par], pars_val[par])
            for var in self.vars:
                self.opti.set_initial(self.vars[var], vars_init[var])
            try:
                sol = self.opti.solve()
                info = {}
                get_value = lambda o: sol.value(o)
            except Exception as ex1:
                try:
                    info = {'error': self.opti.debug.stats()['return_status']}
                    # + ' (' + str(ex1).replace('\n', ' ') + ')'}
                    get_value = lambda o: self.opti.debug.value(o)
                except Exception as ex2:
                    raise RuntimeError(
                        'error during handling of first '
                        f'exception.\nEx. 1: {ex1}\nEx. 2: {ex2}')
            return {name: get_value(var).reshape(var.shape)
                    for name, var in self.vars.items()}, info
        return _f
