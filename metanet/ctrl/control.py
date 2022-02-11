import casadi as cs
import numpy as np
from copy import deepcopy
import itertools

from typing import Union, Callable, Tuple, List, Dict, Any

from ..blocks.origins import Origin, OnRamp
from ..blocks.links import Link, LinkWithVms
from ..blocks.networks import Network
from ..sim.simulations import Simulation
from ..util import pad, repinterl


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
    sim.net._check_unique_names()

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
                opti.subject_to(u >= 0.2)
                opti.subject_to(u <= 1)
        for link, _ in net.links_with_vms:
            v_ctrl = cs.vec(vars[f'v_ctrl_{link}'])
            if disable_vms:
                opti.subject_to(v_ctrl == 9e3)  # or v_free
            else:
                opti.subject_to(v_ctrl >= 20)
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

    def to_func(self) -> Callable[[Dict[str, float], Dict[str, float]],
                                  Tuple[Dict[str, float], Dict[str, str]]]:
        '''
        Returns a callable function to automatically run the MPC optimization.
        '''
        def _f(vars_init: Dict[str, float], pars_val: Dict[str, float]):
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
                        f'exception.\nEx. 1: {ex1}\nEx. 2: {ex2}') from ex2
                        
            info['f'] = float(get_value(self.opti.f))
            return {name: get_value(var).reshape(var.shape)
                    for name, var in self.vars.items()}, info
        return _f


class NlpSolver:
    def __init__(self, sim: Simulation, Np: int, Nc: int,
                 cost: Callable[[Simulation, Dict[str, cs.SX], Dict[str, cs.SX]], cs.SX],
                 M: int = 1,
                 solver: str = 'ipopt',
                 solver_opts: Dict[str, Any] = None,
                 disable_ramp_metering: bool = False,
                 disable_vms: bool = False) -> None:

        vars, lbx, ubx, inps_ext, pars = self.__create_vars_and_pars(
            sim.net, Np, Nc, M, disable_ramp_metering, disable_vms)
        g, lbg, ubg = self.__create_constraints(
            sim, vars, inps_ext, pars, Np, M)
        f = cost(sim, vars, pars)

        # save to self
        self.sim = sim
        self.Np, self.Nc, self.M = Np, Nc, M
        self.solver_name = solver
        self.solver_opts = {} if solver_opts is None else solver_opts
        self.vars, self.inps_ext, self.pars = vars, inps_ext, pars
        self.lbx, self.ubx = lbx, ubx
        self.g, self.lbg, self.ubg = g, lbg, ubg
        self.f = f

    def __create_vars_and_pars(self, net: Network, Np: int, Nc: int,
                               M: int,
                               disable_ramp_metering: bool,
                               disable_vms: bool
                               ) -> Tuple[Dict[str, cs.SX], ...]:
        vars, lbx, ubx = {}, [], []
        inps_ext = {}
        pars = {}

        def add_var(name, lb, ub, *size):
            var = cs.SX.sym(name, *size)
            vars[name] = var
            if isinstance(lb, (float, int)):
                lb = np.full(var.shape, lb)
            if isinstance(ub, (float, int)):
                ub = np.full(var.shape, ub)
            assert ub.shape == var.shape == lb.shape
            lbx.append(lb)
            ubx.append(ub)
            return var

        def add_par(name, *size):
            par = cs.SX.sym(name, *size)
            pars[name] = par
            return par

        # create state
        for origin in net.origins:
            add_var(f'w_{origin}', 0, np.inf, 1, M * Np + 1)
        for link in net.links:
            add_var(f'rho_{link}', 0, np.inf, link.nb_seg, M * Np + 1)
            add_var(f'v_{link}', 0, np.inf, link.nb_seg, M * Np + 1)

        # create control and last values of control (ramp rates and vms speeds)
        for origin, _ in net.onramps:
            if disable_ramp_metering:
                u = add_var(f'r_{origin}', 1, 1, 1, Nc)
            else:
                u = add_var(f'r_{origin}', 0.2, 1, 1, Nc)
            inps_ext[f'r_{origin}'] = pad(repinterl(u, 1, M), (0, 0),
                                          (0, M * (Np - Nc)), mode='edge')
        for link, _ in net.links_with_vms:
            if disable_vms:
                v_ctrl = add_var(f'v_ctrl_{link}', 9e3, 9e3, link.nb_vms, Nc)
            else:
                v_ctrl = add_var(f'v_ctrl_{link}', 20, link.v_free,
                                 link.nb_vms, Nc)
            inps_ext[f'v_ctrl_{link}'] = pad(repinterl(v_ctrl, 1, M), (0, 0),
                                             (0, M * (Np - Nc)), mode='edge')

        # create parameters (demands, initial conditions, last action)
        for origin in net.origins:
            add_par(f'd_{origin}', 1, M * Np)
            add_par(f'w0_{origin}', 1, 1)
        for link in net.links:
            add_par(f'rho0_{link}', link.nb_seg, 1)
            add_par(f'v0_{link}', link.nb_seg, 1)
        for origin, _ in net.onramps:
            add_par(f'r_{origin}_last', 1, 1)
        for link, _ in net.links_with_vms:
            add_par(f'v_ctrl_{link}_last', link.nb_vms, 1)
        return vars, lbx, ubx, inps_ext, pars

    def __create_constraints(self, sim: Simulation, vars: Dict[str, cs.SX],
                             vars_ext: Dict[str, cs.SX],
                             pars: Dict[str, cs.SX],
                             Np: int, M: int) -> Tuple[List[cs.SX], ...]:
        net = sim.net
        g, lbg, ubg = [], [], []

        def add_constr(lb, con, ub):
            if isinstance(lb, (float, int)):
                lb = np.full(con.shape, lb)
            if isinstance(ub, (float, int)):
                ub = np.full(con.shape, ub)
            assert ub.shape == con.shape == lb.shape
            lbg.append(lb)
            g.append(con)
            ubg.append(ub)

        # set initial conditions constraint
        for origin in net.origins:
            c = vars[f'w_{origin}'][:, 0] - pars[f'w0_{origin}']
            add_constr(0, c, 0)
        for link in net.links:
            c = vars[f'rho_{link}'][:, 0] - pars[f'rho0_{link}']
            add_constr(0, c, 0)
            c = vars[f'v_{link}'][:, 0] - pars[f'v0_{link}']
            add_constr(0, c, 0)

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
                c = vars[f'w_{origin}'][:, k + 1] - outs[i]
                add_constr(0, c, 0)
                i += 1
            for link in net.links:
                c = vars[f'rho_{link}'][:, k + 1] - outs[i]
                add_constr(0, c, 0)
                c = vars[f'v_{link}'][:, k + 1] - outs[i + 1]
                add_constr(0, c, 0)
                i += 2

        return g, lbg, ubg

    def add_constraint(self, lb: float, con: cs.SX, ub: float) -> None:
        if isinstance(lb, (float, int)):
            lb = np.full(con.shape, lb)
        if isinstance(ub, (float, int)):
            ub = np.full(con.shape, ub)
        assert lb.shape == con.shape == ub.shape
        self.lbg.append(lb)
        self.g.append(con)
        self.ubg.append(ub)

    def to_func(self) -> Callable[[Dict[str, float], Dict[str, float]],
                                  Tuple[Dict[str, float], Dict[str, str]]]:
        # flatten into vectos
        self.x = cs.vertcat(*(cs.vec(o) for o in self.vars.values()))
        self.p = cs.vertcat(*(cs.vec(o) for o in self.pars.values()))
        for attr in ('lbx', 'ubx', 'g', 'lbg', 'ubg'):
            setattr(self, attr,
                    cs.vertcat(*(cs.vec(o) for o in getattr(self, attr))))

        # build solver
        nlp = {'x': self.x, 'f': self.f, 'g': self.g, 'p': self.p}
        solver = cs.nlpsol('solver', self.solver_name, nlp, self.solver_opts)
        self.solver = solver

        # prepare stuff
        Nx, Nc = self.M * self.Np + 1, self.Nc
        net = self.sim.net

        def _f(vars_init: Dict[str, float], pars_val: Dict[str, float]):
            # sourcery skip: for-append-to-extend, list-comprehension, merge-list-appends-into-extend
            # same order as creation
            p, x0 = [], []
            for origin in net.origins:
                p.append(pars_val[f'd_{origin}'])
                p.append(pars_val[f'w0_{origin}'])
                x0.append(vars_init[f'w_{origin}'])
            for link in net.links:
                p.append(pars_val[f'rho0_{link}'])
                p.append(pars_val[f'v0_{link}'])
                x0.append(vars_init[f'rho_{link}'])
                x0.append(vars_init[f'v_{link}'])
            for onramp, _ in net.onramps:
                p.append(pars_val[f'r_{onramp}_last'])
                x0.append(vars_init[f'r_{onramp}'])
            for link, _ in net.links_with_vms:
                p.append(pars_val[f'v_ctrl_{link}_last'])
                x0.append(vars_init[f'v_ctrl_{link}'])
            p = cs.vertcat(*[cs.vec(o) for o in p])
            x0 = cs.vertcat(*[cs.vec(o) for o in x0])
            assert p.shape == self.p.shape
            assert x0.shape == self.x.shape

            sol = solver(x0=x0, p=p, lbx=self.lbx, ubx=self.ubx,
                         lbg=self.lbg, ubg=self.ubg)

            info = {float(sol['f'])}
            status = solver.stats()['return_status']
            if status != 'Solve_Succeeded':
                info['error'] = status

            x_opt = sol['x']
            i, out = 0, {}
            for origin in net.origins:
                out[f'w_{origin}'] = x_opt[i:i + Nx].reshape((1, Nx))
                i += Nx
            for link in net.links:
                out[f'rho_{link}'] = x_opt[i:i + link.nb_seg * Nx
                                           ].reshape((link.nb_seg, Nx))
                i += link.nb_seg * Nx
                out[f'v_{link}'] = x_opt[i:i + link.nb_seg * Nx
                                         ].reshape((link.nb_seg, Nx))
                i += link.nb_seg * Nx
            for ornamp, _ in net.onramps:
                out[f'r_{ornamp}'] = x_opt[i:i + Nc
                                           ].reshape((1, Nc))
                i += Nc
            for link, _ in net.links_with_vms:
                out[f'v_ctrl_{link}'] = x_opt[i:i + link.nb_vms * Nc
                                              ].reshape((link.nb_vms, Nc))
                i += link.nb_vms * Nc
            return out, info

        return _f


# def run_sim_with_MPC(sim: Simulation, mpc: Union[MPC, NlpSolver], K: int,
#                      use_tqdm: bool = False) -> None:
#     '''
#     experimental: automatically run the simulation with an MPC.
#     Be sure to set the initial conditions before calling this method.
#     '''

#     if use_tqdm:
#         from tqdm import tqdm
#     else:
#         def tqdm(iter, **kwargs):
#             return iter

#     # create functions
#     F = sim2func(sim, out_nonneg=True)
#     MPC = mpc.to_func()
#     M, Np, Nc = mpc.M, mpc.Np, mpc.Nc

#     # initialize true and nominal last solutions
#     vars_last = {
#         **{f'w_{o}': cs.repmat(o.queue[0], 1, M * Np + 1)
#            for o in sim.net.origins},
#         **{f'rho_{l}': cs.repmat(l.density[0], 1, M * Np + 1)
#            for l in sim.net.links},
#         **{f'v_{l}': cs.repmat(l.speed[0], 1, M * Np + 1)
#            for l in sim.net.links},
#         **{f'r_{o}': np.ones((1, Nc)) for o, _ in sim.net.onramps},
#     }

#     # simulation main loop
#     for k in tqdm(range(K), total=K):
#         if k % M == 0:
#             # get future demands (only in true model)
#             dist = {}
#             for origin in sim.net.origins:
#                 d = origin.demand[k:k + M * Np]
#                 dist[f'd_{origin}'] = np.pad(d, (0, M * Np - len(d)),
#                                              mode='edge').reshape(1, -1)

#             # run MPC
#             vars_init = {
#                 var: shift(val, axis=2)
#                 for var, val in vars_last.items()
#             }
#             pars_val = {
#                 **dist,
#                 **{f'w0_{o}': o.queue[k] for o in sim.net.origins},
#                 **{f'rho0_{l}': l.density[k] for l in sim.net.links},
#                 **{f'v0_{l}': l.speed[k] for l in sim.net.links},
#                 **{f'r_{o}_last': vars_last[f'r_{o}'][0, 0]
#                     for o, _ in sim.net.onramps},
#                 **{f'v_ctrl_{l}_last': vars_last[f'v_ctrl_{l}_last'][:, 0]
#                     for l, _ in sim.net.links_with_vms},
#             }
#             vars_last, info = MPC(vars_init, pars_val)
#             if 'error' in info:
#                 tqdm.write(f'{k:{len(str(K))}}: ({sim.net.name}) '
#                            + info['error'] + '.')

#         # set onramp metering rate and vms speed control
#         for onramp, _ in sim.net.onramps:
#             onramp.rate[k] = vars_last[f'r_{onramp}'][0, 0]
#         for link, _ in sim.net.links_with_vms:
#             v_ctrl = vars_last[f'v_ctrl_{link}'][:, 0]
#             link.v_ctrl[k] = v_ctrl.reshape((link.nb_vms, 1))

#         (sim.net.O1.flow[k], sim.net.O1.queue[k + 1],
#             sim.net.O2.flow[k], sim.net.O2.queue[k + 1],
#             sim.net.L1.flow[k], sim.net.L1.density[k + 1],
#             sim.net.L1.speed[k + 1],
#             sim.net.L2.flow[k], sim.net.L2.density[k + 1],
#             sim.net.L2.speed[k + 1]
#          ) = F(sim.net.O1.queue[k],
#                sim.net.O2.queue[k],
#                sim.net.L1.density[k], sim.net.L1.speed[k],
#                sim.net.L2.density[k], sim.net.L2.speed[k],
#                sim.net.O2.rate[k],
#                sim.net.O1.demand[k], sim.net.O2.demand[k])
