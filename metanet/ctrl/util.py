import casadi as cs
import numpy as np
from copy import deepcopy

from typing import Union, List, Dict, Callable, Any

from ..blocks.origins import Origin, OnRamp
from ..blocks.links import Link, LinkWithVms
from ..sim.simulations import Simulation
from ..util import shift


def __create_sim2func_args(in_states: bool,
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


def __create_sim2func_outs(out_nonneg: bool,
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
    args = __create_sim2func_args(
        in_states, in_inputs, in_disturbances, in_origin_params,
        in_link_params, in_sim_params, sym_sim, origins, onramps, links,
        links_with_vms, csXX, k)

    # perform one step
    sym_sim.step(k)

    # gather outputs
    outs = __create_sim2func_outs(out_nonneg, out_nonstate, origins, links, k)

    # create function from state k to state k+1
    F = cs.Function(funcname,
                    list(args.values()),
                    list(outs.values()),
                    list(args.keys()),
                    list(outs.keys()))
    return ((F, args, outs) if return_args else F)


def run_sim_with_MPC(
    sim: Simulation,
    MPC: Union['MPC', 'NlpSolver'],
    K: int,
    use_tqdm: bool = False,
    *cbs: Callable[[int, Simulation, Dict[str, float], Dict[str, Any]], None]
) -> None:
    '''
    cb(k, sim, vars_last, info)
    Automatically run the simulation with an MPC (not necessarily created with 
    the same sim). Be sure to set the initial conditions before calling this 
    method.

    Parameters
    ----------
        sim : metanet.Simulation
            Simulation to run.

        mpc : metanet.control.MPC or metanet.control.NlpSolver 
            MPC controller to run along with the simulation.

        K : int
            Total simulation steps.

        use_tqdm : bool, optional
            Whether to use tqdm to display progress. Defaults to False.

        cbs : Callable[iter, sim, vars, info]
            Callbacks called at the end of each iteration.
            NB: should also pass the dict of parameters to update their values
    '''

    if use_tqdm:
        from tqdm import tqdm
    else:
        def tqdm(iter, *args, **kwargs):
            return iter

    # create functions
    F = sim2func(sim, out_nonstate=True, out_nonneg=True)

    # save some stuff
    M, Np, Nc = mpc.M, mpc.Np, mpc.Nc
    name = sim.net.name
    origins = list(sim.net.origins.keys())
    onramps = list(map(lambda o: o[0], sim.net.onramps))
    links = list(sim.net.links.keys())
    links_with_vms = list(map(lambda o: o[0], sim.net.links_with_vms))

    # initialize true and nominal last solutions
    vars_last = {
        **{f'w_{o}': cs.repmat(o.queue[0], 1, M * Np + 1) for o in origins},
        **{f'rho_{l}': cs.repmat(l.density[0], 1, M * Np + 1) for l in links},
        **{f'v_{l}': cs.repmat(l.speed[0], 1, M * Np + 1) for l in links},
        **{f'r_{o}': np.ones((1, Nc)) for o in onramps},
        **{f'v_ctrl_{l}': np.full((l.nb_vms, Nc), 0.7 * l.v_free)
           for l in links_with_vms}
    }

    # simulation main loop
    for k in tqdm(range(K), total=K):
        if k % M == 0:
            # get future demands (only in true model)
            dist = {}
            for origin in origins:
                d = origin.demand[k:k + M * Np]
                dist[f'd_{origin}'] = np.pad(d, (0, M * Np - len(d)),
                                             mode='edge').reshape(1, -1)

            # run MPC
            vars_init = {
                var: shift(val, axis=2) for var, val in vars_last.items()
            }
            pars_val = {
                **dist,
                **{f'w0_{o}': o.queue[k] for o in origins},
                **{f'rho0_{l}': l.density[k] for l in links},
                **{f'v0_{l}': l.speed[k] for l in links},
                **{f'r_{o}_last': vars_last[f'r_{o}'][0, 0] for o in onramps},
                **{f'v_ctrl_{l}_last': vars_last[f'v_ctrl_{l}'][:, 0]
                    for l in links_with_vms},
            }
            vars_last, info = MPC(vars_init, pars_val)
            if 'error' in info:
                tqdm.write(f'{k:{len(str(K))}}: ({name}) '
                           + info['error'] + '.')

        # set onramp metering rate and vms speed control
        for onramp in onramps:
            onramp.rate[k] = vars_last[f'r_{onramp}'][0, 0]
        for l in links_with_vms:
            l.v_ctrl[k] = vars_last[f'v_ctrl_{l}'][:, 0].reshape((l.nb_vms, 1))

        # example of F args and outs
        # args: w_O1;w_O2|rho_L1[4],v_L1[4];rho_L2[2],v_L2[2]|r_O2;v_ctrl_L1[2]|d_O1;d_O2
        # outs: q_O1,w+_O1;q_O2,w+_O2|q_L1[4],rho+_L1[4],v+_L1[4];q_L2[2],rho+_L2[2],v+_L2[2]

        x, u, d = [], [], []
        for origin in origins:
            x.append(origin.queue[k])
        for link in links:
            x.append(link.density[k])
            x.append(link.speed[k])
        for onramp in onramps:
            u.append(onramp.rate[k])
        for link in links_with_vms:
            u.append(link.v_ctrl[k])
        for origin in origins:
            d.append(origin.demand[k])

        outs = F(*x, *u, *d)

        i = 0
        for origin in origins:
            origin.flow[k] = outs[i]
            origin.queue[k + 1] = outs[i + 1]
            i += 2
        for link in links:
            link.flow[k] = outs[i]
            link.density[k + 1] = outs[i + 1]
            link.speed[k + 1] = outs[i + 2]
            i += 3

        # at the end of each iteration, call the callbacks
        for cb in cbs:
            cb(k, sim, vars_last, info)  # arguments to be defined


def multistart(MPC: Union['MPC', 'NlpSolver'],
               vars_init: Dict[str, float], pars_val: Dict[str, float],
               n: int = 10, noise: str = 'norm', mul: float = 0.33):
    # add noise to the initial conditions
    rng = np.random.default_rng()
    if noise == 'norm':
        def corrupt(x):
            std = mul * np.max(np.abs(x) + 0.25)
            return x + rng.normal(scale=std, size=x.shape)
    elif noise == 'unif':
        def corrupt(x):
            half = (mul * np.max(np.abs(x) + 0.25)) / 2
            return x + rng.uniform(low=-half, high=half, size=x.shape)
    else:
        raise ValueError(
            f'Noise expected to be \'norm\' or \'unif\'; got {noise} instead.')

    vars_init_noisy = [{k: corrupt(v) for k, v in vars_init.items()}
                       for _ in range(n - 1)]
    vars_init_noisy.insert(0, vars_init)  # remember to include original

    # import multiprocessing
    # pool_obj = multiprocessing.Pool()
    # results = pool_obj.starmap(MPC, args, chunksize=1)
    # pool_obj.close()
    # pool_obj.join()
    # return results

    # from joblib import Parallel, delayed
    # r = Parallel(n_jobs=n)(delayed(MPC)(vars[i], pars_val) for i in range(n))

    f_best, vars_best, info_best = float('+inf'), None, None
    for i in range(n):
        vars, info = MPC(vars_init_noisy[i], pars_val)
        f = info['f']
        if f < f_best:
            f_best, vars_best, info_best = f, vars, info

    return vars_best, info_best
