import casadi as cs

from .origins import MainstreamOrigin, OnRamp
from .networks import Network
from . import functional as F


class Simulation:
    def __init__(self, net: Network, T: float, rho_max: float, eta: float, tau: float,
                 kappa: float, delta: float, alpha: float = 0.0) -> None:
        '''
        Creates a simulation environment

        Parameters
        ----------
            net : metanet.Network
                Model of the network. Its topology cannot be changed after
                starting the simulation

            T : float
                Simulation time-step.

            eta, tau, kappa, delta : float
                Model parameters. TODO: better description.

            rho_max : float
                Maximum density of the network.

            alpha : float
                Adherence to Variable Message Sign speed limits. Represents a
                percentual increase over this limits (i.e., 0.1 increases the
                actual limit by 10%).
        '''
        # save parameters
        self.net = net
        self.T = T
        self.eta = eta
        self.tau = tau
        self.kappa = kappa
        self.delta = delta
        self.rho_max = rho_max
        self.alpha = alpha

        # some checks
        self.__check_normalized_turnrates()
        self.__check_main_origins_and_destinations()

    def __check_normalized_turnrates(self):
        # check turnrates are normalized
        sums_per_node = {}
        for (node, _), rate in self.net.turnrates.items():
            s = sums_per_node.get(node, 0)
            sums_per_node[node] = s + rate
        for node, s in sums_per_node.items():
            if s != 1.0:
                raise ValueError(f'Turn-rates at node {node.name} do not sum '
                                 f'to 1; got {s} instead.')

    def __check_main_origins_and_destinations(self):
        # all nodes that have a mainstream origin must have no entering links
        for origin, nodedata in self.net.origins.items():
            if (isinstance(origin, MainstreamOrigin)
                    and len(nodedata.links_in) != 0):
                raise ValueError(
                    f'Node {nodedata.node.name} should have no entering links '
                    'since it is connected to the mainstream origin '
                    f'{origin.name}; got {len(nodedata.links_in)} entering '
                    'links instead.')

        # all nodes that have a destination must have no exiting links
        for destination, nodedata in self.net.destinations.items():
            if (isinstance(origin, MainstreamOrigin)
                    and len(nodedata.links_out) != 0):
                raise ValueError(
                    f'Node {nodedata.node.name} should have no exiting links '
                    'since it is connected to the destination '
                    f'{destination.name}; got {len(nodedata.links_out)} '
                    'exiting links instead.')

    def _check_unique_names(self):
        names = set()
        for o in (list(self.net.nodes.keys())
                  + list(self.net.links.keys())
                  + list(self.net.origins.keys())
                  + list(self.net.destinations.keys())):
            name = o.name
            if name in names:
                raise ValueError(
                    f'{o.__class__.__name__} {name} has a non-unique name.')
            names.add(name)

    def set_init_cond(self, links_init=None, origins_init=None,
                      reset=False):
        '''
        Reset internal variables and sets the initial conditions for the
        model's simulation.

        Parameters
        ----------
            links_init : dict[metanet.Link, (array, array)], optional
                Dictionary initial conditions from link to (density, speed).

            origins_init : dict[metanet.Origin, float], optional
                Dictionary initial conditions from origin to queue.

            reset : bool, optional
                Resets all the internal quantities. Defaults to False.
        '''

        if links_init is not None:
            for link, (rho, v) in links_init.items():
                if reset:
                    link.reset()
                link.density[-1] = rho.reshape((link.nb_seg, 1))
                link.speed[-1] = v.reshape((link.nb_seg, 1))

        if origins_init is not None:
            for origin, w in origins_init.items():
                if reset:
                    origin.reset()
                origin.queue[-1] = w.reshape((1, 1))

    def step(self, k: int):
        '''
        Steps the model's simulation one timestep.

        Parameters
        ----------
            k : int
                Simulation timestep index.
        '''
        self.__step_origins(k)
        self.__step_links(k)

    def __step_origins(self, k: int):
        for origin, nodedata in self.net.origins.items():
            # first link after origin - assume only one link out
            link = nodedata.links_out[0].link

            # compute flow
            if isinstance(origin, MainstreamOrigin):
                origin.flow[k] = F.get_mainorigin_flow(
                    d=origin.demand[k],
                    w=origin.queue[k],
                    v_ctrl=link.v_ctrl[k][0],
                    v_first=link.speed[k][0],
                    rho_crit=link.rho_crit,
                    v_free=link.v_free,
                    a=link.a,
                    lanes=link.lanes,
                    alpha=self.alpha,
                    T=self.T)
            elif isinstance(origin, OnRamp):
                origin.flow[k] = F.get_onramp_flow(
                    d=origin.demand[k],
                    w=origin.queue[k],
                    C=origin.capacity,
                    r=origin.rate[k],
                    rho_max=self.rho_max,
                    rho_first=link.density[k][0],
                    rho_crit=link.rho_crit,
                    T=self.T)
            else:
                raise ValueError(f'Unknown origin type {origin.__class__}.')

            # step queue
            origin.queue[k + 1] = F.step_origin_queue(
                w=origin.queue[k],
                d=origin.demand[k],
                q=origin.flow[k],
                T=self.T)

    def __step_links(self, k: int):
        for link, linkdata in self.net.links.items():
            node_up = linkdata.node_up
            node_down = linkdata.node_down

            # compute flow
            link.flow[k] = F.get_link_flow(
                rho=link.density[k],
                v=link.speed[k],
                lanes=link.lanes)

            # compute NODE & BOUNDARY conditions
            q_up = node_up.origin.flow[k] if node_up.origin is not None else 0
            if len(node_up.links_in) == 0:  # i.e., mainstream origin boundary
                v_up = link.speed[k][0]
            else:
                v_lasts, q_lasts = tuple(map(lambda o: cs.vertcat(*o), zip(
                    *[(L.link.speed[k][-1], L.link.flow[k][-1])
                      for L in node_up.links_in])))
                q_up += F.get_upstream_flow(
                    q_lasts, self.net.turnrates[node_up.node, link])
                v_up = F.get_upstream_speed(v_lasts, q_lasts)

            if len(node_down.links_out) == 0:  # i.e., destination boundary
                rho_down = cs.fmin(link.rho_crit, link.density[k][-1])
            else:
                rho_firsts = cs.vertcat(
                    *[L.link.density[k][0] for L in node_down.links_out])
                rho_down = F.get_downstream_density(rho_firsts)

            # put in vector form
            q_up = cs.vertcat(q_up, link.flow[k][:-1])
            v_up = cs.vertcat(v_up, link.speed[k][:-1])
            rho_down = cs.vertcat(link.density[k][1:], rho_down)

            # step density
            link.density[k + 1] = F.step_link_density(
                rho=link.density[k],
                q=link.flow[k],
                q_up=q_up,
                lanes=link.lanes,
                L=link.lengths,
                T=self.T)

            # compute equivalent speed
            V = F.Veq_ext(rho=link.density[k],
                          v_free=link.v_free,
                          a=link.a,
                          rho_crit=link.rho_crit,
                          v_ctrl=link.v_ctrl[k],
                          alpha=self.alpha)

            # step speed
            onramp = node_up.origin  # possible onramp connected to the link
            q_r = (onramp.flow[k]
                   if onramp is not None and isinstance(onramp, OnRamp) else
                   0)
            link.speed[k + 1] = F.step_link_speed(
                v=link.speed[k],
                v_up=v_up,
                rho=link.density[k],
                rho_down=rho_down,
                V=V,
                lanes=link.lanes,
                L=link.lengths,
                tau=self.tau,
                eta=self.eta,
                kappa=self.kappa,
                q_r=q_r,
                delta=self.delta,
                T=self.T)


def sim2func(sim: Simulation,
             states: bool = True,
             inputs: bool = True,
             disturbances: bool = True,
             origin_params: bool = False,
             link_params: bool = False,
             sim_params: bool = False,
             out_nonstate: bool = False,
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

        states, inputs, disturbances, 
        link_params origin_params, sim_params : bool, optional
            Whether to make various components symbolic or not.

        out_nonstate : bool, optional
            Whether the function should output also other quantities apart 
            from the next states, namely origin and link flows (not in state).
            Note that these quantities belong to the current timestep k, while
            the next states are from time k + 1. Defaults to False.

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

    # create a copy of the simulation which will be entirely symbolic
    from copy import deepcopy
    sym_sim = deepcopy(sim)
    origins, links = sym_sim.net.origins, sym_sim.net.links

    # create the function arguments
    args = {}

    def add_arg(name, *size):
        var = csXX.sym(name, *size)
        args[name] = var
        return var

    if states:
        for origin in origins:
            origin.queue[k] = add_arg(f'w_{origin.name}', 1, 1)
        for link in links:
            link.density[k] = add_arg(f'rho_{link.name}', link.nb_seg, 1)
            link.speed[k] = add_arg(f'v_{link.name}', link.nb_seg, 1)

    if inputs:
        for origin in origins:
            if isinstance(origin, OnRamp):
                origin.rate[k] = add_arg(f'r_{origin.name}', 1, 1)
        for link in links:
            link.v_ctrl[k] = add_arg(f'v_ctrl_{link.name}', link.nb_seg, 1)

    if disturbances:
        for origin in origins:
            origin.demand[k] = add_arg(f'd_{origin.name}', 1, 1)

    if origin_params:
        for origin in origins:
            if isinstance(origin, OnRamp):
                origin.capacity = add_arg(f'C_{origin.name}', 1, 1)

    if link_params:
        for link in links:
            for name in ['lanes', 'lengths', 'v_free', 'rho_crit', 'a']:
                setattr(link, name, add_arg(f'{name}_{link.name}', 1, 1))

    if sim_params:
        for name in ['rho_max', 'eta', 'tau', 'kappa', 'delta', 'alpha', 'T']:
            setattr(sym_sim, name, add_arg(name, 1, 1))

    # perform one step
    sym_sim.step(k)

    # gather outputs - queue for each ramp,
    outs = {}  # sourcery skip: dict-comprehension
    for origin in origins:
        if out_nonstate:
            outs[f'q_{origin.name}'] = origin.flow[k]
        outs[f'w+_{origin.name}'] = origin.queue[k + 1]
    for link in links:
        if out_nonstate:
            outs[f'q_{link.name}'] = link.flow[k]
        outs[f'rho+_{link.name}'] = link.density[k + 1]
        outs[f'v+_{link.name}'] = link.speed[k + 1]

    # create function from state k to state k+1
    F = cs.Function(funcname,
                    list(args.values()),
                    list(outs.values()),
                    list(args.keys()),
                    list(outs.keys()))
    return ((F, args, outs) if return_args else F)
