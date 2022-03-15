import casadi as cs
import numpy as np
from itertools import product, cycle

from typing import Tuple, Dict, Union

from ..blocks.origins import Origin, MainstreamOrigin, OnRamp
from ..blocks.links import Link, LinkWithVms
from ..blocks.networks import Network
from . import functional as F


colors = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#A2142F']


class Simulation:
    def __init__(self, net: Network, T: float, rho_max: float, eta: float,
                 tau: float, kappa: float, delta: float,
                 alpha: float = 0.0) -> None:
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
                Model parameters.

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
        self.net._check_unique_names()

        # containers for other quantities to be saved
        self.objective = []
        self.slacks = {}

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
        # for origin, nodedata in self.net.origins.items():
        for origin, nodedata in self.net.mainstream_origins:
            if len(nodedata.links_in) != 0:
                raise ValueError(
                    f'Node {nodedata.node.name} should have no entering links '
                    'since it is connected to the mainstream origin '
                    f'{origin.name}; got {len(nodedata.links_in)} entering '
                    'links instead.')

        # all nodes that have a destination must have no exiting links
        for destination, nodedata in self.net.destinations.items():
            if len(nodedata.links_out) != 0:
                raise ValueError(
                    f'Node {nodedata.node.name} should have no exiting links '
                    'since it is connected to the destination '
                    f'{destination.name}; got {len(nodedata.links_out)} '
                    'exiting links instead.')

    def set_init_cond(
            self, links_init: Dict[Link, Tuple[np.ndarray, np.ndarray]],
            origins_init: Dict[Origin, np.ndarray],
            ctrl_init: Dict[Union[LinkWithVms, OnRamp], np.ndarray],
            reset_demands: bool = False) -> None:
        '''
        Reset internal variables and sets the initial conditions for the
        model's simulation.

        Parameters
        ----------
            links_init : dict[metanet.Link, (array, array)], optional
                Dictionary holding initial conditions (density, speed) for each
                link. 

            origins_init : dict[metanet.Origin, array], optional
                Dictionary holding initial conditions (queue) for each origin.

            ctrl_init : dict[metanet.LinkVms or Onramp, array], optional
                Dictionary holding initial control actions for each link with 
                vms and/or ramp. 

            reset_demands : bool, optional
                Whether to reset also the demands. Defaults to False.
        '''

        for link in self.net.links:
            link.density.clear()
            link.speed.clear()
            link.flow.clear()
            rho, v = links_init[link]
            link.density[0] = rho.reshape((link.nb_seg, 1))
            link.speed[0] = v.reshape((link.nb_seg, 1))
            if isinstance(link, LinkWithVms):
                link.v_ctrl.clear()
                link.v_ctrl[0] = ctrl_init[link].reshape((link.nb_vms, 1))
        for origin in self.net.origins:
            origin.queue.clear()
            origin.flow.clear()
            if reset_demands:
                origin.demand.clear()
            origin.queue[0] = origins_init[origin].reshape((1, 1))
            if isinstance(origin, OnRamp):
                origin.rate.clear()
                origin.rate[0] = ctrl_init[origin].reshape((1, 1))

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
                kwargs = {
                    'd': origin.demand[k],
                    'w': origin.queue[k],
                    'v_first': link.speed[k][0],
                    'rho_crit': link.rho_crit,
                    'v_free': link.v_free,
                    'a': link.a,
                    'lanes': link.lanes,
                    'alpha': self.alpha,
                    'T': self.T
                }
                if isinstance(link, LinkWithVms) and link.has_vms[0]:
                    origin.flow[k] = F.get_mainorigin_flow(
                        v_ctrl=link.v_ctrl_at(k, 0), **kwargs)
                else:
                    origin.flow[k] = F.get_mainorigin_flow_no_ctrl(**kwargs)
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
        # compute flows first
        for link in self.net.links:
            link.flow[k] = F.get_link_flow(
                rho=link.density[k],
                v=link.speed[k],
                lanes=link.lanes)

        # now compute quantities for the next step
        for link, linkdata in self.net.links.items():
            node_up = linkdata.node_up
            node_down = linkdata.node_down

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

            # put in vector form (vertcatt with empty SX inserts 00, which
            # unexpectedly increases size of vector)
            if link.nb_seg > 1:
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

            # compute equivalent speed - overwrite those link segments
            # with speed control
            kwargs = {
                'v_free': link.v_free,
                'a': link.a,
                'rho_crit': link.rho_crit
            }
            V = F.Veq(rho=link.density[k], **kwargs)
            if isinstance(link, LinkWithVms):
                for i in link.vms:
                    V[i] = F.Veq_ext(rho=link.density[k][i],
                                     v_ctrl=link.v_ctrl_at(k, i),
                                     alpha=self.alpha, **kwargs)

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

    def plot(self,
             t: np.ndarray = None,
             fig: 'Figure' = None,
             axs: np.ndarray = None,
             sharex: bool = False,
             add_labels: bool = True,
             **plot_kwargs) -> Tuple['Figure', np.ndarray]:
        '''
        Plots the simulation outcome.

        Parameters
        ----------
            t : np.ndarray. optional
                Time vector used for plotting in the x-axis.

            fig : matplotlib.figure, optional
                Figure where to plot to.

            axs : 2d array of matplotlib.axis, optional
                Axes to be used for plotting. If not given, axes are 
                automatically constructed.

            add_labels : bool, optional
                Whether to automatically add labels.

            sharex : bool, optional
                Whether the axes should share the x. Defaults to True.

        Returns
        -------
            fig : matplotlib.figure
                The figure created ad hoc if axes were not specified; 
                otherwise None.

            axs : 2d np.ndarray of matplotlib.axis
                The axes used for plotting.
        '''

        if fig is None:
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(10, 7), constrained_layout=True)

        if axs is None:
            from matplotlib.gridspec import GridSpec
            from matplotlib.ticker import FormatStrFormatter

            gs = GridSpec(5, 2, figure=fig)
            axs = np.array(
                [fig.add_subplot(gs[i, j]) for i, j in product(
                    range(gs.nrows),
                    range(gs.ncols))]).reshape(gs.nrows, gs.ncols)

            for ax in axs.flatten():
                ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))

        # create array of maxima - used to set y axis limits
        maxs = np.zeros(axs.shape)

        # create time vector
        if t is None:
            t = np.arange(len(next(iter(self.net.links)).flow)) * self.T

        # reduce linewidth
        if 'linewidth' not in plot_kwargs:
            plot_kwargs['linewidth'] = 1

        def plot(loc, x, c, lbl=None):
            maxs[loc] = max(maxs[loc], x.max())
            if add_labels:
                return axs[loc].plot(t, x, color=c, label=lbl, **plot_kwargs)
            return axs[loc].plot(t, x, color=c, **plot_kwargs)

        # add plots
        any_onramp, any_vms = False, False

        colors_ = cycle(colors)
        for link in self.net.links:
            v = np.hstack(link.speed[:-1])
            rho = np.hstack(link.density[:-1])
            q = np.hstack(link.flow)
            for i in range(link.nb_seg):
                c = next(colors_)
                plot((0, 0), v[i], c, f'$v_{{{link.name}, {i + 1}}}$')
                plot((0, 1), q[i], c, f'$q_{{{link.name}, {i + 1}}}$')
                plot((1, 0), rho[i], c, f'$\\rho_{{{link.name}, {i + 1}}}$')
            if isinstance(link, LinkWithVms):
                any_vms = True
                v_ctrl = np.hstack(link.v_ctrl)
                for i, s in enumerate(link.vms):
                    plot((3, 1), v_ctrl[i], c,
                         f'$v^{{ctrl}}_{{{link.name}, {s + 1}}}$')

        for origin, c in zip(self.net.origins, cycle(colors)):
            w = np.vstack(origin.queue[:-1])
            q = np.vstack(origin.flow)
            d = np.vstack(origin.demand)
            plot((2, 0), d, c, origin.name)
            plot((3, 0), q, c, f'$q_{{{origin.name}}}$')
            plot((2, 1), w, c, f'$\\omega_{{{origin.name}}}$')
            if isinstance(origin, OnRamp):
                any_onramp = True
                r = np.vstack(origin.rate)
                plot((3, 1), r, c, origin.name)

        # plot objective
        if len(self.objective) > 0:
            plot((4, 0), np.squeeze(self.objective), colors[0], f'$J$')

        # plot slacks variables
        for (name, data), c in zip(self.slacks.items(), cycle(colors)):
            plot((4, 1), np.squeeze(data), c)

        # embellish axes
        excluded = set()
        axs[0, 0].set_ylabel('speed (km/h)')
        axs[0, 1].set_ylabel('flow (veh/h)')
        axs[1, 0].set_ylabel('density (veh/km)')
        if any_vms:
            axs[1, 1].set_ylabel('dynamic speed limit (km/h)')
        else:
            excluded.add((1, 1))
        axs[2, 0].set_ylabel('origin demand (veh/h)')
        axs[2, 1].set_ylabel('queue length (veh)')
        axs[3, 0].set_ylabel('origin flow (veh/h)')
        if any_onramp:
            axs[3, 1].set_ylabel('metering rate')
        else:
            excluded.add((3, 1))
        axs[4, 0].set_ylabel('objective (veh h)')
        axs[4, 1].set_ylabel('slack')

        for i, j in product(range(axs.shape[0]), range(axs.shape[1])):
            if (i, j) in excluded:
                axs[i, j].set_axis_off()
            else:
                if sharex:
                    axs[i, j].sharex(axs[0, 0])
                axs[i, j].spines['top'].set_visible(False)
                axs[i, j].spines['right'].set_visible(False)
                axs[i, j].tick_params(direction="in")
                axs[i, j].set_xlabel('time (h)')
                axs[i, j].set_xlim(0, t[-1])
                axs[i, j].autoscale_view()
                axs[i, j].set_ylim(0, max(axs[i, j].get_ylim()[1],
                                          maxs[i, j] * 1.1))
                axs[i, j].legend()

        return fig, axs
