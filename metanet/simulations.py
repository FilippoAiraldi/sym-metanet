import casadi as cs
import numpy as np
from itertools import product

from typing import Tuple

from .origins import MainstreamOrigin, OnRamp
from .links import LinkWithVms
from .networks import Network
from . import functional as F


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
        self.net._check_unique_names()

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

    def plot(self, t: np.ndarray, axs: np.ndarray = None,
             sharex: bool = True) -> Tuple['Figure', np.ndarray]:
        '''
        Plots the simulation outcome.

        Parameters
        ----------
            t : np.ndarray
                Time vector used for plotting in the x-axis.

            axs : 2d array of matplotlib.axis, optional
                Axes to be used for plotting. If not given, axes are 
                automatically constructed.

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

        if axs is None:
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
            # create figure
            fig = plt.figure(figsize=(10, 7), constrained_layout=True)
            gs = GridSpec(4, 2, figure=fig)
            axs = np.array(
                [fig.add_subplot(gs[i, j]) for i, j in product(
                    range(gs.nrows),
                    range(gs.ncols))]).reshape(gs.nrows, gs.ncols)
        else:
            fig = None

        # add plots
        any_onramp, any_vms = False, False
        for link in self.net.links:
            v = np.hstack(link.speed[:-1])
            rho = np.hstack(link.density[:-1])
            q = np.hstack(link.flow)
            for i in range(link.nb_seg):
                axs[0, 0].plot(t, v[i], label=f'$v_{{{link.name}, {i + 1}}}$')
                axs[0, 1].plot(t, q[i], label=f'$q_{{{link.name}, {i + 1}}}$')
                axs[1, 0].plot(t, rho[i],
                               label=f'$\\rho_{{{link.name}, {i + 1}}}$')
            if isinstance(link, LinkWithVms):
                any_vms = True
                v_ctrl = np.hstack(link.v_ctrl)
                for i, s in enumerate(link.vms):
                    axs[3, 1].plot(t, v_ctrl[i],
                                   label=f'$v^{{ctrl}}_{{{link.name}, {s + 1}}}$')
        for origin in self.net.origins:
            w = np.vstack(origin.queue[:-1])
            q = np.vstack(origin.flow)
            axs[2, 0].plot(t, q, label=f'$q_{{{origin.name}}}$')
            axs[2, 1].plot(t, w, label=f'$\\omega_{{{origin.name}}}$')
            if isinstance(origin, OnRamp):
                any_onramp = True
                r = np.vstack(origin.rate)
                axs[3, 0].plot(t, r, label=origin.name)

        excluded = {(1, 1)}
        axs[0, 0].set_ylabel('speed (km/h)')
        axs[0, 1].set_ylabel('flow (veh/h)')
        axs[1, 0].set_ylabel('density (veh/km)')
        axs[2, 0].set_ylabel('origin flows (veh/h)')
        axs[2, 1].set_ylabel('queue length (veh)')
        if any_onramp:
            axs[3, 0].set_ylabel('metering rate')
        else:
            excluded.add((3, 0))
        if any_vms:
            axs[3, 1].set_ylabel('dynamic speed limit (km/h)')
        else:
            excluded.add((3, 1))

        for i, j in product(range(axs.shape[0]), range(axs.shape[1])):
            if (i, j) in excluded:
                axs[i, j].set_axis_off()
            else:
                if sharex:
                    axs[i, j].sharex(axs[0, 0])
                axs[i, j].set_xlabel('time (h)')
                axs[i, j].set_xlim(0, t[-1])
                axs[i, j].set_ylim(0, axs[i, j].get_ylim()[1])
                axs[i, j].legend()

        return fig, axs

    def savepkl(self, filename: str) -> None:
        '''
        Save the simulation outcomes to a .pkl file. 
        Parameters
        ----------
            flename : str
                The filename where to save the data to.
        '''
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def savemat(self, filename: str, **other_data) -> None:
        '''
        Save the simulation outcomes to a .mat file. The simulation must have
        been already run.

        Parameters
        ----------
            flename : str
                The filename where to save the data to.

            other_data : kwargs
                A dictionary of any additional data to be saved to the file.
        '''

        import platform
        from scipy.io import savemat

        # attributes (not lists and arrays containing states, demands, etc)
        sim_attr = ('T', 'eta', 'tau', 'kappa', 'delta', 'rho_max', 'alpha')
        net_attr = ('name', )
        link_attr = ('name', 'nb_seg', 'lanes', 'lengths', 'v_free',
                     'rho_crit', 'a')
        link_vms_attr = link_attr + ('vms', 'nb_vms')
        origin_data = ('name', )
        onramp_data = origin_data + ('capactiy', )

        # add the platform which the simulation was run on
        data = other_data
        data['platform'] = str(platform.platform())

        # create link data
        link_data = {}
        for link in self.net.links:
            d = {
                'speed': np.hstack(link.speed[:-1]).reshape(link.nb_seg, -1),
                'density': np.hstack(link.density[:-1]
                                     ).reshape(link.nb_seg, -1),
                'flow': np.hstack(link.flow).reshape(link.nb_seg, -1)
            }
            if isinstance(link, LinkWithVms):
                d['v_ctrl'] = np.hstack(link.v_ctrl).reshape(link.nb_vms, -1)
            link_data[link.name] = d

        # create origin data
        origin_data = {}
        for origin in self.net.origins:
            d = {
                'queue': np.vstack(origin.queue[:-1]).reshape(1, -1),
                'flow': np.vstack(origin.flow).reshape(1, -1),
                'demand': np.vstack(origin.demand).reshape(1, -1)
            }
            if isinstance(origin, OnRamp):
                d['rate'] = np.vstack(origin.rate).reshape(1, -1)
            origin_data[origin.name] = d

        # create the simulation data
        data['simulation'] = {
            **{a: getattr(self, a) for a in sim_attr},
            'net': {
                **{a: getattr(self.net, a) for a in net_attr},
                'links': link_data,
                'origins': origin_data
            }
        }
        savemat(filename, data)
