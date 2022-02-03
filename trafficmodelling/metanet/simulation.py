import casadi as cs
from dataclasses import dataclass, field

from ..util import NamedClass
from .links import Link
from .nodes import Node
from .origins import Origin, MainstreamOrigin, OnRamp
from .destinations import Destination
from . import functional as F


@dataclass
class NodeData:
    node: Node
    links_in: list['LinkData'] = field(default_factory=list)
    links_out: list['LinkData'] = field(default_factory=list)
    origin: Origin = None
    destination: Destination = None


@dataclass
class LinkData:
    link: Link
    node_up: NodeData = None
    node_down: NodeData = None


class Network(NamedClass):
    '''METANET traffic network'''

    @property
    def nodes(self) -> dict[Node, NodeData]:
        return self.__nodes

    @property
    def links(self) -> dict[Link, LinkData]:
        return self.__links

    @property
    def origins(self) -> dict[Origin, NodeData]:
        return self.__origins

    @property
    def destinations(self) -> dict[Destination, NodeData]:
        return self.__destinations

    @property
    def turnrates(self) -> dict[(Node, Link), float]:
        return self.__beta

    def __init__(self, name=None) -> None:
        '''
        Creates a METANET traffic network model

        Parameters
        ----------
            name : str, optional
                Name of the model.
        '''
        super().__init__(name=name)
        # self.__g = nx.DiGraph()
        self.__nodes: dict[Node, NodeData] = {}
        self.__links: dict[Link, LinkData] = {}
        self.__origins: dict[Origin, NodeData] = {}
        self.__destinations: dict[Destination, NodeData] = {}
        self.__beta: dict[(Node, Link), float] = {}  # turn rates

    def add_vertex(self, *args, **kwargs):
        '''Alias for add_node'''
        return self.add_node(*args, **kwargs)

    def add_edge(self, *args, **kwargs):
        '''Alias for add_link'''
        return self.add_link(*args, **kwargs)

    def add_node(self, node: Node):
        '''
        Add a node to the network. If already present, does nothing.

        Parameters
        ----------
            node : metanet.Node
                The node to add.
        '''
        if node not in self.__nodes:
            self.__nodes[node] = NodeData(node)

    def add_link(self, node_up: Node, link: Link, node_down: Node,
                 turnrate: float = 1.0):
        '''
        Add a link from the upstream node to the downstream node. Raises if
        already present.

        Parameters
        ----------
            node_up : metanet.Node
                Upstream node.
            link : metanet.Link
                Link connecting the two nodes.
            node_down : metanet.Node
                Downstream node.
            turnrate : float, optional
                Turn rate from the upstream node into this link.

        Raises
        ------
            ValueError : link already in network
                If the link had already been added.
        '''
        if link in self.__links:
            data = self.__links[link]
            raise ValueError(
                f'Link {link.name} already inserted from node '
                f'{data.node_up.name} to {data.node_down.name}.')

        # add link
        self.__links[link] = LinkData(link, self.__nodes[node_up],
                                      self.__nodes[node_down])

        # add nodes
        self.__nodes[node_up].links_out.append(self.__links[link])
        self.__nodes[node_down].links_in.append(self.__links[link])

        # add turn rate
        self.__beta[(node_up, link)] = turnrate

    def add_origin(self, origin: Origin, node: None):
        '''
        Add an origin to the node.

        Parameters
        ----------
            origin : metanet.Origin
                Origin (mainstream, on-ramp) to be connected to the node.
            node : metanet.Node
                Node to be connected to the origin.
        '''
        self.__origins[origin] = self.__nodes[node]
        self.__nodes[node].origin = origin

    def add_destination(self, destination: Destination, node: Node):
        '''
        Add a destination to the node.

        Parameters
        ----------
            destination : metanet.Destination
                Destination to be connected to the node.
            node : metanet.Node
                Node to be connected to the destination.
        '''
        self.__destinations[destination] = self.__nodes[node]
        self.__nodes[node].destination = destination

    def plot(self, reverse_x=False, reverse_y=False, **kwargs):
        import matplotlib.pyplot as plt
        import networkx as nx

        # build the network
        G = nx.DiGraph()
        for link, data in self.__links.items():
            G.add_edge(data.node_up.node, data.node_down.node, object=link)
        for origin, data in self.__origins.items():
            G.add_edge(origin, data.node, object=None)
        for destination, data in self.__destinations.items():
            G.add_edge(data.node, destination, object=None)

        # compute positions
        pos = nx.spectral_layout(G)
        if reverse_x:
            for k in pos.values():
                k[0] *= -1
        if reverse_y:
            for k in pos.values():
                k[1] *= -1

        # nodes
        cmap, labels = [], {}
        for node in G.nodes:
            labels[node] = node.name
            if isinstance(node, MainstreamOrigin):
                cmap.append('tab:red')
            elif isinstance(node, OnRamp):
                cmap.append('tab:orange')
            elif isinstance(node, Destination):
                cmap.append('tab:blue')
            else:
                cmap.append('white')
        nx.draw_networkx_nodes(G, pos, node_color=cmap,
                               edgecolors='k', node_size=600, alpha=0.9)
        nx.draw_networkx_labels(G, pos, labels)

        # links
        cmap, width, labels = [], [], {}
        for u, v in G.edges:
            if link := G.edges[u, v]['object']:
                cmap.append('k')
                width.append(2.0)
                labels[(u, v)] = f'{link.name}\n({self.__beta[(u, link)]:.2f})'
            else:
                cmap.append('grey')
                width.append(1.0)
        nx.draw_networkx_edges(G, pos, width=width, edge_color=cmap,
                               arrowsize=20)
        nx.draw_networkx_edge_labels(G, pos, labels)

        # ax = plt.gca()
        # ax.collections[0].set_edgecolor('#000000')
        plt.tight_layout()
        plt.axis("off")
        plt.show()


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
        self.__check_normalized_turnrates(net)
        self.__check_main_origins_and_destinations(net)

    def __check_normalized_turnrates(self, net: Network):
        # check turnrates are normalized
        sums_per_node = {}
        for (node, _), rate in net.turnrates.items():
            s = sums_per_node.get(node, 0)
            sums_per_node[node] = s + rate
        for node, s in sums_per_node.items():
            if s != 1.0:
                raise ValueError(f'Turn-rates at node {node.name} do not sum '
                                 f'to 1; got {s} instead.')

    def __check_main_origins_and_destinations(self, net: Network):
        # all nodes that have a mainstream origin must have no entering links
        for origin, nodedata in net.origins.items():
            if (isinstance(origin, MainstreamOrigin)
                    and len(nodedata.links_in) != 0):
                raise ValueError(
                    f'Node {nodedata.node.name} should have no entering links '
                    'since it is connected to the mainstream origin '
                    f'{origin.name}; got {len(nodedata.links_in)} entering '
                    'links instead.')

        # all nodes that have a destination must have no exiting links
        for destination, nodedata in net.destinations.items():
            if (isinstance(origin, MainstreamOrigin)
                    and len(nodedata.links_out) != 0):
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

    def to_func(self):
        # NOTE: in future, might turn everything (also params) in sym.

        # save state of each link, origin

        # change initial conditions to casadi symbolic
        # states, inputs, disturbances

        # perform one step

        # create function from state k to state k+1
        pass
