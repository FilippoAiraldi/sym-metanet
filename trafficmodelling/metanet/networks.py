from dataclasses import dataclass, field

from typing import Iterable

from ..util import NamedClass
from .links import Link, LinkWithVms
from .nodes import Node
from .origins import Origin, MainstreamOrigin, OnRamp
from .destinations import Destination


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
    def onramps(self) -> Iterable[tuple[OnRamp, NodeData]]:
        return filter(lambda o: isinstance(o[0], OnRamp), self.origins.items())

    @property
    def mainstream_origins(self) -> Iterable[tuple
                                             [MainstreamOrigin, NodeData]]:
        return filter(lambda o: isinstance(o[0], MainstreamOrigin),
                      self.origins.items())

    @property
    def links_with_vms(self) -> Iterable[tuple[LinkWithVms, LinkData]]:
        return filter(lambda o: isinstance(o[0], LinkWithVms),
                      self.links.items())

    def __init__(self, name=None) -> None:
        '''
        Creates a METANET traffic network model

        Parameters
        ----------
            name : str, optional
                Name of the model.
        '''
        super().__init__(name=name)
        self.nodes: dict[Node, NodeData] = {}
        self.links: dict[Link, LinkData] = {}
        self.origins: dict[Origin, NodeData] = {}
        self.destinations: dict[Destination, NodeData] = {}
        self.turnrates: dict[(Node, Link), float] = {}

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
        if node not in self.nodes:
            self.nodes[node] = NodeData(node)

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
        if link in self.links:
            data = self.links[link]
            raise ValueError(
                f'Link {link.name} already inserted from node '
                f'{data.node_up.name} to {data.node_down.name}.')

        # add link
        self.links[link] = LinkData(link, self.nodes[node_up],
                                    self.nodes[node_down])

        # add nodes
        self.nodes[node_up].links_out.append(self.links[link])
        self.nodes[node_down].links_in.append(self.links[link])

        # add turn rate
        self.turnrates[(node_up, link)] = turnrate

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
        self.origins[origin] = self.nodes[node]
        self.nodes[node].origin = origin

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
        self.destinations[destination] = self.nodes[node]
        self.nodes[node].destination = destination

    def plot(self, reverse_x=False, reverse_y=False, **kwargs):
        import matplotlib.pyplot as plt
        import networkx as nx

        # build the network
        G = nx.DiGraph()
        for link, data in self.links.items():
            G.add_edge(data.node_up.node, data.node_down.node, object=link)
        for origin, data in self.origins.items():
            G.add_edge(origin, data.node, object=None)
        for destination, data in self.destinations.items():
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
                labels[(u, v)] = \
                    f'{link.name}\n({self.turnrates[(u, link)]:.2f})'
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
