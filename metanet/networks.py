from dataclasses import dataclass, field

from typing import Iterable, Dict, List, Tuple

from .util import NamedClass
from .links import Link, LinkWithVms
from .nodes import Node
from .origins import Origin, MainstreamOrigin, OnRamp
from .destinations import Destination


@dataclass
class NodeData:
    node: Node
    links_in: List['LinkData'] = field(default_factory=list)
    links_out: List['LinkData'] = field(default_factory=list)
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
    def onramps(self) -> Iterable[Tuple[OnRamp, NodeData]]:
        return filter(lambda o: isinstance(o[0], OnRamp), self.origins.items())

    @property
    def mainstream_origins(self) -> Iterable[Tuple
                                             [MainstreamOrigin, NodeData]]:
        return filter(lambda o: isinstance(o[0], MainstreamOrigin),
                      self.origins.items())

    @property
    def links_with_vms(self) -> Iterable[Tuple[LinkWithVms, LinkData]]:
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
        self.nodes: Dict[Node, NodeData] = {}
        self.links: Dict[Link, LinkData] = {}
        self.origins: Dict[Origin, NodeData] = {}
        self.destinations: Dict[Destination, NodeData] = {}
        self.turnrates: Dict[(Node, Link), float] = {}

    def add_vertex(self, *args, **kwargs) -> None:
        '''Alias for add_node'''
        return self.add_node(*args, **kwargs)

    def add_edge(self, *args, **kwargs) -> None:
        '''Alias for add_link'''
        return self.add_link(*args, **kwargs)

    def add_node(self, node: Node) -> None:
        '''
        Add a node to the network. If already present, does nothing.

        Parameters
        ----------
            node : metanet.Node
                The node to add.
        '''
        if node not in self.nodes:
            self.nodes[node] = NodeData(node)

    def add_nodes(self, nodes: List[Node]) -> None:
        for n in nodes:
            self.add_node(n)

    def add_link(self, node_up: Node, link: Link, node_down: Node,
                 turnrate: float = 1.0) -> None:
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

    def add_origin(self, origin: Origin, node: None) -> None:
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

    def add_destination(self, destination: Destination, node: Node) -> None:
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

    def plot(self, expanded_view=False,
             reverse_x=False, reverse_y=False) -> None:
        import matplotlib.pyplot as plt
        import networkx as nx
        import uuid

        # build the network
        G = nx.DiGraph()
        for link, data, in self.links.items():
            if expanded_view:
                # create fictitious nodes between pair of real
                nodes = [
                    data.node_up.node,
                    *(f'{uuid.uuid1()}' for _ in range(link.nb_seg - 1)),
                    data.node_down.node,
                ]
                for i in range(link.nb_seg):
                    G.add_edge(nodes[i], nodes[i + 1], object=(link, i))
            else:
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

        # draw proper nodes and fictitious nodes
        cmap, size, labels = [], [], {}
        for node in G.nodes:
            if not expanded_view:
                labels[node] = node.name
                size.append(600)
                if isinstance(node, MainstreamOrigin):
                    cmap.append('tab:red')
                elif isinstance(node, OnRamp):
                    cmap.append('tab:orange')
                elif isinstance(node, Destination):
                    cmap.append('tab:blue')
                else:
                    cmap.append('white')
            else:  # sourcery skip: merge-else-if-into-elif
                if isinstance(node, Node): 
                    # is a proper node
                    cmap.append('white')
                    size.append(600)
                    labels[node] = node.name
                else:
                    # is fictitious, Origin, Destination
                    cmap.append('black')
                    if isinstance(node, (Origin, Destination)):
                        size.append(0)
                        labels[node] = node.name
                    else:
                        size.append(100)
        nx.draw_networkx_nodes(G, pos, node_color=cmap, edgecolors='k',
                               node_size=size)
        nx.draw_networkx_labels(G, pos, labels)

        # draw links
        if not expanded_view:
            cmap, width, labels = [], [], {}
            for u, v in G.edges:
                link = G.edges[u, v]['object']
                if link is not None:  # if link := G.edges[u, v]['object']:
                    cmap.append('k')
                    width.append(2.0)
                    lbl = link.name
                    if isinstance(link, LinkWithVms):
                        lbl += '\nvms:' + ','.join(str(o) for o in link.vms)
                    lbl += f'\n({self.turnrates[(u, link)]:.2f})'
                    labels[(u, v)] = lbl
                else:
                    cmap.append('grey')
                    width.append(1.0)
            nx.draw_networkx_edge_labels(G, pos, labels)
        nx.draw_networkx_edges(G, pos, arrowsize=20)

        if self.name is not None and len(self.name) > 0:
            plt.title(self.name)
        plt.tight_layout()
        plt.axis("off")
        plt.show()
