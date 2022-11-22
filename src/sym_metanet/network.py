from functools import cached_property
from itertools import chain, product
from typing import Dict, Iterable, Tuple, Union, List
import networkx as nx
from sym_metanet.blocks.base import ElementBase, sym_var
from sym_metanet.views import (
    LINKENTRY,
    ORIGINENTRY,
    DESTINATIONENTRY,
    InLinkViewWrapper,
    OutLinkViewWrapper,
)
from sym_metanet.blocks.nodes import Node
from sym_metanet.blocks.links import Link
from sym_metanet.blocks.origins import MeteredOnRamp, Origin
from sym_metanet.blocks.destinations import Destination
from sym_metanet.errors import InvalidNetworkError
from sym_metanet.util.funcs import cache_clearer


class Network(ElementBase[sym_var]):
    '''Highway network.'''

    def __init__(self, name: str = None):
        '''Instantiates an highway network.

        Parameters
        ----------
        name : str, optional
            Name of the network, by default `None`.
        '''
        super().__init__(name=name)
        self._graph = nx.DiGraph(name=name)
        self.nodes_by_name: Dict[str, Node] = {}
        self.links_by_name: Dict[str, Link] = {}
        self.origins_by_name: Dict[str, Origin] = {}
        self.destinations_by_name: Dict[str, Destination] = {}
        self.nodes_by_link: Dict[Link, Tuple[Node, Node]] = {}

    @property
    def G(self) -> nx.DiGraph:
        '''Returns the underlying `networkx`'s graph of the highway.'''
        return self._graph

    @property
    def graph(self) -> nx.DiGraph:
        '''Returns the underlying `networkx`'s graph of the highway.'''
        return self._graph

    @property
    def asgraph(self) -> nx.DiGraph:
        '''Returns the underlying `networkx`'s graph of the highway.'''
        return self._graph

    @property
    def nodes(self) -> nx.classes.reportviews.NodeView:
        '''Returns a view on the nodes of the network.'''
        return self._graph.nodes

    @cached_property
    def links(self) -> OutLinkViewWrapper:
        '''Returns a view on the links of the network.'''
        return OutLinkViewWrapper(self._graph)

    @property
    def out_links(self) -> OutLinkViewWrapper:
        '''Alias of the `links` property.'''
        return self.links

    @cached_property
    def in_links(self) -> InLinkViewWrapper:
        '''Returns a view on the inward links of the network.'''
        return InLinkViewWrapper(self._graph)

    @cached_property
    def origins(self) -> Dict[Origin, Node]:
        return {
            data[ORIGINENTRY]: node for node, data in self._graph.nodes.data()
            if ORIGINENTRY in data
        }

    @cached_property
    def destinations(self) -> Dict[Destination, Node]:
        return {
            data[DESTINATIONENTRY]: node
            for node, data in self._graph.nodes.data()
            if DESTINATIONENTRY in data
        }

    def add_node(self, node: Node) -> 'Network':
        '''Adds a node to the highway network.

        Parameters
        ----------
        node : Node
            Node to be added.

        Returns
        -------
        Network
            A reference to itself.
        '''
        self._graph.add_node(node)
        self.nodes_by_name[node.name] = node
        return self

    def add_nodes(self, nodes: Iterable[Node]) -> 'Network':
        '''Adds multiple nodes. See `Network.add_node`.

        Parameters
        ----------
        nodes : iterable of Nodes
            Nodes to be added.

        Returns
        -------
        Network
            A reference to itself.
        '''
        self._graph.add_nodes_from(nodes)
        self.nodes_by_name.update({node.name: node for node in nodes})
        return self

    def add_link(
            self, node_up: Node, link: Link, node_down: Node) -> 'Network':
        '''Adds a link to the highway network, between two nodes.

        Parameters
        ----------
        node_up : Node
            Upstream node, that is, where traffic is coming from.
        link : Link
            The link to be added connecting the two nodes.
        node_down : Node
            Downstream node, that is, where traffic is going to.

        Returns
        -------
        Network
            A reference to itself.
        '''

        self._graph.add_edge(node_up, node_down, **{LINKENTRY: link})
        self.nodes_by_link[link] = (node_up, node_down)
        self.links_by_name[link.name] = link
        return self

    def add_links(self, links: Iterable[Tuple[Node, Link, Node]]) -> 'Network':
        '''Adds multiple links. See `Network.add_link`.

        Parameters
        ----------
        nodes : iterable of Tuple[Node, Link, Node]
            Links to be added between the corresponding nodes.

        Returns
        -------
        Network
            A reference to itself.
        '''

        def get_edge(linkdata: Tuple[Node, Link, Node]):
            node_up, link, node_down = linkdata
            self.nodes_by_link[link] = (node_up, node_down)
            self.links_by_name[link.name] = link
            return (node_up, node_down, {LINKENTRY: link})

        self._graph.add_edges_from(get_edge(link) for link in links)
        return self

    @cache_clearer(origins)
    def add_origin(self, origin: Origin, node: Node) -> 'Network':
        '''Adds the given traffic origin to the node.

        Parameters
        ----------
        origin : Origin
            Origin to be added to the network.
        node : Node
            Node which the origin is attached to.

        Returns
        -------
        Network
            A reference to itself.
        '''
        if node not in self.nodes:
            self._graph.add_node(node, **{ORIGINENTRY: origin})
        else:
            self.nodes[node][ORIGINENTRY] = origin
        self.origins_by_name[origin.name] = origin
        return self

    @cache_clearer(destinations)
    def add_destination(
            self, destination: Destination, node: Node) -> 'Network':
        '''Adds the given traffic destination to the node.

        Parameters
        ----------
        destination : Destination
            Destination to be added to the network.
        node : Node
            Node which the destination is attached to.

        Returns
        -------
        Network
            A reference to itself.
        '''
        if node not in self.nodes:
            self._graph.add_node(node, **{DESTINATIONENTRY: destination})
        else:
            self.nodes[node][DESTINATIONENTRY] = destination
        self.destinations_by_name[destination.name] = destination
        return self

    def add_path(
        self,
        path: Iterable[Union[Node, Link]],
        origin: Origin = None,
        destination: Destination = None
    ) -> 'Network':
        '''Adds a path of nodes and links between the origin and the
        destination.

        Parameters
        ----------
        path : Iterable[Union[Node, Link]]
            A path consists of an alternating sequence of nodes and links,
            starting from the first node and ending at the last. For example, a
            valid path is: `node1, link1, node2, link2, node3, ..., nodeN`.
        origin : Origin, optional
            The origin where the path starts from. Pass `None` to have no
            origin attached to the first node in `path`.
        destination : Destination, optional
            The destination where the path ends in. Pass `None` to have no
            destination attached to the last node in `path`.

        Returns
        -------
        Network
            A reference to itself.

        Raises
        ------
        TypeError
            Raises if
            - the first or last points in `path` are not a `Node`
            - the alternation of `Link`s and `Node`s is not respected
            - the path has length 1, which is not accepted.
        '''
        path = iter(path)
        first_node = next(path)
        if not isinstance(first_node, Node):
            raise TypeError(
                f'First element of the path must be a `{Node.__name__}`; got '
                f'{type(first_node)} instead.')
        self.add_node(first_node)
        if origin is not None:
            self.add_origin(origin, first_node)
        current_link: List[Union[Node, Link]] = [first_node]

        longer_than_one = False
        for i, point in enumerate(path):
            longer_than_one = True
            current_link.append(point)
            L = len(current_link)
            if L == 2:
                if not isinstance(point, Link):
                    raise TypeError(
                        f'Expected a `{Link.__name__}` at index {i} of the '
                        f'path; got {type(point)} instead.')
            else:  # L == 3
                if not isinstance(point, Node):
                    raise TypeError(
                        f'Expected a `{Node.__name__}` at index {i} of the '
                        f'path; got {type(point)} instead.')
                self.add_node(point)
                self.add_link(*current_link)
                current_link = current_link[-1:]
        if not longer_than_one:
            raise ValueError('Path must be longer than a single node.')

        last_node = point
        if not isinstance(first_node, Node):
            raise TypeError(
                f'Last element of the path must be a `{Node.__name__}`; got '
                f'{type(first_node)} instead.')
        if destination is not None:
            self.add_destination(destination, last_node)
        return self

    def validate(self) -> 'Network':
        '''Checks whether the network is consistent.

        Returns
        -------
        Network
            A reference to itself.

        Raises
        ------
        InvalidNetworkError
            Raises if
             - a node has both an origin and a destination
             - a link, origin or destination is duplicated in the network
             - a node with an origin that is not a ramp has also entering links
             - a node with a destination has also exiting links
        '''

        # nodes must not have origins and destinations
        for node, nodedata in self.nodes.data():
            if ORIGINENTRY in nodedata and DESTINATIONENTRY in nodedata:
                raise InvalidNetworkError(
                    f'Node {node.name} must either have an origin or a '
                    'destination, but not both.')

        # no duplicate elements
        def origin_destination_yielder():
            for data, entry in product(self._graph.nodes.values(),
                                       (ORIGINENTRY, DESTINATIONENTRY)):
                if entry in data:
                    yield data[entry]

        count: Dict[ElementBase, int] = {}
        for o in chain((link[2] for link in self.links),
                       iter(origin_destination_yielder())):
            d = count.get(o, 0) + 1
            if d > 1:
                raise InvalidNetworkError(
                    f'Element {o.name} is duplicated in the network.')
            count[o] = d

        # nodes with origins (that are not a ramps) must have no entering links
        for origin, node in self.origins.items():
            if not isinstance(origin, MeteredOnRamp) and \
                    any(self.in_links(node)):
                raise InvalidNetworkError(
                    f'Expected node {node.name} to have no entering links, as '
                    f'it is connected to origin {origin.name} (only ramps '
                    'support entering links).')

        # nodes with destinations must have no exiting links
        for destination, node in self.destinations.items():
            if any(self.out_links(node)):
                raise InvalidNetworkError(
                    f'Expected node {node.name} to have no exiting links, as '
                    f'it is connected to destination {destination.name}.')
        return self
