from functools import cached_property
from typing import Dict, Iterable, Tuple, Union
import networkx as nx
from pymetanet.util.datastructures import NamedObject
from pymetanet.blocks import Node, Link


class LinkView(nx.classes.reportviews.OutEdgeView):
    '''
    Wrapper around `networkx`'s edge view to facilitate operations with links.
    '''

    def __init__(self, net: 'Network') -> None:
        super().__init__(net._graph)

    def __getitem__(self, e: Tuple[Node, Node]) -> Link:
        return super().__getitem__(e)['link']

    def __iter__(self) -> Tuple[Node, Link, Node]:
        for un, dns in self._nodes_nbrs():
            for dn, l in dns.items():
                yield (un, l['link'], dn)


class Network(NamedObject):
    '''Highway network'''

    def __init__(self, name: str = None):
        '''Instantiates an highway network.

        Parameters
        ----------
        name : str, optional
            Name of the network, by default `None`.
        '''
        NamedObject.__init__(self, name=name)
        self._graph = nx.DiGraph(name=name)
        self.nodes_by_name: Dict[str, Node] = {}
        self.links_by_name: Dict[str, Link] = {}

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
    def links(self) -> LinkView:
        '''Returns a view on the links of the network.'''
        return LinkView(self)

    def add_node(self, node: Node) -> None:
        '''Adds a node to the highway network. 

        Parameters
        ----------
        node : Node
            Node to be added.
        '''
        self._graph.add_node(node)
        self.nodes_by_name[node.name] = node

    def add_nodes(self, *nodes: Node) -> None:
        '''Adds multiple nodes. See `Network.add_node`.'''
        self._graph.add_nodes_from(nodes)
        self.nodes_by_name.update({node.name: node for node in nodes})

    def add_link(self, node_up: Node, link: Link, node_down: Node) -> None:
        '''Adds a link to the highway network, between two nodes. 

        Parameters
        ----------
        node_up : Node
            Upstream node, that is, where traffic is coming from.
        link : Link
            The link to be added connecting the two nodes.
        node_down : Node
            Downstream node, that is, where traffic is going to.
        '''
        self._graph.add_edge(node_up, node_down, link=link)
        self.links_by_name[link.name] = link

    def add_links(self, *links: Tuple[Node, Link, Node]) -> None:
        '''Adds multiple links. See `Network.add_link`.'''
        self._graph.add_edges_from(
            (l[0], l[2], {'link': l[1]}) for l in links)
        self.links_by_name.update({l[1].name: l[1] for l in links})

    def add_path(self, path: Iterable[Union[Node, Link]]) -> None:
        '''Adds a path of nodes and links to the network.

        Parameters
        ----------
        path : Iterable[Union[Node, Link]]
            A path consists of an alternating sequence of nodes and links, 
            starting from the first node and ending at the last. For example, a 
            valid path is: `node1, li|nk1, node2, link2, node3, ..., nodeN`. If
            path ends with a link, this one is not added since a downstream 
            node is missing.

        Raises
        ------
        TypeError
            Raises if the first point in the path is not a `Node`; then, raises 
            if the alternation of `Link`s and `Node`s is not respected.
        '''
        path = iter(path)
        first_node = next(path)
        if not isinstance(first_node, Node):
            raise TypeError(
                'First element of the path must be a `Node`; got '
                f'{type(first_node)} instead.')
        self.add_node(first_node)
        current_link = [first_node]

        for point in path:
            current_link.append(point)
            L = len(current_link)
            if L == 2:
                if not isinstance(point, Link):
                    raise TypeError(
                        'Middle element of each triplet must be a `Link`; got '
                        f'{type(point)} instead.')
            else:  # L == 3
                if not isinstance(point, Node):
                    raise TypeError(
                        'Last element of each triplet must be a `Node`; got '
                        f'{type(point)} instead.')
                self.add_node(point)
                self.add_link(*current_link)
                current_link = current_link[-1:]
