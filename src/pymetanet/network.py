from functools import cached_property
from typing import Dict, Iterable, Tuple, Union, Optional
import networkx as nx
from pymetanet.util.datastructures import NamedObject
from pymetanet.util.funcs import cached_property_clearer
from pymetanet.blocks import Node, Link, Origin, Destination


LINKENTRY = 'link'
ORIGINENTRY = 'origin'
DESTINATIONENTRY = 'destination'


class LinkView(nx.classes.reportviews.OutEdgeView):
    '''
    Wrapper around `networkx`'s edge view to facilitate operations with links.
    '''

    def __init__(self, net: 'Network') -> None:
        super().__init__(net._graph)

    def __getitem__(self, e: Tuple[Node, Node]) -> Link:
        return super().__getitem__(e)[LINKENTRY]

    def __iter__(self) -> Tuple[Node, Link, Node]:
        for un, dns in self._nodes_nbrs():
            for dn, l in dns.items():
                yield (un, l[LINKENTRY], dn)


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
        self.origins_by_name: Dict[str, Origin] = {}

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
            A reference to itself
        '''
        self._graph.add_node(node)
        self.nodes_by_name[node.name] = node
        return self

    def add_nodes(self, *nodes: Node) -> 'Network':
        '''Adds multiple nodes. See `Network.add_node`.
        
        Returns
        -------
        Network
            A reference to itself
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
            A reference to itself
        '''
        self._graph.add_edge(node_up, node_down, link=link)
        self.links_by_name[link.name] = link
        return self

    def add_links(self, *links: Tuple[Node, Link, Node]) -> 'Network':
        '''Adds multiple links. See `Network.add_link`.
        
        Returns
        -------
        Network
            A reference to itself
        '''
        self._graph.add_edges_from(
            (l[0], l[2], {LINKENTRY: l[1]}) for l in links)
        self.links_by_name.update({l[1].name: l[1] for l in links})
        return self

    @cached_property_clearer(origins)
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
            A reference to itself
        '''
        self.nodes[node][ORIGINENTRY] = origin
        self.origins_by_name[origin.name] = origin
        return self

    @cached_property_clearer(destinations)
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
            A reference to itself
        '''
        self.nodes[node][DESTINATIONENTRY] = destination
        self.origins_by_name[destination.name] = destination
        return self


    def add_path(
        self,
        origin: Optional[Origin],
        path: Iterable[Union[Node, Link]],
        destination: Optional[Destination]
    ) -> 'Network':
        '''Adds a path of nodes and links between the origin and the 
        destination.

        Parameters
        ----------
        origin : Origin, optional
            The origin where the path starts from. Pass `None` to have no 
            origin attached to the first node in `path`.
        path : Iterable[Union[Node, Link]]
            A path consists of an alternating sequence of nodes and links, 
            starting from the first node and ending at the last. For example, a 
            valid path is: `node1, link1, node2, link2, node3, ..., nodeN`. 
        destination : Destination, optional
            The destination where the path ends in. Pass `None` to have no 
            destination attached to the last node in `path`.

        Returns
        -------
        Network
            A reference to itself

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
        current_link = [first_node]

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
