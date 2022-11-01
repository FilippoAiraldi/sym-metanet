from typing import Any, Dict, Sequence, Tuple, Union
import networkx as nx
from sym_metanet.blocks.nodes import Node
from sym_metanet.blocks.links import Link


LINKENTRY = 'link'
ORIGINENTRY = 'origin'
DESTINATIONENTRY = 'destination'


class OutLinkViewWrapper(nx.classes.reportviews.OutEdgeView):
    '''Wrapper around `networkx`'s outward edge view to facilitate operations 
    with link'''

    def __getitem__(self, e: Tuple[Node, Node]) -> Link:
        return super().__getitem__(e)[LINKENTRY]

    def __iter__(self) -> Tuple[Node, Node, Link]:
        for un, dns in self._nodes_nbrs():
            for dn, l in dns.items():
                yield (un, dn, l[LINKENTRY])

    def __call__(
        self,
        nbunch: Union[Node, Sequence[Node]] = None,
        data: Union[bool, str] = LINKENTRY,
        default: Dict[str, Any] = None
    ) -> Union[Tuple[Node, Node], Tuple[Node, Node, Link]]:
        return super().__call__(nbunch, data, default)


class InLinkViewWrapper(nx.classes.reportviews.InEdgeView):
    '''Wrapper around `networkx`'s inward edge view to facilitate operations 
    with link'''

    def __getitem__(self, e: Tuple[Node, Node]) -> Link:
        return super().__getitem__(e)[LINKENTRY]

    def __iter__(self) -> Tuple[Node, Node, Link]:
        for un, dns in self._nodes_nbrs():
            for dn, l in dns.items():
                yield (un, dn, l[LINKENTRY])

    def __call__(
        self,
        nbunch: Union[Node, Sequence[Node]] = None,
        data: Union[bool, str] = LINKENTRY,
        default: Dict[str, Any] = None
    ) -> Union[Tuple[Node, Node], Tuple[Node, Node, Link]]:
        return super().__call__(nbunch, data, default)
