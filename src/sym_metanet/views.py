from typing import (
    TYPE_CHECKING,
    Any,
    Collection,
    Dict,
    Generator,
    Iterable,
    Optional,
    Tuple,
    Union,
)

import networkx as nx

if TYPE_CHECKING:
    from sym_metanet.blocks.links import Link
    from sym_metanet.blocks.nodes import Node


LINKENTRY = "link"
ORIGINENTRY = "origin"
DESTINATIONENTRY = "destination"


class OutLinkViewWrapper(nx.classes.reportviews.OutEdgeView):
    """Wrapper around `networkx`'s outward edge view to facilitate operations
    with link."""

    def __getitem__(self, e: Tuple["Node", "Node"]) -> "Link":
        return super().__getitem__(e)[LINKENTRY]

    def __iter__(self) -> Generator[Tuple["Node", "Node", "Link"], None, None]:
        for un, dns in self._nodes_nbrs():
            for dn, l in dns.items():
                yield (un, dn, l[LINKENTRY])

    def __call__(
        self,
        nbunch: Union[None, "Node", Iterable["Node"]] = None,
        data: Union[bool, str] = LINKENTRY,
        default: Optional[Dict[str, Any]] = None,
    ) -> Collection[Tuple["Node", "Node", "Link"]]:
        return super().__call__(nbunch, data, default)


class InLinkViewWrapper(nx.classes.reportviews.InEdgeView):
    """Wrapper around `networkx`'s inward edge view to facilitate operations
    with link."""

    def __getitem__(self, e: Tuple["Node", "Node"]) -> "Link":
        return super().__getitem__(e)[LINKENTRY]

    def __iter__(self) -> Generator[Tuple["Node", "Node", "Link"], None, None]:
        for un, dns in self._nodes_nbrs():
            for dn, l in dns.items():
                yield (un, dn, l[LINKENTRY])

    def __call__(
        self,
        nbunch: Union[None, "Node", Iterable["Node"]] = None,
        data: Union[bool, str] = LINKENTRY,
        default: Optional[Dict[str, Any]] = None,
    ) -> Collection[Tuple["Node", "Node", "Link"]]:
        return super().__call__(nbunch, data, default)
