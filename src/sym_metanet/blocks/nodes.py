from typing import Tuple, TYPE_CHECKING
from sym_metanet.blocks.base import ElementBase, sym_var
from sym_metanet.engines.core import EngineBase, get_current_engine
from sym_metanet.views import ORIGINENTRY, DESTINATIONENTRY
if TYPE_CHECKING:
    from sym_metanet.network import Network
    from sym_metanet.blocks.links import Link
    from sym_metanet.blocks.origins import Origin
    from sym_metanet.blocks.destinations import Destination


class Node(ElementBase[sym_var]):
    '''
    Node of the highway [1, Section 3.2.2] representing, e.g., the connection
    between two links. Nodes do not correspond to actual physical components of
    the highway, but are used to separate links in case there is a major change
    in the link parameters or a junction or bifurcation.

    References
    ----------
    [1] Hegyi, A., 2004, "Model predictive control for integrating traffic
        control measures", Netherlands TRAIL Research School.
    '''

    def init_vars(self, *args, **kwargs) -> None:
        raise RuntimeError('Nodes are virtual elements that do not implement '
                           '`init_vars`.')

    def step(self, *args, **kwargs) -> None:
        raise RuntimeError('Nodes are virtual elements that do not implement '
                           '`step`.')

    def get_downstream_density(
        self,
        link: 'Link',
        net: 'Network',
        engine: EngineBase = None
    ) -> sym_var:
        '''Computes the (virtual) downstream density of the node.

        Parameters
        ----------
        link : Link
            The current link (which enter this node) querying this information
            from the nde.
        net : Network
            The network which node and link belongs to.
        engine : EngineBase, optional
            The engine to be used. If `None`, the current engine is used.

        Returns
        -------
        sym_var
            Returns the (virtual) downstream density.
        '''
        # following the link entering this node, this node can only be a
        # destination or have multiple exiting links
        nodedata = net.nodes[self]
        if DESTINATIONENTRY in nodedata:
            destination: 'Destination' = nodedata[DESTINATIONENTRY]
            return destination.vars['rho'] \
                if destination.has_var('rho') else link.vars['rho'][-1]

        # if no destination, then there must be 1 or more exiting links
        if engine is None:
            engine = get_current_engine()
        down_links = net.out_links(self)
        rho_firsts = engine.vcat(
            *(dlink.vars['rho'][-1] for _, _, dlink in down_links))
        return engine.nodes.get_downstream_density(rho_firsts)

    def get_upstream_flow_and_speed(
        self,
        link: 'Link',
        net: 'Network',
        engine: EngineBase = None
    ) -> Tuple[sym_var, sym_var]:
        '''Computes the (virtual) upstream flow and speed of the node.

        Parameters
        ----------
        link : Link
            The current link (which departs from this node) querying this
            information from the nde.
        net : Network
            The network which node and link belongs to.
        engine : EngineBase, optional
            The engine to be used. If `None`, the current engine is used.

        Returns
        -------
        tuple[sym_var, sym_var]
            Returns the (virtual) upstream flow and speed.
        '''
        if engine is None:
            engine = get_current_engine()

        # check incoming links
        up_links = net.in_links(self)
        if any(up_links):
            v_last = []
            q_last = []
            for _, _, ulink in up_links:
                v_last.append(ulink.vars['v'][-1])
                q_last.append(ulink.vars['q'][-1])
            v_last = engine.vcat(*v_last)
            q_last = engine.vcat(*q_last)
        else:
            v_last = None
            q_last = 0

        # check for origins. If no origin exists, then q_orig is zero;
        # otherwise, take the q_orig from the origin itself, if it has one, or
        # from the link (ideal origin).
        nodedata = net.nodes[self]
        if ORIGINENTRY in nodedata:
            origin: 'Origin' = nodedata[ORIGINENTRY]
            q_orig = origin.vars['q'] \
                if origin.has_var('q') else link.vars['q'][0]
        else:
            q_orig = 0

        # there are always downstream links (includes this link)
        down_links = net.out_links(self)
        betas = engine.vcat(*(dlink.turnrate for _, _, dlink in down_links))

        # if no virtual speed (i.e., no entering link), assume speed is the
        # same as first segment in link
        q = engine.nodes.get_upstream_flow(
            q_lasts=q_last,
            beta=link.turnrate,
            betas=betas,
            q_orig=q_orig
        )
        if v_last is not None:
            v = engine.nodes.get_upstream_speed(
                q_lasts=q_last,
                v_lasts=v_last
            )
        else:
            v = link.vars['v'][0]
        return q, v
