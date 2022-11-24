from typing import Tuple, TYPE_CHECKING
from sym_metanet.blocks.base import ElementBase, sym_var
from sym_metanet.engines.core import EngineBase, get_current_engine
from sym_metanet.util.funcs import first
if TYPE_CHECKING:
    from sym_metanet.network import Network
    from sym_metanet.blocks.links import Link


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
        self, net: 'Network', engine: EngineBase = None, **kwargs
    ) -> sym_var:
        '''Computes the (virtual) downstream density of the node.

        Parameters
        ----------
        net : Network
            The network which node and link belongs to.
        engine : EngineBase, optional
            The engine to be used. If `None`, the current engine is used.

        Returns
        -------
        sym_var
            Returns the (virtual) downstream density.
        '''
        if engine is None:
            engine = get_current_engine()

        # following the link entering this node, this node can only be a
        # destination or have multiple exiting links
        if self in net.destinations_by_node:
            return net.destinations_by_node[self].get_density(
                net=net, engine=engine, **kwargs)

        # if no destination, then there must be 1 or more exiting links
        links_down = net.out_links(self)
        if len(links_down) == 1:
            return first(links_down)[-1].vars['rho'][0]
        rho_firsts = engine.vcat(
            *(dlink.vars['rho'][-1] for _, _, dlink in links_down))
        return engine.nodes.get_downstream_density(rho_firsts)

    def get_upstream_speed_and_flow(
        self,
        net: 'Network',
        link: 'Link',
        engine: EngineBase = None,
        **kwargs
    ) -> Tuple[sym_var, sym_var]:
        '''Computes the (virtual) upstream speed and flow of the node for this
        the current link.

        Parameters
        ----------
        net : Network
            The network which node and link belongs to.
        link : Link
            The current link (which departs from this node) querying this
            information from the node.
        engine : EngineBase, optional
            The engine to be used. If `None`, the current engine is used.

        Returns
        -------
        tuple[sym_var, sym_var]
            Returns the (virtual) upstream speed and flow.
        '''
        if engine is None:
            engine = get_current_engine()

        # the node can have 1 or more entering links, as well as a ramp origin.
        # Speed is dictated by the entering links, if any; otherwise by the
        # origin (same as first segment). Flow is dictated both by entering
        # links and origin.
        links_up = net.in_links(self)
        n_up = len(links_up)
        if self in net.origins_by_node:
            origin = net.origins_by_node[self]
            v_o, q_o = origin.get_speed_and_flow(
                net=net, engine=engine, **kwargs)
        else:
            v_o = None
            q_o = None

        if n_up == 0:
            v = v_o
            q = q_o
        elif n_up == 1:
            link_up = next(iter(links_up))[-1]
            v = link_up.vars['v'][-1]
            q = link_up.get_flow(engine=engine)[-1]
            if q_o is not None:
                q += q_o
        else:
            v_last = []
            q_last = []
            for _, _, link_up in links_up:
                v_last.append(link_up.vars['v'][-1])
                q_last.append(link_up.get_flow(engine=engine)[-1])
            v_last = engine.vcat(*v_last)
            q_last = engine.vcat(*q_last)
            betas = engine.vcat(
                *(dlink.turnrate for _, _, dlink in net.out_links(self)))

            v = engine.nodes.get_upstream_speed(q_lasts=q_last, v_lasts=v_last)
            q = engine.nodes.get_upstream_flow(
                q_lasts=q_last,
                beta=link.turnrate,
                betas=betas,
                q_orig=q_o
            )
        return v, q
