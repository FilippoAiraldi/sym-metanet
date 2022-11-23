from sym_metanet.blocks.base import ElementBase, sym_var


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
