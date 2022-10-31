from sym_metanet.util import NamedObject


class Node(NamedObject):
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
