from pymetanet.util.datastructures import NamedObject


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


class Origin(NamedObject):
    '''
    Dummy, state-less highway origin that conveys to the attached
    link as much flow as the flow in such link.
    '''
    # TODO: do we need bare-bone origins? Maybe they just provide the same
    # flow as the next link


class OnRamp(Origin):
    '''
    On-ramp where cars can queue up before being given access to the attached
    link.
    '''
    # TODO: improve doc

    def __init__(self, capacity: float, name: str = None) -> None:
        '''Instantiates an on-ramp with the given capacity.

        Parameters
        ----------
        capacity : float
            Capacity of the on-ramp, i.e., `C`. 
        name : str, optional
            Name of the on-ramp, by default None.
        '''
        super().__init__(name=name)
        self.C = capacity


class MainstreamOrigin(Origin):
    '''METANET Highway mainstream-origin'''
    ...

    # TODO: improve doc


class Destination(NamedObject):
    '''Congestion-free destination of the highway traffic'''
    # TODO: improve doc


class Link(NamedObject):
    '''
    Highway link between two nodes [1, Section 3.2.1]. Links represent stretch
    of highway with similar traffic characteristics and no road changes (e.g., 
    same number of lanes and maximum speed).

    References
    ----------
    [1] Hegyi, A., 2004, "Model predictive control for integrating traffic 
        control measures", Netherlands TRAIL Research School.
    '''

    def __init__(
        self,
        nb_segments: int,
        lanes: int,
        length: float,
        free_flow_velocity: float,
        critical_density: float,
        a: float,
        name: str = None
    ) -> None:
        '''Creates an instance of a METANET link.

        Parameters
        ----------
        nb_segments : int
            Number of segments in this highway link, i.e., `N`.
        lanes : int
            Number of lanes in each segment, i.e., `lam`. 
        lengths : float
            Length of each segment in the link, i.e., `L`. 
        free_flow_velocities : float
            Average speed of cars when traffic is freely flowing, i.e., 
            `v_free`.
        critical_densities : float
            Critical density at which the traffic flow is maximal, i.e., 
            `rho_crit`.
        a : float
            Model parameter in the computations of the equivalent speed
            [1, Equation 3.4]. 
        name : str, optional
            Name of this link, by default `None`.

        References
        ----------
        [1] Hegyi, A., 2004, "Model predictive control for integrating traffic 
            control measures", Netherlands TRAIL Research School.
        '''
        super().__init__(name)
        self.N = nb_segments
        self.lam = lanes
        self.L = length
        self.v_free = free_flow_velocity
        self.rho_crit = critical_density
        self.a = a
