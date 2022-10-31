from sym_metanet.util import NamedObject


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
        turnrate: float = 1.0,
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
        turnrate : float, optional
            Fraction of the total flow that enters this link via the upstream 
            node. Only relevant if multiple exiting links are attached to the 
            same node, in order to split the flow according to these rates. 
            Needs not be normalized. By default, all links have equal rates. 
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
        self.turnrate = turnrate
