from typing import Dict
from sym_metanet.blocks.base import ElementBase, sym_var
from sym_metanet.engines.core import EngineBase, get_current_engine


class Link(ElementBase[sym_var]):
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
        lanes: sym_var,
        length: sym_var,
        free_flow_velocity: sym_var,
        critical_density: sym_var,
        a: sym_var,
        turnrate: sym_var = 1.0,
        name: str = None
    ) -> None:
        '''Creates an instance of a METANET link.

        Parameters
        ----------
        nb_segments : int
            Number of segments in this highway link, i.e., `N`.
        lanes : int or symbolic
            Number of lanes in each segment, i.e., `lam`.
        lengths : float or symbolic
            Length of each segment in the link, i.e., `L`.
        free_flow_velocities : float or symbolic
            Average speed of cars when traffic is freely flowing, i.e.,
            `v_free`.
        critical_densities : float or symbolic
            Critical density at which the traffic flow is maximal, i.e.,
            `rho_crit`.
        a : float or symbolic
            Model parameter in the computations of the equivalent speed
            [1, Equation 3.4].
        turnrate : float or symbolic, optional
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

    def init_vars(
        self,
        init_conditions: Dict[str, sym_var] = None,
        engine: EngineBase = None
    ) -> None:
        '''For each segment in the link, initializes
        -  `rho`: densities (state)
        -  `v`: speeds (state).

        Parameters
        ----------
        init_conditions : dict[str, variable], optional
            Provides name-variable tuples to initialize states, actions and 
            disturbances with specific values. These values must be compatible
            with the symbolic engine in type and shape. If not provided,
            variables are initialized automatically.
        engine : EngineBase, optional
            The engine to be used. If `None`, the current engine is used.
        '''
        if init_conditions is None:
            init_conditions = {}
        if engine is None:
            engine = get_current_engine()
        self.vars = {
            name: (
                init_conditions[name]
                if name in init_conditions else
                engine.var(name, self.N)
            ) for name in ('rho', 'v')
        }
