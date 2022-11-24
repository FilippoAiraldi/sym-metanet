from typing import Dict, Literal, Tuple, TYPE_CHECKING
from sym_metanet.blocks.base import ElementWithVars, sym_var
from sym_metanet.engines.core import EngineBase, get_current_engine
from sym_metanet.util.funcs import first
if TYPE_CHECKING:
    from sym_metanet.network import Network
    from sym_metanet.blocks.links import Link


class Origin(ElementWithVars[sym_var]):
    '''
    Ideal, state-less highway origin that conveys to the attached link as much
    flow as the flow in such link.
    '''

    def init_vars(self, *args, **kwargs) -> None:
        '''Initializes no variable in the ideal origin.'''
        pass

    def step_dynamics(self, *args, **kwargs) -> None:
        '''No dynamics to steps in the ideal origin.'''
        pass

    def get_speed_and_flow(
        self,
        net: 'Network',
        engine: EngineBase = None,
        **kwargs
    ) -> Tuple[sym_var, sym_var]:
        '''Computes the (upstream) speed and flow induced by the ideal origin.

        Parameters
        ----------
        net : Network
            The network this destination belongs to.
        engine : EngineBase, optional
            The engine to be used. If `None`, the current engine is used.

        Returns
        -------
        tuple[sym_var, sym_var]
            The origin's upstream speed and flow.
        '''
        link_down = self._get_exiting_link(net=net)
        return link_down.states['v'][0], link_down.get_flow(engine=engine)[0]

    def _get_exiting_link(self, net: 'Network') -> 'Link':
        '''Internal utility to fetch the link leaving this destination (can
        only be one).'''
        links_down = net.out_links(net.origins[self])
        assert len(links_down) == 1, \
            'Internal error. Only one link can leave an origin.'
        return first(links_down)[-1]


class MeteredOnRamp(Origin[sym_var]):
    '''
    On-ramp where cars can queue up before being given access to the attached
    link. For reference, look at [1], in particular, Section 3.2.1 and
    Equations 3.5 and 3.6.

    References
    ----------
    [1] Hegyi, A., 2004, "Model predictive control for integrating traffic
        control measures", Netherlands TRAIL Research School.
    '''

    def __init__(
        self,
        capacity: sym_var,
        flow_eq_type: Literal['in', 'out'] = 'out',
        name: str = None
    ) -> None:
        '''Instantiates an on-ramp with the given capacity.

        Parameters
        ----------
        capacity : float or symbolic
            Capacity of the on-ramp, i.e., `C`.
        flow_eq_type : 'in' or 'out', optional
            Type of flow equation for the ramp. See
            `engine.origins.get_ramp_flow` for more details.
        name : str, optional
            Name of the on-ramp, by default None.
        '''
        super().__init__(name=name)
        self.C = capacity
        self.flow_eq_type = flow_eq_type

    def init_vars(
        self,
        init_conditions: Dict[str, sym_var] = None,
        engine: EngineBase = None,
    ) -> None:
        '''Initializes
        - `w`: queue length (state)
        - `r`: ramp metering rate (control action)
        - `d`: demand (disturbance).

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
        self.states = {
            'w': init_conditions['w']
            if 'w' in init_conditions else
            engine.var(f'w_{self.name}')
        }
        self.actions = {
            'r': init_conditions['r']
            if 'r' in init_conditions else
            engine.var(f'r_{self.name}')
        }
        self.disturbances = {
            'd': init_conditions['d']
            if 'd' in init_conditions else
            engine.var(f'd_{self.name}')
        }

    def step_dynamics(
        self,
        net: 'Network',
        T: sym_var,
        engine: EngineBase = None,
        **kwargs
    ) -> Dict[str, sym_var]:
        '''Steps the dynamics of this origin.

        Parameters
        ----------
        net : Network
            The network the origin belongs to.
        T : sym_var
            Sampling time.
        engine : EngineBase, optional
            The engine to be used. If `None`, the current engine is used.

        Returns
        -------
        Dict[str, sym_var]
            A dict with the states of the origin (queue) at the next time step.
        '''
        if engine is None:
            engine = get_current_engine()
        q = self.get_speed_and_flow(net=net, T=T, engine=engine, **kwargs)[-1]
        w_next = engine.origins.step_queue(
            w=self.states['w'], d=self.disturbances['d'], q=q, T=T)
        return {'w': w_next}

    def get_speed_and_flow(
        self,
        net: 'Network',
        T: sym_var,
        engine: EngineBase = None,
        **kwargs
    ) -> Tuple[sym_var, sym_var]:
        '''Computes the (upstream) speed and flow induced by the metered ramp.

        Parameters
        ----------
        net : Network
            The network this destination belongs to.
        T : sym variable
            Sampling time of the simulation.
        engine : EngineBase, optional
            The engine to be used. If `None`, the current engine is used.

        Returns
        -------
        tuple[sym_var, sym_var]
            The origin's upstream speed and flow.
        '''
        if engine is None:
            engine = get_current_engine()
        link_down = self._get_exiting_link(net=net)
        v = link_down.states['v'][0]
        q = engine.origins.get_ramp_flow(
            d=self.disturbances['d'],
            w=self.states['w'],
            C=self.C,
            r=self.actions['r'],
            rho_max=link_down.rho_max,
            rho_crit=link_down.rho_crit,
            rho_first=link_down.states['rho'][0],
            T=T,
            type=self.flow_eq_type
        )
        return v, q


class SimpleMeteredOnRamp(MeteredOnRamp[sym_var]):
    '''
    A simplified version of the vanilla on-ramp, where the flow of vehicles on
    the ramp is the direct control action (instead of controlling the metering
    rate that in turns dictates the car flow on the ramp).

    See `MeteredOnRamp` for the original version.
    '''

    def __init__(self, capacity: sym_var, name: str = None) -> None:
        super().__init__(capacity=capacity, name=name)
        del self.flow_eq_type

    def init_vars(
        self,
        init_conditions: Dict[str, sym_var] = None,
        engine: EngineBase = None
    ) -> None:
        '''Initializes
        - `w`: queue length (state)
        - `q`: ramp flow (control action)
        - `d`: demand (disturbance).

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
        self.states = {
            'w': init_conditions['w']
            if 'w' in init_conditions else
            engine.var(f'w_{self.name}')
        }
        self.actions = {
            'q': init_conditions['q']
            if 'q' in init_conditions else
            engine.var(f'q_{self.name}')
        }
        self.disturbances = {
            'd': init_conditions['d']
            if 'd' in init_conditions else
            engine.var(f'd_{self.name}')
        }

    def get_speed_and_flow(
        self,
        net: 'Network',
        **kwargs
    ) -> Tuple[sym_var, sym_var]:
        '''Computes the (upstream) speed and flow induced by the simple-metered
        ramp.

        Parameters
        ----------
        net : Network
            The network this destination belongs to.
        engine : EngineBase, optional
            The engine to be used. If `None`, the current engine is used.

        Returns
        -------
        tuple[sym_var, sym_var]
            The origin's upstream speed and flow.
        '''
        return \
            self._get_exiting_link(net=net).states['v'][0], self.actions['q']
