from typing import Dict, Literal, Tuple, TYPE_CHECKING
from sym_metanet.blocks.base import ElementBase, sym_var
from sym_metanet.engines.core import EngineBase, get_current_engine
if TYPE_CHECKING:
    from sym_metanet.network import Network
    from sym_metanet.blocks.links import Link


class Origin(ElementBase[sym_var]):
    '''
    Ideal, state-less highway origin that conveys to the attached link as much
    flow as the flow in such link.
    '''

    def init_vars(self, *args, **kwargs) -> None:
        '''Initializes no variable in the ideal origin.'''
        pass

    def step(self, *args, **kwargs) -> None:
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
        down_link = self._get_exiting_link(net=net)
        return down_link.vars['v'][0], down_link.get_flow(engine=engine)[0]

    def _get_exiting_link(self, net: 'Network') -> 'Link':
        '''Internal utility to fetch the link leaving this destination (can
        only be one).'''
        down_links = net.out_links(net.origins[self])
        assert len(down_links) == 1, \
            'Internal error. Only one link can leave an origin.'
        return next(iter(down_links))[-1]


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
        self.vars = {
            name: (
                init_conditions[name]
                if name in init_conditions else
                engine.var(f'{name}_{self.name}')
            ) for name in ('w', 'r', 'd')
        }

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
        down_link = self._get_exiting_link(net=net)
        v = down_link.vars['v'][0]
        q = engine.origins.get_ramp_flow(
            d=self.vars['d'],
            w=self.vars['w'],
            C=self.C,
            r=self.vars['r'],
            rho_max=down_link.rho_max,
            rho_crit=down_link.rho_crit,
            rho_first=down_link.vars['rho'][0],
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
        self.vars = {
            name: (
                init_conditions[name]
                if name in init_conditions else
                engine.var(f'{name}_{self.name}')
            ) for name in ('w', 'q', 'd')
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
        return self._get_exiting_link(net=net).vars['v'][0], self.vars['q']
