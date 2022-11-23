from typing import Dict, TYPE_CHECKING
from sym_metanet.blocks.base import ElementBase, sym_var
from sym_metanet.engines.core import EngineBase, get_current_engine
if TYPE_CHECKING:
    from sym_metanet.blocks.links import Link


class Origin(ElementBase[sym_var]):
    '''
    Ideal, state-less highway origin that conveys to the attached link as much
    flow as the flow in such link.
    '''

    def __init__(self, name: str = None) -> None:
        super().__init__(name)

    def init_vars(self, *args, **kwargs) -> None:
        pass



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

    def __init__(self, capacity: sym_var, name: str = None) -> None:
        '''Instantiates an on-ramp with the given capacity.

        Parameters
        ----------
        capacity : float or symbolic
            Capacity of the on-ramp, i.e., `C`.
        name : str, optional
            Name of the on-ramp, by default None.
        '''
        super().__init__(name=name)
        self.C = capacity

    def init_vars(
        self,
        init_conditions: Dict[str, sym_var] = None,
        engine: EngineBase = None
    ) -> None:
        '''Initializes
        - `w`: queue length (state)
        - `r`: ramp metering rate (control action)
        - `d`: demand (disturbance).

        Parameters
        ----------
        init_conditions : dict[str, variable], optional
            Provides name-variable tuples to initialize states and actions with
            specific values. These values must be compatible with the symbolic
            engine in type and shape. If not provided, variables are
            initialized automatically.
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
                engine.var(name, (1, 1))
            ) for name in ('w', 'r', 'd')
        }


class SimpleMeteredOnRamp(MeteredOnRamp[sym_var]):
    '''
    A simplified version of the vanilla on-ramp, where the flow of vehicles on
    the ramp is the direct control action (instead of controlling the metering
    rate that in turns dictates the car flow on the ramp).

    See `MeteredOnRamp` for the original version.
    '''

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
            Provides name-variable tuples to initialize states and actions with
            specific values. These values must be compatible with the symbolic
            engine in type and shape. If not provided, variables are
            initialized automatically.
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
                engine.var(name, (1, 1))
            ) for name in ('w', 'q', 'd')
        }
