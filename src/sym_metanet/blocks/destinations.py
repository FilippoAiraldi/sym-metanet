from typing import Dict, TYPE_CHECKING
from sym_metanet.blocks.base import ElementBase, sym_var
from sym_metanet.engines.core import EngineBase, get_current_engine


class Destination(ElementBase[sym_var]):
    '''
    Ideal congestion-free destination, representing a sink where cars can leave
    the highway with no congestion (i.e., no slowing down due to downstream
    density).
    '''

    def init_vars(self, *args, **kwargs) -> None:
        pass


class CongestedDestination(Destination[sym_var]):
    '''
    Destination with a downstream density scenario to emulate congestions, that
    is, cars cannot exit freely the highway but must slow down and, possibly,
    create a congestion.
    '''

    def init_vars(
        self,
        init_conditions: Dict[str, sym_var] = None,
        engine: EngineBase = None
    ) -> None:
        '''Initializes
        - `d`: downstream density scenario (disturbance).

        Parameters
        ----------
        init_d : sym_var, optional
            The initial density scenario. If `None`, it is automatically
            initialized.
        init_conditions : dict[str, variable], optional
            Provides name-variable tuples to initialize states, actions and 
            disturbances with specific values. These values must be compatible
            with the symbolic engine in type and shape. If not provided,
            variables are initialized automatically.
        engine : EngineBase, optional
            The engine to be used. If `None`, the current engine is used.
        '''
        if engine is None:
            engine = get_current_engine()
        self.vars = {
            'd': engine.var('d', (1, 1))
            if init_conditions is None or 'd' not in init_conditions else
            init_conditions['d']
        }
