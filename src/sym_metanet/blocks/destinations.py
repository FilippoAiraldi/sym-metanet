from typing import Dict, TYPE_CHECKING
from sym_metanet.blocks.base import ElementBase, sym_var
from sym_metanet.engines.core import EngineBase, get_current_engine
if TYPE_CHECKING:
    from sym_metanet.network import Network
    from sym_metanet.blocks.links import Link


class Destination(ElementBase[sym_var]):
    '''
    Ideal congestion-free destination, representing a sink where cars can leave
    the highway with no congestion (i.e., no slowing down due to downstream
    density).
    '''

    def init_vars(self, *args, **kwargs) -> None:
        '''Initializes no variable in the ideal destination.'''
        pass

    def step(self, *args, **kwargs) -> None:
        '''No dynamics to steps in the ideal destination.'''
        pass


class CongestedDestination(Destination[sym_var]):
    '''
    Destination with a downstream density scenario to emulate congestions, that
    is, cars cannot exit freely the highway but must slow down and, possibly,
    create a congestion.
    '''

    def init_vars(
        self,
        net: 'Network',
        init_conditions: Dict[str, sym_var] = None,
        engine: EngineBase = None,
        **kwargs
    ) -> None:
        '''Initializes
        - `d`: downstream density scenario (disturbance).

        Parameters
        ----------
        net : Network
            The network this destination belongs to.
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
        d = (
            engine.var(f'd_{self.name}')
            if init_conditions is None or 'd' not in init_conditions else
            init_conditions['d']
        )
        up_links = net.out_links(net.origins[self])
        assert len(up_links) == 1, \
            'Internal error. Only one link can enter a destination.'
        up_link: 'Link' = next(iter(up_links))[-1]
        rho = engine.destinations.get_congested_downstream_density(
            rho_last=up_link.vars['rho'][-1],
            rho_crit=up_link.rho_crit,
            rho_destination=d
        )
        self.vars = {'d': d, 'rho': rho}
