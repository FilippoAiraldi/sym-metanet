from sym_metanet.blocks.base import ElementBase, sym_var

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from sym_metanet.blocks.links import Link


class Origin(ElementBase[sym_var]):
    '''
    Ideal, state-less highway origin that conveys to the attached link as much
    flow as the flow in such link.
    '''

    def __init__(self, name: str = None) -> None:
        super().__init__(name)

    def init_vars(self, link_down: 'Link[sym_var]') -> None:
        '''Initializes the ideal origin variables by borrowing them from the
        first link's segment attached the origin.

        Parameters
        ----------
        link_down : Link
            Downstream link from the origin.
        '''
        self.vars = {n: link_down.vars[n][0] for n in ('v', 'q')}


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


class SimpleMeteredOnRamp(MeteredOnRamp[sym_var]):
    '''
    A simplified version of the vanilla on-ramp where the flow of vehicle on
    the ramp is the direct control action, instead of controlling the metering
    rate that in turns dictates the car flow on the ramp.

    See `MeteredOnRamp` for the original version.
    '''
