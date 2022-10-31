from sym_metanet.util import NamedObject


class Origin(NamedObject):
    '''
    Ideal, state-less highway origin that conveys to the attached link as much 
    flow as the flow in such link.
    '''

    def __init__(self, name: str = None) -> None:
        super().__init__(name)

    # TODO: do we need bare-bone origins? Maybe they just provide the same
    # flow as the next link


class MeteredOnRamp(Origin):
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


class SimpleMeteredOnRamp(MeteredOnRamp):
    '''
    A simplified version of the vanilla on-ramp where the flow of vehicle on 
    the ramp is the direct control action, instead of controlling the metering 
    rate that in turns dictates the car flow on the ramp.

    See `MeteredOnRamp` for the original version.
    '''


class MainstreamOrigin(Origin):
    '''METANET Highway mainstream-origin'''
    ...

    # TODO: improve doc
