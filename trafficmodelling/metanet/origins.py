from ..util import NamedClass, SmartList


class Origin(NamedClass):
    '''METANET origin'''

    def __init__(self, name=None) -> None:
        super().__init__(name=name)
        self.__reset()

    def __reset(self):
        # initialize state (queue) and other quantities (flow, demand)
        self.queue = SmartList.from_list([0.0])
        self.flow = SmartList()
        self.demand = SmartList()


class OnRamp(Origin):
    '''METANET on-ramp'''

    def __init__(self, capacity: float, name=None) -> None:
        super().__init__(name=name)
        self.capacity = capacity
        self.__reset()

    def __reset(self):
        # initialize other quantities (metering rate)
        self.rate = SmartList.from_list([1.0])


class MainstreamOrigin(Origin):
    '''METANET mainstream-origin'''
    pass
