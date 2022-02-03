from ..util import NamedClass, SmartList


class Origin(NamedClass):
    '''METANET origin'''

    @property
    def queue(self) -> SmartList:
        return self.__w

    @property
    def flow(self) -> SmartList:
        return self.__q

    def __init__(self, name=None) -> None:
        super().__init__(name=name)
        self.reset()

    def reset(self):
        # initialize state (queue) and other quantities (flow, demand)
        self.__w = SmartList()
        self.__w.append(0.0)
        self.__q = SmartList()
        self.demand = SmartList()


class OnRamp(Origin):
    '''METANET on-ramp'''

    @property
    def rate(self) -> SmartList:
        return self.__r

    def __init__(self, capacity, name=None) -> None:
        super().__init__(name=name)
        self.capacity = capacity
        self.reset()

    def reset(self):
        # initialize other quantities (metering rate)
        super().reset()
        self.__r = SmartList()
        self.__r.append(1.0)


class MainstreamOrigin(Origin):
    '''METANET mainstream-origin'''

    def __init__(self, name=None) -> None:
        super().__init__(name=name)
