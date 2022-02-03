from ..util import NamedClass, SmartList


class Origin(NamedClass):
    '''METANET origin'''

    def __init__(self, name=None) -> None:
        super().__init__(name=name)
        self.reset()

    def reset(self):
        # initialize state (queue) and other quantities (flow, demand)
        self.queue = SmartList()
        self.flow = SmartList()
        self.demand = SmartList()
        self.queue.append(0.0)


class OnRamp(Origin):
    '''METANET on-ramp'''

    def __init__(self, capacity, name=None) -> None:
        super().__init__(name=name)
        self.capacity = capacity
        self.reset()

    def reset(self):
        # initialize other quantities (metering rate)
        super().reset()
        self.rate = SmartList()
        self.rate.append(1.0)


class MainstreamOrigin(Origin):
    '''METANET mainstream-origin'''

    def __init__(self, name=None) -> None:
        super().__init__(name=name)
