from .util import NamedClass


class Destination(NamedClass):
    '''METANET destination'''

    def __init__(self, name=None) -> None:
        super().__init__(name=name)
