from .util import NamedClass


class Node(NamedClass):
    '''METANET node'''

    def __init__(self, name=None) -> None:
        super().__init__(name=name)
