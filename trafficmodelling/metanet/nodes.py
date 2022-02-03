from ..util import NamedClass


# class WithDownNode:
#     __node_down = None

#     @property
#     def node_down(self):
#         return self.__node_down

#     def _set_node_down(self, node_down):
#         if self.__node_down is not None:
#             raise ValueError(
#                 f'{self.__class__.__name__} {self.name} already had '
#                 f'downstream node {self.__node_down.name}; tried '
#                 f'to set {node_down.name} node instead.')
#         self.__node_down = node_down


# class WithUpNode:
#     __node_up = None

#     @property
#     def node_up(self):
#         return self.__node_up

#     def _set_node_up(self, node_up):
#         if self.__node_up is not None:
#             raise ValueError(
#                 f'{self.__class__.__name__} {self.name} already had '
#                 f'upstream node {self.__node_up.name}; tried '
#                 f'to set {node_up.name} node instead.')
#         self.__node_up = node_up


class Node(NamedClass):
    '''METANET node'''

    def __init__(self, name=None) -> None:
        super().__init__(name=name)
