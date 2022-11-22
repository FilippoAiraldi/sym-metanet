from abc import ABC, abstractmethod
from itertools import count
from typing import Dict, TypeVar


sym_var = TypeVar('sym_var')
sym_var.__doc__ = \
    'Variable that can also be numerical or symbolic, depending on the engine.'


class ElementBase(ABC):
    '''Base class for any element for a highway modelled in METANET.'''

    __ids: Dict[type, count] = {}

    def __init__(self, name: str = None) -> None:
        '''Instantiates the element with the given `name` attribute.

        Parameters
        ----------
        name : str, optional
            Name of the element. If `None`, one is automatically created from a
            counter of the class' instancies.
        '''
        cls = self.__class__
        if cls in self.__ids:
            _id = self.__ids[cls]
        else:
            _id = count(0)
            self.__ids[cls] = _id
        self.name = name or f'{cls.__name__}{next(_id)}'

    # @abstractmethod
    def init_vars(self, *args, **kwargs) -> None:
        raise NotImplementedError('Variable initialization not supported '
                                  f'for {self.__class__.__name__}.')

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f'<{self.name}: {self.__class__.__name__}>'
