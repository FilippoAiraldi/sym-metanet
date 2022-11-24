from abc import ABC, abstractmethod
from itertools import count
from typing import Dict, Generic, TypeVar, Optional


class ElementBase:
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

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f'<{self.name}: {self.__class__.__name__}>'


sym_var = TypeVar('sym_var')
sym_var.__doc__ = ('Variable that can also be numerical or symbolic, '
                   'depending on the engine. Should be indexable as an array '
                   'in case of vector quantities.')

NO_VARS = None


class ElementWithVars(ElementBase, Generic[sym_var], ABC):
    '''Base class for any element with states, actions or disturbances.'''

    def __init__(self, name: str = None) -> None:
        '''Instantiates the element with the given `name` attribute.

        Parameters
        ----------
        name : str, optional
            Name of the element. If `None`, one is automatically created from a
            counter of the class' instancies.
        '''
        super().__init__(name=name)
        self.states: Optional[Dict[str, sym_var]] = NO_VARS
        self.next_states: Optional[Dict[str, sym_var]] = NO_VARS
        self.actions: Optional[Dict[str, sym_var]] = NO_VARS
        self.disturbances: Optional[Dict[str, sym_var]] = NO_VARS

    @property
    def has_states(self) -> bool:
        '''Gets whether this element has state variables.'''
        return self.states is not NO_VARS

    @property
    def has_next_states(self) -> bool:
        '''Gets whether this element has state variables (computed by stepping
        the dynamics).'''
        return self.next_states is not NO_VARS

    @property
    def has_actions(self) -> bool:
        '''Gets whether this element has control action variables.'''
        return self.actions is not NO_VARS

    @property
    def has_disturbances(self) -> bool:
        '''Gets whether this element has disturbance variables.'''
        return self.disturbances is not NO_VARS

    @abstractmethod
    def init_vars(self, *args, **kwargs) -> None:
        '''Initializes the variable dicts (`states`, `actions`, `disturbances`)
        of this element.'''
        raise NotImplementedError('Variable initialization not supported for '
                                  + self.__class__.__name__ + '.')

    @abstractmethod
    def step_dynamics(self, *args, **kwargs) -> Dict[str, sym_var]:
        '''Internal method for stepping the element's dynamics by one time
        step.

        Returns
        -------
        Dict[str, sym_var]
            A dict with the states at the next time step.
        '''
        raise NotImplementedError('Stepping the dynamics not supported for '
                                  + self.__class__.__name__ + '.')

    def step(self, *args, **kwargs) -> None:
        '''Steps the dynamics of this element.'''
        assert self.states is not NO_VARS, 'States not initialized.'
        next_states = self.step_dynamics(*args, **kwargs)
        if self.next_states is NO_VARS:
            self.next_states = {}
        for name in self.states:
            self.next_states[name] = next_states[name]

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f'<{self.name}: {self.__class__.__name__}>'
