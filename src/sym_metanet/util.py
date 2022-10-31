from functools import cached_property, wraps
from itertools import count
from typing import Callable, Dict


class NamedObject:
    '''
    Class with a `name` variable that is automatically created, 
    if not provided.
    '''

    __ids: Dict[type, count] = {}

    def __init__(self, name: str = None) -> None:
        '''Instantiates the object with the given `name` attribute.

        Parameters
        ----------
        name : str, optional
            Name of the object. If `None`, one is automatically created from a 
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
        return self.__repr__()

    def __repr__(self) -> str:
        return f'<{self.name}: {self.__class__.__name__}>'


def cached_property_clearer(cproperty: cached_property) -> Callable:
    '''Decorator that allows to enhance a method with the ability, when caleld,
    to clear the cached of a target property. This is especially useful to 
    reset the cache of a given cached property when a method makes changes to 
    the underlying data, thus compromising the cached results.

    Returns
    -------
    decorated_func : Callable
        Returns the function wrapped with this decorator.

    Raises
    ------
    ValueError
        Raises if the given property is not an instance of 
        `functools.cached_property`.
    '''

    if not isinstance(cproperty, cached_property):
        raise ValueError(
            'The specified property must be an instance of '
            '`functools.cached_property`.'
        )

    # use a double decorator as it is a trick to allow passing arguments to it
    def actual_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            self = args[0]
            n = cproperty.attrname
            if n is not None and n in self.__dict__:
                del self.__dict__[n]
            return func(*args, **kwargs)
        return wrapper
    return actual_decorator
