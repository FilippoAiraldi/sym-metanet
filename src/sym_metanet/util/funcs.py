from functools import _lru_cache_wrapper, cached_property, wraps
from typing import Callable, Iterable, List, TypeVar, Union

T = TypeVar('T')


def first(o: Iterable[T]) -> T:
    '''Returns the first item of an iterable. Use `next` for iterators.

    Parameters
    ----------
    o : iterable of T
        Object that can be iterated over.

    Returns
    -------
    T
        The first element in the iterable.
    '''
    return next(iter(o))


def cache_clearer(
    *callables: Union[cached_property, _lru_cache_wrapper]
) -> Callable:
    '''Decorator that allows to enhance a method with the ability, when
    called, to clear the cached of some target methods/properties. This is
    especially useful to reset the cache of a given cached method/property when
    another method makes changes to the underlying data, thus compromising the
    cached results.

    Thanks to https://github.com/FilippoAiraldi/casadi-nlp/blob/dev/src/csnlp/util/funcs.py

    Parameters
    ----------
    callables : cached_property or lru cache wrapper
        The cached properties or methods to be reset in this decorator.

    Returns
    -------
    decorated_func : Callable[[Any], Any]
        Returns the function wrapped with this decorator.

    Raises
    ------
    TypeError
        Raises if the given inputs are not instances of
        `functools.cached_property` or `functools._lru_cache_wrapper`.
    '''
    cps: List[cached_property] = []
    lrus: List[_lru_cache_wrapper] = []
    for p in callables:
        if isinstance(p, cached_property):
            cps.append(p)
        elif isinstance(p, _lru_cache_wrapper):
            lrus.append(p)
        else:
            raise TypeError('Expected cached properties or lru wrappers; got '
                            f'{p.__class__.__name__} instead.')

    def actual_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            self = args[0]
            for prop in cps:
                n = prop.attrname
                if n is not None and n in self.__dict__:
                    del self.__dict__[n]
            for lru in lrus:
                lru.cache_clear()
            return func(*args, **kwargs)
        return wrapper

    return actual_decorator
