from abc import ABC, abstractmethod
from typing import Dict, Generic, TypeVar, Union
import sym_metanet


T = TypeVar('T')


class SymEngineNotFoundError(Exception):
    '''
    Exception raised when no symbolic engine is found among the available ones.
    To see which are avaiable, see `sym_metanet.engines.get_available_engines`.
    '''


class SymEngineNotFoundWarning(Warning):
    '''
    Warning raised when no symbolic engine is found among the available ones.
    To see which are avaiable, see `sym_metanet.engines.get_available_engines`.
    '''


class SymEngineBase(ABC, Generic[T]):
    @abstractmethod
    def Veq(self, rho: T, v_free: T, a: T, rho_crit: T) -> T:
        pass
    

def get_available_engines() -> Dict[str, str]:
    '''Returns the available symbolic engines for METANET modelling, for which
    an implementation exists.

    Returns
    -------
    Dict[str, str]
        The available engines in the form of a `dict` whose keys are the 
        engine class names, and the values are the modules in which they lie.
    '''
    return {
        'CasadiEngine': 'sym_metanet.engines.casadi'
    }


def get_current_engine() -> SymEngineBase:
    '''Gets the current symbolic engine.

    Returns
    -------
    SymEngineBase
        The current symbolic engine.
    '''
    return sym_metanet.engine


def use(engine: Union[str, SymEngineBase], *args, **kwargs) -> SymEngineBase:
    '''Uses the given symbolic engine for computations.

    Parameters
    ----------
    engine : str or instance of engine
        If a string, then `engine` must represent the class name of one of the 
        available engines, so that it can be instantiated with args and kwargs.
        (see `sym_metanet.engines.get_available_engines`). Otherwise, it must 
        be an instance of an engine itself that inherits from 
        `sym_metanet.engines.SymEngineBase`.
    args, kwargs
        Passed to the engine constructor in case `engine` is a string.

    Returns
    -------
    SymEngineBase
        A reference to the new engine, if `engine` is a string, or the 
        reference to the same instance, otherwise. 

    Raises
    ------
    ValueError
        Raises in case `engine` is neither a string nor a `SymEngineBase` 
        instance.
    SymEngineNotFoundError
        Raises in case `engine` is a string but matches no available engines.
    '''
    if isinstance(engine, SymEngineBase):
        sym_metanet.engine = engine
    engines = get_available_engines()
    if engine not in engines:
        raise SymEngineNotFoundError(
            f'Engine class must be in {{{", ".join(engines)}}}; got '
            f'{engine} instead.')
    elif isinstance(engine, str):
        from importlib import import_module
        cls = getattr(import_module(engines[engine]), engine)
        sym_metanet.engine = cls(*args, **kwargs)
    else:
        raise ValueError('Expected `engine` to be either a string or an '
                         f'instance of an engine; got {engine} instead.')
    return sym_metanet.engine
