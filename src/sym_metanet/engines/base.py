from abc import ABC, abstractmethod
from typing import Dict, Generic, Tuple, TypeVar, Union
import sym_metanet


T = TypeVar('T')


class EngineNotFoundError(Exception):
    '''
    Exception raised when no symbolic engine is found among the available ones.
    To see which are avaiable, see `sym_metanet.engines.get_available_engines`.
    '''


class EngineNotFoundWarning(Warning):
    '''
    Warning raised when no symbolic engine is found among the available ones.
    To see which are avaiable, see `sym_metanet.engines.get_available_engines`.
    '''


class EngineBase(ABC, Generic[T]):
    '''
    Abstract class of a symbolic engine for modelling highways via the METANET
    framework. The methods of this class implement the various equations 
    proposed in the framework, which can be found in [1].

    References
    ----------
    [1] Hegyi, A., 2004, "Model predictive control for integrating traffic 
        control measures", Netherlands TRAIL Research School.
    '''

    @abstractmethod
    def var(self, name: str, shape: Tuple[int, ...], *args, **kwargs) -> T:
        '''Creates a variable.

        Parameters
        ----------
        name : str
            Name of the variable.
        shape : Tuple[int, ...]
            Shape of the variable.

        Returns
        -------
        sym variable
            The symbolic variable (T is generic).
        '''
        pass

    @abstractmethod
    def Veq(self, rho: T, v_free: T, rho_crit: T, a: T) -> T:
        '''Computes the equilibrium speed of the link, according to 
        [1, Equation 3.4].

        Parameters
        ----------
        rho
            Densities of the link.
        v_free
            Free-flow speed of the link.
        rho_crit
            Critical density of the link.
        a
            Model parameter in the equilibrium speed exponent. 

        Returns
        -------
        Veq
            The equilibrium speed of the link.
        '''
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


def get_current_engine() -> EngineBase:
    '''Gets the current symbolic engine.

    Returns
    -------
    SymEngineBase
        The current symbolic engine.
    '''
    return sym_metanet.engine


def use(engine: Union[str, EngineBase], *args, **kwargs) -> EngineBase:
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
    if isinstance(engine, EngineBase):
        sym_metanet.engine = engine
    engines = get_available_engines()
    if engine not in engines:
        raise EngineNotFoundError(
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
