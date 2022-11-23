from typing import Callable, Literal, Type, Tuple, Union
import numpy as np
from sym_metanet.engines.core import (
    NodesEngineBase,
    LinksEngineBase,
    OriginsEngineBase,
    DestinationsEngineBase,
    EngineBase
)


class NodesEngine(NodesEngineBase):
    '''NumPy implementation of `sym_metanet.engines.core.NodesEngineBase`.'''

    @staticmethod
    def get_upstream_flow(q_lasts: np.ndarray, beta: np.ndarray) -> np.ndarray:
        return beta * np.sum(q_lasts, axis=0)

    @staticmethod
    def get_upstream_speed(q_lasts: np.ndarray,
                           v_lasts: np.ndarray) -> np.ndarray:
        return (v_lasts.T @ q_lasts) / np.sum(q_lasts, axis=0)

    @staticmethod
    def get_downstream_density(rho_firsts: np.ndarray) -> np.ndarray:
        return (rho_firsts.T @ rho_firsts) / np.sum(rho_firsts, axis=0)


class LinksEngine(LinksEngineBase):
    '''NumPy implementation of `sym_metanet.engines.core.LinksEngineBase`.'''

    @staticmethod
    def get_flow(rho: np.ndarray, v: np.ndarray,
                 lanes: np.ndarray) -> np.ndarray:
        return rho * v * lanes

    @staticmethod
    def step_density(
            rho: np.ndarray, q: np.ndarray, q_up: np.ndarray,
            lanes: np.ndarray, L: np.ndarray, T: np.ndarray) -> np.ndarray:
        return rho + (T / lanes / L) * (q_up - q)

    @staticmethod
    def step_speed(
        v: np.ndarray, v_up: np.ndarray, rho: np.ndarray,
        rho_down: np.ndarray, Veq: np.ndarray, lanes: int,
        L: int, tau: float, eta: float, kappa: float, T: float,
        q_ramp: np.ndarray = None, delta: float = None,
        lanes_drop: int = None, phi: float = None, rho_crit: float = None
    ) -> np.ndarray:
        relaxation = (T / tau) * (Veq - v)
        convection = T * v / L * (v_up - v)
        anticipation = (eta * T / tau) * (rho_down - rho) / (L * (rho + kappa))
        v_next = v + relaxation + convection - anticipation
        if q_ramp is not None and delta is not None:
            v_next[0] -= \
                (delta * T * q_ramp * v[0]) / (L * lanes * (rho[0] + kappa))
        if lanes_drop is not None and phi is not None and rho_crit is not None:
            v_next[-1] -= (phi * T * lanes_drop * rho[-1] * v[-1]**2) / \
                (L * lanes * rho_crit)
        return v_next

    @staticmethod
    def Veq(rho: np.ndarray, v_free: np.ndarray,
            rho_crit: np.ndarray, a: np.ndarray) -> np.ndarray:
        return v_free * np.exp((-1 / a) * np.power(rho / rho_crit, a))


class OriginsEngine(OriginsEngineBase):
    '''
    NumPy implementation of `sym_metanet.engines.core.OriginsEngineBase`.
    '''

    @staticmethod
    def step_queue(w: np.ndarray, d: np.ndarray, q: np.ndarray,
                   T: np.ndarray) -> np.ndarray:
        return w + T * (d - q)

    @staticmethod
    def get_ramp_flow(
            d: np.ndarray, w: np.ndarray, C: np.ndarray, r: np.ndarray,
            rho_max: np.ndarray, rho_first: np.ndarray, rho_crit: np.ndarray,
            T: np.ndarray, type: Literal['in', 'out'] = 'out') -> np.ndarray:
        term1 = d + w / T
        term3 = (rho_max - rho_first) / (rho_max - rho_crit)
        if type == 'in':
            return np.minimum(term1, C * np.minimum(r, term3))
        return r * np.minimum(term1, C * np.minimum(1, term3))


class DestinationsEngine(DestinationsEngineBase):
    '''
    NumPy implementation of `sym_metanet.engines.core.DestinationsEngineBase`.
    '''

    @staticmethod
    def get_congested_downstream_density(
            rho_last: np.ndarray, rho_destination: np.ndarray,
            rho_crit: np.ndarray) -> np.ndarray:
        return np.maximum(np.minimum(rho_last, rho_crit), rho_destination)


class Engine(EngineBase):
    '''Symbolic engine implemented with NumPy.'''

    def __init__(self, var_type: Union[str, np.ndarray] = 'empty') -> None:
        '''Instantiates a NumPy engine.

        Parameters
        ----------
        var_type : str or array_like, optional
            How numerical values inside each variable should be initialized
                - 'empty': variable vectors are initialized as empty
                - 'rand': variable vectors are initialized randomly (uniform)
                - 'randn': variable vectors are initialized randomly (natural)
            or initialized with the given array value. By default, 'empty' is
            selected.

        Raises
        ------
        ValueError
            Raises if the specified type of variable initialization is invalid.
        '''
        super().__init__()
        self.var_type = var_type

    @property
    def var_type(self) -> Union[str, np.ndarray]:
        '''Gets the variable initialization strategy.'''
        return self._var_type

    @var_type.setter
    def var_type(self, val: Union[str, np.ndarray]) -> None:
        '''Sets the variable initialization strategy.

        Parameters
        ----------
        var_type : str or array_like, optional
            How numerical values inside each variable should be initialized
                - 'empty': variable vectors are initialized as empty
                - 'rand': variable vectors are initialized randomly (uniform)
                - 'randn': variable vectors are initialized randomly (natural)
            or initialized with the given array value.

        Raises
        ------
        ValueError
            Raises if the specified type of variable initialization is invalid.
        '''
        if isinstance(val, str):
            if val == 'empty':
                def gen(shape): return np.empty(shape, val)
            elif val == 'rand':
                def gen(shape): return np.random.rand(*shape)
            elif val == 'randn':
                def gen(shape): return np.random.randn(*shape)
            else:
                raise ValueError('Invalid variable initialization method.')
        else:
            def gen(shape): return np.full(shape, val)
        self._var_gen: Callable[[Tuple[int, ...]], np.ndarray] = gen
        self._var_type = val

    @property
    def nodes(self) -> Type[NodesEngine]:
        return NodesEngine

    @property
    def links(self) -> Type[LinksEngine]:
        return LinksEngine

    @property
    def origins(self) -> Type[OriginsEngine]:
        return OriginsEngine

    @property
    def destinations(self) -> Type[DestinationsEngine]:
        return DestinationsEngine

    def var(
        self, name: str, shape: Tuple[int, ...], *args, **kwargs
    ) -> np.ndarray:
        return self._var_gen(shape)

    def __str__(self) -> str:
        return f'{self.__class__.__name__}(numpy)'

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(numpy)'
