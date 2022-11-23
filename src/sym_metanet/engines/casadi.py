from typing import Dict, Literal, Type, Tuple, Generic, TypeVar, Union
import casadi as cs
from sym_metanet.engines.core import (
    NodesEngineBase,
    LinksEngineBase,
    OriginsEngineBase,
    DestinationsEngineBase,
    EngineBase
)


csTYPES: Dict[str, Type] = {
    'SX': cs.SX,
    'MX': cs.MX,
}
csXX = TypeVar('csXX', cs.SX, cs.MX)


class NodesEngine(NodesEngineBase, Generic[csXX]):
    '''CasADi implementation of `sym_metanet.engines.core.NodesEngineBase`.'''

    @staticmethod
    def get_upstream_flow(q_lasts: csXX, beta: csXX) -> csXX:
        return beta * cs.sum1(q_lasts)

    @staticmethod
    def get_upstream_speed(q_lasts: csXX, v_lasts: csXX) -> csXX:
        return (v_lasts.T @ q_lasts) / cs.sum1(q_lasts)

    @staticmethod
    def get_downstream_density(rho_firsts: csXX) -> csXX:
        return (rho_firsts.T @ rho_firsts) / cs.sum1(rho_firsts)


class LinksEngine(LinksEngineBase, Generic[csXX]):
    '''CasADi implementation of `sym_metanet.engines.core.LinksEngineBase`.'''

    @staticmethod
    def get_flow(rho: csXX, v: csXX, lanes: csXX) -> csXX:
        return rho * v * lanes

    @staticmethod
    def step_density(
            rho: csXX, q: csXX, q_up: csXX,
            lanes: csXX, L: csXX, T: csXX) -> csXX:
        return rho + (T / lanes / L) * (q_up - q)

    @staticmethod
    def step_speed(v: csXX, v_up: csXX, rho: csXX, rho_down: csXX,
                   Veq: csXX, lanes: csXX, L: csXX, tau: csXX, eta: csXX,
                   kappa: csXX, T: csXX,
                   q_ramp: csXX = None, delta: csXX = None,
                   lanes_drop: csXX = None, phi: csXX = None,
                   rho_crit: csXX = None) -> csXX:
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
    def Veq(rho: csXX, v_free: csXX, rho_crit: csXX, a: csXX) -> csXX:
        return v_free * cs.exp((-1 / a) * cs.power(rho / rho_crit, a))


class OriginsEngine(OriginsEngineBase, Generic[csXX]):
    '''
    CasADi implementation of `sym_metanet.engines.core.OriginsEngineBase`.
    '''

    @staticmethod
    def step_queue(w: csXX, d: csXX, q: csXX, T: csXX) -> csXX:
        return w + T * (d - q)

    @staticmethod
    def get_ramp_flow(d: csXX, w: csXX, C: csXX, r: csXX, rho_max: csXX,
                      rho_first: csXX, rho_crit: csXX, T: csXX,
                      type: Literal['in', 'out'] = 'out') -> csXX:
        term1 = d + w / T
        term3 = (rho_max - rho_first) / (rho_max - rho_crit)
        if type == 'in':
            return cs.fmin(term1, C * cs.fmin(r, term3))
        return r * cs.fmin(term1, C * cs.fmin(1, term3))


class DestinationsEngine(DestinationsEngineBase, Generic[csXX]):
    '''
    CasADi implementation of `sym_metanet.engines.core.DestinationsEngineBase`.
    '''

    @staticmethod
    def get_congested_downstream_density(
            rho_last: csXX, rho_destination: csXX, rho_crit: csXX) -> csXX:
        return cs.fmax(cs.fmin(rho_last, rho_crit), rho_destination)


class Engine(EngineBase, Generic[csXX]):
    '''Symbolic engine implemented with the CasADi framework'''

    def __init__(self, sym_type: Literal['SX', 'MX'] = 'SX') -> None:
        '''Instantiates a CasADi engine.

        Parameters
        ----------
        sym_type : {'SX', 'MX'}, optional
            A string that tells the engine with type of symbolic variables to
            use. Must be either `'SX'` or `'MX'`, at which point the engine
            employes `casadi.SX` or `casadi.MX` variables, respectively. By
            default, `'SX'` is used.

        Raises
        ------
        ValueError
            Raises if the provided string `type` is not valid.
        '''
        super().__init__()
        if sym_type not in csTYPES:
            raise ValueError(
                f'CasADi symbolic type must be in {{{", ".join(csTYPES)}}}; '
                f'got {sym_type} instead.')
        self._csXX: Union[Type[cs.SX], Type[cs.MX]] = csTYPES[sym_type]

    @property
    def nodes(self) -> Type[NodesEngine[csXX]]:
        return NodesEngine[csXX]

    @property
    def links(self) -> Type[LinksEngine[csXX]]:
        return LinksEngine[csXX]

    @property
    def origins(self) -> Type[OriginsEngine[csXX]]:
        return OriginsEngine[csXX]

    @property
    def destinations(self) -> Type[DestinationsEngine[csXX]]:
        return DestinationsEngine[csXX]

    def var(self, name: str, shape: Tuple[int, int], *args, **kwargs) -> csXX:
        return self._csXX.sym(name, *shape)

    def __str__(self) -> str:
        return f'{self.__class__.__name__}(casadi)'

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(casadi, type={self._csXX.__name__})'
