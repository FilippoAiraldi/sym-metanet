from itertools import product
from typing import Any, Dict, Literal, Type, Generic, TypeVar, Union, \
    TYPE_CHECKING
import casadi as cs
from sym_metanet.engines.core import (
    NodesEngineBase,
    LinksEngineBase,
    OriginsEngineBase,
    DestinationsEngineBase,
    EngineBase,
)
if TYPE_CHECKING:
    from sym_metanet.network import Network


csTYPES: Dict[str, Type] = {
    'SX': cs.SX,
    'MX': cs.MX,
}
csXX = TypeVar('csXX', cs.SX, cs.MX)


class NodesEngine(NodesEngineBase, Generic[csXX]):
    '''CasADi implementation of `sym_metanet.engines.core.NodesEngineBase`.'''

    @staticmethod
    def get_upstream_flow(
        q_lasts: csXX, beta: csXX, betas: csXX, q_orig: csXX = None
    ) -> csXX:
        Q = cs.sum1(q_lasts)
        if q_orig is not None:
            Q += q_orig
        return (beta / cs.sum1(betas)) * Q

    @staticmethod
    def get_upstream_speed(q_lasts: csXX, v_lasts: csXX) -> csXX:
        return cs.sum1(v_lasts * q_lasts) / cs.sum1(q_lasts)

    @staticmethod
    def get_downstream_density(rho_firsts: csXX) -> csXX:
        return cs.sum1(rho_firsts**2) / cs.sum1(rho_firsts)


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

    def var(self, name: str, n: int = 1, *args, **kwargs) -> csXX:
        return self._csXX.sym(name, n, 1)

    def vcat(self, *arrays):
        return cs.vertcat(*arrays)

    def to_function(
        self,
        net: 'Network',
        compact: int = 0,
        more_out: bool = False,
        force_positive_speed: bool = True,
        parameters: Dict[str, csXX] = None,
        **other_parameters
    ) -> cs.Function:
        '''Converts the network's dynamics to a CasADi Function.

        Parameters
        ----------
        net : Network
            The network whose dynamics must be translated into a function.
        compact : int, optional
            The compactness of input and output arguments. The levels are

            - <= 0: no aggregation of arguments, i.e., the function keeps
            states, action or disturbances for each element separate.

            - == 1: some aggregation, i.e., same types of variables are clumped
            together.

            -  > 1: most aggregation, i.e., states, action and disturbances are
            aggregated in a single vector each.

        more_out : bool, optional
            Includes flows of links and origins in the output. By default
            `False`.
        force_positive_speed : bool, optional
            If `True`, the links speeds at the next time step are forced to be
            positive as `v+ = max(0, v+)`. METANET is in fact known to sometime
            yield negative speeds, which are infeasible.
        parameters : dict[str, casadi.SX or MX], optional
            Symbolic network parameters to be included in the function, by
            default None.
        **other_parameters
            Other parameters (numerical or symbolical) required during the
            computations, e.g., sampling time T is usually required.

        Returns
        -------
        cs.Function
            The CasADi Function representing the network's dynamics.

        Raises
        ------
        RuntimeError
            Raises if variables have not yet been initialized; or if the
            dynamics have not been stepped yet, so no state at the next time
            instant is found.
        '''
        for el, group in product(net.elements,
                                 ['_states', '_actions', '_disturbances']):
            if any(getattr(el, group)) and not getattr(el, f'has{group}'):
                raise RuntimeError(
                    f'Found no {group[1:-1]} in {el.name}; perhaps variables '
                    'have not been initialized via `net.init_vars`?')
            if any(el._states) and not el.has_next_states:
                raise RuntimeError(
                    f'Found no next state in {el.name}; perhaps dynamics have '
                    'not been stepped via `net.step`?')

        if parameters is None:
            parameters = {}

        # process inputs
        x = {el: _filter_vars(vars) for el, vars in net.states.items()}
        u = {el: _filter_vars(vars) for el, vars in net.actions.items()}
        d = {el: _filter_vars(vars) for el, vars in net.disturbances.items()}

        # gather inputs
        names_in, args_in = [], []
        if compact <= 0:
            for vars_in in (x, u, d):
                for el, vars in vars_in.items():
                    for varname, var in vars.items():
                        names_in.append(f'{varname}_{el.name}')
                        args_in.append(var)
        else:
            states, actions, disturbances = {}, {}, {}
            for vars_in, group in [
                    (x, states), (u, actions), (d, disturbances)]:
                for el, vars in vars_in.items():
                    for varname, var in vars.items():
                        if varname in group:
                            group[varname].append(var)
                        else:
                            group[varname] = [var]

            for group in (states, actions, disturbances):
                for varname, vars in group.items():
                    group[varname] = cs.vertcat(*vars)

            if compact == 1:
                names_in = list(states.keys()) + list(actions.keys()) + \
                    list(disturbances.keys())
                args_in = list(states.values()) + list(actions.values()) + \
                    list(disturbances.values())
            else:
                names_in = ['x', 'u', 'd']
                args_in = [
                    cs.vertcat(*states.values()),
                    cs.vertcat(*actions.values()),
                    cs.vertcat(*disturbances.values())
                ]

        # process outputs
        x_next = {el: _filter_vars(vars, independent=False)
                  for el, vars in net.next_states.items()}
        if force_positive_speed:
            for vars in x_next.values():
                if 'v' in vars:
                    vars['v'] = cs.fmax(0, vars['v'])

        # gather outputs
        names_out, args_out = [], []
        if compact <= 0:
            for el, vars in x_next.items():
                for varname, var in vars.items():
                    names_out.append(f'{varname}_{el.name}+')
                    args_out.append(var)
        else:
            next_states = {}
            for el, vars in x_next.items():
                for varname, var in vars.items():
                    varname += '+'
                    if varname in next_states:
                        next_states[varname].append(var)
                    else:
                        next_states[varname] = [var]

            for varname, vars in next_states.items():
                next_states[varname] = cs.vertcat(*vars)

            if compact == 1:
                names_out = list(next_states.keys())
                args_out = list(next_states.values())
            else:
                names_out = ['x+']
                args_out = [cs.vertcat(*next_states.values())]

        # add link and origin flows (q_l, q_o) to output
        if more_out:
            names_link, flows_link = [], []
            names_origins, flows_origins = [], []
            for _, _, link in net.links:
                names_link.append(f'q_{link.name}')
                flows_link.append(link.get_flow(engine=self))
            for origin in net.origins:
                names_origins.append(f'q_o_{origin.name}')
                flows_origins.append(
                    origin.get_flow(net=net, engine=self,
                                    **parameters, **other_parameters))

            if compact > 0:
                names_link = ['q']
                flows_link = [cs.vertcat(*flows_link)]
                names_origins = ['q_o']
                flows_origins = [cs.vertcat(*flows_origins)]
            if compact > 1:
                names_link = ['q']
                flows_link = [cs.vertcat(flows_link[0], flows_origins[0])]
                names_origins, flows_origins = [], []
            names_out += names_link + names_origins
            args_out += flows_link + flows_origins

        # add parameters
        if len(parameters) > 0:
            if compact <= 0:
                names_in.extend(parameters.keys())
                args_in.extend(parameters.values())
            else:
                names_in.append('p')
                args_in.append(cs.vertcat(*parameters.values()))

        return cs.Function('F', args_in, args_out, names_in, names_out)

    def __str__(self) -> str:
        return f'{self.__class__.__name__}(casadi)'

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(casadi, type={self._csXX.__name__})'


def _filter_vars(
    vars: Dict[str, Union[csXX, Any]],
    symbolic: bool = True,
    independent: bool = True
) -> Dict[str, csXX]:
    '''Internal utility to filter out symbols that are either only symbolic
    and/or independent (and thus can be inputs to  `casadi.Function`).'''

    def is_ok(var: Union[csXX, Any]) -> bool:
        if not symbolic:
            return True
        if not isinstance(var, (cs.SX, cs.MX)):
            return False
        if independent:
            return all(var[i].n_dep() == 0 for i in range(var.shape[0]))
        return True

    return {name: var for name, var in vars.items() if is_ok(var)}
