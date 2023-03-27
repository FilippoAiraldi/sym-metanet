from itertools import chain, product
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import casadi as cs

from sym_metanet.blocks.base import ElementWithVars
from sym_metanet.engines.core import (
    DestinationsEngineBase,
    EngineBase,
    LinksEngineBase,
    NodesEngineBase,
    OriginsEngineBase,
)

if TYPE_CHECKING:
    from sym_metanet.blocks.links import Link
    from sym_metanet.network import Network


VarType = TypeVar("VarType", cs.SX, cs.MX)


class NodesEngine(NodesEngineBase, Generic[VarType]):
    """CasADi implementation of `sym_metanet.engines.core.NodesEngineBase`."""

    @staticmethod
    def get_upstream_flow(
        q_lasts: VarType,
        beta: VarType,
        betas: VarType,
        q_orig: Optional[VarType] = None,
    ) -> VarType:
        Q = cs.sum1(q_lasts)
        if q_orig is not None:
            Q += q_orig
        return (beta / cs.sum1(betas)) * Q

    @staticmethod
    def get_upstream_speed(q_lasts: VarType, v_lasts: VarType) -> VarType:
        return cs.sum1(v_lasts * q_lasts) / cs.sum1(q_lasts)

    @staticmethod
    def get_downstream_density(rho_firsts: VarType) -> VarType:
        return cs.sum1(rho_firsts**2) / cs.sum1(rho_firsts)


class LinksEngine(LinksEngineBase, Generic[VarType]):
    """CasADi implementation of `sym_metanet.engines.core.LinksEngineBase`."""

    @staticmethod
    def get_flow(rho: VarType, v: VarType, lanes: VarType) -> VarType:
        return rho * v * lanes

    @staticmethod
    def step_density(
        rho: VarType, q: VarType, q_up: VarType, lanes: VarType, L: VarType, T: VarType
    ) -> VarType:
        return rho + (T / lanes / L) * (q_up - q)

    @staticmethod
    def step_speed(
        v: VarType,
        v_up: VarType,
        rho: VarType,
        rho_down: VarType,
        Veq: VarType,
        lanes: VarType,
        L: VarType,
        tau: VarType,
        eta: VarType,
        kappa: VarType,
        T: VarType,
        q_ramp: Optional[VarType] = None,
        delta: Optional[VarType] = None,
        lanes_drop: Optional[VarType] = None,
        phi: Optional[VarType] = None,
        rho_crit: Optional[VarType] = None,
    ) -> VarType:
        relaxation = (T / tau) * (Veq - v)
        convection = T * v / L * (v_up - v)
        anticipation = (eta * T / tau) * (rho_down - rho) / (L * (rho + kappa))
        v_next = v + relaxation + convection - anticipation
        if q_ramp is not None and delta is not None:
            v_next[0] -= (delta * T * q_ramp * v[0]) / (L * lanes * (rho[0] + kappa))
        if lanes_drop is not None and phi is not None and rho_crit is not None:
            v_next[-1] -= (phi * T * lanes_drop * rho[-1] * v[-1] ** 2) / (
                L * lanes * rho_crit
            )
        return v_next

    @staticmethod
    def Veq(rho: VarType, v_free: VarType, rho_crit: VarType, a: VarType) -> VarType:
        return v_free * cs.exp((-1 / a) * cs.power(rho / rho_crit, a))


class OriginsEngine(OriginsEngineBase, Generic[VarType]):
    """CasADi implementation of `sym_metanet.engines.core.OriginsEngineBase`."""

    @staticmethod
    def step_queue(w: VarType, d: VarType, q: VarType, T: VarType) -> VarType:
        return w + T * (d - q)

    @staticmethod
    def get_ramp_flow(
        d: VarType,
        w: VarType,
        C: VarType,
        r: VarType,
        rho_max: VarType,
        rho_first: VarType,
        rho_crit: VarType,
        T: VarType,
        type: Literal["in", "out"] = "out",
    ) -> VarType:
        term1 = d + w / T
        term3 = (rho_max - rho_first) / (rho_max - rho_crit)
        if type == "in":
            return cs.fmin(term1, C * cs.fmin(r, term3))
        return r * cs.fmin(term1, C * cs.fmin(1, term3))

    @staticmethod
    def get_simplifiedramp_flow(
        qdes: VarType,
        d: VarType = None,
        w: VarType = None,
        C: VarType = None,
        rho_max: VarType = None,
        rho_first: VarType = None,
        rho_crit: VarType = None,
        T: VarType = None,
        type: Literal["limited", "unlimited"] = "limited",
    ) -> VarType:
        if type == "unlimited":
            return qdes
        term2 = d + w / T
        term3 = C * cs.fmin(1, (rho_max - rho_first) / (rho_max - rho_crit))
        return cs.fmin(qdes, cs.fmin(term2, term3))


class DestinationsEngine(DestinationsEngineBase, Generic[VarType]):
    """CasADi implementation of `sym_metanet.engines.core.DestinationsEngineBase`."""

    @staticmethod
    def get_congested_downstream_density(
        rho_last: VarType, rho_destination: VarType, rho_crit: VarType
    ) -> VarType:
        return cs.fmax(cs.fmin(rho_last, rho_crit), rho_destination)


class Engine(EngineBase, Generic[VarType]):
    """Symbolic engine implemented with the CasADi framework"""

    def __init__(self, sym_type: Literal["SX", "MX"] = "SX") -> None:
        """Instantiates a CasADi engine.

        Parameters
        ----------
        sym_type : {'SX', 'MX'}, optional
            A string that tells the engine with type of symbolic variables to use. Must
            be either `'SX'` or `'MX'`, at which point the engine employes `casadi.SX`
            or `casadi.MX` variables, respectively. By default, `'SX'` is used.

        Raises
        ------
        AttributeError
            Raises if `sym_type` is not valid.
        """
        super().__init__()
        self.sym_type: Union[Type[cs.SX], Type[cs.MX]] = getattr(cs, sym_type)

    @property
    def nodes(self) -> Type[NodesEngine[VarType]]:
        return NodesEngine[VarType]

    @property
    def links(self) -> Type[LinksEngine[VarType]]:
        return LinksEngine[VarType]

    @property
    def origins(self) -> Type[OriginsEngine[VarType]]:
        return OriginsEngine[VarType]

    @property
    def destinations(self) -> Type[DestinationsEngine[VarType]]:
        return DestinationsEngine[VarType]

    def var(self, name: str, n: int = 1, *args, **kwargs) -> VarType:
        return self.sym_type.sym(name, n, 1)

    def vcat(self, *arrays):
        return cs.vertcat(*arrays)

    def to_function(  # type: ignore[override]
        self,
        net: "Network",
        compact: int = 0,
        more_out: bool = False,
        force_positive_speed: bool = True,
        force_positive_density: bool = False,
        force_positive_queue: bool = False,
        parameters: Optional[Dict[str, VarType]] = None,
        **other_parameters: Any,
    ) -> cs.Function:
        """Converts the network's dynamics to a CasADi Function.

        Parameters
        ----------
        net : Network
            The network whose dynamics must be translated into a function.
        compact : int, optional
            The compactness of input and output arguments. The levels are

            - <= 0: no aggregation of arguments, i.e., the function keeps states, action
            or disturbances for each element separate.

            - == 1: some aggregation, i.e., same variable types are clumped together.

            -  > 1: most aggregation, i.e., states, action and disturbances are
            aggregated in a single vector each.

        more_out : bool, optional
            Includes flows of links and origins in the output. By default `False`.
        force_positive_speed : bool, optional
            If `True`, the links speeds at the next time step are forced to be positive
            as `v+ = max(0, v+)`. METANET is in fact known to sometime yield negative
            speeds, which are infeasible.
        force_positive_density : bool, optional
            Same as `force_positive_speed`, but for densities. By default, `False`.
        force_positive_queue : bool, optional
            Same as `force_positive_speed`, but for queues. By default, `False`.
        parameters : dict[str, casadi.SX or MX], optional
            Symbolic network parameters to be included in the function, by default None.
        **other_parameters
            Other parameters (numerical or symbolical) required during the computations,
            e.g., sampling time T is usually required.

        Returns
        -------
        cs.Function
            The CasADi Function representing the network's dynamics.

        Raises
        ------
        RuntimeError
            Raises if variables have not yet been initialized; or if the dynamics have
            not been stepped yet, so no state at the next time instant is found.
        """
        for el, group in product(
            net.elements, ["_states", "_actions", "_disturbances"]
        ):
            if any(getattr(el, group)) and not getattr(el, f"has{group}"):
                raise RuntimeError(
                    f"Found no {group[1:-1]} in {el.name}; perhaps variables "
                    "have not been initialized via `net.init_vars`?"
                )
            if any(el._states) and not el.has_next_states:
                raise RuntimeError(
                    f"Found no next state in {el.name}; perhaps dynamics have "
                    "not been stepped via `net.step`?"
                )

        if parameters is None:
            parameters = {}

        # process inputs
        x = {el: _filter_vars(vars) for el, vars in net.states.items()}
        u = {el: _filter_vars(vars) for el, vars in net.actions.items()}
        d = {el: _filter_vars(vars) for el, vars in net.disturbances.items()}

        # process outputs
        x_next = {
            el: _filter_vars(vars, independent=False)
            for el, vars in net.next_states.items()
        }
        if force_positive_speed or force_positive_density or force_positive_queue:
            for vars in x_next.values():
                if force_positive_speed and "v" in vars:
                    vars["v"] = cs.fmax(0, vars["v"])
                if force_positive_density and "rho" in vars:
                    vars["rho"] = cs.fmax(0, vars["rho"])
                if force_positive_queue and "w" in vars:
                    vars["w"] = cs.fmax(0, vars["w"])

        # gather inputs/outputs
        names_in, args_in = _gather_inputs(x, u, d, compact)
        names_out, args_out = _gather_outputs(x_next, compact)
        if parameters:
            _add_parameters_to_inputs(names_in, args_in, parameters, compact)
        if more_out:
            _add_flows_to_outputs(
                names_out, args_out, self, net, parameters, other_parameters, compact
            )

        # finally create function
        return cs.Function("F", args_in, args_out, names_in, names_out)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(casadi)"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(casadi, type={self.sym_type.__name__})"


def _filter_vars(
    vars: Dict[str, Union[VarType, Any]],
    symbolic: bool = True,
    independent: bool = True,
) -> Dict[str, VarType]:
    """Internal utility to filter out symbols that are either only symbolic
    and/or independent (and thus can be inputs to `casadi.Function`)."""

    def is_ok(var: Union[VarType, Any]) -> bool:
        # sourcery skip: assign-if-exp, reintroduce-else
        if not symbolic:
            return True
        if isinstance(var, cs.SX):
            if independent:
                return all(var[i].n_dep() == 0 for i in range(var.shape[0]))
            return True
        if isinstance(var, cs.MX):
            if independent:
                return var.n_dep() == 0
            return True
        return False

    return {name: var for name, var in vars.items() if is_ok(var)}


def _gather_inputs(
    x: Dict[ElementWithVars, Dict[str, VarType]],
    u: Dict[ElementWithVars, Dict[str, VarType]],
    d: Dict[ElementWithVars, Dict[str, VarType]],
    compact: int,
) -> Tuple[List[str], List[VarType]]:
    """Internal utility to gather inputs for `casadi.Function`."""

    if compact <= 0:
        # no aggregation
        names_in, args_in = [], []
        for vars_in in (x, u, d):
            for el, vars in vars_in.items():  # type: ignore[attr-defined]
                for varname, var in vars.items():
                    names_in.append(f"{varname}_{el.name}")
                    args_in.append(var)
        return names_in, args_in

    # group variables as (name, list of vars)
    states: Dict[str, List[VarType]] = {}
    actions: Dict[str, List[VarType]] = {}
    disturbances: Dict[str, List[VarType]] = {}
    for vars_in, group in [(x, states), (u, actions), (d, disturbances)]:
        for el, vars in vars_in.items():  # type: ignore[attr-defined]
            for varname, var in vars.items():
                if varname in group:
                    group[varname].append(var)
                else:
                    group[varname] = [var]

    # group variables as (name, symbol)
    for group in (states, actions, disturbances):
        for varname, list_of_vars in group.items():
            group[varname] = cs.vcat(list_of_vars)

    # add to names and args
    if compact == 1:
        names_in = list(chain(states.keys(), actions.keys(), disturbances.keys()))
        args_in = list(chain(states.values(), actions.values(), disturbances.values()))
    else:
        names_in = ["x", "u", "d"]
        args_in = [
            cs.vcat(states.values()),
            cs.vcat(actions.values()),
            cs.vcat(disturbances.values()),
        ]
    return names_in, args_in


def _gather_outputs(
    x_next: Dict[ElementWithVars, Dict[str, VarType]],
    compact: int,
) -> Tuple[List[str], List[VarType]]:
    """Internal utility to gather outputs for `casadi.Function`."""

    if compact <= 0:
        # no aggregation
        names_out, args_out = [], []
        for el, vars in x_next.items():
            for varname, var in vars.items():
                names_out.append(f"{varname}_{el.name}+")
                args_out.append(var)
        return names_out, args_out

    # group variables as (name, list of vars)
    next_states: Dict[str, List[VarType]] = {}
    for vars in x_next.values():
        for varname, var in vars.items():
            varname += "+"
            if varname in next_states:
                next_states[varname].append(var)
            else:
                next_states[varname] = [var]

    # group variables as (name, symbol)
    for varname, list_of_vars in next_states.items():
        next_states[varname] = cs.vcat(list_of_vars)

    # add to names and args
    if compact == 1:
        names_out = list(next_states.keys())
        args_out = list(next_states.values())
    else:
        names_out = ["x+"]
        args_out = [cs.vcat(next_states.values())]
    return names_out, args_out


def _add_parameters_to_inputs(
    names_in: List[str],
    args_in: List[VarType],
    parameters: Dict[str, VarType],
    compact: int,
) -> None:
    """Internal utility to add parameters to inputs for `casadi.Function`."""
    if compact <= 0:
        names_in.extend(parameters.keys())
        args_in.extend(parameters.values())
    else:
        names_in.append("p")
        args_in.append(cs.vcat(parameters.values()))


def _add_flows_to_outputs(
    names_out: List[str],
    args_out: List[VarType],
    engine: Engine,
    net: "Network",
    parameters: Dict[str, VarType],
    other_parameters: Dict[str, Any],
    compact: int,
) -> None:
    """Internal utility to add even more outputs for `casadi.Function`."""

    # add link and origin flows (q, q_o) to output
    names_link: List[str] = []
    flows_link: List[VarType] = []
    names_origins, flows_origins = [], []
    link: "Link[VarType]"
    for _, _, link in net.links:
        names_link.append(f"q_{link.name}")
        flows_link.append(link.get_flow(engine))
    for origin in net.origins:
        names_origins.append(f"q_o_{origin.name}")
        flows_origins.append(
            origin.get_flow(net, engine=engine, **parameters, **other_parameters)
        )

    if compact > 0:
        names_link = ["q"]
        flows_link = [cs.vcat(flows_link)]
        names_origins = ["q_o"]
        flows_origins = [cs.vcat(flows_origins)]
    if compact > 1:
        names_link = ["q"]
        flows_link = [cs.vertcat(flows_link[0], flows_origins[0])]
        names_origins, flows_origins = [], []

    names_out.extend(names_link + names_origins)
    args_out.extend(flows_link + flows_origins)
