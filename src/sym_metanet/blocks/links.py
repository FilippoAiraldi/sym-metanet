from typing import TYPE_CHECKING, Collection, Dict, Optional, Tuple, Union

from sym_metanet.blocks.base import ElementWithVars
from sym_metanet.blocks.origins import MeteredOnRamp
from sym_metanet.engines.core import EngineBase, get_current_engine
from sym_metanet.util.funcs import first
from sym_metanet.util.types import VarType

if TYPE_CHECKING:
    from sym_metanet.blocks.nodes import Node
    from sym_metanet.network import Network


class Link(ElementWithVars[VarType]):
    """Highway link between two nodes [1, Section 3.2.1]. Links represent stretch of
    highway with similar traffic characteristics and no road changes (e.g., same number
    of lanes and maximum speed).

    References
    ----------
    [1] Hegyi, A., 2004, "Model predictive control for integrating traffic control
        measures", Netherlands TRAIL Research School.
    """

    __slots__ = ("N", "lam", "L", "rho_max", "rho_crit", "v_free", "a", "turnrate")
    _states = {"rho", "v"}

    def __init__(
        self,
        nb_segments: int,
        lanes: Union[VarType, int],
        length: Union[VarType, float],
        maximum_density: Union[VarType, float],
        critical_density: Union[VarType, float],
        free_flow_velocity: Union[VarType, float],
        a: Union[VarType, float],
        turnrate: Union[VarType, float] = 1.0,
        name: Optional[str] = None,
    ) -> None:
        """Creates an instance of a METANET link.

        Parameters
        ----------
        nb_segments : int
            Number of segments in this highway link, i.e., `N`.
        lanes : int or variable
            Number of lanes in each segment, i.e., `lam`.
        lengths : float or variable
            Length of each segment in the link, i.e., `L`.
        maximum density : float or variable
            Maximum density that the link can withstand, i.e., `rho_max`.
        critical_densities : float or variable
            Critical density at which the traffic flow is maximal, i.e., `rho_crit`.
        free_flow_velocities : float or variable
            Average speed of cars when traffic is freely flowing, i.e., `v_free`.
        a : float or variable
            Model parameter in the computations of the equivalent speed [1, Equation
            3.4].
        turnrate : float or variable, optional
            Fraction of the total flow that enters this link via the upstream node. Only
            relevant if multiple exiting links are attached to the same node, in order
            to split the flow according to these rates. Needs not be normalized. By
            default, all links have equal rates.
        name : str, optional
            Name of this link, by default `None`.

        References
        ----------
        [1] Hegyi, A., 2004, "Model predictive control for integrating traffic control
            measures", Netherlands TRAIL Research School.
        """
        super().__init__(name)
        self.N = nb_segments
        self.lam = lanes
        self.L = length
        self.rho_max = maximum_density
        self.rho_crit = critical_density
        self.v_free = free_flow_velocity
        self.a = a
        self.turnrate = turnrate

    def init_vars(
        self,
        init_conditions: Optional[Dict[str, VarType]] = None,
        engine: Optional[EngineBase] = None,
        **_,
    ) -> None:
        """For each segment in the link, initializes
         - `rho`: densities (state)
         - `v`: speeds (state).

        Parameters
        ----------
        init_conditions : dict[str, variable], optional
            Provides name-variable tuples to initialize states, actions and disturbances
            with specific values. These values must be compatible with the symbolic
            engine in type and shape. If not provided, variables are initialized
            automatically.
        engine : EngineBase, optional
            The engine to be used. If `None`, the current engine is used.
        """
        if init_conditions is None:
            init_conditions = {}
        if engine is None:
            engine = get_current_engine()
        self.states: Dict[str, VarType] = {
            name: (
                init_conditions[name]
                if name in init_conditions
                else engine.var(f"{name}_{self.name}", self.N)
            )
            for name in ("rho", "v")
        }

    def get_flow(self, engine: Optional[EngineBase] = None, **kwargs) -> VarType:
        """Gets the flow in this link's segments.

        Parameters
        ----------
        engine : EngineBase, optional
            The engine to be used. If `None`, the current engine is used.

        Returns
        -------
        variable
            The flow in this link.
        """
        if engine is None:
            engine = get_current_engine()
        return engine.links.get_flow(self.states["rho"], self.states["v"], self.lam)

    def step_dynamics(
        self,
        net: "Network",
        tau: Union[VarType, float],
        eta: Union[VarType, float],
        kappa: Union[VarType, float],
        T: Union[VarType, float],
        delta: Union[None, VarType, float] = None,
        phi: Union[None, VarType, float] = None,
        engine: Optional[EngineBase] = None,
        **_,
    ) -> Dict[str, VarType]:
        """Steps the dynamics of this link.

        Parameters
        ----------
        net : Network
            The network the link belongs to.
        tau : float or variable
            Model parameter for the speed relaxation term.
        eta : float or variable
            Model parameter for the speed anticipation term.
        kappa : float or variable
            Model parameter for the speed anticipation term.
        T : float or variable
            Sampling time.
        delta : float or variable, optional
            Model parameter for merging phenomenum. By default, not considered.
        phi : float or variable, optional
            Model parameter for lane drop phenomenum. By defaul, not considered.
        engine : EngineBase, optional
            The engine to be used. If `None`, the current engine is used.

        Returns
        -------
        Dict[str, variable]
            A dict with the states of the link (speeds and densities) at the next time
            step.
        """
        if engine is None:
            engine = get_current_engine()

        node_up, node_down = net.nodes_by_link[self]  # type: ignore[index]
        rho = self.states["rho"]
        v = self.states["v"]
        q = self.get_flow(engine)

        # get upstream flow and speed, and downstream density
        v0, q0 = node_up.get_upstream_speed_and_flow(net, self, engine, T=T)
        rhoN_1 = node_down.get_downstream_density(net, engine)
        if self.N > 1:
            q_up = engine.vcat(q0, q[:-1])
            v_up = engine.vcat(v0, v[:-1])
            rho_down = engine.vcat(rho[1:], rhoN_1)
        else:
            q_up = q0
            v_up = v0
            rho_down = rhoN_1

        # check for ramp merging in this link's upstream node with other
        # entering links.
        q_ramp = None
        if (
            delta is not None
            and node_up in net.origins_by_node
            and any(net.in_links(node_up))
        ):
            origin = net.origins_by_node[node_up]  # type: ignore[index]
            if isinstance(origin, MeteredOnRamp):
                q_ramp = origin.get_flow(net, T, engine)

        # check for lane drops in the next link (only if one link downstream)
        lanes_drop = None
        if phi is not None:
            links_down: Collection[
                Tuple["Node", "Node", "Link[VarType]"]
            ] = net.out_links(node_down)
            if len(links_down) == 1:
                link_down = first(links_down)[-1]
                lanes_drop = self.lam - link_down.lam  # type: ignore[operator]
            if lanes_drop == 0:
                lanes_drop = None

        # step densities
        rho_next = engine.links.step_density(rho, q, q_up, self.lam, self.L, T)

        # step speeds
        Veq = engine.links.Veq(rho, self.v_free, self.rho_crit, self.a)
        v_next = engine.links.step_speed(
            v,
            v_up,
            rho,
            rho_down,
            Veq,
            self.lam,
            self.L,
            tau,
            eta,
            kappa,
            T,
            q_ramp,
            delta,
            lanes_drop,
            phi,
            self.rho_crit,
        )
        return {"rho": rho_next, "v": v_next}
