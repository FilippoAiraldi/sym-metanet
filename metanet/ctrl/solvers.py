import casadi as cs
import numpy as np
import itertools

from typing import Callable, Tuple, List, Dict, Any

from ..blocks.networks import Network
from ..sim.simulations import Simulation
from .util import sim2func
from ..util import pad, repinterl


class MPC:
    def __init__(self, sim: Simulation, Np: int, Nc: int,
                 cost: Callable[[Simulation, Dict[str, cs.SX], Dict[str, cs.SX]], cs.SX],
                 M: int = 1,
                 disable_ramp_metering: bool = False,
                 disable_vms: bool = False,
                 solver: str = 'ipopt',
                 plugin_opts: Dict[str, Any] = None,
                 solver_opts: Dict[str, Any] = None) -> None:
        '''
        Instantiates an MPC for METANET control.

        Parameters
        ----------
            sim : metanet.Simulation
                The simulation to control via MPC.

            Np, Nc : int
                Prediction and control horizon. Must be positive and Np >= Nc.

            cost : Callable
                The cost function to be minimized.

            M : int, optional
                Control action multiplier, i.e., one action free every M steps,
                or, in other words, every action is kept constant for M steps.

            disable_ramp_metering, disable_vms : bool
                Whether to disable control of ramp metering rates and Variable
                Sign Message speed control. 
        '''
        opti = cs.Opti()
        vars, vars_ext, pars = self.__create_vars_and_pars(
            sim.net, opti, Np, Nc, M)
        self.__create_constraints(
            sim, opti, vars, vars_ext, pars, Np, M,
            disable_ramp_metering, disable_vms)
        self.__create_objective_and_others(
            sim, opti, vars, pars, cost, solver, plugin_opts, solver_opts)

        # save to self
        self.opti = opti
        self.vars, self.vars_ext, self.pars = vars, vars_ext, pars
        self.sim = sim
        self.Np, self.Nc, self.M = Np, Nc, M

    def __create_vars_and_pars(self, net: Network, opti: cs.Opti, Np: int,
                               Nc: int, M: int) -> Tuple[Dict[str, cs.SX], ...]:
        vars, vars_ext, pars = {}, {}, {}

        # create state
        for origin in net.origins:
            vars[f'w_{origin}'] = opti.variable(1, M * Np + 1)
        for link in net.links:
            vars[f'rho_{link}'] = opti.variable(link.nb_seg, M * Np + 1)
            vars[f'v_{link}'] = opti.variable(link.nb_seg, M * Np + 1)

        # create control and last values of control (ramp rates and vms speeds)
        for origin, _ in net.onramps:
            u = opti.variable(1, Nc)
            vars[f'r_{origin}'] = u
            vars_ext[f'r_{origin}'] = pad(repinterl(u, 1, M), (0, 0),
                                          (0, M * (Np - Nc)), mode='edge')
        for link, _ in net.links_with_vms:
            v_ctrl = opti.variable(link.nb_vms, Nc)
            vars[f'v_ctrl_{link}'] = v_ctrl
            vars_ext[f'v_ctrl_{link}'] = pad(repinterl(v_ctrl, 1, M), (0, 0),
                                             (0, M * (Np - Nc)), mode='edge')

        # create parameters (demands, initial conditions, last action)
        for origin in net.origins:
            pars[f'd_{origin}'] = opti.parameter(1, M * Np)
            pars[f'w0_{origin}'] = opti.parameter(1, 1)
        for link in net.links:
            pars[f'rho0_{link}'] = opti.parameter(link.nb_seg, 1)
            pars[f'v0_{link}'] = opti.parameter(link.nb_seg, 1)
        for origin, _ in net.onramps:
            pars[f'r_{origin}_last'] = opti.parameter(1, 1)
        for link, _ in net.links_with_vms:
            pars[f'v_ctrl_{link}_last'] = opti.parameter(link.nb_vms, 1)

        return vars, vars_ext, pars

    def __create_constraints(
            self, sim: Simulation, opti: cs.Opti, vars: Dict[str, cs.SX],
            vars_ext: Dict[str, cs.SX], pars: Dict[str, cs.SX],
            Np: int, M: int,
            disable_ramps: bool, disable_vms: bool) -> None:
        net = sim.net

        # set input constraints
        for origin, _ in net.onramps:
            u = cs.vec(vars[f'r_{origin}'])
            if disable_ramps:
                opti.subject_to(u == 1)
            else:
                opti.subject_to(u >= 0.2)
                opti.subject_to(u <= 1)
        for link, _ in net.links_with_vms:
            v_ctrl = cs.vec(vars[f'v_ctrl_{link}'])
            if disable_vms:
                opti.subject_to(v_ctrl == 9e3)  # or v_free
            else:
                opti.subject_to(v_ctrl >= 20)
                opti.subject_to(v_ctrl <= link.v_free)

        # set state positivity constraints
        for origin in net.origins:
            opti.subject_to(cs.vec(vars[f'w_{origin}']) >= 0)
        for link in net.links:
            opti.subject_to(cs.vec(vars[f'rho_{link}']) >= 0)
            opti.subject_to(cs.vec(vars[f'v_{link}']) >= 0)

        # set initial conditions constraint
        for origin in net.origins:
            opti.subject_to(vars[f'w_{origin}'][:, 0] == pars[f'w0_{origin}'])
        for link in net.links:
            opti.subject_to(vars[f'rho_{link}'][:, 0] == pars[f'rho0_{link}'])
            opti.subject_to(vars[f'v_{link}'][:, 0] == pars[f'v0_{link}'])

        # create x+ = F(x, u, d)
        # Example
        # (w_O1,w_O2,rho_L1[4],v_L1[4],rho_L2[2],v_L2[2],r_O2,d_O1,d_O2) ->
        # (w+_O1,w+_O2,rho+_L1[4],v+_L1[4],rho+_L2[2],v+_L2[2])
        F = sim2func(sim, out_nonneg=True)

        # set trajectory evolution constraint
        for k in range(M * Np):
            outs = F(
                *(vars[f'w_{o}'][:, k] for o in net.origins),
                *itertools.chain.from_iterable(
                    (vars[f'rho_{l}'][:, k], vars[f'v_{l}'][:, k])
                    for l in net.links),
                *(vars_ext[f'r_{o}'][:, k] for o, _ in net.onramps),
                *(vars_ext[f'v_ctrl_{l}'][:, k]
                  for l, _ in net.links_with_vms),
                *(pars[f'd_{o}'][:, k] for o in net.origins))
            i = 0
            for origin in net.origins:
                opti.subject_to(vars[f'w_{origin}'][:, k + 1] == outs[i])
                i += 1
            for link in net.links:
                opti.subject_to(vars[f'rho_{link}'][:, k + 1] == outs[i])
                opti.subject_to(vars[f'v_{link}'][:, k + 1] == outs[i + 1])
                i += 2

    def __create_objective_and_others(
            self, sim: Simulation, opti: cs.Opti, vars: Dict[str, cs.SX],
            pars: Dict[str, cs.SX],
            cost: Callable, solver: str,
            plugin_opts: Dict[str, Any], solver_opts: Dict[str, Any]) -> None:
        # optimization criterion
        opti.minimize(cost(sim, vars, pars))

        # set solver
        opti.solver(solver,
                    {} if plugin_opts is None else plugin_opts,
                    {} if solver_opts is None else solver_opts)

    def to_func(self) -> Callable[[Dict[str, float], Dict[str, float]],
                                  Tuple[Dict[str, float], Dict[str, Any]]]:
        '''
        Returns a callable function to automatically run the MPC optimization.
        '''
        def _f(vars_init: Dict[str, float], pars_val: Dict[str, float]):
            for par in self.pars:
                self.opti.set_value(self.pars[par], pars_val[par])
            for var in self.vars:
                self.opti.set_initial(self.vars[var], vars_init[var])

            try:
                sol = self.opti.solve()
                info = {}
                get_value = lambda o: sol.value(o)
            except Exception as ex1:
                try:
                    info = {'error': self.opti.debug.stats()['return_status']}
                    # + ' (' + str(ex1).replace('\n', ' ') + ')'}
                    get_value = lambda o: self.opti.debug.value(o)
                except Exception as ex2:
                    raise RuntimeError(
                        'error during handling of first '
                        f'exception.\nEx. 1: {ex1}\nEx. 2: {ex2}') from ex2

            info['f'] = float(get_value(self.opti.f))
            return {name: get_value(var).reshape(var.shape)
                    for name, var in self.vars.items()}, info
        return _f


class NlpSolver:
    def __init__(self, sim: Simulation, Np: int, Nc: int,
                 cost: Callable[[Simulation, Dict[str, cs.SX], Dict[str, cs.SX]], cs.SX],
                 M: int = 1,
                 solver: str = 'ipopt',
                 solver_opts: Dict[str, Any] = None,
                 disable_ramp_metering: bool = False,
                 disable_vms: bool = False) -> None:

        vars, lbx, ubx, inps_ext, pars = self.__create_vars_and_pars(
            sim.net, Np, Nc, M, disable_ramp_metering, disable_vms)
        g, lbg, ubg = self.__create_constraints(
            sim, vars, inps_ext, pars, Np, M)
        f = cost(sim, vars, pars)

        # save to self
        self.sim = sim
        self.Np, self.Nc, self.M = Np, Nc, M
        self.solver_name = solver
        self.solver_opts = {} if solver_opts is None else solver_opts
        self.vars, self.inps_ext, self.pars = vars, inps_ext, pars
        self.lbx, self.ubx = lbx, ubx
        self.g, self.lbg, self.ubg = g, lbg, ubg
        self.f = f

    def __create_vars_and_pars(self, net: Network, Np: int, Nc: int,
                               M: int,
                               disable_ramp_metering: bool,
                               disable_vms: bool
                               ) -> Tuple[Dict[str, cs.SX], ...]:
        vars, lbx, ubx = {}, [], []
        inps_ext = {}
        pars = {}

        def add_var(name, lb, ub, *size):
            var = cs.SX.sym(name, *size)
            vars[name] = var
            if isinstance(lb, (float, int)):
                lb = np.full(var.shape, lb)
            if isinstance(ub, (float, int)):
                ub = np.full(var.shape, ub)
            assert ub.shape == var.shape == lb.shape
            lbx.append(lb)
            ubx.append(ub)
            return var

        def add_par(name, *size):
            par = cs.SX.sym(name, *size)
            pars[name] = par
            return par

        # create state
        for origin in net.origins:
            add_var(f'w_{origin}', 0, np.inf, 1, M * Np + 1)
        for link in net.links:
            add_var(f'rho_{link}', 0, np.inf, link.nb_seg, M * Np + 1)
            add_var(f'v_{link}', 0, np.inf, link.nb_seg, M * Np + 1)

        # create control and last values of control (ramp rates and vms speeds)
        for origin, _ in net.onramps:
            if disable_ramp_metering:
                u = add_var(f'r_{origin}', 1, 1, 1, Nc)
            else:
                u = add_var(f'r_{origin}', 0.2, 1, 1, Nc)
            inps_ext[f'r_{origin}'] = pad(repinterl(u, 1, M), (0, 0),
                                          (0, M * (Np - Nc)), mode='edge')
        for link, _ in net.links_with_vms:
            if disable_vms:
                v_ctrl = add_var(f'v_ctrl_{link}', 9e3, 9e3, link.nb_vms, Nc)
            else:
                v_ctrl = add_var(f'v_ctrl_{link}', 20, link.v_free,
                                 link.nb_vms, Nc)
            inps_ext[f'v_ctrl_{link}'] = pad(repinterl(v_ctrl, 1, M), (0, 0),
                                             (0, M * (Np - Nc)), mode='edge')

        # create parameters (demands, initial conditions, last action)
        for origin in net.origins:
            add_par(f'd_{origin}', 1, M * Np)
            add_par(f'w0_{origin}', 1, 1)
        for link in net.links:
            add_par(f'rho0_{link}', link.nb_seg, 1)
            add_par(f'v0_{link}', link.nb_seg, 1)
        for origin, _ in net.onramps:
            add_par(f'r_{origin}_last', 1, 1)
        for link, _ in net.links_with_vms:
            add_par(f'v_ctrl_{link}_last', link.nb_vms, 1)
        return vars, lbx, ubx, inps_ext, pars

    def __create_constraints(self, sim: Simulation, vars: Dict[str, cs.SX],
                             vars_ext: Dict[str, cs.SX],
                             pars: Dict[str, cs.SX],
                             Np: int, M: int) -> Tuple[List[cs.SX], ...]:
        net = sim.net
        g, lbg, ubg = [], [], []

        def add_constr(lb, con, ub):
            if isinstance(lb, (float, int)):
                lb = np.full(con.shape, lb)
            if isinstance(ub, (float, int)):
                ub = np.full(con.shape, ub)
            assert ub.shape == con.shape == lb.shape
            lbg.append(lb)
            g.append(con)
            ubg.append(ub)

        # set initial conditions constraint
        for origin in net.origins:
            c = vars[f'w_{origin}'][:, 0] - pars[f'w0_{origin}']
            add_constr(0, c, 0)
        for link in net.links:
            c = vars[f'rho_{link}'][:, 0] - pars[f'rho0_{link}']
            add_constr(0, c, 0)
            c = vars[f'v_{link}'][:, 0] - pars[f'v0_{link}']
            add_constr(0, c, 0)

        # create x+ = F(x, u, d)
        # Example
        # (w_O1,w_O2,rho_L1[4],v_L1[4],rho_L2[2],v_L2[2],r_O2,d_O1,d_O2) ->
        # (w+_O1,w+_O2,rho+_L1[4],v+_L1[4],rho+_L2[2],v+_L2[2])
        F = sim2func(sim, out_nonneg=True)

        # set trajectory evolution constraint
        for k in range(M * Np):
            outs = F(
                *(vars[f'w_{o}'][:, k] for o in net.origins),
                *itertools.chain.from_iterable(
                    (vars[f'rho_{l}'][:, k], vars[f'v_{l}'][:, k])
                    for l in net.links),
                *(vars_ext[f'r_{o}'][:, k] for o, _ in net.onramps),
                *(vars_ext[f'v_ctrl_{l}'][:, k]
                  for l, _ in net.links_with_vms),
                *(pars[f'd_{o}'][:, k] for o in net.origins))

            i = 0
            for origin in net.origins:
                c = vars[f'w_{origin}'][:, k + 1] - outs[i]
                add_constr(0, c, 0)
                i += 1
            for link in net.links:
                c = vars[f'rho_{link}'][:, k + 1] - outs[i]
                add_constr(0, c, 0)
                c = vars[f'v_{link}'][:, k + 1] - outs[i + 1]
                add_constr(0, c, 0)
                i += 2

        return g, lbg, ubg

    def add_constraint(self, lb: float, con: cs.SX, ub: float) -> None:
        if isinstance(lb, (float, int)):
            lb = np.full(con.shape, lb)
        if isinstance(ub, (float, int)):
            ub = np.full(con.shape, ub)
        assert lb.shape == con.shape == ub.shape
        self.lbg.append(lb)
        self.g.append(con)
        self.ubg.append(ub)

    def to_func(self) -> Callable[[Dict[str, float], Dict[str, float]],
                                  Tuple[Dict[str, float], Dict[str, str]]]:
        # flatten into vectos
        self.x = cs.vertcat(*(cs.vec(o) for o in self.vars.values()))
        self.p = cs.vertcat(*(cs.vec(o) for o in self.pars.values()))
        for attr in ('lbx', 'ubx', 'g', 'lbg', 'ubg'):
            v = getattr(self, attr)
            if isinstance(v, list):
                setattr(self, attr, cs.vertcat(*(cs.vec(o) for o in v)))

        # build solver
        nlp = {'x': self.x, 'f': self.f, 'g': self.g, 'p': self.p}
        solver = cs.nlpsol('solver', self.solver_name, nlp, self.solver_opts)
        self.solver = solver

        # prepare stuff
        Nx, Nc = self.M * self.Np + 1, self.Nc
        net = self.sim.net

        def _f(vars_init: Dict[str, float], pars_val: Dict[str, float]):
            # sourcery skip: for-append-to-extend, list-comprehension, merge-list-appends-into-extend
            # same order as creation
            p, x0 = [], []
            for origin in net.origins:
                p.append(pars_val[f'd_{origin}'])
                p.append(pars_val[f'w0_{origin}'])
                x0.append(vars_init[f'w_{origin}'])
            for link in net.links:
                p.append(pars_val[f'rho0_{link}'])
                p.append(pars_val[f'v0_{link}'])
                x0.append(vars_init[f'rho_{link}'])
                x0.append(vars_init[f'v_{link}'])
            for onramp, _ in net.onramps:
                p.append(pars_val[f'r_{onramp}_last'])
                x0.append(vars_init[f'r_{onramp}'])
            for link, _ in net.links_with_vms:
                p.append(pars_val[f'v_ctrl_{link}_last'])
                x0.append(vars_init[f'v_ctrl_{link}'])
            p = cs.vertcat(*[cs.vec(o) for o in p])
            x0 = cs.vertcat(*[cs.vec(o) for o in x0])
            assert p.shape == self.p.shape
            assert x0.shape == self.x.shape

            sol = solver(x0=x0, p=p, lbx=self.lbx, ubx=self.ubx,
                         lbg=self.lbg, ubg=self.ubg)

            info = {'f': float(sol['f'])}
            status = solver.stats()['return_status']
            if status != 'Solve_Succeeded':
                info['error'] = status

            x_opt = sol['x']
            i, out = 0, {}
            for origin in net.origins:
                out[f'w_{origin}'] = x_opt[i:i + Nx].reshape((1, Nx))
                i += Nx
            for link in net.links:
                out[f'rho_{link}'] = x_opt[i:i + link.nb_seg * Nx
                                           ].reshape((link.nb_seg, Nx))
                i += link.nb_seg * Nx
                out[f'v_{link}'] = x_opt[i:i + link.nb_seg * Nx
                                         ].reshape((link.nb_seg, Nx))
                i += link.nb_seg * Nx
            for ornamp, _ in net.onramps:
                out[f'r_{ornamp}'] = x_opt[i:i + Nc
                                           ].reshape((1, Nc))
                i += Nc
            for link, _ in net.links_with_vms:
                out[f'v_ctrl_{link}'] = x_opt[i:i + link.nb_vms * Nc
                                              ].reshape((link.nb_vms, Nc))
                i += link.nb_vms * Nc
            return out, info

        return _f
