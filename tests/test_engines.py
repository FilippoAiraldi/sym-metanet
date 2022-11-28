import unittest

import casadi as cs
import numpy as np

import sym_metanet as metanet
from sym_metanet import (
    CongestedDestination,
    Link,
    MeteredOnRamp,
    Network,
    Node,
    SimpleMeteredOnRamp,
    engines,
)
from sym_metanet.engines.casadi import Engine as CasadiEngine
from sym_metanet.engines.numpy import Engine as NumpyEngine
from sym_metanet.errors import EngineNotFoundError
from sym_metanet.util.funcs import first


def get_net():
    L = 1
    lanes = 2
    C = (3500, 2000)
    tau = 18 / 3600
    kappa = 40
    eta = 60
    rho_max = 180
    delta = 0.0122
    T = 10 / 3600
    a_sym = cs.SX.sym("a")
    v_free_sym = cs.SX.sym("v_free")
    rho_crit_sym = cs.SX.sym("rho_crit_sym")
    N1 = Node(name="N1")
    N2 = Node(name="N2")
    N3 = Node(name="N3")
    L1 = Link(2, lanes, L, rho_max, rho_crit_sym, v_free_sym, a_sym, name="L1")
    L2 = Link(1, lanes, L, rho_max, rho_crit_sym, v_free_sym, a_sym, name="L2")
    O1 = MeteredOnRamp(C[0], name="O1")
    O2 = SimpleMeteredOnRamp(C[1], name="O2")
    D1 = CongestedDestination(name="D1")
    net = (
        Network(name="A1")
        .add_path(origin=O1, path=(N1, L1, N2, L2, N3), destination=D1)
        .add_origin(O2, N2)
    )
    sym_pars = {"rho_crit": rho_crit_sym, "a": a_sym, "v_free": v_free_sym}
    others = {"T": T, "tau": tau, "eta": eta, "kappa": kappa, "delta": delta}
    return net, sym_pars, others


class TestEngines(unittest.TestCase):
    def test_get_current_engine__gets_correct_engine(self):
        self.assertIs(engines.get_current_engine(), metanet.engine)

    def test_get_available_engines__gets_available_engines_correctly(self):
        engines_ = engines.get_available_engines()
        self.assertIsInstance(engines_, dict)
        self.assertIsInstance(first(engines_.keys()), str)

    def test_use__sets_the_engine_correctly(self):
        old_engine = metanet.engine
        new_engine = engines.use("casadi")
        self.assertIsNot(metanet.engine, old_engine)
        self.assertIs(metanet.engine, new_engine)

    def test_use__raises__when_engine_not_found(self):
        invalid_engine = object()
        with self.assertRaises(EngineNotFoundError):
            engines.use(invalid_engine)


class TestCasadiEngine(unittest.TestCase):
    def test_to_function__fails__with_uninit_vars(self):
        net, sym_pars, other_pars = get_net()
        self.assertTrue(net.is_valid(raises=False)[0])
        engine = engines.use("casadi", sym_type="SX")

        with self.assertRaises(RuntimeError):
            engine.to_function(
                net=net,
                parameters=sym_pars,
                T=other_pars["T"],
                compact=1,
                more_out=True,
            )

    def test_to_function__fails__with_unstepped_dynamics(self):
        net, sym_pars, other_pars = get_net()
        self.assertTrue(net.is_valid(raises=False)[0])
        engine = engines.use("casadi", sym_type="SX")
        for el in net.elements:
            el.init_vars(engine=engine)
        with self.assertRaises(RuntimeError):
            engine.to_function(
                net=net,
                parameters=sym_pars,
                T=other_pars["T"],
                compact=1,
                more_out=True,
            )

    def test_to_function__numerically_works(self):
        L = 1
        lanes = 2
        C = (3500, 2000)
        tau = 18 / 3600
        kappa = 40
        eta = 60
        rho_max = 180
        delta = 0.0122
        T = 10 / 3600
        a = cs.SX.sym("a")
        v_free = cs.SX.sym("v_free")
        rho_crit = cs.SX.sym("rho_crit_sym")
        N1 = Node(name="N1")
        N2 = Node(name="N2")
        N3 = Node(name="N3")
        L1 = Link(2, lanes, L, rho_max, rho_crit, v_free, a, name="L1")
        L2 = Link(1, lanes, L, rho_max, rho_crit, v_free, a, name="L2")
        O1 = MeteredOnRamp(C[0], name="O1")
        O2 = SimpleMeteredOnRamp(C[1], name="O2")
        D1 = CongestedDestination(name="D1")
        net = (
            Network()
            .add_path(origin=O1, path=(N1, L1, N2, L2, N3), destination=D1)
            .add_origin(O2, N2)
        )
        net.is_valid(raises=True)
        net.step(T=T, tau=tau, eta=eta, kappa=kappa, delta=delta)
        args = {
            "net": net,
            "more_out": True,
            "T": T,
            "parameters": {"rho_crit": rho_crit, "a": a, "v_free": v_free},
        }
        F = metanet.engine.to_function(**args, compact=1)

        p = [33, 1.8, 130]
        rho = [15, 20, 25]
        v = [90, 80, 70]
        w = [50, 30]
        d = [2e3, 1e3, 50]
        q = 800
        rho_next, v_next, w_next, q, q_o = F(rho, v, w, 1, q, d, p)

        for name, x, y in [
            ("rho", rho_next, [16.11111111, 19.30555556, 25.69444444]),
            ("v", v_next, [100.10991104, 92.63849965, 71.7779459]),
            ("w", w_next, [45.83333333, 30.55555556]),
            ("q", q, [2700, 3200, 3500]),
            ("q_o", q_o, [3500, 800]),
        ]:
            np.testing.assert_allclose(
                x.full().flatten(), y, atol=1e-6, rtol=1e-6, err_msg=name
            )


class TestCasadiVsNumpyEngine(unittest.TestCase):
    N = 7
    NE = NumpyEngine()
    CE = CasadiEngine[cs.DM]()

    def test_nodes__get_upstream_flow(self):
        N = self.N
        q_lasts = np.maximum(0, np.random.randn(N) * 50 + 200)
        betas = np.random.randint(low=1, high=10, size=N)
        q_orig = np.maximum(0, np.random.randn() * 50 + 200)
        args = [q_lasts, betas[0], betas, q_orig]
        np.testing.assert_allclose(
            self.NE.nodes.get_upstream_flow(*args),
            self.CE.nodes.get_upstream_flow(*map(cs.DM, args)).full().squeeze(),
        )

    def test_nodes__get_upstream_speed(self):
        N = self.N
        q_lasts = np.maximum(0, np.random.randn(N) * 50 + 200)
        v_lasts = np.maximum(0, np.random.randn(N) * 10 + 60)
        args = [q_lasts, v_lasts]
        np.testing.assert_allclose(
            self.NE.nodes.get_upstream_speed(*args),
            self.CE.nodes.get_upstream_speed(*map(cs.DM, args)).full().squeeze(),
        )

    def test_nodes__get_downstream_density(self):
        N = self.N
        rho_firsts = np.maximum(0, np.random.randn(N) * 5 + 30)
        np.testing.assert_allclose(
            self.NE.nodes.get_downstream_density(rho_firsts),
            self.CE.nodes.get_downstream_density(cs.DM(rho_firsts)).full().squeeze(),
        )

    def test_links__get_flow(self):
        N = self.N
        rho = np.maximum(0, np.random.randn(N) * 5 + 30)
        v = np.maximum(0, np.random.randn(N) * 10 + 80)
        lanes = np.random.randint(3, 7)
        args = [rho, v, lanes]
        np.testing.assert_allclose(
            self.NE.links.get_flow(*args),
            self.CE.links.get_flow(*map(cs.DM, args)).full().squeeze(),
        )

    def test_links__step_density(self):
        N = self.N
        rho = np.maximum(0, np.random.randn(N) * 5 + 30)
        q = np.maximum(0, np.random.randn(N) * 10 + 800)
        q_up = np.maximum(0, np.random.randn(N) * 10 + 800)
        lanes = np.random.randint(3, 7)
        L = np.random.rand() * 1000 + 1000
        T = np.random.rand()
        args = [rho, q, q_up, lanes, L, T]
        np.testing.assert_allclose(
            self.NE.links.step_density(*args),
            self.CE.links.step_density(*map(cs.DM, args)).full().squeeze(),
        )

    def test_links__step_speed(self):
        N = self.N
        v = np.maximum(0, np.random.randn(N) * 10 + 100)
        v_up = np.maximum(0, np.random.randn(N) * 10 + 100)
        rho = np.maximum(0, np.random.randn(N) * 5 + 30)
        rho_down = np.maximum(0, np.random.randn(N) * 5 + 30)
        Veq = np.maximum(0, np.random.randn(N) * 10 + 100)
        lanes = np.random.randint(3, 7)
        L = np.random.rand() * 1000 + 1000
        tau = np.random.rand()
        eta = T = np.random.rand()
        kappa = np.random.rand()
        T = np.random.rand()
        q_ramp = np.maximum(0, np.random.randn() * 100 + 800)
        delta = np.random.rand()
        lanes_drop = np.random.randint(1, 3)
        phi = np.random.rand()
        rho_crit = np.random.rand() * 10 + 30
        args = [
            v,
            v_up,
            rho,
            rho_down,
            Veq,
            lanes,
            L,
            tau,
            eta,
            kappa,
            T,
            q_ramp,
            delta,
            lanes_drop,
            phi,
            rho_crit,
        ]
        np.testing.assert_allclose(
            self.NE.links.step_speed(*args),
            self.CE.links.step_speed(*map(cs.DM, args)).full().squeeze(),
        )

    def test_links__Veq(self):
        N = self.N
        rho = np.maximum(0, np.random.randn(N) * 5 + 30)
        v_free = np.random.rand() * 40 + 80
        rho_crit = np.random.rand() * 10 + 30
        a = np.random.rand() + 1
        args = [rho, v_free, rho_crit, a]
        np.testing.assert_allclose(
            self.NE.links.Veq(*args),
            self.CE.links.Veq(*map(cs.DM, args)).full().squeeze(),
        )

    def test_origins__step_queue(self):
        w = np.random.randint(low=1, high=3) * 100 + 300
        d = np.random.randint(low=1, high=3) * 100 + 300
        q = np.maximum(0, np.random.randn() * 200 + 800)
        T = np.random.rand()
        args = [w, d, q, T]
        np.testing.assert_allclose(
            self.NE.origins.step_queue(*args),
            self.CE.origins.step_queue(*map(cs.DM, args)).full().squeeze(),
        )

    def test_origins__get_ramp_flow(self):
        w = np.random.randint(low=1, high=3) * 100 + 300
        d = np.random.randint(low=1, high=3) * 100 + 300
        C = np.maximum(100, np.random.randn() * 500 + 2000)
        r = np.random.rand()
        rho_max = (np.random.rand() * 30 + 130) * 2
        rho_first = np.random.rand() * 20 + 20
        rho_crit = np.random.rand() * 20 + 20
        T = np.random.rand()
        args = [d, w, C, r, rho_max, rho_first, rho_crit, T]
        for type_ in ("in", "out"):
            np.testing.assert_allclose(
                self.NE.origins.get_ramp_flow(*args, type_),
                self.CE.origins.get_ramp_flow(*map(cs.DM, args), type_)
                .full()
                .squeeze(),
            )

    def test_destinations__get_ramp_flow(self):
        rho_last = np.random.rand() * 20 + 20
        rho_destination = np.random.rand() * 30 + 20
        rho_crit = np.random.rand() * 20 + 20
        args = [rho_last, rho_destination, rho_crit]
        np.testing.assert_allclose(
            self.NE.destinations.get_congested_downstream_density(*args),
            self.CE.destinations.get_congested_downstream_density(*map(cs.DM, args))
            .full()
            .squeeze(),
        )


if __name__ == "__main__":
    unittest.main()
