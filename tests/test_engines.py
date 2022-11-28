import sys
import unittest

import casadi as cs
import numpy as np

sys.path.insert(1, "src")
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
            ("v", v_next, [100.10991104,  92.63849965,  71.7779459]),
            ("w", w_next, [45.83333333, 30.55555556]),
            ("q", q, [2700, 3200, 3500]),
            ("q_o", q_o, [3500, 800]),
        ]:
            np.testing.assert_allclose(
                x.full().flatten(), y, atol=1e-6, rtol=1e-6, err_msg=name
            )


if __name__ == "__main__":
    unittest.main()
