import pickle
import unittest
from typing import Any, Dict, Tuple

import casadi as cs
import numpy as np

from sym_metanet import (
    CongestedDestination,
    Destination,
    Link,
    MeteredOnRamp,
    Network,
    Node,
    Origin,
    SimplifiedMeteredOnRamp,
    engine,
    engines,
)

with open("tests/data_test_examples.pkl", "rb") as f:
    RESULTS = pickle.load(f)


def create_demands(time: np.ndarray) -> np.ndarray:
    return np.stack(
        (
            np.interp(time, (2.0, 2.25), (3500, 1000)),
            np.interp(time, (0.0, 0.15, 0.35, 0.5), (500, 1500, 1500, 500)),
        )
    )


def get_net() -> Tuple[Network, Dict[str, Any]]:
    T = 10 / 3600
    Tfin = 2.5
    time = np.arange(0, Tfin, T)
    L = 1
    lanes = 2
    C = (4000, 2000)
    tau = 18 / 3600
    kappa = 40
    eta = 60
    rho_max = 180
    delta = 0.0122
    a = 1.867
    rho_crit = 33.5
    v_free = 102

    # build network
    N1 = Node(name="N1")
    N2 = Node(name="N2")
    N3 = Node(name="N3")
    O1 = MeteredOnRamp(C[0], name="O1")
    O2 = MeteredOnRamp(C[1], name="O2")
    D1 = Destination(name="D1")
    L1 = Link(4, lanes, L, rho_max, rho_crit, v_free, a, name="L1")
    L2 = Link(2, lanes, L, rho_max, rho_crit, v_free, a, name="L2")
    net = (
        Network(name="A1")
        .add_path(origin=O1, path=(N1, L1, N2, L2, N3), destination=D1)
        .add_origin(O2, N2)
    )
    net.is_valid(raises=True)
    return net, {
        "T": T,
        "L": L,
        "lanes": lanes,
        "C": C,
        "tau": tau,
        "kappa": kappa,
        "eta": eta,
        "rho_max": rho_max,
        "delta": delta,
        "a": a,
        "rho_crit": rho_crit,
        "v_free": v_free,
        "time": time,
    }


class TestExamples(unittest.TestCase):
    def test_dynamics_example(self):
        engines.use("casadi", sym_type="SX")
        net, pars = get_net()
        net.step(
            T=pars["T"],
            tau=pars["tau"],
            eta=pars["eta"],
            kappa=pars["kappa"],
            delta=pars["delta"],
        )
        F = engine.to_function(
            net=net,
            more_out=True,
            compact=1,
            T=pars["T"],
        )
        demands = create_demands(pars["time"]).T
        rho = cs.DM([22, 22, 22.5, 24, 30, 32])
        v = cs.DM([80, 80, 78, 72.5, 66, 62])
        w = cs.DM([0, 0])
        r = cs.DM.ones(2, 1)  # constant ramp metering rates
        RHO, V, W, Q, Q_o = [], [], [], [], []
        for d in demands:
            rho, v, w, q, q_o = F(rho, v, w, r, d)
            RHO.append(rho)
            V.append(v)
            W.append(w)
            Q.append(q)
            Q_o.append(q_o)
        RHO, V, W, Q, Q_o = (np.squeeze(o) for o in (RHO, V, W, Q, Q_o))  # type: ignore
        tts = pars["T"] * sum(
            (rho * pars["L"] * pars["lanes"]).sum() + w.sum() for rho, w in zip(RHO, W)
        )

        RHO_, V_, W_, Q_, Q_o_, tts_ = RESULTS["test_dyn"]
        np.testing.assert_allclose(RHO, RHO_)
        np.testing.assert_allclose(V, V_)
        np.testing.assert_allclose(W, W_)
        np.testing.assert_allclose(Q, Q_)
        np.testing.assert_allclose(Q_o, Q_o_)
        np.testing.assert_allclose(tts, tts_)


if __name__ == "__main__":
    unittest.main()
