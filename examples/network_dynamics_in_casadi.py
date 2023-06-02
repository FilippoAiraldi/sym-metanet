"""
Reproduces the results in Figure 6.5 of [1].

References
----------
[1] Hegyi, A., 2004. Model predictive control for integrating traffic control measures.
    Netherlands TRAIL Research School.
"""


import casadi as cs
import matplotlib.pyplot as plt
import numpy as np

import sym_metanet as metanet
from sym_metanet import Destination, Link, MeteredOnRamp, Network, Node, engines


def create_demands(time: np.ndarray) -> np.ndarray:
    return np.stack(
        (
            np.interp(time, (2.0, 2.25), (3500, 1000)),
            np.interp(time, (0.0, 0.15, 0.35, 0.5), (500, 1500, 1500, 500)),
        )
    )


# parameters
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
O1 = MeteredOnRamp[cs.SX](C[0], name="O1")
O2 = MeteredOnRamp[cs.SX](C[1], name="O2")
D1 = Destination[cs.SX](name="D1")
L1 = Link[cs.SX](4, lanes, L, rho_max, rho_crit, v_free, a, name="L1")
L2 = Link[cs.SX](2, lanes, L, rho_max, rho_crit, v_free, a, name="L2")
net = (
    Network(name="A1")
    .add_path(origin=O1, path=(N1, L1, N2, L2, N3), destination=D1)
    .add_origin(O2, N2)
)

# make a casadi function out of the network
engines.use("casadi", sym_type="SX")
net.is_valid(raises=True)
net.step(T=T, tau=tau, eta=eta, kappa=kappa)
F = metanet.engine.to_function(
    net=net,
    more_out=True,
    compact=1,
    T=T,
)
# F: (rho[6], v[6], w[2], r[2], d[2]) -> (rho+[6], v+[6], w+[2], q[6], q_o[2])

# create demands
demands = create_demands(time).T

# create initial conditions
rho = cs.DM([22, 22, 22.5, 24, 30, 32])
v = cs.DM([80, 80, 78, 72.5, 66, 62])
w = cs.DM([0, 0])

# simulate
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


# compute TTS metric (Total-Time-Spent)
tts = T * sum((rho * L * lanes).sum() + w.sum() for rho, w in zip(RHO, W))
print(f"TTS = {tts:.3f} veh.h")

# plot
_, axs = plt.subplots(3, 2, constrained_layout=True, sharex=True)
axs[0, 0].plot(time, V)
axs[0, 0].set_ylabel("speed")
axs[0, 1].plot(time, Q)
axs[0, 1].set_ylabel("flow")
axs[1, 0].plot(time, RHO)
axs[1, 0].set_ylabel("density")
axs[1, 1].plot(time, demands)
axs[1, 1].set_ylabel("origin demands")
axs[2, 0].plot(time, Q_o)
axs[2, 0].set_ylabel("origin flow")
axs[2, 1].plot(time, W)
axs[2, 1].set_ylabel("queue")
axs[0, 0].set_xlim(0, Tfin)
plt.show()
