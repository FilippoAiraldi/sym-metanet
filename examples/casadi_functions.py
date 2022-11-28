import casadi as cs

try:
    import sym_metanet as metanet
except ImportError:
    import sys

    sys.path.insert(1, "src")
    import sym_metanet as metanet


# set casadi as the symbolic engine
metanet.engines.use("casadi", sym_type="SX")


# network parameters
L = 1  # length of links (km)
lanes = 2  # lanes per link (adim)
C = (3500, 2000)  # on-ramp capacities (veh/h/lane)
tau = 18 / 3600  # model parameter (s)
kappa = 40  # model parameter (veh/km/lane)
eta = 60  # model parameter (km^2/lane)
rho_max = 180  # maximum capacity (veh/km/lane)
delta = 0.0122  # merging phenomenum parameter
a = 1.867  # model parameter (adim)
v_free = 102  # free flow speed (km/h)
rho_crit = 33.5  # critical capacity (veh/km/lane)
T = 10 / 3600  # simulation step size (h)

a_sym = cs.SX.sym("a")
v_free_sym = cs.SX.sym("v_free")
rho_crit_sym = cs.SX.sym("rho_crit_sym")


# build components
N1 = metanet.Node(name="N1")
N2 = metanet.Node(name="N2")
N3 = metanet.Node(name="N3")
L1 = metanet.Link(2, lanes, L, rho_max, rho_crit_sym, v_free_sym, a_sym, name="L1")
L2 = metanet.Link(1, lanes, L, rho_max, rho_crit_sym, v_free_sym, a_sym, name="L2")
O1 = metanet.MeteredOnRamp(C[0], name="O1")
O2 = metanet.SimpleMeteredOnRamp(C[1], name="O2")
D3 = metanet.CongestedDestination(name="D3")


# build and validate network
net = (
    metanet.Network()
    .add_path(origin=O1, path=(N1, L1, N2, L2, N3), destination=D3)
    .add_origin(O2, N2)
)
net.is_valid(raises=True)


# get dynamics as symbolical function
net.step(T=T, tau=tau, eta=eta, kappa=kappa, delta=delta)
parameters = {"rho_crit": rho_crit_sym, "a": a_sym, "v_free": v_free_sym}
F = metanet.engine.to_function(
    net=net, T=T, parameters=parameters, more_out=True, compact=1
)


# run the dynamics once
# states
rho = [25, 20, 15]  # first two are densities in L1, last is density in L2
v = [80, 90, 100]  # same as for rho
w = [5, 10]  # first is queue at O1, second is queue at O2

# control actions
r = 1  # ramp metering rate of O1 (1 -> fully opened)\
q = 1000  # flow allowed at O2

# disturbances
d = [
    2e3,
    1e3,
    5e2,
]  # first two are demands at O1 and O2, last is the congestion scenario at D3

# compute next states and other output quantities
(
    rho_next,  # densities at the next time instant
    v_next,  # speeds at the next time instant
    w_next,  # queues at the next time instant
    q,  # flows in the links
    q_o,  # flows at the origins
) = F(rho, v, w, r, q, d, [rho_crit, a, v_free])
