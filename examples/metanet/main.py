###############################################################################
############################### METANET EXAMPLE ###############################
###############################################################################
import numpy as np

import sys, os
sys.path.append(os.path.expanduser('~\\Documents\\git\\traffic-modelling'))
# import sys
# sys.path.append('path\to\traffic-modelling')
from trafficmodelling import metanet, util as tm_util

from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


#################################### MODEL ####################################


# simulation params
Tfin = 2.5                      # final simulation time (h)
T = 10 / 3600                   # simulation step size (h)
t = np.arange(0, Tfin + T, T)   # time vector (h)
K = t.size                      # simulation steps

# segments
L = 1                           # length of links (km)
lanes = 2                       # lanes per link (adim)

# on-ramp O2
C2 = 2000                       # on-ramp capacity (veh/h/lane)
max_queue = 100

# Variable speed limits
alpha = 0.1                     # adherence to limit

# model parameters
tau = 18 / 3600                 # model parameter (s)
kappa = 40                      # model parameter (veh/km/lane)
eta = 60                        # model parameter (km^2/lane)
rho_max = 180                   # maximum capacity (veh/km/lane)
delta = 0.0122                  # merging phenomenum parameter
a = 1.867                       # model parameter (adim)
v_free = 102                    # free flow speed (km/h)
rho_crit = 33.5                 # critical capacity (veh/km/lane)

# create components
N1 = metanet.Node(name='N1')
N2 = metanet.Node(name='N2')
N3 = metanet.Node(name='N3')
L1 = metanet.Link(4, lanes, L, v_free, rho_crit, a, name='L1')
L2 = metanet.Link(2, lanes, L, v_free, rho_crit, a, name='L2')
O1 = metanet.MainstreamOrigin(name='O1')
O2 = metanet.OnRamp(C2, name='O2')
D1 = metanet.Destination(name='D1')

# assemble network
net = metanet.Network(name='Small example')
for n in (N1, N2, N3):
    net.add_node(n)
net.add_link(N1, L1, N2)
net.add_link(N2, L2, N3)
net.add_origin(O1, N1)
net.add_origin(O2, N2)
net.add_destination(D1, N3)
# net.plot(reverse_x=True) # requires networkx

# instantiate simulation
sim = metanet.Simulation(net, T, rho_max, eta, tau, kappa, delta, alpha=alpha)


################################# DISTURBANCE #################################


O1.demand = tm_util.create_profile(t, [2, 2.25], [3500, 1000])
O2.demand = tm_util.create_profile(
    t, [0, .125, .375, 0.5], [500, 1500, 1500, 500])

plt.figure(figsize=(4, 3), constrained_layout=True)
plt.plot(t, O1.demand, '-', label='$O_1$')
plt.plot(t, O2.demand, '--', label='$O_2$')
plt.xlabel('time (h)')
plt.ylabel('demand (veh/h)')
plt.xlim(0, Tfin)
plt.ylim(0, 4000)
plt.legend()


################################# SIMULATION ##################################


# initial state
q0_L1, v0_L1 = 1e3 * np.array([3.5, 3.5, 3.5, 3.5]), np.array([81, 81, 79, 76])
q0_L2, v0_L2 = 1e3 * np.array([4, 4]), np.array([66, 62])
sim.set_init_cond({
    L1: (q0_L1 / (L1.lanes * v0_L1), v0_L1),
    L2: (q0_L2 / (L1.lanes * v0_L2), v0_L2)
}, {
    O1: np.array(0),
    O2: np.array(0)
})

# simulaiton main loop
for k in tqdm(range(K), total=K):
    # set links' speed controls (there is one for each segment)
    L1.v_ctrl[k] = np.full((L1.nb_seg, 1), np.inf)
    # ...set L1's 3rd speed control here...
    # ...set L1's 4th speed control here...
    L2.v_ctrl[k] = np.full((L2.nb_seg, 1), np.inf)

    # set onramp metering rate
    O2.rate[k] = 1

    # step simulation
    sim.step(k)


################################### PLOTTING ##################################


fig = plt.figure(figsize=(10, 7), constrained_layout=True)
gs = GridSpec(3, 2, figure=fig)
axs = [fig.add_subplot(gs[0, 0]),
       fig.add_subplot(gs[0, 1]),
       fig.add_subplot(gs[1, 0]),
       fig.add_subplot(gs[1, 1]),
       fig.add_subplot(gs[2, 0]),
       fig.add_subplot(gs[2, 1])]

for link in net.links:
    v = np.hstack(link.speed[:-1])
    rho = np.hstack(link.density[:-1])
    q = np.hstack(link.flow)

    for i in range(link.nb_seg):
        axs[0].plot(t, v[i], label=f'$v_{{{link.name}, {i + 1}}}$')
        axs[1].plot(t, q[i], label=f'$q_{{{link.name}, {i + 1}}}$')
        axs[2].plot(t, rho[i], label=f'$\\rho_{{{link.name}, {i + 1}}}$')

for origin in net.origins:
    w = np.vstack(origin.queue[:-1])
    q = np.vstack(origin.flow)

    axs[4].plot(t, q, label=f'$q_{{{origin.name}}}$')
    axs[5].plot(t, w, label=f'$\\omega_{{{origin.name}}}$')

axs[0].set_ylabel('speed (km/h)')
axs[1].set_ylabel('flow (veh/h)')
axs[2].set_ylabel('density (veh/km)')
axs[3].set_axis_off()
axs[4].set_ylabel('origin flows (veh/h)')
axs[5].set_ylabel('queue length (veh)')


for ax in axs:
    ax.sharex(axs[0])
    ax.set_xlabel('time (h)')
    ax.set_xlim(0, Tfin)
    ax.set_ylim(0, ax.get_ylim()[1])
    ax.legend()

plt.show()
# fig.savefig('test.eps')
