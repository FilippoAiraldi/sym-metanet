# Traffic Modelling

Traffic Modelling (TM) is a Python collection of tools to model traffic 
networks. TM is free software; you can redistribute it and/or modify it under 
the terms of the GNU General Public License as published by the Free Software Foundation; either version 3, or (at your option) any later version.

TM is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
PARTICULAR PURPOSE.

## Installation

Currently, no `*setup.py` file is implemented. The library must be included 
in the Python path and then imported.

To use TM, you will need a recent versions of

* Numpy
* CasADi ([download here](http://files.casadi.org))

## Supported Frameworks

The currently supported traffic modelling frameworks are

* METANET

##  Usage

The TM package contains the methods needed to simulate a discrete system representing a traffic network. An explanatory example of such a network is 
represented below:

![A small traffic network](/img/example.png "A small traffic network")

Here, an on-ramp is connected to the traffic main road, which later downstream experiences a lane drop.

The code to simulate this network with the METANET framework is explained below. First, all necessary imports

```python
import numpy as np
import casadi as cs

import sys
sys.path.append('path\to\traffic-modelling')
from trafficmodelling import metanet, util as tm_util
```
Then, we can initialize the METANET model with all its parameters. The <code>metanet.init</code> must be called to initialize the internal variables, otherwise an error will be thrown.
```python
# simulation
Tfin = 4                        # final simulation time (h)
T = 10 / 3600                   # simulation step size (h)
t = np.arange(0, Tfin + T, T)   # time vector (h)
K = t.size                      # simulation steps

# segments
L = np.array([2, 2, 1, 0.5])    # length of links (km)
lanes = np.array([3, 3, 3, 2])  # lanes per link (adim)
I = L.size                      # number of segments

# on-ramp
C0 = 1500                       # on-ramp links capacity (veh/h/lane)
O = 1                           # number of on-ramps

# model parameters
tau = 18                        # model parameter (s)
kappa = 40                      # model parameter (veh/km/lane)
eta = 60                        # model parameter (km^2/lane)
rho_max = 180                   # maximum capacity (veh/km/lane)
delta = 0.0122                  # on-ramp merging model parameter (adim)
phi = 2.98                      # lane drop model parameter (adim)
rho_crit = 33.5                 # critical capacity (veh/km/lane)
v_free = 110                    # free flow speed (km/h)
a = 1.636                       # model parameter (adim)

# initialize METANET functions
metanet.init(metanet.Config(
    O=O, I=I, C0=C0, v_free=v_free, rho_crit=rho_crit,
    rho_max=rho_max, a=a, delta=delta, eta=eta, kappa=kappa,
    tau=tau, phi=phi, lanes=lanes, L=L, T=T))
```
Finally, we can write the discrete state function
$x(k+1) = f \left( x(k), u(k), d_{ramp}(k), d_{link}(k) \right)$,
where 
* $x(k)$ and $x(k + 1)$ are the current and next states, respectively (here the state collects the ramp's queue and links' densities and speeds 
$x = [\omega^T, \rho^T, v^T]^T$)
* $u(k) \in [0,1]$ is the control action (the metering rate of the ramp in this case)
* $d_{ramp}(k)$ and $d_{link}(k)$ are the disturbance flows entering the ramp 
and the boundary at the first link, respectively
```python
@tm_util.force_inputs_2d
def f(x, u, d_ramp, d_link):
    # unpack
    w, rho, v = metanet.F.util.x2q(x)

    # step ramp queues
    rho_first = rho[1]
    r = metanet.F.ramps.get_flow(u, w, d_ramp, rho_first)
    w_next = metanet.F.ramps.step_queue(w, d_ramp, r)

    # step link densities
    q = metanet.F.links.get_flow(rho, v)
    q_up = cs.vertcat(d_link, q[:-1])
    r_ramps = cs.vertcat(0, r, 0, 0)
    s_ramps = np.zeros(r_ramps.shape)
    rho_next = metanet.F.links.step_density(rho, q, q_up, r_ramps, s_ramps)

    # step link velocities
    v_up = cs.vertcat(metanet.F.links.get_upstream_speed(v[0]), v[:-1])
    rho_down = cs.vertcat(rho[1:],
                          metanet.F.links.get_downstream_density(rho[-1]))
    Veq = metanet.F.links.get_Veq(rho)
    v_next = metanet.F.links.step_speed(v, v_up, Veq, rho, rho_down, r_ramps)

    return metanet.F.util.q2x(w_next, rho_next, v_next)
```

If we then call $f(\cdot)$ iteratively in a loop, we can compute the trajectory 
of the state $x$, given the input $u$, thus simulating the behaviour 
of the model according to METANET.

---
## Authors

Filippo Airaldi  
<f.airaldi@tudelft.nl>   
Delft University of Technology - Delft Center for Systems and Control
