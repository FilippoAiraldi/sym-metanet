# Symbolic Modelling of Highway Traffic Networks with METANET

<div align="center">
  <img src="https://raw.githubusercontent.com/FilippoAiraldi/sym-metanet/dev/resources/example.jpg" alt="network-example" height="180">
</div>

**sym-metanet** is a Python package to model traffic networks with the METANET framework, a collection of tools to mathematically model the macroscopic behaviour of traffic in highway systems (see [[1]](#1) and [[2]](#2) for more details).

[![PyPI version](https://badge.fury.io/py/sym-metanet.svg)](https://badge.fury.io/py/sym-metanet)
[![Source Code License](https://img.shields.io/badge/license-MIT-blueviolet)](https://github.com/FilippoAiraldi/casadi-nlp/blob/release/LICENSE)
![Python 3.8](https://img.shields.io/badge/python->=3.8-green.svg)

[![Tests](https://github.com/FilippoAiraldi/sym-metanet/actions/workflows/ci.yml/badge.svg)](https://github.com/FilippoAiraldi/sym-metanet/actions/workflows/ci.yml)
[![Downloads](https://static.pepy.tech/badge/sym-metanet)](https://www.pepy.tech/projects/sym-metanet)
[![Maintainability](https://api.codeclimate.com/v1/badges/c2725b1b8012a72db289/maintainability)](https://codeclimate.com/github/FilippoAiraldi/sym-metanet/maintainability)
[![Test Coverage](https://api.codeclimate.com/v1/badges/c2725b1b8012a72db289/test_coverage)](https://codeclimate.com/github/FilippoAiraldi/sym-metanet/test_coverage)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## Installation

To install the package, run

```bash
pip install sym-metanet
```

**sym-metanet** has the following dependencies

- Python 3.8 or higher
- [NetworkX](https://networkx.org/)

and optionally

- [NumPy](https://pypi.org/project/numpy/)
- [CasADi](https://pypi.org/project/casadi/).

For playing around with the source code instead, run

```bash
git clone https://github.com/FilippoAiraldi/sym-metanet.git
```

---

## Usage

In METANET, a highway network is represented as a directed graph whose edges are stretches of highways, a.k.a. links. Each link bridges between nodes (that have no physical meaning), which in turn can host origins (sources of traffic flow) and destinations (sinks of flow). For instance, to create a very simple network consisting of a single link connecting an origin to a destination, we can do as follows:

```python
import sym_metanet as metanet

...

N1 = metanet.Node(name="N1")
N2 = metanet.Node(name="N2")
L1 = metanet.Link(segments, lanes, L, rho_max, rho_crit, v_free, a, name="L1")
O1 = metanet.Origin(C[0], name="O1")
D3 = metanet.Destination(name="D3")

net = metanet.Network().add_path(origin=O1, path=(N1, L1, N2, L2, N3), destination=D3)
```

Once the network is built, we can validate it and use, e.g., [CasADi](https://pypi.org/project/casadi/), to symbolically construct a function that represents the dynamics governing this network (according to METANET).

```python
net.is_valid(raises=True)

T = 10 / 3600  # sampling time
metanet.engines.use("casadi", sym_type="SX")
F = metanet.engine.to_function(net=net, T=T)
```

Function `F` can be then used to simulate the state transitions of the network in the context of, e.g., highway traffic control (see [[2]](#2) for more details).

## Examples

Our [examples](examples) folder contains an example on how to get started with this package.

## Extensions

This code is symbolic-engine-agnostic, in the sense that it does not rely on a particular implementation of the underlying engine for symbolic computations. In other words, it is relatively easy to create a new engine for modelling networks with a new specific symbolic library (e.g., [SimPy](https://www.sympy.org/en/index.html)) by implementing the abstract class [`sym_metanet.engines.core.EngineBase`](src\sym_metanet\engines\core.py#EngineBase). An engine implemented in [CasADi](https://pypi.org/project/casadi/) is already shipped but requires the symbolic library to be installed. Additionally, the engine is also implemented in [NumPy](https://pypi.org/project/numpy/) (does not allow symbolic computations though).

---

## License

The repository is provided under the MIT License. See the LICENSE file included with this repository.

---

## Author

[Filippo Airaldi](https://www.tudelft.nl/staff/f.airaldi/), PhD Candidate [f.airaldi@tudelft.nl | filippoairaldi@gmail.com]

> [Delft Center for Systems and Control](https://www.tudelft.nl/en/3me/about/departments/delft-center-for-systems-and-control/) in [Delft University of Technology](https://www.tudelft.nl/en/)

Copyright (c) 2023 Filippo Airaldi.

Copyright notice: Technische Universiteit Delft hereby disclaims all copyright interest in the program “sym-metanet” (Symbolic Modelling of Highway Traffic Networks with METANET) written by the Author(s). Prof. Dr. Ir. Fred van Keulen, Dean of 3mE.

---

## References

<a id="1">[1]</a>
A. Messmer and Papageorgiou, “METANET: a macroscopic simulation program for motorway networks,” Traffic Engineering and Control, vol. 31, pp. 466–470, 1990.

<a id="2">[2]</a>
Hegyi, A., 2004. Model predictive control for integrating traffic control measures. Netherlands TRAIL Research School.
