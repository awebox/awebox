# AWEbox

[![build](https://github.com/awebox/awebox/actions/workflows/python-app.yml/badge.svg?branch=master)](https://github.com/awebox/awebox/actions/workflows/python-app.yml)
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)

AWEbox is a Python toolbox for modelling and optimal control of multiple-kite systems for Airborne Wind Energy (AWE). It provides interfaces that aim to take away from the user the burden of

* generating optimization-friendly system dynamics for different combinations of modeling options.
* formulating optimal control problems for common multi-kite trajectory types.
* solving the trajectory optimization problem reliably
* postprocessing and visualizing the solution and performing quality checks 
* tracking MPC design and solver generation for (mostly offline) closed-loop simulations

The main focus of the toolbox are _rigid-wing_, _lift_- and _drag_-mode multiple-kite systems.

Single-kite optimal trajectory             |  Dual-kite optimal trajectory (reel-out)
:-------------------------:|:-------------------------:
<img src="https://github.com/jdeschut/awebox/blob/README-figures/docs/single_kite.png" width="400">  |  <img src="https://github.com/jdeschut/awebox/blob/README-figures/docs/dual_kites.png" width="400">


## Installation

`awebox` runs on Python 3. It depends heavily on the modeling language CasADi, which is a symbolic framework for algorithmic differentiation. CasADi also provides the interface to the NLP solver IPOPT.  
It is optional but highly recommended to use HSL linear solvers as a plugin with IPOPT.

1.   Get a local copy of the latest `awebox` release:

     ```
     git clone https://github.com/awebox/awebox.git
     ```

2.   Install using pip

     ```
     pip3 install awebox/
     ```

3.   In order to get the HSL solvers and render them visible to CasADi, follow these [instructions](https://github.com/casadi/casadi/wiki/Obtaining-HSL). Additional installation instructions can be found [here](https://github.com/awebox/awebox/blob/develop/INSTALLATION.md).


## Getting started

To run one of the examples from the `awebox` root folder:

```
python3 examples/ampyx_ap2_trajectory.py
```

## Acknowledgments

This software has been developed in collaboration with the company Kiteswarms Ltd. The company has also supported the project through research funding.

This project has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Sklodowska-Curie grant agreement No 642682 (_AWESCO_)

## Citing `awebox`
Please use the following citation: 

"_De Schutter, J.; Leuthold, R.; Bronnenmeyer, T.; Malz, E.; Gros, S.; Diehl, M. AWEbox: An Optimal Control Framework for Single- and Multi-Aircraft Airborne Wind Energy Systems. Energies 2023, 16, 1900. https://doi.org/10.3390/en16041900_"
