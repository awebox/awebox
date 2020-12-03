# awebox

`awebox` is a Python toolbox for modelling and optimal control of multiple-kite systems for Airborne Wind Energy (AWE). It provides interfaces that aim to take away from the user the burden of

* generating optimization-friendly system dynamics for different combinations of modeling options.
* formulating optimal control problems for common multi-kite trajectory types.
* solving the trajectory optimization problem reliably
* postprocessing and visualizing the solution and performing quality checks 
* tracking MPC design and handling for offline closed-loop simulations

The main focus of the toolbox are _rigid-wing_, _lift_- and _drag_-mode multiple-kite systems.

## Installation

`awebox` runs on Python 3. It depends heavily on the modeling language CasADi, which is a symbolic framework for algorithmic differentiation. CasADi also provides the interface to the NLP solver IPOPT.  
It is optional but highly recommended to use HSL linear solvers as a plugin with IPOPT.

1.   Get a local copy of the latest `awebox` release:

     ```
     git clone https://github.com/awebox/awebox.git
     ```

2.   Install CasADI version **3.5** for Python 3, following these [installation instructions](https://github.com/casadi/casadi/wiki/InstallationInstructions).

3.   In order to get the HSL solvers and render them visible to CasADi, follow these [instructions](https://github.com/casadi/casadi/wiki/Obtaining-HSL).

Additional installation instructions can be found [here](https://github.com/awebox/awebox/blob/develop/INSTALLATION.md).


## Getting started

Add awebox to the PYTHONPATH environment variable (add those lines to your .bashrc or .zshrc to set the paths permanently).

```
export PYTHONPATH=<path_to_awebox_root_folder>:$PYTHONPATH
```


To run one of the examples from the `awebox` root folder:

```
python3 examples/single_kite_lift_mode_simple.py
```

## Options

For an overview of the available options, first have a look at the different examples.  
An exhaustive overview can be found in `awebox/opts/default.py`.
In order to alter non-user options: generate the `Options`-object with internal access rights switched on:

```python
import awebox as awe
options = awe.Options(internal_access = True)
```

and set the according fields in the `Options`-subdicts to the desired values.

## Acknowledgments

This software has been developed in collaboration with the company [Kiteswarms Ltd](http://www.kiteswarms.com). The company has also supported the project through research funding.

This project has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Sklodowska-Curie grant agreement No 642682 (_AWESCO_)

## Citing `awebox`
Please use the following citation: 

"_awebox: Modelling and optimal control of single- and multiple-kite systems for airborne wind energy. https://github.com/awebox/awebox_"

## Literature

### `awebox`-based research

[Optimal Control of Stacked Multi-Kite Systems for Utility-Scale Airborne Wind Energy](https://cdn.syscop.de/publications/DeSchutter2019.pdf) \
De Schutter et al. / IEEE Conference on Decision and Control (CDC) 2019

[Wake Characteristics of Pumping Mode Airborne Wind Energy Systems](https://cdn.syscop.de/publications/Haas2019.pdf) \
Haas et al. / Journal of Physics: Conference Series 2019

[Operational Regions of a Multi-Kite AWE System](https://cdn.syscop.de/publications/Leuthold2018.pdf) \
Leuthold et al. / European Control Conference (ECC) 2018

[Optimal Control for Multi-Kite Emergency Trajectories](https://cdn.syscop.de/publications/Bronnenmeyer2018.pdf) \
Bronnenmeyer (Masters thesis) / University of Stuttgart 2018

### Models

**Induction models**\
[Engineering Wake Induction Model For Axisymmetric Multi-Kite Systems](https://www.researchgate.net/publication/334616920_Engineering_Wake_Induction_Model_For_Axisymmetric_Multi-Kite_Systems) \
Leuthold et al. / Wake Conference 2019

**Point-mass model**\
[Airborne Wind Energy Based on Dual Airfoils](https://cdn.syscop.de/publications/Zanon2013a.pdf) \
Zanon et al. / IEEE Transactions on Control Systems Technology 2013

### Methods

**Homotopy strategy** \
[A Relaxation Strategy for the Optimization of Airborne Wind Energy Systems](https://cdn.syscop.de/publications/Gros2013a.pdf) \
Gros et al. / European Control Conference (ECC) 2013

**Trajectory optimization** \
[Numerical Trajectory Optimization for Airborne Wind Energy Systems Described by High Fidelity Aircraft Models](https://cdn.syscop.de/publications/Horn2013.pdf) \
Horn et al. / Airborne Wind Energy 2013

### Software

**IPOPT**\
[On the Implementation of a Primal-Dual Interior Point Filter Line Search Algorithm for Large-Scale Nonlinear Programming](http://cepac.cheme.cmu.edu/pasilectures/biegler/ipopt.pdf) \
Wächter et al. / Mathematical Programming 106 (2006) 25-57

**CasADi**\
[CasADi - A software framework for nonlinear optimization and optimal control](https://cdn.syscop.de/publications/Andersson2018.pdf) \
Andersson et al. / Mathematical Programming Computation 2018