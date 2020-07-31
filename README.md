# awebox

`awebox` is a Python toolbox for modelling and optimal control of multiple-kite systems for Airborne Wind Energy (AWE). It provides interfaces that aim to take away from the user the burden of

* generating optimization-friendly system dynamics for different combinations of modeling options.
* formulating optimal control problems for common multi-kite trajectory types.
* solving the optimization problem reliably
* postprocessing the solution and performing quality checks 

At the moment, the main focus of the toolbox are _rigid-wing_, _lift-mode_ multiple-kite systems.

## Installation

`awebox` runs on Python 3. It depends heavily on the modeling language CasADi, which is a symbolic framework for algorithmic differentiation. CasADi also provides the interface to the NLP solver IPOPT.  
It is optional but highly recommended to use HSL linear solvers as a plugin with IPOPT.

1.   Get a local copy of the latest `awebox` release:

     ```
     git clone https://github.com/awebox/awebox.git
     ```

2.   Install CasADI version **3.4.5** for Python 3, following these [installation instructions](https://github.com/casadi/casadi/wiki/InstallationInstructions).

3.   In order to get the HSL solvers and render them visible to CasADi, follow these [instructions](https://github.com/casadi/casadi/wiki/Obtaining-HSL).

##### possible variations to the HSL installation instructions, for linux installation in the /usr/ directory, using anaconda: 
 
- At Instruction  6.  (to avoid Step 2 and the use of LD_LIBRARY_PATH to let IPOPT know where to find libhsl.so, which does not seem to work):
```
./configure --prefix=/usr LIBS="-llapack" --with-blas="-L/usr/lib -lblas" CXXFLAGS="-g -O2 -fopenmp" FCFLAGS="-g -O2 -fopenmp" CFLAGS="-g -O2 -fopenmp"
```
 
- At Instruction 7. (very important, as the metis library is not being linked otherwise) :
```
make LDFLAGS="-lmetis"  
```
 
- At Instruction 8. (because of /usr chosen as install directory)  :
```
sudo make install
```
 

- At Instruction 9. :
```
ln -s usr/lib/libcoinhsl.so (anaconda folder where CasaDI is installed)/libhsl.so
```
The anaconda folder may possibly be at: /home/user_name/anaconda3/lib/python3.7/site-packages/casadi



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

For an overview of the different (user and non-user) options, first have a look at the examples.  
An exhaustive overview can be found in `awebox/opts/default.py`, where all the default options are set.  
In order to alter non-user options: generate the `Options`-object with internal access rights switched on:

```python
import awebox as awe
options = awe.Options(internal_access = True)
```

and set the according fields in the `Options`-subdicts to the desired values.

## Acknowledgments

This software has been developed in collaboration with the company [Kiteswarms Ltd](http://www.kiteswarms.com). The company has also supported the project through research funding.

This project has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Sklodowska-Curie grant agreement No 642682 (_AWESCO_)

## How to cite the `awebox`
Please cite the `awebox` using the following citation: 

```
awebox: Modelling and optimal control of single- and multiple-kite systems for airborne wind energy. https://github.com/awebox/awebox
```

## Literature

### `awebox`-based research

Operational Regions of a Multi-Kite AWE System \
R. Leuthold, J. De Schutter, E Malz, G. Licitra, S. Gros, M. Diehl \
European Control Conference (ECC) 2018

Optimal Control for Multi-Kite Emergency Trajectories \
T. Bronnenmeyer (Masters thesis) \
University of Stuttgart 2018

### Models

**Induction models**\
Engineering Wake Induction Model For Axisymmetric Multi-Kite Systems \
R. Leuthold, C. Crawford, S. Gros, M. Diehl \
Wake Conference 2019 (accepted)

**Point-mass model**\
Airborne Wind Energy Based on Dual Airfoils \
M. Zanon, S. Gros, J. Andersson, M. Diehl \
IEEE Transactions on Control Systems Technology 2013

### Methods

**Homotopy strategy** \
A Relaxation Strategy for the Optimization of Airborne Wind Energy Systems \
S. Gros, M. Zanon, M. Diehl \
Proceedings of the European Control Conference (ECC) 2013

**Trajectory optimization** \
Numerical Trajectory Optimization for Airborne Wind Energy Systems Described by High Fidelity Aircraft Models \
G. Horn, S. Gros, M. Diehl \
Airborne Wind Energy 2013

### Software

**IPOPT**\
On the Implementation of a Primal-Dual Interior Point Filter Line Search Algorithm for Large-Scale Nonlinear Programming \
A. Wächter, L.T. Biegler \
Mathematical Programming 106 (2006) 25-57

**CasADi**\
CasADi - A software framework for nonlinear optimization and optimal control \
J.A.E. Andersson, J. Gillis, G. Horn, J.B. Rawlings, M. Diehl \
Mathematical Programming Computation, 2018