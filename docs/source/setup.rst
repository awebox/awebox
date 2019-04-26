
==================================================
Getting started
==================================================

Installation
------------

AWEbox runs on Python 2.7. It depends heavily on the modeling language CasADi, which is a symbolic framework for algorithmic differentiation. CasADi also provides the interface to the NLP solver IPOPT.  
It is optional but highly recommended to use HSL linear solvers as a plugin with IPOPT.

1.   Get a local copy of the latest AWEbox release:

    .. code-block:: bash

        $ git clone git@gitlab.syscop.de:Jochem.De.Schutter/AWEbox.git

2.   Install CasADI version **3.4.5** for Python 2.7, following these `installation instructions <https://github.com/casadi/casadi/wiki/InstallationInstructions>`_.

3.   In order to get the HSL solvers and render them visible to CasADi, follow these `instructions <https://github.com/casadi/casadi/wiki/Obtaining-HSL>`_.


Getting started
---------------

Grab dependencies by installing

.. code-block:: bash

    $ apt install python-tk
    $ pip install scipy
    $ pip install ipdb


Add  AWEbox to the PYTHONPATH environment variable (add those lines to your .bashrc or .zshrc to set the paths permanently):

.. code-block:: bash

    $ export PYTHONPATH=<path_to_awebox_root_folder>:$PYTHONPATH

To run one of the examples from the AWEbox root folder:

.. code-block:: bash

    $ python examples/single_kite_lift_mode_simple.py


Options
-------

For an overview of the different (user and non-user) options, first have a look at the examples.  
An exhaustive overview can be found in `AWEbox/options_dir/options_default.py`, where all the default options are set.  
In order to alter non-user options: generate the `Options`-object with internal access rights switched on: ::

    import AWEbox as awe
    options = awe.Options(internal_access = True)

and set the according fields in the `Options`-subdicts to the desired values.
