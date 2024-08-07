#!/usr/bin/python3
"""
Circular pumping trajectory for the 6DOF megAWES reference rigid-wing aircraft.

Aircraft dimensions adapted from:
"Six-degrees-of-freedom simulation model for future multi-megawatt airborne wind energy systems",
Dylan Eijkelhof, Roland Schmehl
Renewable Energy, Vol.196, pp. 137-150, 2022.

Aerodynamic model and constraints from BORNE project (Ghent University, UCLouvain, 2024)

:author: Thomas Haas, Ghent University, 2024 (adapted from Jochem De Schutter)
"""

import awebox as awe
from megawes_settings import set_megawes_path_generation_settings
import matplotlib.pyplot as plt
import numpy as np

# ----------------- user-specific options ----------------- #

# indicate aerodynamic model of aircraft: options are 'VLM', 'ALM', and 'CFD'
aero_model = 'VLM'

# indicate desired system architecture
options = {}
options['user_options.system_model.architecture'] = {1:0}
options = set_megawes_path_generation_settings(aero_model, options)

# indicate desired operation mode
options['user_options.trajectory.type'] = 'power_cycle'
options['user_options.trajectory.system_type'] = 'lift_mode'
options['user_options.trajectory.lift_mode.phase_fix'] = 'single_reelout' # positive (or null) reel-out speed during power generation
options['user_options.trajectory.lift_mode.windings'] = 1 # number of loops
options['model.system_bounds.theta.t_f'] = [1., 20.] # cycle period [s]

# indicate desired wind environment
options['user_options.wind.model'] = 'log_wind'
options['user_options.wind.u_ref'] = 12.
options['params.wind.z_ref'] = 100.
options['params.wind.log_wind.z0_air'] = 0.0002

# indicate numerical nlp details
options['nlp.n_k'] = 40 # approximately 40 per loop
options['nlp.collocation.u_param'] = 'zoh' # constant control inputs
options['solver.linear_solver'] = 'ma57' # if HSL is installed, otherwise 'mumps'
options['nlp.collocation.ineq_constraints'] = 'collocation_nodes' # default is 'shooting_nodes'

# ----------------- solve OCP ----------------- #

# build and optimize the NLP (trial)
trial = awe.Trial(options, 'MegAWES')
trial.build()
trial.optimize()
trial.write_to_csv('megawes_trajectory_'+aero_model, rotation_representation='euler')

# plot results
trial.plot(['isometric', 'states', 'controls', 'constraints'])

# extract information from the solution for independent plotting or post-processing
# here: plot relevant system outputs, compare to [Licitra2019, Fig 11].
plot_dict = trial.visualization.plot_dict
outputs = plot_dict['outputs']
time = plot_dict['time_grids']['ip']
avg_power = plot_dict['power_and_performance']['avg_power']/1e3
print('======================================')
print('Average power: {} kW'.format(avg_power))
print('======================================')

