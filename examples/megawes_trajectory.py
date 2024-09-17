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
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

# ----------------- user-specific options ----------------- #

# indicate aerodynamic model of aircraft
aero_model = 'VLM' # options are 'VLM', 'ALM', and 'CFD'

# indicate desired system architecture
options = {}
options['user_options.system_model.architecture'] = {1:0}
options = set_megawes_path_generation_settings(aero_model, options)

# indicate desired operation mode
options['user_options.trajectory.type'] = 'power_cycle'
options['user_options.trajectory.system_type'] = 'lift_mode'
options['user_options.trajectory.lift_mode.phase_fix'] = 'single_reelout' # positive (or null) reel-out speed during power generation
options['user_options.trajectory.lift_mode.windings'] = 1 # number of loops
options['model.system_bounds.theta.t_f'] = [1., 40.] # cycle period [s]

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
trial.write_to_csv('outputs_megawes_trajectory_'+aero_model.lower()+'_results', rotation_representation='dcm')

# extract information from the solution for independent plotting or post-processing
# here: plot relevant system outputs, compare to [Licitra2019, Fig 11].
plot_dict = trial.visualization.plot_dict
outputs = plot_dict['outputs']
time = plot_dict['time_grids']['ip']
avg_power = plot_dict['power_and_performance']['avg_power']/1e3
print('======================================')
print('Average power: {} kW'.format(avg_power))
print('======================================')

# ----------------- specific plots ----------------- #

# plot 3D flight path
trial.plot(['isometric'])
fig = plt.gcf()
fig.set_size_inches(8,8)
fig.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95)
ax = fig.get_axes()[0]
ax.tick_params(labelsize=12)
ax.set_xlabel(ax.get_xlabel(), fontsize=12)
ax.set_ylabel(ax.get_ylabel(), fontsize=12)
ax.set_zlabel(ax.get_zlabel(), fontsize=12)
ax.set_xlim([0,600])
ax.set_ylim([-200,200])
ax.set_zlim([0,600])
ax.view_init(azim=-70, elev=20)
l = ax.get_lines()
l[0].set_color('b')
ax.get_legend().remove()
ax.legend([l[0]], ['reference ('+aero_model+', P='+'{:.2f}'.format(avg_power.full()[0][0]/1e3)+'MW)'], fontsize=12)
fig.suptitle("")
fig.savefig('outputs_megawes_trajectory_'+aero_model.lower()+'_plot_3dpath.png')

# plot power profile
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
fig.subplots_adjust(top=0.95, bottom=0.1, left=0.15, right=0.95)
ax.plot(trial.visualization.plot_dict['time_grids']['ip'], 1e-6*trial.visualization.plot_dict['outputs']['performance']['p_current'][0], 'b')
ax.legend(['reference ('+aero_model+', P='+'{:.2f}'.format(avg_power.full()[0][0]/1e3)+'MW)'], fontsize=12)
ax.tick_params(axis='both', labelsize=12)
ax.set_xlabel('t [s]', fontsize=12)
ax.set_ylabel('P [MW]', fontsize=12)
ax.grid()
fig.savefig('outputs_megawes_trajectory_'+aero_model.lower()+'_plot_power.png')
plt.show()

