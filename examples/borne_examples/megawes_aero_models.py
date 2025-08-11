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
import numpy as np
import matplotlib.pyplot as plt

# ----------------- user-specific options ----------------- #

# indicate desired system architecture
options = {}
options['user_options.system_model.architecture'] = {1:0}
options = set_megawes_path_generation_settings('VLM', options)

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

# ----------------- 1. generate path with VLM aero model ----------------- #

# set aero model (options are 'VLM', 'ALM', and 'CFD')
options['user_options.kite_standard'] = awe.megawes_data.data_dict(aero_model='VLM')

# build and optimize the NLP
trial = awe.Trial(options, 'MegAWES')
trial.build()
trial.optimize()
avg_power = [trial.visualization.plot_dict['power_and_performance']['avg_power'].full()[0][0]/1e6]
q_vlm = np.array(trial.visualization.plot_dict['x']['q10']).T
time_vlm = trial.visualization.plot_dict['time_grids']['ip']
p_current_vlm = trial.visualization.plot_dict['outputs']['performance']['p_current'][0]

# ----------------- 2. generate path with ALM aero model ----------------- #

# set aero model (options are 'VLM', 'ALM', and 'CFD')
options['user_options.kite_standard'] = awe.megawes_data.data_dict(aero_model='ALM')

# optimize the NLP
trial.optimize(options)
avg_power += [trial.visualization.plot_dict['power_and_performance']['avg_power'].full()[0][0]/1e6]
q_alm = np.array(trial.visualization.plot_dict['x']['q10']).T
time_alm = trial.visualization.plot_dict['time_grids']['ip']
p_current_alm = trial.visualization.plot_dict['outputs']['performance']['p_current'][0]

# ----------------- 3. generate path with CFD aero model ----------------- #

# set aero model (options are 'VLM', 'ALM', and 'CFD')
options['user_options.kite_standard'] = awe.megawes_data.data_dict(aero_model='CFD')

# build and optimize the NLP
trial.optimize(options)
avg_power += [trial.visualization.plot_dict['power_and_performance']['avg_power'].full()[0][0]/1e6]
q_cfd = np.array(trial.visualization.plot_dict['x']['q10']).T
time_cfd = trial.visualization.plot_dict['time_grids']['ip']
p_current_cfd = trial.visualization.plot_dict['outputs']['performance']['p_current'][0]

# ----------------- specific plots ----------------- #

# legend labels
legend_labs = []
for k, (model, power) in enumerate(zip(['VLM', 'ALM', 'CFD'], avg_power)):
    legend_labs += ["model: "+model+", P="+"{:.2f}".format(power)+"MW"]

# plot 3D flight path
fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
fig.set_size_inches(8,8)
fig.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95)
ax.plot(q_vlm[:,0], q_vlm[:,1], q_vlm[:,2])
ax.plot(q_alm[:,0], q_alm[:,1], q_alm[:,2])
ax.plot(q_cfd[:,0], q_cfd[:,1], q_cfd[:,2])
ax.tick_params(labelsize=12)
ax.set_xlabel(ax.get_xlabel(), fontsize=12)
ax.set_ylabel(ax.get_ylabel(), fontsize=12)
ax.set_zlabel(ax.get_zlabel(), fontsize=12)
ax.set_xlim([0,600])
ax.set_ylim([-300,300])
ax.set_zlim([0,400])
ax.view_init(azim=-70, elev=20)
l = ax.get_lines()
l[0].set_color('b')
l[-2].set_color('g')
l[-1].set_color('r')
ax.legend([l[0],l[-2],l[-1]], legend_labs, fontsize=12)
fig.suptitle("")
fig.savefig('outputs_megawes_aero_models_plot_3dpath.png')

# plot power profile
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
fig.subplots_adjust(top=0.95, bottom=0.1, left=0.15, right=0.95)
ax.plot(time_vlm, 1e-6*p_current_vlm, 'b')
ax.plot(time_alm, 1e-6*p_current_alm, 'g')
ax.plot(time_cfd, 1e-6*p_current_cfd, 'r')
ax.legend(legend_labs, fontsize=12)
ax.tick_params(axis='both', labelsize=12)
ax.set_xlabel('t [s]', fontsize=12)
ax.set_ylabel('P [MW]', fontsize=12)
ax.grid()
fig.savefig('outputs_megawes_aero_models_plot_power.png')
plt.show()

