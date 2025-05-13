#!/usr/bin/python3
"""
Circular pumping trajectory for the Ampyx AP2 aircraft.
Model and constraints as in:

"Performance assessment of a rigid wing Airborne Wind Energy pumping system",
G. Licitra, J. Koenemann, A. Bürger, P. Williams, R. Ruiterkamp, M. Diehl
Energy, Vol.173, pp. 569-585, 2019.

:author: Jakob Harzer
"""

from typing import List, Dict
import awebox as awe
import awebox.opts.kite_data.ampyx_ap2_settings as ampyx_ap2_settings
from examples.paper_benchmarks.reference_options import set_reference_options
import matplotlib.pyplot as plt
import numpy as np

# set the logger level to 'DEBUG' to see IPOPT output
from awebox.logger.logger import Logger as awelogger
awelogger.logger.setLevel(10)

# dual kite example?
DUAL_KITES = False

# indicate desired system architecture
# here: single kite with 6DOF Ampyx AP2 model
options = {}


options['user_options.system_model.architecture'] = {1: 0}
# options = ampyx_ap2_settings.set_ampyx_ap2_settings(options)
options = set_reference_options(options)
options['user_options.trajectory.lift_mode.phase_fix'] = 'single_reelout'

# # indicate desired operation mode
# # here: lift-mode system with pumping-cycle operation, with a one winding trajectory
# options['user_options.trajectory.type'] = 'power_cycle'
# options['user_options.trajectory.system_type'] = 'lift_mode'

#
# # indicate desired environment
# # here: wind velocity profile according to power-law
# options['params.wind.z_ref'] = 100.0
# options['params.wind.power_wind.exp_ref'] = 0.15
# options['user_options.wind.model'] = 'power'
# options['user_options.wind.u_ref'] = 10.

# larger kite?
# bref = options['user_options.kite_standard']['geometry']['b_ref']
# mref = options['user_options.kite_standard']['geometry']['m_k']
# jref = options['user_options.kite_standard']['geometry']['j']
# kappa = 2.4
# b = 5.5
# options['user_options.kite_standard']['geometry']['b_ref'] = b
# options['user_options.kite_standard']['geometry']['s_ref'] = b ** 2 / options['user_options.kite_standard']['geometry'][
#     'ar']
# options['user_options.kite_standard']['geometry']['c_ref'] = b / options['user_options.kite_standard']['geometry']['ar']
# options['user_options.kite_standard']['geometry']['m_k'] = mref * (b / bref) ** kappa
# options['user_options.kite_standard']['geometry']['j'] = jref * (b / bref) ** (kappa + 2)
# options['user_options.trajectory.fixed_params'] = {} # the tether diameter is fixed in the AmpyxAP2 problem, we free it again
# # options['user_options.trajectory.fixed_params'] = {'diam_t': 8e-3}

# print the just set options:
# print(f"b_ref:{options['user_options.kite_standard']['geometry']['b_ref']}")
# print(f"s_ref:{options['user_options.kite_standard']['geometry']['s_ref']}")
# print(f"c_ref:{options['user_options.kite_standard']['geometry']['c_ref']}")
# print(f"m_k:{options['user_options.kite_standard']['geometry']['m_k']}")
# print(f"j:{options['user_options.kite_standard']['geometry']['j']}")
# print(f"diam_t:{options['user_options.trajectory.fixed_params']['diam_t']}")

# indicate numerical nlp details
# here: nlp discretization, with a zero-order-hold control parametrization, and a simple phase-fixing routine. also, specify a linear solver to perform the Newton-steps within ipopt.
# (experimental) set to "True" to significantly (factor 5 to 10) decrease construction time
# note: this may result in slightly slower solution timings
options['nlp.compile_subfunctions'] = False
# smooth the reel in phase (this increases convergence speed x10)
options['nlp.cost.beta'] = False # penalize side-slip (can improve convergence)
options['model.integration.method'] = 'constraints'  # use enery as a state, works better with SAM

# indicate numerical nlp details
options['nlp.SAM.use'] = True
options['nlp.SAM.N'] = 5 # the number of full cycles approximated
options['nlp.SAM.d'] = 3 # the number of cycles actually computed

# SAM Regularization
single_regularization_param = 1E-1
options['nlp.SAM.Regularization.AverageStateFirstDeriv'] = 1E1*single_regularization_param
options['nlp.SAM.Regularization.AverageStateThirdDeriv'] = 1E0*single_regularization_param
options['nlp.SAM.Regularization.AverageAlgebraicsThirdDeriv'] = 0*single_regularization_param
options['nlp.SAM.Regularization.SimilarMicroIntegrationDuration'] = 1E-2*single_regularization_param

# Number of discretization points
n_k = 15 * (options['nlp.SAM.d']) * 2
options['nlp.n_k'] = n_k

# initialization
# options['solver.initialization.l_t'] = 200.

# model bounds
# options['model.system_bounds.x.dl_t'] = [-50.0, 20.0]  # [m/s]=
# options['model.system_bounds.x.l_t'] = [10.0, 2500.0]  # [m]
# options['model.system_bounds.x.q'] = [np.array([0, -np.inf, 10.0]), np.array([np.inf, np.inf, np.inf])]  # [m]
if DUAL_KITES:
    options['model.system_bounds.theta.t_f'] = [5, 10 * options['nlp.SAM.N']]  # [s]
else:
    options['model.system_bounds.theta.t_f'] = [50, 150 + options['nlp.SAM.N'] * 30]  # [s]



options['solver.linear_solver'] = 'ma27'
options['visualization.cosmetics.interpolation.n_points'] = 100* options['nlp.SAM.N'] # high plotting resolution

# build and optimize the NLP (trial)
trial = awe.Trial(options, 'DualKitesLongHorizon')
trial.build()
trial.optimize()



# %% Postprocessing and plotting preparations
from awebox.tools.struct_operations import calculate_SAM_regions

plot_dict_SAM = trial.visualization.plot_dict_SAM
plot_dict_REC = trial.visualization.plot_dict
time_plot_SAM = plot_dict_SAM['time_grids']['ip']
ip_regions_SAM = plot_dict_SAM['SAM_regions_ip']
time_grid_SAM = plot_dict_SAM['time_grids']
time_grid_SAM_x = time_grid_SAM['x']
regions_indeces = calculate_SAM_regions(trial.nlp.options)
avg_power_REC = plot_dict_SAM['power_and_performance']['avg_power'] / 1e3
avg_power_SAM = plot_dict_REC['power_and_performance']['avg_power'] / 1e3

print('======================================')
print('Average power SAM: {} kW'.format(avg_power_SAM))
print('Average power REC: {} kW'.format(avg_power_REC))
print('======================================')

# print the costs:
cost_dict = trial.visualization.plot_dict['cost']
print('\n======================================')
print('Costs:')
for key, value in cost_dict.items():
    val = float(value)
    if np.abs(val) > 1e-10:
        print(f'\t {key}:  {val:0.4f}')
print('======================================')

# %% plot integral states
if options['model.integration.method'] != 'constraints':
    import casadi as ca
    e_opt = ca.vertcat(*trial.solution_dict['integral_output_vals']['opt']['int_out',:,'e']).full().flatten()
    # betaI_opt = ca.vertcat(*trial.solution_dict['integral_output_vals']['opt']['int_out',:,'beta']).full().flatten()


    plt.figure()
    plt.plot(plot_dict_SAM['time_grids']['x'],e_opt,'o-')
    # plt.plot(plot_dict_SAM['time_grids']['x'],betaI_opt,'o-')
    plt.xlim([0,plot_dict_SAM['time_grids']['x'][-1]])
    plt.ylim([0,np.max(e_opt)])
    plt.show()

# %% constraints
trial.plot('constraints')


# %% Plot Invariants
import casadi as ca
time_plot_REC = plot_dict_REC['time_grids']['ip']
invariants_REC = plot_dict_REC['outputs']['invariants']

time_plot_SAM_xcoll = plot_dict_SAM['time_grids']['x_coll']

x_coll_output_vals = plot_dict_SAM['output_vals']['opt']
x_coll_invariants = trial.model.outputs.repeated(x_coll_output_vals.full())

# get the keys of the invariants to plot
invariants_to_plot = [key for key in trial.model.outputs_dict['invariants'].keys() if
                                key.startswith(tuple(['c', 'dc', 'orthonormality']))]

plt.figure()
for region_index in np.arange(0, trial.options['nlp']['SAM']['d']+1):
    indices = np.where(plot_dict_SAM['SAM_regions_x_coll'][0:-1] == region_index)[0]
    for index,key in enumerate(invariants_to_plot):
        _invariants = ca.vertcat(*x_coll_invariants[:,'invariants',key]).full()
        plt.plot(time_plot_SAM_xcoll[indices], np.abs(_invariants[indices]), f'C{index}.-')

for index,key in enumerate(invariants_to_plot):
    plt.plot([],[],f'C{index}.-',label=key)
plt.plot([],[],'k-',alpha=0.3,label='REC')

# add phase switches
for region in regions_indeces:
    plt.gca().axvspan(time_grid_SAM_x[region[0]], time_grid_SAM_x[region[-1]]+(time_grid_SAM_x[region[-1]]-time_grid_SAM_x[region[-2]]), color='k', alpha=0.1)

plt.xlim([0,time_plot_REC[-1]])
plt.title(f"Invariants: {invariants_to_plot}")
plt.legend()
plt.yscale('log')
plt.xlabel('Time [s]')
plt.show()


# %% Plot the states
plt.figure(figsize=(10, 10))
t_ip = plot_dict_REC['time_grids']['ip']

# decide which states to plot
kite_name_to_plot = 'q21' if DUAL_KITES else 'q10'
# plot_states = [kite_name_to_plot, f'd{kite_name_to_plot}', 'l_t', 'dl_t', 'e']
plot_states = [kite_name_to_plot, f'd{kite_name_to_plot}', 'l_t', 'dl_t']

for index, state_name in enumerate(plot_states):
    plt.subplot(3, 2, index + 1)
    state_traj = np.vstack([plot_dict_SAM['x'][state_name][i] for i in range(plot_dict_SAM['x'][state_name].__len__())]).T

    d = trial.options['nlp']['SAM']['d']
    for region_index in range(d+1):
        plt.plot(time_grid_SAM['ip'][np.where(ip_regions_SAM == region_index)],
                    state_traj[np.where(ip_regions_SAM == region_index)],  '-')
        plt.gca().set_prop_cycle(None)  # reset color cycle

    plt.plot([], [], label=state_name)

    plt.gca().set_prop_cycle(None)  # reset color cycle

    state_recon = np.vstack([plot_dict_REC['x'][state_name][i] for i in range(plot_dict_REC['x'][state_name].__len__())]).T
    plt.plot(t_ip, state_recon, label=state_name + '_recon', linestyle='--')

    # add phase switches
    for region in regions_indeces:
        plt.axvline(x=time_grid_SAM_x[region[0]],color='k',linestyle='--')
        plt.axvline(x=time_grid_SAM_x[region[-1]]+(time_grid_SAM_x[region[-1]]-time_grid_SAM_x[region[-2]]),color='k',linestyle='--')
    plt.xlabel('time [s]')

    plt.legend()
plt.tight_layout()
plt.show()
# %% plot the results
import mpl_toolkits.mplot3d as a3

plt.figure(figsize=(10, 10))
ax = plt.axes(projection='3d')

# plot the reconstructed trajectory
q_kite = plot_dict_REC['x'][kite_name_to_plot]
ax.plot3D(q_kite[0], q_kite[1], q_kite[2], 'C0-', alpha=0.3)

# plot the average reel-out trajcetory
Q_opt = plot_dict_SAM['X'][kite_name_to_plot]
time_X = plot_dict_SAM['time_grids']['ip_X']
ax.plot3D(Q_opt[0], Q_opt[1], Q_opt[2], 'C0--', alpha=1)

# plot the individual micro-integration
q_kite_SAM = plot_dict_SAM['x'][kite_name_to_plot]
ip_regions_SAM = plot_dict_SAM['SAM_regions_ip']

for region_index, color in zip(np.arange(0, trial.options['nlp']['SAM']['d']+1), [f'C{i}' for i in range(20)]):
    ax.plot3D(q_kite_SAM[0][np.where(ip_regions_SAM == region_index)],
              q_kite_SAM[1][np.where(ip_regions_SAM == region_index)],
              q_kite_SAM[2][np.where(ip_regions_SAM == region_index)]
              , '-', color=color,
                  alpha=1, markersize=3)


# set bounds for nice view
q10_REC_all = np.vstack([q_kite[0], q_kite[1], q_kite[2]])
meanpos = np.mean(q10_REC_all, axis=1)
bblenght = np.max(np.abs(q10_REC_all - meanpos.reshape(3, 1)))
ax.set_xlim3d(meanpos[0] - bblenght, meanpos[0] + bblenght)
ax.set_ylim3d(meanpos[1] - bblenght, meanpos[1] + bblenght)
ax.set_zlim3d(meanpos[2] - bblenght, meanpos[2] + bblenght)

# add an arrow for the wind
ax.quiver(meanpos[0] - bblenght / 2, meanpos[1] - bblenght / 2, meanpos[2] - bblenght, 1, 0, 0, length=40, color='g')
ax.text(meanpos[0] - bblenght / 2, meanpos[1] - bblenght / 2, meanpos[2] - bblenght, "Wind", 'x', color='g', size=15)

# labels and view
ax.set_xlabel(r'$x$ in m')
ax.set_ylabel(r'$y$ in m')
ax.set_zlabel(r'$z$ in m')
ax.view_init(elev=23., azim=-45)
plt.tight_layout()
plt.show()
