#!/usr/bin/python3
"""
Circular pumping trajectory for the Ampyx AP2 aircraft.
Model and constraints as in:

"Performance assessment of a rigid wing Airborne Wind Energy pumping system",
G. Licitra, J. Koenemann, A. Bürger, P. Williams, R. Ruiterkamp, M. Diehl
Energy, Vol.173, pp. 569-585, 2019.

:author: Jochem De Schutter
:edited: Rachel Leuthold
"""

from typing import List, Dict

import numpy

import awebox as awe
import awebox.opts.kite_data.ampyx_ap2_settings as ampyx_ap2_settings
import matplotlib.pyplot as plt
import numpy as np

# set the logger level to 'DEBUG' to see IPOPT output
from awebox.logger.logger import Logger as awelogger
awelogger.logger.setLevel(10)

# indicate desired system architecture
# here: single kite with 6DOF Ampyx AP2 model
options = {}
options['user_options.system_model.architecture'] = {1: 0}
options = ampyx_ap2_settings.set_ampyx_ap2_settings(options)

# indicate desired operation mode
# here: lift-mode system with pumping-cycle operation, with a one winding trajectory
options['user_options.trajectory.type'] = 'power_cycle'
options['user_options.trajectory.system_type'] = 'lift_mode'

# indicate desired environment
# here: wind velocity profile according to power-law
options['params.wind.z_ref'] = 100.0
options['params.wind.power_wind.exp_ref'] = 0.15
options['user_options.wind.model'] = 'power'
options['user_options.wind.u_ref'] = 10.

# indicate numerical nlp details
# here: nlp discretization, with a zero-order-hold control parametrization, and a simple phase-fixing routine. also, specify a linear solver to perform the Newton-steps within ipopt.
options['model.system_bounds.x.l_t'] = [200.0, 1500.0]  # [m]

# (experimental) set to "True" to significantly (factor 5 to 10) decrease construction time
# note: this may result in slightly slower solution timings
options['nlp.compile_subfunctions'] = False
options['model.integration.method'] = 'constraints'  # use enery as a state, works better with SAM

options['nlp.collocation.u_param'] = 'zoh'
options['nlp.SAM.use'] = True
options['nlp.SAM.MaInt_type'] = 'legendre'
options['nlp.SAM.N'] = 2 # the number of full cycles approximated
options['nlp.SAM.d'] = 3 # the number of cycles actually computed
options['nlp.SAM.ADAtype'] = 'CD'  # the approximation scheme
options['user_options.trajectory.lift_mode.windings'] =  options['nlp.SAM.d'] + 1 # todo: set this somewhere else


# SAM Regularization
single_regularization_param = 1E0
options['nlp.SAM.Regularization.AverageStateFirstDeriv'] = 1E-2*single_regularization_param
options['nlp.SAM.Regularization.AverageStateThirdDeriv'] = 1*single_regularization_param
options['nlp.SAM.Regularization.AverageAlgebraicsThirdDeriv'] = 0*single_regularization_param
options['nlp.SAM.Regularization.SimilarMicroIntegrationDuration'] = 1E-1*single_regularization_param

# smooth the reel in phase (this increases convergence speed x10)
options['nlp.cost.beta'] = False # penalize side-slip (can improve convergence)
# options['solver.cost.beta.0'] = 8e0
# options['solver.cost.u_regularisation.0'] = 1e0
# options['solver.max_iter'] = 0
# options['solver.max_iter_hippo'] = 0

# Number of discretization points
n_k = 20 * (options['nlp.SAM.d'] + 2)
options['nlp.n_k'] = n_k
# options['nlp.collocation.d'] = 4
options['model.system_bounds.theta.t_f'] = [40, 40*options['nlp.SAM.N']] # [s]

options['solver.linear_solver'] = 'ma27'

options['visualization.cosmetics.interpolation.n_points'] = 200* options['nlp.SAM.N'] # high plotting resolution

# build and optimize the NLP (trial)
trial = awe.Trial(options, 'DualKitesLongHorizon')
trial.build()
trial.optimize()
# trial.save(fn=f'trial_save_SAM_{"dual" if DUAL_KITES else "single"}Kite')
solution_dict = trial.solution_dict

# draw some of the pre-coded plots for analysis

# extract information from the solution for independent plotting or post-processing
# here: plot relevant system outputs, compare to [Licitra2019, Fig 11].
plot_dict = trial.visualization.plot_dict
outputs = plot_dict['outputs']
time = plot_dict['time_grids']['ip']
avg_power = plot_dict['power_and_performance']['avg_power'] / 1e3

print('======================================')
print('Average power: {} kW'.format(avg_power))
print('======================================')

# # %%
# trial.plot(['states'])
# plt.gcf().tight_layout()



# print(asdf)
# %% Post-Processing


import casadi as ca
from awebox.ocp.discretization_averageModel import OthorgonalCollocation, construct_time_grids_SAM_reconstruction
from awebox.tools.struct_operations import calculate_SAM_regions, evaluate_cost_dict

d_SAM = options['nlp.SAM.d']
N_SAM = options['nlp.SAM.N']

Vopt = solution_dict['V_opt']
Vref = solution_dict['V_ref']
# Vinit = trial.optimization.V_init
# Vopt = trial.optimization.V_init

d = trial.nlp.time_grids['x'](Vopt['theta', 't_f']).full().flatten()

regions_indeces = calculate_SAM_regions(trial.nlp.options)
strobo_indeces = [region_indeces[0] for region_indeces in regions_indeces[1:]]
model = trial.model
macroIntegrator = OthorgonalCollocation(np.array(ca.collocation_points(d_SAM, options['nlp.SAM.MaInt_type'])))

t_f_opt = Vopt['theta', 't_f']

# %% Reconstuct into large V structure of the FULL trajectory
# %% plot the results
import matplotlib
import mpl_toolkits.mplot3d as a3
plot_dict_REC = trial.visualization.plot_dict
plot_dict_SAM = trial.visualization.plot_dict_SAM
plt.figure(figsize=(10, 10))
ax = plt.axes(projection='3d')

_raw_vertices = np.array([[-1.2, 0, -0.4, 0],
                          [0, -1, 0, 1],
                          [0, 0, 0, 0]])
_raw_vertices = _raw_vertices - np.mean(_raw_vertices, axis=1).reshape((3, 1))


def drawKite(pos, rot, wingspan, color='C0', alpha=1):
    rot = np.reshape(rot, (3, 3)).T

    vtx = _raw_vertices * wingspan / 2  # -np.array([[0.5], [0], [0]]) * sizeKite
    vtx = rot @ vtx + pos
    tri = a3.art3d.Poly3DCollection([vtx.T])
    tri.set_color(matplotlib.colors.to_rgba(color, alpha - 0.1))
    tri.set_edgecolor(matplotlib.colors.to_rgba(color, alpha))
    # tri.set_alpha(alpha)
    # tri.set_edgealpha(alpha)
    ax.add_collection3d(tri)


nk_reelout = int(options['nlp.n_k'] * trial.options['nlp']['phase_fix_reelout'])
nk_cut = round(options['nlp.n_k'] * trial.options['nlp']['phase_fix_reelout'])

q10_REC = plot_dict_REC['x']['q10']
ax.plot3D(q10_REC[0], q10_REC[1], q10_REC[2], 'C0-', alpha=0.3)

q10_opt = plot_dict_SAM['x']['q10']
ip_regions_SAM = plot_dict_SAM['SAM_regions_ip']


for region_index, color in zip(np.arange(0, trial.options['nlp']['SAM']['d']), [f'C{i}' for i in range(20)]):
    ax.plot3D(q10_opt[0][np.where(ip_regions_SAM == region_index)],
              q10_opt[1][np.where(ip_regions_SAM == region_index)],
              q10_opt[2][np.where(ip_regions_SAM == region_index)]
              , '-', color=color,
                  alpha=1, markersize=3)

# ax.plot3D(q10_opt[0][np.where(ip_regions_SAM == trial.options['nlp']['SAM']['d'])],
#           q10_opt[1][np.where(ip_regions_SAM == trial.options['nlp']['SAM']['d'])],
#           q10_opt[2][np.where(ip_regions_SAM == trial.options['nlp']['SAM']['d'])], 'C0.',
#           alpha=0.3, markersize=6)
# # ax.plot3D(q10_opt_average[0], q10_opt_average[1], q10_opt_average[2], 'b-', alpha=1)
# ax.plot3D(q10_opt_average_strobo[0], q10_opt_average_strobo[1], q10_opt_average_strobo[2], 'bs', alpha=1)
#
# # reconstructed trajectory
# ax.plot3D(q10_reconstruct[0], q10_reconstruct[1], q10_reconstruct[2], 'C1-', alpha=0.2)

# average trajectory
# ax.plot3D(X_macro_coll[0,:], X_macro_coll[1,:], X_macro_coll[2,:], 'b.-', alpha=1, markersize=6)


# plot important points
# ax.plot3D(q1_opt[0,0], q1_opt[1,0], q1_opt[2,0], 'C1o', alpha=1)
# ax.plot3D(q1_opt[0,strobo_indeces[-1]], q1_opt[1,strobo_indeces[-1]], q1_opt[2,strobo_indeces[-1]], 'C1o', alpha=1)


# tether10 = np.hstack([q21_opt_plot[:, [-1]], np.zeros((3, 1))])
# tether21 = np.hstack([q2_opt[:, [-1]], q1_opt[:, [-1]]])
# tether31 = np.hstack([q3_opt[:, [-1]], q1_opt[:, [-1]]])
# ax.plot3D(tether21[0], tether21[1], tether21[2], '-',color='black')
# ax.plot3D(tether31[0], tether31[1], tether31[2], '-',color='black')
# ax.plot3D(tether10[0], tether10[1], tether10[2], '-',color='black')


# set bounds for nice view
q10_REC_all = np.vstack([q10_REC[0],q10_REC[1],q10_REC[2]])
meanpos = np.mean(q10_REC_all, axis=1)

bblenght = np.max(np.abs(q10_REC_all - meanpos.reshape(3, 1)))
ax.set_xlim3d(meanpos[0] - bblenght, meanpos[0] + bblenght)
ax.set_ylim3d(meanpos[1] - bblenght, meanpos[1] + bblenght)
ax.set_zlim3d(meanpos[2] - bblenght, meanpos[2] + bblenght)

ax.quiver(meanpos[0] - bblenght / 2, meanpos[1] - bblenght / 2, meanpos[2] - bblenght, 1, 0, 0, length=40, color='g')
ax.text(meanpos[0] - bblenght / 2, meanpos[1] - bblenght / 2, meanpos[2] - bblenght, "Wind", 'x', color='g', size=15)

ax.set_xlabel(r'$x$ in m')
ax.set_ylabel(r'$y$ in m')
ax.set_zlabel(r'$z$ in m')

# ax.legend()

# plt.axis('off')
ax.view_init(elev=23., azim=-45)

# plt.legend()
plt.tight_layout()
# plt.savefig('3DReelout.pdf')
plt.show()


# %% Fake the AWEbox into recalibrating its visualz with the reconstructed trajectory
V_reconstruct = trial.visualization.plot_dict['V_plot']
trial.options['nlp']['SAM']['flag_SAM_reconstruction'] = True
trial.options['nlp']['SAM']['use'] = False
n_k_total = len(V_reconstruct['x']) - 1
trial.visualization.plot_dict['n_k'] = n_k_total
# print(calculate_kdx_SAM_reconstruction(trial.options['nlp'], V_reconstruct,30))

# OVERWRITE VOPT OF THE TRIAL
trial.optimization.V_opt = V_reconstruct
trial.optimization.V_final_si = trial.visualization.plot_dict['V_plot_si']

# %% MPC SIMULATION
import copy

# awelogger.logger.setLevel('INFO')

# from awebox.logger.logger import Logger as awelogger
# awelogger.logger.setLevel('INFO')
time_grid_MPC = trial.visualization.plot_dict['time_grids']['x']
T_opt = float(time_grid_MPC[-1])


# set-up closed-loop simulation
T_mpc = 3 # seconds
N_mpc = 20 # MPC horizon
ts = T_mpc/N_mpc # sampling time

#SAM reconstruct options
options['nlp.SAM.flag_SAM_reconstruction'] = True
options['nlp.SAM.use'] = False

# MPC options
options['mpc.scheme'] = 'radau'
options['mpc.d'] = 3
options['mpc.jit'] = False
options['mpc.cost_type'] = 'tracking'
options['mpc.expand'] = True
options['mpc.linear_solver'] = 'ma27'
options['mpc.max_iter'] = 600
options['mpc.max_cpu_time'] = 2000
options['mpc.N'] = N_mpc
options['mpc.plot_flag'] = False
options['mpc.ref_interpolator'] = 'poly'
options['mpc.homotopy_warmstart'] = True
options['mpc.terminal_point_constr'] = False

# simulation options
options['sim.number_of_finite_elements'] = 50 # integrator steps within one sampling time
options['sim.sys_params'] = copy.deepcopy(trial.options['solver']['initialization']['sys_params_num'])

weights_x = model.variables_dict['x'](1E-6)
weights_x['q10'] = 1
weights_x['dq10'] = 1
weights_x['r10'] = 1
weights_x['e'] = 0

additionalMPCoptions = {}
additionalMPCoptions['Q'] = weights_x
additionalMPCoptions['R'] = model.variables_dict['u'](1)
additionalMPCoptions['P'] = weights_x
additionalMPCoptions['Z'] = model.variables_dict['z'](1E-6)

# make simulator
from awebox import sim

closed_loop_sim = sim.Simulation(trial,'closed_loop', ts, options, additional_mpc_options=additionalMPCoptions)

#  Run the closed-loop simulation

# T_sim = T_opt//40 # seconds
T_sim = T_opt # seconds
N_sim = int(T_sim/ts)  # closed-loop simulation steps
# tion steps

startTime = 0
closed_loop_sim.run(N_sim, startTime= startTime)
# plt.show()



# %% Debug Plot
# # plot_t_grid = np.array(closed_loop_sim.visualization.plot_dict['time_grids']['ip']).flatten()
#
# #evaluate the interpolator of the closed loop simulation
interpolator = closed_loop_sim.mpc.interpolator_si
time_grid_MPC_ref = np.mod(np.linspace(0, T_opt-0.001, 100), T_opt)
x_ref = trial.model.variables_dict['x'].repeated(interpolator(time_grid_MPC_ref,'x'))
# q21_ref = np.vstack([interpolator(time_grid_MPC.full().flatten(),'q21',0,'x').full().flatten(),
#                      interpolator(time_grid_MPC.full().flatten(),'q21',1,'x').full().flatten(),
#                      interpolator(time_grid_MPC.full().flatten(),'q21',2,'x').full().flatten()])
# plt.figure(figsize=(10, 10))
# states_to_plot = ['q10','dq10','l_t','dl_t']
# for index,state in enumerate(states_to_plot):
#     plt.subplot(2, 2, index+1)
#     plt.plot(trial.visualization.plot_dict['time_grids']['x'].full().flatten(), ca.horzcat(*V_reconstruct['x',:,state]).full().T,'.-',alpha=0.2)
#
#     # reset color cycle
#     plt.gca().set_prop_cycle(None)
#     plt.plot(time_grid_MPC_ref, ca.horzcat(*x_ref[:,state]).full().T,'--')
# plt.show()
# #
# # print(asdf)

#  plot the interpolated reference trajectory
plot_t_grid = np.array(closed_loop_sim.visualization.plot_dict['time_grids']['ip']).flatten()

# trajectories
q10_MPC = np.vstack([np.array(closed_loop_sim.visualization.plot_dict['x']['q10'][0]).flatten(),
                     np.array(closed_loop_sim.visualization.plot_dict['x']['q10'][1]).flatten(),
                     np.array(closed_loop_sim.visualization.plot_dict['x']['q10'][2]).flatten()])

plot_dict_CLSIM = closed_loop_sim.visualization.plot_dict
ip_grid = plot_dict_CLSIM['time_grids']['ip']

# reference trajectory of the PMCP at the start
startTime_updated = ip_grid[0]
pmpc_first_time_grid = closed_loop_sim.mpc._Pmpc__compute_time_grids(startTime_updated)
pmpc_first_ref = closed_loop_sim.mpc.get_reference(*pmpc_first_time_grid)

# % Plot the STATES
plt.figure(figsize=(10,10))

# plot the reference
# plt.plot(closed_loop_sim.visualization.plot_dict['time_grids']['ref']['x'].full(), closed_loop_sim.visualization.plot_dict['ref']['x']['q10'][0], label='reference_MPC')

states_to_plot = ['q10','dq10','r10','l_t','dl_t','e']
for index_state, name_state in enumerate(states_to_plot):
    plt.subplot(int(np.ceil((len(states_to_plot)+1)//2)), 2, index_state + 1)

    plt.plot(ip_grid,
             np.vstack(plot_dict_CLSIM['x'][name_state]).T)
    plt.plot([],[],'k-',label='sim')
    # reset color cycle
    plt.gca().set_prop_cycle(None)

    traj_state =  np.vstack(plot_dict_CLSIM['ref_si']['x'][name_state])
    plt.plot(ip_grid,traj_state.T,'--',)
    plt.plot([],[],'k--',label=' mpc reference_recorded')

    # plot reconstructed trajectory
    # plt.gca().set_prop_cycle(None)
    # plt.plot(trial.visualization.plot_dict['time_grids']['x'].full().flatten(), ca.horzcat(*trial.visualization.plot_dict['V_plot']['x',:,name_state]).full().T,'.-',alpha=0.2)
    plt.plot(trial.visualization.plot_dict['time_grids']['x'].full().flatten(), ca.horzcat(*trial.visualization.plot_dict['V_plot_si']['x',:,name_state]).full().T,'.-',alpha=0.2)
    plt.plot([],[],'k.-',label='rec+ip')

    # reset color cycle
    plt.gca().set_prop_cycle(None)
    plt.plot(time_grid_MPC_ref, ca.horzcat(*x_ref[:,name_state]).full().T,'--',alpha=0.3,linestyle='dotted')
    plt.plot([],[],'k',linestyle='dotted',label='mpc interpol. eval')

    # plot first mpc reference trajectory
    # plt.gca().set_prop_cycle(None)
    # plt.plot(pmpc_first_time_grid[1][0::4], ca.horzcat(*pmpc_first_ref['x',:,name_state]).full().T,'-.')

    plt.ylabel(name_state)
    plt.legend()
plt.tight_layout()
plt.show()


# %% 3D plot of the tracked trajectory
import matplotlib
import mpl_toolkits.mplot3d as a3

plt.figure(figsize=(10, 10))
ax = plt.axes(projection='3d')

_raw_vertices = np.array([[-1.2, 0, -0.4, 0],
                          [0, -1, 0, 1],
                          [0, 0, 0, 0]])
_raw_vertices = _raw_vertices - np.mean(_raw_vertices, axis=1).reshape((3, 1))


def drawKite(pos, rot, wingspan, color='C0', alpha=1):
    rot = np.reshape(rot, (3, 3)).T

    vtx = _raw_vertices * wingspan / 2  # -np.array([[0.5], [0], [0]]) * sizeKite
    vtx = rot @ vtx + pos
    tri = a3.art3d.Poly3DCollection([vtx.T])
    tri.set_color(matplotlib.colors.to_rgba(color, alpha - 0.1))
    tri.set_edgecolor(matplotlib.colors.to_rgba(color, alpha))
    # tri.set_alpha(alpha)
    # tri.set_edgealpha(alpha)
    ax.add_collection3d(tri)


# nk_reelout = int(options['nlp.n_k'] * options['nlp.phase_fix_reelout'])
# nk_cut = round(options['nlp.n_k'] * options['nlp.phase_fix_reelout'])
#

# else:
q10_REC = trial.visualization.plot_dict['x']['q10']
ax.plot3D(q10_REC[0], q10_REC[1], q10_REC[2], 'C1-', alpha=0.2)
ax.plot3D(q10_MPC[0], q10_MPC[1], q10_MPC[2], 'C0-', alpha=1)


# set bounds for nice view
meanpos = np.mean(q10_MPC[:], axis=1) + np.array([50, 0, 0])

bblenght = np.max(np.abs(q10_MPC - meanpos.reshape(3, 1)))
ax.set_xlim3d(meanpos[0] - bblenght, meanpos[0] + bblenght)
ax.set_ylim3d(meanpos[1] - bblenght, meanpos[1] + bblenght)
ax.set_zlim3d(meanpos[2] - bblenght, meanpos[2] + bblenght)

ax.quiver(meanpos[0] - bblenght / 2, meanpos[1] - bblenght / 2, meanpos[2] - bblenght, 1, 0, 0, length=40, color='g')
ax.text(meanpos[0] - bblenght / 2, meanpos[1] - bblenght / 2, meanpos[2] - bblenght, "Wind", 'x', color='g', size=15)

ax.set_xlabel(r'$x$ in m')
ax.set_ylabel(r'$y$ in m')
ax.set_zlabel(r'$z$ in m')

# ax.legend()
# plt.axis('off')
ax.view_init(elev=23., azim=-45)

# plt.legend()
plt.tight_layout()
# plt.savefig('3DReelout.pdf')
plt.show()


# %% Export Trajectories for fancier Plotting

# all the stuff to be plotted from SAM
export_dict_SAM = {}
export_dict_SAM['regions'] = ip_regions_SAM
export_dict_SAM['time'] = plot_dict_SAM['time_grids']['ip']
export_dict_SAM['time_X'] = plot_dict_SAM['time_grids']['ip_X']
export_dict_SAM['x'] = plot_dict_SAM['x']
export_dict_SAM['X'] = plot_dict_SAM['X']
export_dict_SAM['d'] = trial.options['nlp']['SAM']['d']
export_dict_SAM['N'] = trial.options['nlp']['SAM']['N']
export_dict_SAM['regularizationValue'] = single_regularization_param

export_dict_REC = {}
export_dict_REC['time'] = plot_dict_REC['time_grids']['ip']
export_dict_REC['x'] = plot_dict_REC['x']

export_dict_MPC = {}
export_dict_MPC['time'] = plot_dict_CLSIM['time_grids']['ip']
export_dict_MPC['x'] = plot_dict_CLSIM['x']

export_dict = {'SAM': export_dict_SAM, 'REC': export_dict_REC, 'MPC': export_dict_MPC}

# save the data
from datetime import datetime
datestr = datetime.now().strftime('%Y%m%d_%H%M')
filename= f'{datestr}_AWE_SAM_N{trial.options['nlp']['SAM']['N']}_d{trial.options['nlp']['SAM']['d']}'
np.savez(f'_export/{filename}.npz', **export_dict)



