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
# awelogger.logger.setLevel(10)


def run_SAM_MPC_experiment(d=3, N=5):

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

    # (experimental) set to "True" to significantly (factor 5 to 10) decrease construction time
    # note: this may result in slightly slower solution timings
    options['nlp.compile_subfunctions'] = False
    options['model.integration.method'] = 'constraints'  # use enery as a state, works better with SAM

    options['nlp.collocation.u_param'] = 'zoh'
    options['nlp.SAM.use'] = True
    options['nlp.SAM.MaInt_type'] = 'legendre'
    options['nlp.SAM.N'] = N  # the number of full cycles approximated
    options['nlp.SAM.d'] = d  # the number of cycles actually computed
    options['nlp.SAM.ADAtype'] = 'CD'  # the approximation scheme
    options['user_options.trajectory.lift_mode.windings'] = options['nlp.SAM.d'] + 1  # todo: set this somewhere else

    # model bounds
    # options['model.system_bounds.x.l_t'] = [10.0, 2500.0]  # [m]
    # options['model.system_bounds.theta.t_f'] = [50, 150 + options['nlp.SAM.N'] * 20]  # [s]

    # smooth the reel in phase (this increases convergence speed x10)
    options['nlp.cost.beta'] = False  # penalize side-slip (can improve convergence)

    # SAM Regularization
    single_regularization_param = 1E-1
    options['nlp.SAM.Regularization.AverageStateFirstDeriv'] = 1E1 * single_regularization_param
    options['nlp.SAM.Regularization.AverageStateThirdDeriv'] = 1E0 * single_regularization_param
    # options['nlp.SAM.Regularization.AverageAlgebraicsThirdDeriv'] = 1E3*single_regularization_param
    options['nlp.SAM.Regularization.AverageAlgebraicsThirdDeriv'] = 0 * single_regularization_param
    options['nlp.SAM.Regularization.SimilarMicroIntegrationDuration'] = 1E-2 * single_regularization_param

    # Number of discretization points
    n_k = 20 * (options['nlp.SAM.d']) * 2
    # n_k = 70 + 30 * (options['nlp.SAM.d'])
    options['nlp.n_k'] = n_k

    # model bounds
    options['model.system_bounds.x.l_t'] = [10.0, 2500.0]  # [m]
    options['model.system_bounds.theta.t_f'] = [50, 50 + options['nlp.SAM.N'] * 20]  # [s]

    options['solver.linear_solver'] = 'ma27'

    options['visualization.cosmetics.interpolation.n_points'] = 300 * options['nlp.SAM.N']  # high plotting resolution

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

    weights_x = trial.model.variables_dict['x'](1E-6)
    weights_x['q10'] = 1
    weights_x['dq10'] = 1
    weights_x['r10'] = 1
    weights_x['e'] = 0

    additionalMPCoptions = {}
    additionalMPCoptions['Q'] = weights_x
    additionalMPCoptions['R'] = trial.model.variables_dict['u'](1)
    additionalMPCoptions['P'] = weights_x
    additionalMPCoptions['Z'] = trial.model.variables_dict['z'](1E-6)

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


    # %% Export Trajectories for fancier Plotting

    #SAM solver stats
    solver_stats = {
        'N_var': trial.nlp.V.size,
        'N_eq': trial.nlp.g.size,
        't_wall': trial.optimization.t_wall,
        'N_iter': trial.optimization.iterations,
    }

    # all the stuff to be plotted from SAM
    plot_dict_SAM = trial.visualization.plot_dict_SAM
    export_dict_SAM = {}
    export_dict_SAM['regions'] =  plot_dict_SAM['SAM_regions_ip']
    export_dict_SAM['time'] = plot_dict_SAM['time_grids']['ip']
    export_dict_SAM['time_X'] = plot_dict_SAM['time_grids']['ip_X']
    export_dict_SAM['x'] = plot_dict_SAM['x']
    export_dict_SAM['X'] = plot_dict_SAM['X']
    export_dict_SAM['d'] = trial.options['nlp']['SAM']['d']
    export_dict_SAM['N'] = trial.options['nlp']['SAM']['N']
    export_dict_SAM['regularizationValue'] = single_regularization_param
    export_dict_SAM['n_k'] = trial.options['nlp']['n_k']
    export_dict_SAM['solver_stats'] = solver_stats

    export_dict_REC = {}
    plot_dict_REC = trial.visualization.plot_dict
    export_dict_REC['time'] = plot_dict_REC['time_grids']['ip']
    export_dict_REC['x'] = plot_dict_REC['x']

    export_dict_MPC = {}
    plot_dict_CLSIM = closed_loop_sim.visualization.plot_dict
    export_dict_MPC['time'] = plot_dict_CLSIM['time_grids']['ip']
    export_dict_MPC['x'] = plot_dict_CLSIM['x']

    export_dict = {'SAM': export_dict_SAM, 'REC': export_dict_REC, 'MPC': export_dict_MPC}

    # save the data
    from datetime import datetime
    datestr = datetime.now().strftime('%Y%m%d_%H%M')
    filename= f'{datestr}_AWE_SAM_N{trial.options['nlp']['SAM']['N']}_d{trial.options['nlp']['SAM']['d']}'
    np.savez(f'_export/{filename}.npz', **export_dict)

    awelogger.logger.info(f'Exported data to _export/{filename}.npz')

    return export_dict
