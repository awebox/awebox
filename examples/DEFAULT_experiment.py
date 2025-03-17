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
import awebox as awe
import awebox.opts.kite_data.ampyx_ap2_settings as ampyx_ap2_settings
import numpy as np

# set the logger level to 'DEBUG' to see IPOPT output
from awebox.logger.logger import Logger as awelogger
# awelogger.logger.setLevel(10)


def run_DEFAULT_MPC_experiment(N):
    # indicate desired system architecture
    # here: single kite with 6DOF Ampyx AP2 model
    options = {}
    options['user_options.system_model.architecture'] = {1: 0}
    options = ampyx_ap2_settings.set_ampyx_ap2_settings(options)
    # options['model.system_bounds.theta.t_f'] = [5., 30.]  # [s]

    # indicate desired operation mode
    # here: lift-mode system with pumping-cycle operation, with a one winding trajectory
    options['user_options.trajectory.type'] = 'power_cycle'
    options['user_options.trajectory.system_type'] = 'lift_mode'
    options['user_options.trajectory.lift_mode.windings'] = N + 1

    # indicate desired environment
    # here: wind velocity profile according to power-law
    options['params.wind.z_ref'] = 100.0
    options['params.wind.power_wind.exp_ref'] = 0.15
    options['user_options.wind.model'] = 'power'
    options['user_options.wind.u_ref'] = 10.

    # indicate numerical nlp details
    # here: nlp discretization, with a zero-order-hold control parametrization, and a simple phase-fixing routine. also, specify a linear solver to perform the Newton-steps within ipopt.
    options['model.system_bounds.x.l_t'] = [10.0, 2500.0]  # [m]
    options['model.system_bounds.theta.t_f'] = [50, 50 + N * 20]  # [s]


    # a simple phase-fixing routine. also, specify a linear solver to perform the Newton-steps
    # within ipopt.
    options['nlp.n_k'] = 80 + N * 20
    options['nlp.collocation.u_param'] = 'zoh'
    options['nlp.cost.output_quadrature'] = False  # use enery as a state, works better with SAM
    options['user_options.trajectory.lift_mode.phase_fix'] = 'single_reelout'  # 'single_reelout'
    options['solver.linear_solver'] = 'ma27'  # if HSL is installed, otherwise 'mumps'
    options['nlp.cost.beta'] = False  # penalize side-slip (can improve convergence)

    # (experimental) set to "True" to significantly (factor 5 to 10) decrease construction time
    # note: this may result in slightly slower solution timings
    options['nlp.compile_subfunctions'] = False
    options['model.integration.method'] = 'constraints'  # use enery as a state, works better with SAM

    # for option_name, option_val in overwrite_options.items():
    #     options[option_name] = option_val

    # build and optimize the NLP (trial)
    trial = awe.Trial(options, 'Ampyx_AP2')
    trial.build()
    trial.optimize()

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
    T_sim = T_opt # seconds
    N_sim = int(T_sim/ts)  # closed-loop simulation steps
    closed_loop_sim.run(N_sim, startTime= 0)


    # %% Export Trajectories for fancier Plotting

    # solver stats
    solver_stats = {
        'N_var': trial.nlp.V.size,
        'N_eq': trial.nlp.g.size,
        't_wall': trial.optimization.t_wall,
        'N_iter': trial.optimization.iterations,
    }

    # all the stuff to be plotted from SAM
    plot_dict_DEFAULT = trial.visualization.plot_dict
    export_dict_DEFAULT = {}
    export_dict_DEFAULT['time'] = plot_dict_DEFAULT['time_grids']['ip']
    export_dict_DEFAULT['x'] = plot_dict_DEFAULT['x']
    export_dict_DEFAULT['N'] = N
    export_dict_DEFAULT['n_k'] = trial.options['nlp']['n_k']
    export_dict_DEFAULT['solver_stats'] = solver_stats


    export_dict_MPC = {}
    plot_dict_CLSIM = closed_loop_sim.visualization.plot_dict
    export_dict_MPC['time'] = plot_dict_CLSIM['time_grids']['ip']
    export_dict_MPC['x'] = plot_dict_CLSIM['x']

    export_dict = {'DEFAULT': export_dict_DEFAULT, 'MPC': export_dict_MPC}

    # save the data
    from datetime import datetime
    datestr = datetime.now().strftime('%Y%m%d_%H%M')
    filename= f'{datestr}_DEFAULT_N{N}'
    np.savez(f'_export/{filename}.npz', **export_dict)

    awelogger.logger.info(f'Exported data to _export/{filename}.npz')

    return export_dict
