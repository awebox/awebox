#!/usr/bin/python3
from platform import architecture

import matplotlib
matplotlib.use('TkAgg')

import awebox as awe

import matplotlib.pyplot as plt
import pickle
import copy
import numpy as np
import csv
import os

from datetime import date
import random


import awebox.trial as awe_trial
import awebox.opts.kite_data.ampyx_data as ampyx_data
import awebox.opts.kite_data.ampyx_ap2_settings as ampyx_ap2_settings

import awebox.tools.vector_operations as vect_op
import awebox.tools.struct_operations as struct_op
import awebox.tools.print_operations as print_op
import awebox.tools.save_operations as save_op

import awebox.viz.wake as wake_viz
import awebox.opti.initialization_dir.initialization as initialization
import awebox.mdl.aero.induction_dir.vortex_dir.alg_repr_dir.initialization as alg_initialization


from awebox.logger.logger import Logger as awelogger
import casadi.tools as cas

awelogger.logger.setLevel(10)




def build_unique_trial_name(base_name, inputs):

    trial_name_baseline = base_name    
    for name, val in inputs.items():
        trial_name_baseline += '_' + name + '_' + str(val)
    	
    today = date.today()
    rand = random.randint(1000000, 9000000)
    trial_name_baseline += '_' + str(today) + '_' + str(rand)
    return trial_name_baseline
    
def toggle_baseline_options(options):
    options['user_options.induction_model'] = 'not_in_use'
    options['user_options.trajectory.lift_mode.phase_fix'] = 'single_reelout' # the default option
    options['visualization.cosmetics.plot_ref'] = False # the 'refernence' is just a circle.
    options['solver.hippo_strategy'] = True # the default interior-point homotopy embedding
    options['solver.linear_solver'] = 'ma57' # the repeatable option
    options['solver.homotopy_method.put_fictitious_before_induction'] = True
    return options
    
def toggle_vortex_options(options):
    options['user_options.induction_model'] = 'vortex'
    #options['user_options.trajectory.lift_mode.phase_fix'] = 'simple' # it's really tempting to remove inequality constraints by switching the phase-fixing method, but that upsets the warmstart generation so we're not doing it.
    options['visualization.cosmetics.plot_ref'] = True # the 'reference' here is the baseline problem
    options['solver.hippo_strategy'] = False # save memory by only requring one casadi solver
    options['solver.linear_solver'] = 'ma86' # the parallelized but non-repeatable option
    options['solver.homotopy_method.put_fictitious_before_induction'] = False # we need the fictitious forces to still be enabled, so that we can fly the simulation/reference trajectory with different aerodynamics
    return options

def get_list_of_plots():
    list_of_plots = ['power', 'states', 'algebraic_variables', 'controls', 'constraints', 'animation_snapshot',  'isometric', 'projected_xy', 'projected_yz', 'projected_xz', 'wake_isometric', 'wake_xy', 'wake_xz', 'wake_yz', 'wake_legend', 'velocity_deficits', 'velocity_distribution', 'aero_dimensionless', 'relative_radius', 'relative_radius_of_curvature', 'aero_coefficients', 'circulation', 'local_induction_factor_all_projections']
    #'induction_wind_tunnel',  'induction_contour_normal_wind', 
    #'induction_contour_normal_normal', 'induction_contour_wind_wind',
    list_of_plots = list(set(list_of_plots)) # double-check that we don't waste time producing the same plot twice
    return list_of_plots


def turn_off_inequalities_except_time(options):
    options['model.system_bounds.theta.diam_t'] = [-cas.inf, cas.inf]
    options['model.system_bounds.x.l_t'] = [-cas.inf, cas.inf]
    options['model.system_bounds.x.dl_t'] = [-cas.inf, cas.inf]
    options['model.system_bounds.x.ddl_t'] = [-cas.inf, cas.inf]
    options['model.system_bounds.u.dddl_t'] = [-cas.inf, cas.inf]
    options['model.system_bounds.x.q'] = [np.array([-cas.inf, -cas.inf, -cas.inf]), np.array([cas.inf, cas.inf, cas.inf])]
    options['model.system_bounds.x.dq'] = [np.array([-cas.inf, -cas.inf, -cas.inf]), np.array([cas.inf, cas.inf, cas.inf])]
    options['model.system_bounds.x.coeff'] = [np.array([-cas.inf, -cas.inf]), np.array([cas.inf, cas.inf])]
    options['model.system_bounds.u.dcoeff'] = [np.array([-cas.inf, -cas.inf]), np.array([cas.inf, cas.inf])]
    options['model.system_bounds.x.omega'] = [np.array([-cas.inf, -cas.inf, -cas.inf]), np.array([cas.inf, cas.inf, cas.inf])]
    options['model.system_bounds.z.lambda'] = [-cas.inf, cas.inf]

    options['model.system_bounds.theta.diam_t'] = [-cas.inf, cas.inf]
    options['model.system_bounds.theta.diam_s'] = [-cas.inf, cas.inf]
    options['model.system_bounds.theta.diam_c'] = [-cas.inf, cas.inf]   
    options['model.system_bounds.x.l_t'] = [-cas.inf, cas.inf]
    options['model.system_bounds.x.dl_t'] = [-cas.inf, cas.inf]    
    options['model.system_bounds.x.ddl_t'] = [-cas.inf, cas.inf]    
    options['model.system_bounds.u.dddl_t'] = [-cas.inf, cas.inf]                
    options['model.system_bounds.theta.l_s'] = [-cas.inf, cas.inf]
    options['model.system_bounds.theta.l_i'] = [-cas.inf, cas.inf]
    options['model.system_bounds.theta.l_c'] = [-cas.inf, cas.inf]       
    options['model.system_bounds.x.q'] = [np.array([-cas.inf, -cas.inf, -cas.inf]), np.array([cas.inf, cas.inf, cas.inf])]
    options['model.system_bounds.x.dq'] = [np.array([-cas.inf, -cas.inf, -cas.inf]), np.array([cas.inf, cas.inf, cas.inf])]
    options['model.system_bounds.x.omega'] = [np.array([-cas.inf, -cas.inf, -cas.inf]), np.array([cas.inf, cas.inf, cas.inf])]
    #options['model.system_bounds.theta.t_f'] = [-cas.inf, cas.inf] # we cannot use bounds on the time, because this makes a singularity in the gradient.
    options['model.system_bounds.z.lambda'] = [-cas.inf, cas.inf]    
    options['model.system_bounds.u.dkappa'] = [-cas.inf, cas.inf]    
    options['model.system_bounds.x.coeff'] = [np.array([-cas.inf, -cas.inf]), np.array([cas.inf, cas.inf])]
    options['model.system_bounds.u.dcoeff'] = [np.array([-cas.inf, -cas.inf]), np.array([cas.inf, cas.inf])]

    options['model.geometry.overwrite.delta_max'] = np.array([cas.inf, cas.inf, cas.inf])
    options['model.geometry.overwrite.ddelta_max'] = np.array([cas.inf, cas.inf, cas.inf])
    
    options['model.model_bounds.tether_stress.include'] = False
    options['model.model_bounds.tether_force.include'] = False
    options['model.model_bounds.airspeed.include'] = False
    options['model.model_bounds.aero_validity.include'] = False
    options['model.model_bounds.anticollision.include'] = False
    options['model.model_bounds.acceleration.include'] = False
    options['model.model_bounds.rotation.include'] = False
    
    return options
    
def adjust_weights_for_tracking(trial_baseline, options):

    options['solver.cost.beta.0'] = 0.
    options['nlp.cost.beta'] = False

    options['solver.weights.vortex'] = 0.
    options['solver.cost.fictitious.0'] = 1.e-10
    options['solver.cost.fictitious.1'] = 1.e-10

    extra_much_more_important = 1e3
    much_more_important = 1e2
    more_important = 1e1
    less_important = 1e-1
    much_less_important = 1e-2
    would_be_zero_except_sosc = 1e-4
    baseline_options = trial_baseline.options
        
    #options['solver.cost.tracking.0'] = 1e-2 * baseline_options['solver']['cost']['tracking'][0]
    
    options['solver.weights.q'] = extra_much_more_important * baseline_options['solver']['weights']['q']
    options['solver.weights.dq'] = more_important * baseline_options['solver']['weights']['dq']
    options['solver.weights.r'] = more_important * baseline_options['solver']['weights']['r']
    options['solver.weights.omega'] = more_important * baseline_options['solver']['weights']['omega']
    
    options['solver.weights.coeff'] = extra_much_more_important * baseline_options['solver']['weights']['coeff']
    options['solver.weights.delta'] = much_more_important * baseline_options['solver']['weights']['delta']
    options['solver.weights.ddelta'] = extra_much_more_important * baseline_options['solver']['weights']['ddelta']

    options['solver.cost.t_f.0'] = less_important * baseline_options['solver']['cost']['theta_regularisation'][0] # penalizes switching time
    options['solver.cost.u_regularisation.0'] = more_important * baseline_options['solver']['cost']['u_regularisation'][
        0]
    options['solver.cost.tracking.0'] = more_important * baseline_options['solver']['cost']['tracking'][0]


    #options['solver.weights.l_t'] = 100. * much_more_important * baseline_options['solver']['weights']['l_t']
    #options['solver.weights.dl_t'] = 10. * much_more_important * baseline_options['solver']['weights']['dl_t']

    #options['solver.cost.theta_regularisation.0'] = much_more_important * baseline_options['solver']['cost']['theta_regularisation'][0]
    
    # from august 6th
    #options['solver.weights.q'] = 1e2
    #options['solver.weights.dq'] = 1e2
    #options['solver.weights.r'] = 1e2
    #options['solver.weights.omega'] = 1e2
    #options['solver.cost.tracking.0'] = 1e2 #1
    #options['solver.cost.fictitious.0'] = 1.e-6
    #options['solver.cost.beta.0'] = 1.e-6
    #options['solver.cost.u_regularisation.0'] = 1.e-6
    #options['solver.cost.xdot_regularisation.0'] = 1.e-6
    #options['solver.weights.vortex'] = 1.e-10

    
    
    # # from haas 2019
    # options['solver.cost.tracking.0'] = 1e1
    # options['solver.cost.fictitious.0'] = 1.e-10
    # options['solver.cost.u_regularisation.0'] = 1.e1
    # options['solver.cost.xdot_regularisation.0'] = 1.e-2
    # options['solver.weights.vortex'] = 0.
    # options['solver.weights.q'] = 1e4
    # options['solver.weights.dq'] = 1e3
    # options['solver.weights.r'] = 1e2
    # options['solver.weights.omega'] = 1e3
    # options['solver.weights.coeff'] = 1e1
    # options['solver.weights.delta'] = 1e2
    #
    
    
    # from reattempt
    #options['solver.weights.q'] = 1e3
    #options['solver.weights.dq'] = 1e2
    #options['solver.weights.r'] = 1e1
    #options['solver.weights.omega'] = 1e2
    #options['solver.weights.l_t'] = 1e4
    #options['solver.weights.dl_t'] = 1e3
    #options['solver.weights.delta'] = 1e2
    #options['solver.weights.coeff'] = 1e2
    
    #options['solver.weights.ddelta'] = 1e-3
    #options['solver.weights.ddl_t'] = 1e2
    
    #options['solver.cost.tracking.0'] = 1e-1 #1e-1
    #options['solver.cost.u_regularisation.0'] = 1e2 #1e-6
    ###options['solver.cost.xdot_regularisation.0'] = 1e-8
    ###options['solver.cost.theta_regularisation.0'] = 1e0
    ##options['solver.cost.t_f.0'] = 1e4
    #options['solver.cost.fictitious.0'] = 1e-8
    
    #options['solver.cost.t_f.0'] = options['solver.cost.theta_regularisation.0'] # because this is included by combined q and dq regularization
    
    # options['solver.cost.u_regularisation.0'] = more_important * baseline_options['solver']['cost']['u_regularisation'][0]
    ## options['solver.cost.xdot_regularisation.0'] = less_important * baseline_options['solver']['cost']['xdot_regularisation'][0]
    
    #options['solver.cost.fictitious.0'] = would_be_zero_except_sosc * baseline_options['solver']['cost']['fictitious'][0]
    #options['solver.weights.vortex'] = would_be_zero_except_sosc * baseline_options['solver']['weights']['vortex']

    #options['model.scaling.other.position_scaling_method'] = 'radius'
    #options['model.scaling.other.force_scaling_method'] = 'synthesized'
    #options['model.scaling.other.flight_radius_estimate'] = 'centripetal'
    #options['model.scaling.other.tension_estimate'] = 'average_force'

    return options
    
def fix_params_to_baseline(trial_baseline, options):

    fixed_params = {}
    
    V_baseline_si = trial_baseline.optimization.V_final_si
    for var_name in trial_baseline.model.variables_dict['theta'].keys():
        if var_name != 't_f':
            fixed_params[var_name] = V_baseline_si['theta', var_name]
    
    time_period = trial_baseline.optimization.global_outputs_opt['time_period'].full()[0][0]
    fixed_params['t_f'] = time_period
    
    options['user_options.trajectory.fixed_params'] = fixed_params

    return options

def construct_vortex_initial_guess(trial_baseline, trial_vortex, inequalities_are_off=True):

    print('updating reference info for warmstarting...')

    solution_dict_local = trial_baseline.solution_dict

    print('import variable values...')
    solution_dict_local['final_homotopy_step'] = 'initial'

    V_baseline_si = trial_baseline.optimization.V_final_si
    V_local_si = trial_vortex.nlp.V(0.)
    for ldx in range(V_baseline_si.shape[0]):
        print_op.print_progress(ldx, V_baseline_si.shape[0])
        try:
            canonical = V_baseline_si.getCanonicalIndex(ldx)
            V_local_si[canonical] = V_baseline_si.cat[ldx]
        except:
            pass
    print_op.close_progress()

    lam_x0_local = trial_vortex.nlp.V(0.)
    if not inequalities_are_off:
        print('import variable multipliers...')
        if 'lam_x0' in solution_dict_local['opt_arg'].keys():
            lam_x0_baseline = solution_dict_local['opt_arg']['lam_x0']
            for ldx in range(V_baseline_si.shape[0]):
                print_op.print_progress(ldx, V_baseline_si.shape[0])
                try:
                    lam_x0_local[V_baseline_si.getCanonicalIndex(ldx)] = lam_x0_baseline[ldx]
                except:
                    pass
        print_op.close_progress()

    # reset the homotopy parameters to their start values
    V_local_si['phi'] = cas.DM.ones(V_local_si['phi'].shape)

    print('import problem parameters...')
    p_local = trial_vortex.nlp.P(0.)
    p_fix_num_baseline = trial_baseline.optimization.p_fix_num
    for pdx in range(p_fix_num_baseline.shape[0]):
        print_op.print_progress(pdx, p_fix_num_baseline.shape[0])
        try:
            p_local[p_fix_num_baseline.getCanonicalIndex(pdx)] = p_fix_num_baseline.cat[pdx]
        except:
            pass
    print_op.close_progress()

    # try our best to match the time period, if the phase_fixing method isn't the same. notice, switching phase-fixing methods doesn't presently work well.
    time_period = trial_baseline.optimization.global_outputs_opt['time_period'].full()[0][0]
    if V_baseline_si['theta', 't_f'].shape[0] > V_local_si['theta', 't_f'].shape[0]:
        V_local_si['theta', 't_f'] = time_period * cas.DM.ones(V_local_si['theta', 't_f'].shape)
        p_local['p', 'ref', 'theta', 't_f', 0] = time_period

    print('use the vortex variable initialization routine...')
    try:
       V_local_si = alg_initialization.get_initialization(trial_vortex.options['solver']['initialization'], V_local_si, p_local, trial_vortex.nlp, trial_vortex.model)
    except:
       pass

    V_local_scaled = struct_op.si_to_scaled(V_local_si, trial_vortex.model.scaling)

    print('save the warmstart and reference information...')
    solution_dict_local['V_opt'] = V_local_scaled
    solution_dict_local['V_ref'] = V_local_scaled
    solution_dict_local['opt_arg']['lam_x0'] = lam_x0_local
    solution_dict_local['opt_arg']['lam_g0'] = cas.DM.zeros(trial_vortex.nlp.g.shape)
    
    return solution_dict_local
    


def save_results_including_figures(trial, options):

    if trial.optimization.solve_succeeded:
        save_and_print_info(trial, options)
        trial.plot(get_list_of_plots())
        plt.show(block=False)
        try:
            plt.close('all')
        except:
            pass

    else:
        filename = trial.name + '.csv'
        report = {}
        report['trial_name'] = trial.name
        report['solve'] = trial.optimization.solve_succeeded
        if report['solve']:
            report['tests'] = trial.quality.all_tests_passed()
        save_op.write_or_append_two_column_dict_to_csv(report, filename)

    return None




   
def make_comparison_power_plot(trial_vortex, trial_baseline):
    # extract the power profile from the solutions, then plot
    vortex_plot_dict = trial_vortex.visualization.plot_dict
    baseline_plot_dict = trial_baseline.visualization.plot_dict
    baseline_time = baseline_plot_dict['time_grids']['ip']
    vortex_time = vortex_plot_dict['time_grids']['ip']

    fig, ax = plt.subplots()
    vortex_power_with_fictitious = vortex_plot_dict['interpolation_si']['outputs']['performance']['p_current'][0]
    vortex_power = vortex_plot_dict['interpolation_si']['outputs']['performance']['p_current_without_fictitious'][0]
    baseline_power = baseline_plot_dict['interpolation_si']['outputs']['performance']['p_current'][0]
    plt.plot(vortex_time, vortex_power, label='RLL Simulation OCP (without fictitious forces)')
    plt.plot(vortex_time, vortex_power_with_fictitious, label='RLL Simulation OCP (with fictitious forces)')
    plt.plot(baseline_time, baseline_power, label='baseline AWE OCP')
    plt.title('power in (B) and (C)')
    plt.legend()
    plt.grid(True)
    fig.savefig(trial_vortex.name + '_power_comparison.pdf')
    return None

def include_val(local_val):
    return isinstance(local_val, str) or isinstance(local_val, int) or vect_op.is_numeric_scalar(local_val)


def save_and_print_info(trial, options):

    # this one saves all of the interpolated variable and output information as time-series
    try:
        trial.write_to_csv()
    except:
        message = 'something went wrong with write_to_csv'
        print(message)

    # this one saves the actual pickled trial object
    try:
        trial.save()
    except:
        message = 'something went wrong with trial.save'
        print(message)

    # everything below saves a summary file, so that you don't have to load and re-average the full time-series datafile

    plot_dict = trial.visualization.plot_dict

    report = {}
    report['trial_name'] = trial.name

    report['count'] = 0
    report['n_k'] = options['nlp.n_k']
    try:
        report['p_t'] = float(options['model.aero.vortex.wake_nodes'] - 1.) / options['nlp.n_k']
    except:
        pass

    report['solve'] = trial.optimization.solve_succeeded
    if report['solve']:
        report['tests'] = trial.quality.all_tests_passed()

    report['model_variables'] = np.prod(trial.model.variables.shape)
    if hasattr(trial, 'model') and hasattr(trial.model, 'dimensions_dict'):
        for local_name in trial.model.dimensions_dict.keys():
            local_val = trial.model.dimensions_dict[local_name]
            if include_val(local_val):
                report['model' + '_' + local_name] = local_val

    report['nlp_variables'] = np.prod(trial.nlp.V.shape)
    if hasattr(trial, 'nlp') and hasattr(trial.nlp, 'dimensions_dict'):
        for local_name in trial.nlp.dimensions_dict.keys():
            local_val = trial.nlp.dimensions_dict[local_name]
            if include_val(local_val):
                report['nlp' + '_' + local_name] = local_val

    if hasattr(trial, 'quality') and hasattr(trial.quality, 'results'):
        for local_name in trial.quality.results.keys():
            local_val = trial.quality.results[local_name]
            if include_val(local_val):
                report['quality' + '_' + local_name] = local_val

    if hasattr(trial, 'optimization') and hasattr(trial.optimization, 'stats') and hasattr(trial.optimization.stats, 'keys'):
        for local_name in trial.optimization.stats.keys():
            local_val = trial.optimization.stats[local_name]
            if include_val(local_val):
                report['stats' + '_' + local_name] = local_val

    if hasattr(trial, 'optimization') and hasattr(trial.optimization, 'iterations') and hasattr(trial.optimization.iterations, 'keys'):
        for local_name in trial.optimization.iterations.keys():
            local_val = trial.optimization.iterations[local_name]
            if include_val(local_val):
                report['iterations' + '_' + local_name] = local_val

    if hasattr(trial, 'optimization') and hasattr(trial.optimization, 'timings') and hasattr(trial.optimization.timings, 'keys'):
        for local_name in trial.optimization.timings.keys():
            local_val = trial.optimization.timings[local_name]
            if include_val(local_val):
                report['timings' + '_' + local_name] = local_val

    if hasattr(trial, 'optimization') and hasattr(trial.optimization, 'cumulative_max_memory') and hasattr(trial.optimization.cumulative_max_memory, 'keys'):
        for local_name in trial.optimization.cumulative_max_memory.keys():
            local_val = trial.optimization.cumulative_max_memory[local_name]
            if include_val(local_val):
                report['cumulative_max_memory' + '_' + local_name] = local_val

    if hasattr(trial, 'optimization') and hasattr(trial.optimization, 'global_outputs_opt') and hasattr(trial.optimization.global_outputs_opt, 'keys'):
        for odx in range(trial.optimization.global_outputs_opt.shape[0]):
            local_val = trial.optimization.global_outputs_opt.cat[odx]
            if include_val(local_val):
                report[trial.optimization.global_outputs_opt.labels()[odx]] = local_val

    interesting_output_types = ['vortex', 'geometry', 'aerodynamics']
    for interesting_output in interesting_output_types:
        if interesting_output in plot_dict['interpolation_si']['outputs'].keys():
            for local_name, local_val in plot_dict['interpolation_si']['outputs'][interesting_output].items():
                if len(local_val) == 1:
                    local_interest = {}
                    local_interest['avg'] = np.mean(local_val)
                    local_interest['stdev'] = np.std(local_val)
                    local_interest['min'] = np.min(local_val)
                    local_interest['max'] = np.max(local_val)
                    for interest_key in ['avg', 'stdev', 'min', 'max']:
                        report[interesting_output + '_' + local_name + '_' + interest_key] = local_interest[interest_key]

    time_period = trial.optimization.global_outputs_opt['time_period'].full()[0][0]
    avg_power_watts = trial.optimization.global_outputs_opt['avg_power_watts'].full()[0][0]
    e_final_joules = trial.optimization.global_outputs_opt['e_final_joules'].full()[0][0]
    if include_val(e_final_joules):
        report['solution_total_energy_joules'] = e_final_joules
    if include_val(avg_power_watts):
        report['solution_average_power_kw'] = avg_power_watts * 1.e-3
    if include_val(time_period):
        report['solution_time_period'] = time_period

    local_name = 'l_t'
    local_val = plot_dict['interpolation_si']['x'][local_name]
    local_interest = {}
    local_interest['avg'] = np.mean(local_val)
    local_interest['stdev'] = np.std(local_val)
    local_interest['min'] = np.min(local_val)
    local_interest['max'] = np.max(local_val)
    for interest_key in ['avg', 'stdev', 'min', 'max']:
        report['solution_' + local_name + '_' + interest_key] = local_interest[interest_key]

    for local_name in trial.model.variables_dict['theta'].keys():
        if local_name != 't_f':
            local_val = trial.optimization.V_final_si['theta', local_name]
            if include_val(local_val):
                report['solution_' + local_name] = local_val

    for local_name, local_val in trial.visualization.plot_dict['power_and_performance'].items():
        if include_val(local_val):
            report['p&p_' + local_name] = local_val

    for local_name in options.keys():
        local_val = options[local_name]
        if include_val(local_val):
            report[local_name] = local_val
    print_op.print_dict_as_table(report, level='info')

    filename = get_summary_csv_filename(trial)
    save_op.write_or_append_two_column_dict_to_csv(report, filename)
    return None

def get_summary_csv_filename(trial):
    return 's-' + trial.name + '.csv'

