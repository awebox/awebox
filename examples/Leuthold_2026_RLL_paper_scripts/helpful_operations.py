#!/usr/bin/python3
from platform import architecture

import matplotlib
# matplotlib.use('TkAgg')

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


def get_basic_options_for_convergence_expense_and_comparison(options):

    # base problem definition
    options['user_options.system_model.architecture'] = {1: 0, 2: 1, 3: 1}
    options = ampyx_ap2_settings.set_ampyx_ap2_settings(options)
    options['user_options.system_model.kite_dof'] = 6
    options['model.system_bounds.theta.t_f'] = [5., 35.]  # [s]
    options['user_options.trajectory.lift_mode.windings'] = 1

    options['user_options.induction_model'] = 'vortex'
    options['model.aero.vortex.far_wake_element_type'] = 'semi_infinite_filament'
    options['model.aero.vortex.representation'] = 'alg'
    options['model.aero.vortex.convection_type'] = 'rigid'
    options['model.aero.vortex.core_to_chord_ratio'] = 0.05
    options['model.aero.actuator.a_ref'] = 0.1
    options['model.aero.vortex.wu_ind_scaling_method'] = 'infty_ref'

    options['model.model_bounds.tether_stress.include'] = True
    options['model.model_bounds.tether_force.include'] = False
    options['user_options.trajectory.fixed_params'] = {}
    
    options['model.model_bounds.anticollision.safety_factor'] = 2

    options['solver.linear_solver'] = 'ma86'
    options['solver.hippo_strategy'] = True
    options['solver.homotopy_method.advance_despite_max_iter'] = False
    options['solver.raise_error_at_max_time'] = True
    options['solver.max_iter'] = 2e4
    options['solver.max_iter_hippo'] = options['solver.max_iter']
    options['solver.max_cpu_time'] = 1e10 * 60. * 60.  # set the max cpu time ridiculously high so that it won't kill the large problems

    options['solver.tol'] = 1e-8
    options['solver.mu_hippo'] = 1e-1
    
    options['visualization.cosmetics.induction.n_points_contour'] = 300
    options['visualization.cosmetics.interpolation.n_points'] = 300
    options['model.aero.actuator.geometry_overwrite'] = 'averaged' 

    # these options gave the best output over a tuning sweep
    options['model.scaling.other.flight_radius_estimate'] = 'anticollision'
    options['model.scaling.other.period_estimate'] = 't_f_bounds'
    options['model.scaling.other.position_scaling_method'] = 'radius_and_tether'
    options['model.scaling.other.force_scaling_method'] = 'aero'
    options['model.scaling.other.tension_estimate'] = 'power'
    options['model.scaling.other.power_estimate'] = 'loyd'
    options["model.scaling.other.power_estimate"] = 'synthesized'
    options["model.aero.vortex.position_scaling_method"] = 'b_ref'
    options["model.aero.vortex.rate_of_change_scaling_factor"] = 0.001
    options["model.aero.actuator.a_ref"] = 0.3
    options['solver.cost_factor.power'] = 10.
    options['solver.cost.psi.1'] = 100.
    options["solver.cost.u_regularisation.0"] = 1e-6

    options['nlp.phase_fix_reelout'] = 0.55

    #options['solver.linear_solver'] = 'ma86' # 'ma86'
    #options['solver.max_iter'] = 2e3
    #options['solver.max_iter_hippo'] = options['solver.max_iter']


    return options


def from_wake_nodes_to_periods_tracked(n_k, wake_nodes): 
    periods_tracked = float(wake_nodes - 1.) / float(n_k)
    return periods_tracked


def from_periods_tracked_to_wake_nodes(n_k, periods_tracked):
    wake_nodes = int(np.ceil(n_k * periods_tracked + 1))
    return wake_nodes


def build_unique_trial_name(base_name, inputs):

    trial_name_baseline = base_name    
    for name, val in inputs.items():
        trial_name_baseline += '_' + name + '_' + str(val)

    today = date.today()
    rand = random.randint(1000000, 9000000)    
    unique_addition = '_' + str(today) + '_' + str(rand)
    
    file_extension_length = 5
    contd_addition = '_contd_'    
    max_characters_total = 240 # 255 max characters in linux

    max_name_length = max_characters_total - len(unique_addition) - file_extension_length
    if len(trial_name_baseline) > max_name_length:
        trial_name_baseline = trial_name_baseline[:max_name_length-len(contd_addition)] + contd_addition
    	
    trial_name_baseline += unique_addition
    return trial_name_baseline
    
def toggle_baseline_options(options):
    options['user_options.induction_model'] = 'not_in_use'
    options['user_options.trajectory.lift_mode.phase_fix'] = 'single_reelout' # the default option
    options['visualization.cosmetics.plot_ref'] = False # the 'refernence' is just a circle.
    options['solver.hippo_strategy'] = True # the default interior-point homotopy embedding
    options['solver.linear_solver'] = 'ma57' # the repeatable option
    options['solver.homotopy_method.put_induction_step_after'] = 'fictitious'
    return options
    
def toggle_simulation_options(options):
    options['user_options.induction_model'] = 'vortex'
    #options['user_options.trajectory.lift_mode.phase_fix'] = 'simple' # notice that the -inf <= dl_t <= inf bounds, will achieve this, without upsetting the warmstart.
    options['visualization.cosmetics.plot_ref'] = True # the 'reference' here is the baseline problem
    options['solver.hippo_strategy'] = False # save memory by only requring one casadi solver
    options['solver.linear_solver'] = 'ma86' # the parallelized but non-repeatable option
    options['solver.homotopy_method.put_induction_step_after'] = 'initial'  # we need the fictitious forces to still be enabled, so that we can perfectly fly the simulation/reference trajectory with different aerodynamics. The fictitious forces and moments are not passed to the constraints that determine the circulation.
    return options
    
def toggle_tracking_options(options):
    options['user_options.induction_model'] = 'vortex'
    #options['user_options.trajectory.lift_mode.phase_fix'] = 'simple' # notice that the -inf <= dl_t <= inf bounds, will achieve this, without upsetting the warmstart.
    options['visualization.cosmetics.plot_ref'] = True # the 'reference' here is the baseline problem
    options['solver.hippo_strategy'] = False # save memory by only requring one casadi solver
    options['solver.linear_solver'] = 'ma86' # the parallelized but non-repeatable option
    options['solver.homotopy_method.put_induction_step_after'] = 'fictitious' # in the tracking problem, we want a 'physical' trajectory, meaning: no fictitious forces.
    return options

def toggle_basic_health_options(options):
    options['nlp.collocation.u_param'] = 'zoh'
    options['solver.hippo_strategy'] = False

    options['solver.health_check.when'] = 'success'
    options['nlp.collocation.name_constraints'] = True
    options['solver.health_check.help_with_debugging'] = False
    options['model.scaling.other.print_help_with_scaling'] = True

    options['solver.homotopy_method.advance_despite_max_iter'] = False
    options['solver.homotopy_method.advance_despite_ill_health'] = False
    options['solver.homotopy_method.consider_restoration_as_failure'] = False #True
    options['solver.health_check.raise_exception'] = False #True
    options['solver.initialization.check_reference'] = True
    options['solver.initialization.check_feasibility.raise_exception'] = False #True
    options['solver.max_iter'] = 2000
    options['solver.max_iter_hippo'] = 2000
    options['solver.ipopt.autoscale'] = False
    options['solver.health_check.spy_matrices'] = False
    options['quality.when'] = 'never'
    options['visualization.cosmetics.variables.si_or_scaled'] = 'si'
    options['solver.health_check.save_health_indicators'] = True
    options['solver.health_check.thresh.condition_number'] = 1e10
    return options
    

def get_list_of_plots():
    list_of_plots = ['power', 'states', 'algebraic_variables', 'controls', 'constraints', 'animation_snapshot',  'isometric', 'projected_xy', 'projected_yz', 'projected_xz', 'wake_isometric', 'wake_xy', 'wake_xz', 'wake_yz', 'wake_legend', 'velocity_deficits', 'velocity_distribution', 'aero_dimensionless', 'relative_radius', 'relative_radius_of_curvature', 'aero_coefficients', 'circulation', 'local_induction_factor_all_projections', 'induction_wind_tunnel',  'induction_contour_normal_wind']
    # 'induction_contour_wind_wind']
    #'induction_contour_normal_normal', ,
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


    
def adjust_weights_for_tracking(trial_baseline, options, ratio_power_to_tracking=1e-6):

    # everything the same, as in the default tracking part of the awebox homotopy, except with adjusted weights on variously power or pathfollowing vars.
    power_vars = ['lambda', 'l_t', 'dl_t']
    tracking_vars = ['q', 'dq', 'r', 'omega']

    factor_power = ratio_power_to_tracking**0.5
    factor_tracking = 1./factor_power

    for power_var_name in power_vars:
        local_opt_name = 'solver.weights.' + power_var_name
        options[local_opt_name] = factor_power * trial_baseline.options['solver']['weights'][power_var_name]
    for tracking_var_name in tracking_vars:
        local_opt_name = 'solver.weights.' + tracking_var_name
        options[local_opt_name] = factor_tracking * trial_baseline.options['solver']['weights'][tracking_var_name]

    return options


    
def adjust_weights_for_simulation(trial_baseline, options):

    options['solver.cost.beta.0'] = 0.
    options['nlp.cost.beta'] = False

    options['solver.weights.vortex'] = 0.
    options['solver.cost.fictitious.0'] = 1.e-10
    options['solver.cost.fictitious.1'] = 1.e-10

    really_really_extra_more_important = 1e4
    extra_much_more_important = 1e3
    much_more_important = 1e2
    more_important = 1e1
    less_important = 1e-1
    much_less_important = 1e-2
    would_be_zero_except_sosc = 1e-4
    baseline_options = trial_baseline.options
        
    options['solver.weights.q'] = extra_much_more_important * baseline_options['solver']['weights']['q']
    options['solver.weights.dq'] = more_important * baseline_options['solver']['weights']['dq']
    options['solver.weights.r'] = much_more_important * baseline_options['solver']['weights']['r']
    options['solver.weights.omega'] = more_important * baseline_options['solver']['weights']['omega']
    
    options['solver.weights.coeff'] = extra_much_more_important * baseline_options['solver']['weights']['coeff']
    options['solver.weights.delta'] = much_more_important * baseline_options['solver']['weights']['delta']
    options['solver.weights.ddelta'] = extra_much_more_important * baseline_options['solver']['weights']['ddelta']

    options['solver.weights.l_t'] = more_important * baseline_options['solver']['weights']['l_t']
    options['solver.weights.dl_t'] = more_important * baseline_options['solver']['weights']['dl_t']
    options['solver.weights.lambda'] = 1e-10

    options['solver.cost.t_f.0'] = more_important * baseline_options['solver']['cost']['theta_regularisation'][0] # penalizes switching time, which is important so that time matches from pathfollowing -> trajectory tracking
    options['solver.cost.u_regularisation.0'] = less_important * baseline_options['solver']['cost']['u_regularisation'][
        0]
    options['solver.cost.tracking.0'] = 1. #more_important * baseline_options['solver']['cost']['tracking'][0]

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
            try:
                report['tests'] = trial.quality.all_tests_passed()
            except:
                report['tests'] = 'na'
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


def add_from_trial_object_to_report(report, trial, trial_object_name, attr_name):
    if hasattr(trial, trial_object_name):
        trial_object = getattr(trial, trial_object_name)
        if hasattr(trial_object, attr_name):
            local_dict = getattr(trial_object, attr_name)
            if isinstance(local_dict, dict):
                for local_name, local_val in local_dict.items():
                    if include_val(local_val):
                        report[trial_object_name + '_' + attr_name + '_' + local_name] = local_val
    return report

def save_and_print_info(trial, options):

    # this one saves all of the interpolated variable and output information as time-series
    try:
        trial.write_to_csv()
    except:
        message = 'something went wrong with write_to_csv'
        print(message)

    # everything below saves a summary file, so that you don't have to load and re-average the full time-series datafile

    plot_dict = trial.visualization.plot_dict

    report = {}
    report['trial_name'] = trial.name

    report['count'] = 0
    report['n_k'] = options['nlp.n_k']
    report['d'] = trial.options['nlp']['collocation']['d']
    try:
        n_k = report['n_k']
        wake_nodes = trial.options['model']['aero']['vortex']['wake_nodes']
        p_t = float(wake_nodes - 1.) / n_k
        report['p_t'] = p_t
        report['tol'] = trial.options['solver']['tol']
        number_of_kites = trial.model.architecture.number_of_kites
        at_slice = 3 * number_of_kites * wake_nodes
        report['vortex_elements_at_slice'] = at_slice
        report['vortex_elements_all_time'] = at_slice * (n_k + 1 + n_k * report['d'])
    except:
        pass

    phase_fix_reelout = trial.options['nlp']['phase_fix_reelout']
    n_k_reelout = round(n_k * phase_fix_reelout)
    t_f = trial.optimization.V_final_si['theta', 't_f']
    t_switch = float(t_f[0] * n_k_reelout / n_k)
    report['t_switch'] = t_switch
    
    time_period = trial.optimization.global_outputs_opt['time_period'].full()[0][0]
    report['phi_switch'] = t_switch/time_period

    report['solve'] = trial.optimization.solve_succeeded
    if report['solve']:
        try:
            report['tests'] = trial.quality.all_tests_passed()
        except:
            report['tests'] = 'na'

    report['model_variables'] = np.prod(trial.model.variables.shape)
    report = add_from_trial_object_to_report(report, trial, 'model', 'dimensions_dict')

    report['nlp_variables'] = np.prod(trial.nlp.V.shape)
    report = add_from_trial_object_to_report(report, trial, 'nlp', 'dimensions_dict')

    report = add_from_trial_object_to_report(report, trial, 'quality', 'results')
    
    for optimization_attr in ['stats', 'iterations', 'timings', 'cpu_timings', 't_wall', 't_proc', 'cumulative_max_memory']:
        report = add_from_trial_object_to_report(report, trial, 'optimization', optimization_attr)

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

