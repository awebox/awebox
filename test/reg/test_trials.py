#!/usr/bin/python3
"""Template for trial tests

@author: Thilo Bronnenmeyer, kiteswarms 2018

- edit: Rachel Leuthold, ALU-FR 2020
"""

import collections
import copy
import logging

import awebox.opts.kite_data.ampyx_data as ampyx_data
import awebox.opts.kite_data.bubbledancer_data as bubbledancer_data
import awebox.opts.kite_data.boeing747_data as boeing747_data
import awebox.opts.options as options
import awebox.trial as awe_trial
import awebox.tools.print_operations as print_op

logging.basicConfig(filemode='w',format='%(levelname)s:    %(message)s', level=logging.WARNING)

def generate_options_dict():
    """
    Set options for the trials that should be tested and store them in dictionary
    :return: dictionary with trial options
    """

    # set options
    single_kite_options = options.Options(internal_access = True)
    single_kite_options['user_options']['system_model']['architecture'] = {1:0}
    single_kite_options['user_options']['trajectory']['lift_mode']['windings'] = 3
    single_kite_options['user_options']['kite_standard'] = ampyx_data.data_dict()
    single_kite_options['user_options']['system_model']['kite_dof'] = 3
    single_kite_options['user_options']['induction_model'] = 'not_in_use'
    single_kite_options['user_options']['tether_drag_model'] = 'split'

    drag_mode_options = copy.deepcopy(single_kite_options)
    drag_mode_options['user_options']['trajectory']['system_type'] = 'drag_mode'

    save_trial_options = copy.deepcopy(single_kite_options)
    save_trial_options['solver']['save_trial'] = True

    multi_tether_options = copy.deepcopy(single_kite_options)
    multi_tether_options['user_options']['tether_drag_model'] = 'multi'

    dual_kite_options = copy.deepcopy(single_kite_options)
    dual_kite_options['user_options']['system_model']['architecture'] = {1:0, 2:1, 3:1}

    dual_kite_6_dof_options = copy.deepcopy(dual_kite_options)
    dual_kite_6_dof_options['user_options']['system_model']['kite_dof'] = 6

    small_dual_kite_options = copy.deepcopy(dual_kite_6_dof_options)
    small_dual_kite_options['user_options']['kite_standard'] = bubbledancer_data.data_dict()
    small_dual_kite_options['params']['ground_station']['r_gen'] = 0.1
    small_dual_kite_options['params']['ground_station']['m_gen'] = 5.
    small_dual_kite_options['user_options']['trajectory']['lift_mode']['windings'] = 1

    actuator_qaxi_options = copy.deepcopy(dual_kite_6_dof_options)
    actuator_qaxi_options['user_options']['induction_model'] = 'actuator'
    actuator_qaxi_options['model']['aero']['actuator']['steadyness'] = 'quasi-steady'
    actuator_qaxi_options['model']['aero']['actuator']['symmetry'] = 'axisymmetric'
    actuator_qaxi_options['user_options']['trajectory']['lift_mode']['windings'] = 1
    actuator_qaxi_options['nlp']['n_k'] = 20

    actuator_uaxi_options = copy.deepcopy(actuator_qaxi_options)
    actuator_uaxi_options['model']['aero']['actuator']['steadyness'] = 'unsteady'

    actuator_qasym_options = copy.deepcopy(actuator_qaxi_options)
    actuator_qasym_options['model']['aero']['actuator']['symmetry'] = 'asymmetric'
    actuator_qasym_options['model']['aero']['actuator']['a_range'] = [-0.06, 0.06]

    actuator_uasym_options = copy.deepcopy(actuator_qasym_options)
    actuator_uasym_options['model']['aero']['actuator']['steadyness'] = 'unsteady'
    actuator_uasym_options['nlp']['n_k'] = 15

    actuator_comparison_options = copy.deepcopy(actuator_qaxi_options)
    actuator_comparison_options['model']['aero']['actuator']['steadyness_comparison'] = ['q', 'u']

    dual_kite_tracking_options = copy.deepcopy(dual_kite_6_dof_options)
    dual_kite_tracking_options['user_options']['trajectory']['type'] = 'tracking'
    dual_kite_tracking_options['user_options']['trajectory']['lift_mode']['windings'] = 1
    dual_kite_tracking_options['user_options']['trajectory']['tracking']['fix_tether_length'] = True
    dual_kite_tracking_options['nlp']['n_k'] = 20

    dual_kite_tracking_winch_options = copy.deepcopy(dual_kite_tracking_options)
    dual_kite_tracking_winch_options['user_options']['trajectory']['tracking']['fix_tether_length'] = False

    # nominal landing
    nominal_landing_options = copy.deepcopy(dual_kite_options)
    nominal_landing_options['user_options']['trajectory']['type'] = 'nominal_landing'
    nominal_landing_options['user_options']['trajectory']['transition']['initial_trajectory'] = 'dual_kite_trial.dict'
    nominal_landing_options['solver']['initialization']['initialization_type'] = 'modular'

    # compromised landing
    compromised_landing_options = copy.deepcopy(nominal_landing_options)
    compromised_landing_options['user_options']['trajectory']['type'] = 'compromised_landing'
    compromised_landing_options['model']['model_bounds']['dcoeff_compromised_factor'] = 0.0
    compromised_landing_options['user_options']['trajectory']['compromised_landing']['emergency_scenario'] = ('broken_roll', 2)
    compromised_landing_options['user_options']['trajectory']['compromised_landing']['xi_0_initial'] = 0.8

    # define options list
    options_dict = collections.OrderedDict()
    options_dict['single_kite_trial'] = single_kite_options
    options_dict['drag_mode_trial'] = drag_mode_options
    options_dict['save_trial'] = save_trial_options
    # options_dict['multi_tether_trial'] = multi_tether_options
    options_dict['dual_kite_trial'] = dual_kite_options
    options_dict['small_dual_kite_trial'] = small_dual_kite_options
    options_dict['dual_kite_6_dof_trial'] = dual_kite_6_dof_options
    options_dict['actuator_qaxi_trial'] = actuator_qaxi_options
    options_dict['actuator_uaxi_trial'] = actuator_uaxi_options
    options_dict['actuator_qasym_trial'] = actuator_qasym_options
    options_dict['actuator_uasym_trial'] = actuator_uasym_options
    options_dict['actuator_comparison_trial'] = actuator_comparison_options
    options_dict['dual_kite_tracking_trial'] = dual_kite_tracking_options
    options_dict['dual_kite_tracking_winch_trial'] = dual_kite_tracking_winch_options
    # options_dict['nominal_landing_trial'] = nominal_landing_options
    # options_dict['compromised_landing_trial'] = compromised_landing_options

    return options_dict

def generate_options_dict_for_trials_that_we_dont_expect_to_solve():

    vortex_options = options.Options(internal_access = True)
    vortex_options['user_options']['system_model']['architecture'] = {1:0, 2:1, 3:1}
    vortex_options['user_options']['kite_standard'] = ampyx_data.data_dict()
    vortex_options['user_options']['system_model']['kite_dof'] = 6
    vortex_options['user_options']['induction_model'] = 'vortex'
    vortex_options['user_options']['tether_drag_model'] = 'split'
    vortex_options['nlp']['n_k'] = 3
    vortex_options['user_options']['trajectory']['lift_mode']['windings'] = 1
    vortex_options['model']['aero']['vortex']['wake_nodes'] = 3
    vortex_options['solver']['max_iter'] = 2

    options_dict = collections.OrderedDict()
    options_dict['vortex_trial'] = vortex_options

    return options_dict


def test_trials():
    """
    Test all trials that are defined in options_dict
    :return: None
    """

    # generate options_dict
    options_dict = generate_options_dict()

    # loop over trials
    for trial_name in list(options_dict.keys()):
        trial_options = options_dict[trial_name]
        solve_and_check(trial_options, trial_name)

    options_dict_dont_solve = generate_options_dict_for_trials_that_we_dont_expect_to_solve()

    # loop over trials
    for trial_name in list(options_dict_dont_solve.keys()):
        trial_options = options_dict_dont_solve[trial_name]
        solve_trial(trial_options, trial_name)

    return None

def solve_and_check(trial_options, trial_name):
    """
    Solve one individual trial and run tests on it
    :param trial_options: options of a single trial
    :param trial_name: name of the trial
    :param test_param_dict: dictionary with test parameters
    :return: None
    """

    # compute trajectory solution
    trial = solve_trial(trial_options, trial_name)

    # evaluate results
    evaluate_results(trial.quality.results, trial_name)

    return None

def evaluate_results(results, trial_name):

    # loop over all results
    for result in list(results.keys()):
        assert results[result] == True, 'Test failed for ' + trial_name + ', Test regarding ' + result + ' failed.'

    return None

def solve_trial(trial_options, trial_name):
    """
    Set up and solve trial
    :return: solved trial
    """

    trial = awe_trial.Trial(trial_options, trial_name)
    trial.build()
    trial.optimize()

    return trial