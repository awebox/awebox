#!/usr/bin/python3
"""Template for trial tests

@author: Thilo Bronnenmeyer, kiteswarms 2018

- edited: Jochem De Schutter ALU-FR 2020-21
- edited: rachel leuthold, alu-fr, 2020-2025
"""

import collections
import copy
import logging

import awebox as awe

import awebox.opts.kite_data.ampyx_data as ampyx_data
import awebox.opts.kite_data.bubbledancer_data as bubbledancer_data
import awebox.opts.kite_data.boeing747_data as boeing747_data
import awebox.trial as awe_trial
import awebox.tools.save_operations as save_op
import awebox.tools.print_operations as print_op
import awebox.tools.vector_operations as vect_op
import awebox.tools.struct_operations as struct_op
import numpy as np

from awebox.logger.logger import Logger as awelogger

import matplotlib.pyplot as plt
awelogger.logger.setLevel(10)



logging.basicConfig(filemode='w',format='%(levelname)s:    %(message)s', level=logging.DEBUG)


# 1
def test_single_kite_basic_health(final_homotopy_step='final', overwrite_options={}):
    trial_name = 'single_kite_basic_health_trial'
    run_test(trial_name, final_homotopy_step=final_homotopy_step, overwrite_options=overwrite_options)
    return None


# 2
def test_single_kite(final_homotopy_step='final', overwrite_options={}):
    trial_name = 'single_kite_trial'
    run_test(trial_name, final_homotopy_step=final_homotopy_step, overwrite_options=overwrite_options)
    return None


# 3
def test_single_kite_6_dof_basic_health(final_homotopy_step='final', overwrite_options={}):
    trial_name = 'single_kite_6_dof_basic_health_trial'
    run_test(trial_name, final_homotopy_step=final_homotopy_step, overwrite_options=overwrite_options)
    return None


# 4
def test_single_kite_6_dof(final_homotopy_step='final', overwrite_options={}):
    trial_name = 'single_kite_6_dof_trial'
    run_test(trial_name, final_homotopy_step=final_homotopy_step, overwrite_options=overwrite_options)
    return None


# 5
def test_poly(final_homotopy_step='final', overwrite_options={}):
    trial_name = 'poly_trial'
    run_test(trial_name, final_homotopy_step=final_homotopy_step, overwrite_options=overwrite_options)
    return None


# 6
def test_drag_mode(final_homotopy_step='final', overwrite_options={}):
    trial_name = 'drag_mode_trial'
    run_test(trial_name, final_homotopy_step=final_homotopy_step, overwrite_options=overwrite_options)
    return None


# 7
def test_save_trial(final_homotopy_step='initial', overwrite_options={}):
    trial_name = 'save_trial'
    run_test(trial_name, final_homotopy_step=final_homotopy_step, overwrite_options=overwrite_options)
    return None


# 8
def test_dual_kite(final_homotopy_step='final', overwrite_options={}):
    trial_name = 'dual_kite_trial'
    run_test(trial_name, final_homotopy_step=final_homotopy_step, overwrite_options=overwrite_options)
    return None


# 9
def test_dual_kite_basic_health(final_homotopy_step='final', overwrite_options={}):
    trial_name = 'dual_kite_basic_health_trial'
    run_test(trial_name, final_homotopy_step=final_homotopy_step, overwrite_options=overwrite_options)
    return None


# 10
def test_dual_kite_6_dof(final_homotopy_step='final', overwrite_options={}):
    trial_name = 'dual_kite_6_dof_trial'
    run_test(trial_name, final_homotopy_step=final_homotopy_step, overwrite_options=overwrite_options)
    return None


# 11
def test_dual_kite_6_dof_basic_health(final_homotopy_step='final', overwrite_options={}):
    trial_name = 'dual_kite_6_dof_basic_health_trial'
    run_test(trial_name, final_homotopy_step=final_homotopy_step, overwrite_options=overwrite_options)
    return None


# 12
def test_dual_kite_tracking(final_homotopy_step='final', overwrite_options={}):
    trial_name = 'dual_kite_tracking_trial'
    run_test(trial_name, final_homotopy_step=final_homotopy_step, overwrite_options=overwrite_options)
    return None


# 13
def test_dual_kite_tracking_winch(final_homotopy_step='final', overwrite_options={}):
    trial_name = 'dual_kite_tracking_winch_trial'
    run_test(trial_name, final_homotopy_step=final_homotopy_step, overwrite_options=overwrite_options)
    return None


# 14
def test_vortex_force_zero_basic_health(final_homotopy_step='final', overwrite_options={}):
    trial_name = 'vortex_force_zero_basic_health_trial'
    run_test(trial_name, final_homotopy_step=final_homotopy_step, overwrite_options=overwrite_options)
    return None


# 15
def test_vortex_force_zero(final_homotopy_step='final', overwrite_options={}):
    trial_name = 'vortex_force_zero_trial'
    run_test(trial_name, final_homotopy_step=final_homotopy_step, overwrite_options=overwrite_options)
    return None


# 16
def test_vortex_basic_health(final_homotopy_step='final', overwrite_options={}):
    trial_name = 'vortex_basic_health_trial'
    run_test(trial_name, final_homotopy_step=final_homotopy_step, overwrite_options=overwrite_options)
    return None


# 17
def test_vortex(final_homotopy_step='final', overwrite_options={}):
    trial_name = 'vortex_trial'
    run_test(trial_name, final_homotopy_step=final_homotopy_step, overwrite_options=overwrite_options)
    return None

# 18
def test_vortex_3_dof(final_homotopy_step='final', overwrite_options={}):
    trial_name = 'vortex_3_dof_trial'
    run_test(trial_name, final_homotopy_step=final_homotopy_step, overwrite_options=overwrite_options)
    return None

# 19
def test_segmented_tether(final_homotopy_step='final', overwrite_options={}):
    trial_name = 'segmented_tether_trial'
    run_test(trial_name, final_homotopy_step=final_homotopy_step, overwrite_options=overwrite_options)
    return None


# def test_small_dual_kite(final_homotopy_step='final', overwrite_options={}):
#     trial_name = 'small_dual_kite_trial'
#     run_test(trial_name, final_homotopy_step=final_homotopy_step, overwrite_options=overwrite_options)
#     return None
#
# def test_small_dual_kite_basic_health(final_homotopy_step='final', overwrite_options={}):
#     trial_name = 'small_dual_kite_basic_health_trial'
#     run_test(trial_name, final_homotopy_step=final_homotopy_step, overwrite_options=overwrite_options)
#     return None
#
# def test_large_dual_kite(final_homotopy_step='final', overwrite_options={}):
#     trial_name = 'large_dual_kite_trial'
#     run_test(trial_name, final_homotopy_step=final_homotopy_step, overwrite_options=overwrite_options)
#     return None
#
# def test_large_dual_kite_basic_health(final_homotopy_step='final', overwrite_options={}):
#     trial_name = 'large_dual_kite_basic_health_trial'
#     run_test(trial_name, final_homotopy_step=final_homotopy_step, overwrite_options=overwrite_options)
#     return None


# def test_actuator_qaxi(final_homotopy_step='final', overwrite_options={}):
#     trial_name = 'actuator_qaxi_trial'
#     run_test(trial_name, final_homotopy_step=final_homotopy_step, overwrite_options=overwrite_options)
#     return None
#
# def test_actuator_qaxi_basic_health(final_homotopy_step='final', overwrite_options={}):
#     trial_name = 'actuator_qaxi_basic_health_trial'
#     run_test(trial_name, final_homotopy_step=final_homotopy_step, overwrite_options=overwrite_options)
#     return None
#
#
# def test_actuator_uaxi(final_homotopy_step='final', overwrite_options={}):
#     trial_name = 'actuator_uaxi_trial'
#     run_test(trial_name, final_homotopy_step=final_homotopy_step, overwrite_options=overwrite_options)
#     return None
#
#
# def test_actuator_qasym(final_homotopy_step='final', overwrite_options={}):
#     trial_name = 'actuator_qasym_trial'
#     run_test(trial_name, final_homotopy_step=final_homotopy_step, overwrite_options=overwrite_options)
#     return None
#
#
# def test_actuator_uasym(final_homotopy_step='final', overwrite_options={}):
#     trial_name = 'actuator_uasym_trial'
#     run_test(trial_name, final_homotopy_step=final_homotopy_step, overwrite_options=overwrite_options)
#     return None
#
#
# def test_actuator_comparison(final_homotopy_step='final', overwrite_options={}):
#     trial_name = 'actuator_comparison_trial'
#     run_test(trial_name, final_homotopy_step=final_homotopy_step, overwrite_options=overwrite_options)
#     return None




def make_basic_health_variant(base_options):
    basic_health_options = copy.deepcopy(base_options)

    basic_health_options['user_options.trajectory.lift_mode.windings'] = 1
    basic_health_options['nlp.n_k'] = 9 #6 #9 # try to decrease this.
    basic_health_options['nlp.collocation.d'] = 3
    basic_health_options['nlp.collocation.u_param'] = 'zoh'
    basic_health_options['solver.hippo_strategy'] = False

    basic_health_options['solver.health_check.when'] = 'success'
    basic_health_options['nlp.collocation.name_constraints'] = True
    basic_health_options['solver.health_check.help_with_debugging'] = True #False

    basic_health_options['solver.homotopy_method.advance_despite_max_iter'] = False
    basic_health_options['solver.homotopy_method.advance_despite_ill_health'] = False
    basic_health_options['solver.health_check.raise_exception'] = True
    basic_health_options['solver.initialization.check_reference'] = True
    basic_health_options['solver.initialization.check_feasibility.raise_exception'] = True
    basic_health_options['solver.max_iter'] = 300
    basic_health_options['solver.ipopt.autoscale'] = False
    basic_health_options['solver.health_check.spy_matrices'] = False
    basic_health_options['quality.when'] = 'never'
    basic_health_options['visualization.cosmetics.variables.si_or_scaled'] = 'si' #'scaled'
    basic_health_options['solver.health_check.save_health_indicators'] = True
    basic_health_options['solver.health_check.thresh.condition_number'] = 1e10


    return basic_health_options


def generate_options_dict():
    """
    Set options for the trials that should be tested and store them in dictionary
    :return: dictionary with trial options
    """

    # set options
    single_kite_options = {}
    single_kite_options['user_options.system_model.architecture'] = {1: 0}
    single_kite_options['user_options.system_model.kite_dof'] = 3
    single_kite_options['user_options.kite_standard'] = awe.ampyx_data.data_dict()
    single_kite_options['user_options.trajectory.system_type'] = 'lift_mode'
    single_kite_options['user_options.trajectory.lift_mode.windings'] = 1
    single_kite_options['model.tether.aero_elements'] = 1
    single_kite_options['user_options.induction_model'] = 'not_in_use'
    single_kite_options['solver.linear_solver'] = 'ma57'
    single_kite_options['nlp.collocation.u_param'] = 'zoh'
    single_kite_options['nlp.n_k'] = 20
    single_kite_options['quality.raise_exception'] = True
    single_kite_options['visualization.cosmetics.plot_eq_constraints'] = True
    single_kite_options['visualization.cosmetics.trajectory.reel_in_linestyle'] = '--'

    single_kite_basic_health_options = make_basic_health_variant(single_kite_options)

    single_kite_6_dof_options = copy.deepcopy(single_kite_options)
    single_kite_6_dof_options['user_options.system_model.kite_dof'] = 6
    # single_kite_6_dof_options['solver.weights.r'] = 1e0

    single_kite_6_dof_basic_health_options = make_basic_health_variant(single_kite_6_dof_options)

    segmented_tether_options = copy.deepcopy(single_kite_options)
    segmented_tether_options['user_options.system_model.architecture'] = {1: 0, 2:1, 3:2, 4:3}

    poly_options = copy.deepcopy(single_kite_options)
    poly_options['nlp.collocation.u_param'] = 'poly'
    poly_options['solver.cost_factor.power'] = 1e1  # 1e4

    drag_mode_options = copy.deepcopy(single_kite_options)
    drag_mode_options['user_options.trajectory.system_type'] = 'drag_mode'
    drag_mode_options['quality.test_param.power_balance_thresh'] = 2.
    drag_mode_options['model.system_bounds.theta.t_f'] = [5., 70.]  # [s]
    drag_mode_options['nlp.n_k'] = 30
    drag_mode_options['solver.cost_factor.power'] = 1e0  # 1e4

    save_trial_options = copy.deepcopy(single_kite_options)
    save_trial_options['solver.save_trial'] = True

    dual_kite_options = copy.deepcopy(single_kite_options)
    dual_kite_options['user_options.system_model.architecture'] = {1: 0, 2: 1, 3: 1}
    dual_kite_options['solver.initialization.theta.l_s'] = 75.
    dual_kite_options['solver.initialization.check_reference'] = True

    dual_kite_basic_health_options = make_basic_health_variant(dual_kite_options)

    dual_kite_6_dof_options = copy.deepcopy(single_kite_6_dof_options)
    dual_kite_6_dof_options['user_options.system_model.kite_dof'] = 6
    dual_kite_6_dof_options['user_options.system_model.architecture'] = {1: 0, 2: 1, 3: 1}
    dual_kite_6_dof_options['solver.initialization.theta.l_s'] = 75.

    dual_kite_6_dof_basic_health_options = make_basic_health_variant(dual_kite_6_dof_options)

    small_dual_kite_options = copy.deepcopy(dual_kite_6_dof_options)
    small_dual_kite_options['user_options.kite_standard'] = bubbledancer_data.data_dict()
    # small_dual_kite_options['model.system_bounds.theta.t_f'] = [5., 50.]
    small_dual_kite_options['solver.initialization.check_reference'] = True

    small_dual_kite_basic_health_options = make_basic_health_variant(small_dual_kite_options)

    large_dual_kite_options = copy.deepcopy(dual_kite_6_dof_options)
    large_dual_kite_options['user_options.kite_standard'] = boeing747_data.data_dict()
    large_dual_kite_options['solver.initialization.theta.l_s'] = 60. * 10.
    large_dual_kite_options['solver.initialization.l_t'] = 2.e3
    large_dual_kite_options['model.system_bounds.theta.t_f'] = [5., 5. * 60.]
    large_dual_kite_options['solver.initialization.groundspeed'] = 100.
    large_dual_kite_options['params.model_bounds.airspeed_limits'] = np.array([77., 273.])
    large_dual_kite_options['model.model_bounds.tether_force.include'] = True
    large_dual_kite_options['model.model_bounds.tether_stress.include'] = False
    large_dual_kite_options['params.model_bounds.tether_force_limits'] = np.array([1e0, 2e6])
    large_dual_kite_options['solver.initialization.check_reference'] = True

    large_dual_kite_basic_health_options = make_basic_health_variant(large_dual_kite_options)

    actuator_qaxi_options = copy.deepcopy(dual_kite_6_dof_options)
    actuator_qaxi_options['user_options.kite_standard'] = ampyx_data.data_dict()
    actuator_qaxi_options['user_options.induction_model'] = 'actuator'
    actuator_qaxi_options['model.aero.actuator.steadyness'] = 'quasi-steady'
    actuator_qaxi_options['model.aero.actuator.symmetry'] = 'axisymmetric'
    actuator_qaxi_options['visualization.cosmetics.trajectory.actuator'] = True
    actuator_qaxi_options['visualization.cosmetics.trajectory.kite_bodies'] = True
    actuator_qaxi_options['model.system_bounds.theta.a'] = [-0., 0.5]
    actuator_qaxi_options['user_options.trajectory.lift_mode.windings'] = 3

    actuator_qaxi_basic_health_options = make_basic_health_variant(actuator_qaxi_options)

    actuator_uaxi_options = copy.deepcopy(actuator_qaxi_options)
    actuator_uaxi_options['model.aero.actuator.steadyness'] = 'unsteady'
    actuator_uaxi_options['model.model_bounds.tether_stress.scaling'] = 10.

    actuator_qasym_options = copy.deepcopy(actuator_qaxi_options)
    actuator_qasym_options['model.aero.actuator.symmetry'] = 'asymmetric'

    actuator_uasym_options = copy.deepcopy(actuator_qaxi_options)
    actuator_uasym_options['model.aero.actuator.steadyness'] = 'unsteady'
    actuator_uasym_options['model.aero.actuator.symmetry'] = 'asymmetric'

    actuator_comparison_options = copy.deepcopy(actuator_qaxi_options)
    actuator_comparison_options['model.aero.actuator.steadyness_comparison'] = ['q', 'u']
    actuator_comparison_options['user_options.system_model.kite_dof'] = 6

    vortex_options = copy.deepcopy(single_kite_6_dof_options)
    vortex_options['user_options.trajectory.lift_mode.windings'] = 1
    vortex_options['user_options.induction_model'] = 'vortex'
    vortex_options['model.aero.vortex.representation'] = 'alg'
    vortex_options['quality.test_param.vortex_truncation_error_thresh'] = 1e20
    vortex_options['model.aero.vortex.far_wake_element_type'] = 'semi_infinite_filament'
    vortex_options['model.aero.vortex.wake_nodes'] = 2
    vortex_options['quality.raise_exception'] = False

    vortex_basic_health_options = make_basic_health_variant(vortex_options)
    vortex_basic_health_options['model.aero.vortex.double_check_wingtip_fixing'] = True

    vortex_force_zero_options = copy.deepcopy(vortex_options)
    vortex_force_zero_options['model.aero.induction.force_zero'] = True
    vortex_force_zero_options['nlp.collocation.d'] = 4
    vortex_force_zero_options['quality.raise_exception'] = True

    vortex_force_zero_basic_health_options = make_basic_health_variant(vortex_force_zero_options)
    vortex_force_zero_basic_health_options['model.aero.vortex.double_check_wingtip_fixing'] = True

    vortex_3_dof_options = copy.deepcopy(vortex_options)
    vortex_3_dof_options['user_options.system_model.kite_dof'] = 3

    dual_kite_tracking_options = copy.deepcopy(dual_kite_6_dof_options)
    dual_kite_tracking_options['user_options.trajectory.type'] = 'tracking'
    dual_kite_tracking_options['user_options.trajectory.lift_mode.windings'] = 1

    dual_kite_tracking_winch_options = copy.deepcopy(dual_kite_tracking_options)
    dual_kite_tracking_winch_options['user_options.trajectory.tracking.fix_tether_length'] = False

    # nominal landing
    nominal_landing_options = copy.deepcopy(dual_kite_options)
    nominal_landing_options['user_options.trajectory.type'] = 'nominal_landing'
    nominal_landing_options['user_options.trajectory.transition.initial_trajectory'] = 'dual_kite_trial.dict'
    nominal_landing_options['solver.initialization.initialization_type'] = 'modular'

    # compromised landing
    compromised_landing_options = copy.deepcopy(nominal_landing_options)
    compromised_landing_options['user_options.trajectory.type'] = 'compromised_landing'
    compromised_landing_options['model.model_bounds.dcoeff_compromised_factor'] = 0.0
    compromised_landing_options['user_options.trajectory.compromised_landing.emergency_scenario'] = ('broken_roll', 2)
    compromised_landing_options['user_options.trajectory.compromised_landing.xi_0_initial'] = 0.8

    # define options list
    options_dict = collections.OrderedDict()
    options_dict['single_kite_trial'] = single_kite_options
    options_dict['single_kite_basic_health_trial'] = single_kite_basic_health_options
    options_dict['single_kite_6_dof_trial'] = single_kite_6_dof_options
    options_dict['single_kite_6_dof_basic_health_trial'] = single_kite_6_dof_basic_health_options
    options_dict['segmented_tether_trial'] = segmented_tether_options
    options_dict['poly_trial'] = poly_options
    options_dict['drag_mode_trial'] = drag_mode_options
    options_dict['save_trial'] = save_trial_options
    options_dict['dual_kite_trial'] = dual_kite_options
    options_dict['dual_kite_basic_health_trial'] = dual_kite_basic_health_options
    options_dict['small_dual_kite_trial'] = small_dual_kite_options
    options_dict['small_dual_kite_basic_health_trial'] = small_dual_kite_basic_health_options
    options_dict['large_dual_kite_trial'] = large_dual_kite_options
    options_dict['large_dual_kite_basic_health_trial'] = large_dual_kite_basic_health_options
    options_dict['dual_kite_6_dof_trial'] = dual_kite_6_dof_options
    options_dict['dual_kite_6_dof_basic_health_trial'] = dual_kite_6_dof_basic_health_options
    options_dict['actuator_qaxi_trial'] = actuator_qaxi_options
    options_dict['actuator_qaxi_basic_health_trial'] = actuator_qaxi_basic_health_options
    options_dict['actuator_uaxi_trial'] = actuator_uaxi_options
    options_dict['actuator_qasym_trial'] = actuator_qasym_options
    options_dict['actuator_uasym_trial'] = actuator_uasym_options
    options_dict['actuator_comparison_trial'] = actuator_comparison_options
    options_dict['vortex_force_zero_trial'] = vortex_force_zero_options
    options_dict['vortex_force_zero_basic_health_trial'] = vortex_force_zero_basic_health_options
    options_dict['vortex_trial'] = vortex_options
    options_dict['vortex_basic_health_trial'] = vortex_basic_health_options
    options_dict['vortex_3_dof_trial'] = vortex_3_dof_options
    options_dict['dual_kite_tracking_trial'] = dual_kite_tracking_options
    options_dict['dual_kite_tracking_winch_trial'] = dual_kite_tracking_winch_options
    # options_dict['nominal_landing_trial'] = nominal_landing_options
    # options_dict['compromised_landing_trial'] = compromised_landing_options

    return options_dict


def run_test(trial_name, final_homotopy_step='final', overwrite_options={}):
    """
    Solve one individual trial and run tests on it
    :param trial_name: name of the trial
    :return: None
    """

    options_dict = generate_options_dict()
    trial_options = options_dict[trial_name]

    for option_name, option_value in overwrite_options.items():
        trial_options[option_name] = option_value
        trial_name += '_' + option_name + '_' + repr(option_value)

    # compute trajectory solution
    trial = awe_trial.Trial(trial_options, trial_name)
    trial.build()

    trial.optimize(final_homotopy_step=final_homotopy_step)
    trial.print_cost_information()

    if not trial.optimization.solve_succeeded:
        message = 'optimization of trial ' + trial_name + ' failed'
        raise Exception(message)

    return None


def this_test_is_intended_to_fail():
    raise ValueError("This test has correctly failed. Good!")


if __name__ == "__main__":

    parallel_or_serial = 'serial'

    types_of_problems = {'single_kites': True,
                         'base_alternatives': True,
                         'dual_kites': True,
                         'tracking': True,
                         'size_alternatives': False,
                         'vortex': True,
                         'actuator': False}

    if parallel_or_serial == 'parallel':

        list_functions = [] #this_test_is_intended_to_fail]
        if types_of_problems['single_kites']:
            list_functions += [test_single_kite_basic_health, test_single_kite, test_single_kite_6_dof_basic_health, test_single_kite_6_dof]
        if types_of_problems['base_alternatives']:
            list_functions += [test_segmented_tether, test_poly, test_drag_mode, test_save_trial]
        if types_of_problems['dual_kites']:
            list_functions += [test_dual_kite_basic_health, test_dual_kite, test_dual_kite_6_dof_basic_health, test_dual_kite_6_dof]
        if types_of_problems['tracking']:
            list_functions += [test_dual_kite_tracking, test_dual_kite_tracking_winch]
        # if types_of_problems['size_alternatives']:
        #     list_functions += [test_small_dual_kite_basic_health, test_small_dual_kite, test_large_dual_kite_basic_health, test_large_dual_kite]
        if types_of_problems['vortex']:
            list_functions += [test_vortex_force_zero_basic_health, test_vortex_force_zero, test_vortex_basic_health, test_vortex, test_vortex_3_dof]
        # if types_of_problems['actuator']:
        #     list_functions += [test_actuator_qaxi_basic_health, test_actuator_qaxi, test_actuator_qasym, test_actuator_uaxi, test_actuator_uasym, test_actuator_comparison]

        from concurrent.futures import ProcessPoolExecutor, wait, FIRST_EXCEPTION
        import multiprocessing

        max_workers = multiprocessing.cpu_count() - 1
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(f) for f in list_functions]

            done, not_done = wait(futures, return_when=FIRST_EXCEPTION)

            for future in done:
                try:
                    future.result()
                except Exception as e:
                    print(f"A function failed: {e}")
                    # Cancel the rest (they may not actually stop if already running)
                    for f in not_done:
                        f.cancel()
                    raise RuntimeError("At least one function failed") from e


    elif parallel_or_serial == 'serial':

        if types_of_problems['single_kites']:
            test_single_kite_basic_health()
            test_single_kite()
            test_single_kite_6_dof_basic_health()
            test_single_kite_6_dof()

        if types_of_problems['base_alternatives']:
            test_segmented_tether()
            test_poly()
            test_drag_mode()
            test_save_trial()

        if types_of_problems['dual_kites']:
            test_dual_kite_basic_health()
            test_dual_kite()
            test_dual_kite_6_dof_basic_health()
            test_dual_kite_6_dof()

        if types_of_problems['tracking']:
            test_dual_kite_tracking()
            test_dual_kite_tracking_winch()

        # if types_of_problems['size_alternatives']:
        #     test_small_dual_kite_basic_health()
        #     test_small_dual_kite()
        #     test_large_dual_kite_basic_health()
        #     test_large_dual_kite()

        if types_of_problems['vortex']:
            test_vortex_basic_health()
            test_vortex_force_zero_basic_health()
            test_vortex()
            test_vortex_force_zero()
            test_vortex_3_dof()

        # if types_of_problems['actuator']:
        #     test_actuator_qaxi_basic_health()
        #     test_actuator_qaxi()
        #     test_actuator_qasym()
        #     test_actuator_uaxi()
        #     test_actuator_uasym()
        #     test_actuator_comparison()


    else:
        message = 'unexpected method of running test_trials trials'
        print_op.log_and_raise_error(message)

    print('done')
