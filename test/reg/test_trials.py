#!/usr/bin/python3
"""Template for trial tests

@author: Thilo Bronnenmeyer, kiteswarms 2018

- edit: Rachel Leuthold, Jochem De Schutter ALU-FR 2020-21
"""

import collections
import copy
import logging
import pdb
import casadi.tools as cas

import awebox.opts.kite_data.ampyx_data as ampyx_data
import awebox.opts.kite_data.bubbledancer_data as bubbledancer_data
import awebox.opts.kite_data.boeing747_data as boeing747_data
from ampyx_ap2_settings import set_ampyx_ap2_settings
import awebox.opts.options as options
import awebox.trial as awe_trial
import awebox.tools.print_operations as print_op

from awebox.logger.logger import Logger as awelogger
import matplotlib.pyplot as plt
awelogger.logger.setLevel(10)



logging.basicConfig(filemode='w',format='%(levelname)s:    %(message)s', level=logging.DEBUG)

def test_single_kite():

    options_dict = generate_options_dict()
    trial_name = 'single_kite_trial'
    solve_and_check(options_dict[trial_name], trial_name)

    return None

def test_zoh():

    options_dict = generate_options_dict()
    trial_name = 'zoh_trial'
    solve_and_check(options_dict[trial_name], trial_name)

    return None


def test_basic_health():
    options_dict = generate_options_dict()
    trial_name = 'basic_health_trial'
    solve_trial(options_dict[trial_name], trial_name)
    return None


def test_drag_mode():

    options_dict = generate_options_dict()
    trial_name = 'drag_mode_trial'
    solve_and_check(options_dict[trial_name], trial_name)

def test_save_trial():

    options_dict = generate_options_dict()
    trial_name = 'save_trial'
    solve_and_check(options_dict[trial_name], trial_name)

    return None

def test_dual_kite():

    options_dict = generate_options_dict()
    trial_name = 'dual_kite_trial'
    solve_and_check(options_dict[trial_name], trial_name)

    return None

def test_dual_kite_6_dof():

    options_dict = generate_options_dict()
    trial_name = 'dual_kite_6_dof_trial'
    solve_and_check(options_dict[trial_name], trial_name)

    return None

def test_small_dual_kite():

    options_dict = generate_options_dict()
    trial_name = 'small_dual_kite_trial'
    solve_and_check(options_dict[trial_name], trial_name)

    return None

def test_actuator_qaxi():

    options_dict = generate_options_dict()
    trial_name = 'actuator_qaxi_trial'
    solve_and_check(options_dict[trial_name], trial_name)

    return None

def test_actuator_uaxi():

    options_dict = generate_options_dict()
    trial_name = 'actuator_uaxi_trial'
    solve_and_check(options_dict[trial_name], trial_name)

    return None

def test_actuator_qasym():

    options_dict = generate_options_dict()
    trial_name = 'actuator_qasym_trial'
    solve_and_check(options_dict[trial_name], trial_name)

    return None

def test_actuator_uasym():

    options_dict = generate_options_dict()
    trial_name = 'actuator_uasym_trial'
    solve_and_check(options_dict[trial_name], trial_name)

    return None

def test_actuator_comparison():

    options_dict = generate_options_dict()
    trial_name = 'actuator_comparison_trial'
    solve_and_check(options_dict[trial_name], trial_name)

    return None

def test_dual_kite_tracking():

    options_dict = generate_options_dict()
    trial_name = 'dual_kite_tracking_trial'
    solve_and_check(options_dict[trial_name], trial_name)

    return None

def test_dual_kite_tracking_winch():

    options_dict = generate_options_dict()
    trial_name = 'dual_kite_tracking_winch_trial'
    solve_and_check(options_dict[trial_name], trial_name)

    return None

def test_vortex():

    options_dict = generate_options_dict()
    trial_name = 'vortex_trial'
    solve_trial(options_dict[trial_name], trial_name)

    return None

def generate_options_dict():
    """
    Set options for the trials that should be tested and store them in dictionary
    :return: dictionary with trial options
    """

    # set options
    single_kite_options = {}
    single_kite_options['user_options.system_model.architecture'] = {1: 0}
    single_kite_options = set_ampyx_ap2_settings(single_kite_options)
    single_kite_options['solver.linear_solver'] = 'ma57'
    single_kite_options['model.system_bounds.x.dl_t'] = [-cas.inf, cas.inf]
    single_kite_options['model.system_bounds.x.ddl_t'] = [-cas.inf, cas.inf]
    single_kite_options['model.scaling.x.l_t'] = 1.e2   # 1.e2
    single_kite_options['solver.weights.dq'] = 1.e1
    single_kite_options['solver.weights.l_t'] = 1e-3  #
    single_kite_options['solver.weights.dl_t'] = 1e-3
    single_kite_options['solver.weights.ddl_t'] = 1e0
    single_kite_options['solver.weights.dddl_t'] = 1e0
    single_kite_options['solver.weights.q'] = 1e0

    zoh_options = copy.deepcopy(single_kite_options)
    zoh_options['nlp.collocation.u_param'] = 'zoh'

    drag_mode_options = copy.deepcopy(single_kite_options)
    drag_mode_options['user_options.trajectory.system_type'] = 'drag_mode'
    drag_mode_options['quality.test_param.power_balance_thresh'] = 2.
    drag_mode_options['model.system_bounds.theta.t_f'] = [20., 70.]  # [s]

    save_trial_options = copy.deepcopy(single_kite_options)
    save_trial_options['solver.save_trial'] = True

    dual_kite_options = copy.deepcopy(single_kite_options)
    dual_kite_options['user_options.system_model.architecture'] = {1: 0, 2: 1, 3: 1}
    dual_kite_options['model.system_bounds.theta.t_f'] = [20., 70.]  # [s]

    dual_kite_6_dof_options = copy.deepcopy(dual_kite_options)
    dual_kite_6_dof_options['user_options.system_model.kite_dof'] = 6

    small_dual_kite_options = copy.deepcopy(dual_kite_6_dof_options)
    small_dual_kite_options['user_options.kite_standard'] = bubbledancer_data.data_dict()
    small_dual_kite_options['user_options.trajectory.lift_mode.windings'] = 1

    actuator_qaxi_options = {}
    actuator_qaxi_options['user_options.system_model.architecture'] = {1: 0, 2: 1, 3: 1}
    actuator_qaxi_options['user_options.kite_standard'] = ampyx_data.data_dict()
    actuator_qaxi_options['user_options.system_model.kite_dof'] = 6
    actuator_qaxi_options['user_options.tether_drag_model'] = 'split'
    actuator_qaxi_options['user_options.induction_model'] = 'actuator'
    actuator_qaxi_options['model.aero.actuator.steadyness'] = 'quasi-steady'
    actuator_qaxi_options['model.aero.actuator.symmetry'] = 'axisymmetric'
    actuator_qaxi_options['user_options.trajectory.lift_mode.windings'] = 1
    actuator_qaxi_options['model.aero.overwrite.alpha_max_deg'] = 20.
    actuator_qaxi_options['model.aero.overwrite.alpha_min_deg'] = -20.
    actuator_qaxi_options['model.aero.overwrite.beta_max_deg'] = 20.
    actuator_qaxi_options['model.aero.overwrite.beta_min_deg'] = -20.
    actuator_qaxi_options['model.model_bounds.tether_stress.scaling'] = 10.
    actuator_qaxi_options['model.tether.lift_tether_force'] = True
    actuator_qaxi_options['model.aero.lift_aero_force'] = True
    actuator_qaxi_options['nlp.collocation.u_param'] = 'zoh'
    actuator_qaxi_options['solver.cost.fictitious.0'] = 1.e3
    actuator_qaxi_options['nlp.n_k'] = 15

    actuator_uaxi_options = copy.deepcopy(actuator_qaxi_options)
    actuator_uaxi_options['model.aero.actuator.steadyness'] = 'unsteady'
    actuator_uaxi_options['model.model_bounds.tether_stress.scaling'] = 10.

    actuator_qasym_options = copy.deepcopy(actuator_qaxi_options)
    actuator_qasym_options['model.aero.actuator.symmetry'] = 'asymmetric'
    actuator_qasym_options['solver.cost.psi.1'] = 1.e1

    actuator_uasym_options = copy.deepcopy(actuator_qaxi_options)
    actuator_uasym_options['model.aero.actuator.symmetry'] = 'asymmetric'
    actuator_uasym_options['model.aero.actuator.symmetry'] = 'asymmetric'
    actuator_uasym_options['solver.cost.psi.1'] = 1.e1


    actuator_comparison_options = copy.deepcopy(actuator_qaxi_options)
    actuator_comparison_options['model.aero.actuator.steadyness_comparison'] = ['q', 'u']
    actuator_comparison_options['user_options.system_model.kite_dof'] = 6

    vortex_options = {}
    vortex_options['user_options.system_model.architecture'] = {1: 0}
    vortex_options['user_options.trajectory.lift_mode.windings'] = 1
    vortex_options['user_options.kite_standard'] = ampyx_data.data_dict()
    vortex_options['user_options.system_model.kite_dof'] = 6
    vortex_options['user_options.induction_model'] = 'vortex'
    vortex_options['user_options.tether_drag_model'] = 'split'
    vortex_options['nlp.n_k'] = 8
    vortex_options['model.aero.vortex.wake_nodes'] = 10
    vortex_options['model.aero.vortex.representation'] = 'alg'
    vortex_options['model.aero.overwrite.alpha_max_deg'] = 20.
    vortex_options['model.aero.overwrite.alpha_min_deg'] = -20.
    vortex_options['model.tether.lift_tether_force'] = True
    vortex_options['model.aero.lift_aero_force'] = True
    vortex_options['nlp.collocation.u_param'] = 'zoh'

    dual_kite_tracking_options = copy.deepcopy(dual_kite_6_dof_options)
    dual_kite_tracking_options['user_options.trajectory.type'] = 'tracking'
    dual_kite_tracking_options['user_options.trajectory.lift_mode.windings'] = 1
    dual_kite_tracking_options['user_options.trajectory.tracking.fix_tether_length'] = True
    dual_kite_tracking_options['nlp.n_k'] = 20

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

    basic_health_options = copy.deepcopy(single_kite_options)
    basic_health_options['user_options.trajectory.lift_mode.windings'] = 1
    basic_health_options['nlp.n_k'] = 10
    basic_health_options['nlp.collocation.name_constraints'] = True
    basic_health_options['solver.health_check.when'] = 'always'
    basic_health_options['solver.health_check.raise_exception'] = True
    basic_health_options['solver.hippo_strategy'] = False
    basic_health_options['solver.health_check.spy_matrices'] = False
    basic_health_options['nlp.collocation.u_param'] = 'zoh'
    basic_health_options['solver.homotopy_method.advance_despite_max_iter'] = False
    basic_health_options['solver.homotopy_method.advance_despite_ill_health'] = False
    basic_health_options['model.system_bounds.x.dl_t'] = [-cas.inf, cas.inf]
    basic_health_options['model.system_bounds.x.ddl_t'] = [-cas.inf, cas.inf]
    basic_health_options['solver.initialization.use_reference_to_check_scaling'] = True
    basic_health_options['model.scaling.x.l_t'] = 1.e2   # 1.e2
    basic_health_options['solver.weights.dq'] = 1.e1
    basic_health_options['solver.weights.l_t'] = 1e-3  #
    basic_health_options['solver.weights.dl_t'] = 1e-3
    basic_health_options['solver.weights.ddl_t'] = 1e0
    basic_health_options['solver.weights.dddl_t'] = 1e0
    basic_health_options['solver.weights.q'] = 1e0

    # define options list
    options_dict = collections.OrderedDict()
    options_dict['single_kite_trial'] = single_kite_options
    options_dict['zoh_trial'] = zoh_options
    options_dict['drag_mode_trial'] = drag_mode_options
    options_dict['save_trial'] = save_trial_options
    options_dict['dual_kite_trial'] = dual_kite_options
    options_dict['small_dual_kite_trial'] = small_dual_kite_options
    options_dict['dual_kite_6_dof_trial'] = dual_kite_6_dof_options
    options_dict['actuator_qaxi_trial'] = actuator_qaxi_options
    options_dict['actuator_uaxi_trial'] = actuator_uaxi_options
    options_dict['actuator_qasym_trial'] = actuator_qasym_options
    options_dict['actuator_uasym_trial'] = actuator_uasym_options
    options_dict['actuator_comparison_trial'] = actuator_comparison_options
    options_dict['vortex_trial'] = vortex_options
    options_dict['dual_kite_tracking_trial'] = dual_kite_tracking_options
    options_dict['dual_kite_tracking_winch_trial'] = dual_kite_tracking_winch_options
    # options_dict['nominal_landing_trial'] = nominal_landing_options
    # options_dict['compromised_landing_trial'] = compromised_landing_options
    options_dict['basic_health_trial'] = basic_health_options

    return options_dict


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

    trial.plot('level_1')
    plt.show()

    return None

def evaluate_results(results, trial_name):

    # loop over all results
    for test_name in list(results.keys()):
        assert results[test_name], 'Test failed for ' + trial_name + ', Test regarding ' + test_name + ' failed.'

    return None

def solve_trial(trial_options, trial_name, final_homotopy_step='final'):
    """
    Set up and solve trial
    :return: solved trial
    """

    trial = awe_trial.Trial(trial_options, trial_name)
    trial.build()
    trial.optimize(final_homotopy_step=final_homotopy_step)

    # trial.plot('level_2')
    # plt.show()

    return trial

test_single_kite()
# test_zoh()
# test_basic_health()
# test_drag_mode()
# test_save_trial()
# test_dual_kite()
# test_small_dual_kite()
# test_dual_kite_6_dof()
# test_actuator_qaxi()
# test_actuator_qasym()
# test_actuator_uaxi()
# test_actuator_uasym()
# test_actuator_comparison()
# test_vortex()
# test_dual_kite_tracking()
# test_dual_kite_tracking_winch()