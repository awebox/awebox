#!/usr/bin/python3
"""Test to check model functionality

@author: Jochem De Schutter,
edit: rachel leuthold, alu-fr 2020
"""


import awebox.opts.kite_data.ampyx_data as ampyx_data
import awebox.opts.options as options
import awebox.trial as awe_trial
import awebox.tools.print_operations as print_op

import awebox as awe
import logging
import casadi as cas
import awebox.mdl.architecture as archi
import numpy as np
import awebox.mdl.system as system
import awebox.tools.print_operations as print_op

logging.basicConfig(filemode='w',format='%(levelname)s:    %(message)s', level=logging.WARNING)
#

def build_and_solve_integration_test_nlp(integration_method='integral_outputs'):
    trial_name = 'integration_test_using_' + integration_method

    # set options
    single_kite_options = {}
    single_kite_options['solver.linear_solver'] = 'ma57'
    single_kite_options['nlp.n_k'] = 5
    single_kite_options['user_options.system_model.architecture'] = {1:0}
    single_kite_options['user_options.trajectory.lift_mode.windings'] = 1
    single_kite_options['user_options.kite_standard'] = ampyx_data.data_dict()
    single_kite_options['user_options.system_model.kite_dof'] = 3
    single_kite_options['user_options.induction_model'] = 'not_in_use'
    single_kite_options['user_options.tether_drag_model'] = 'split'
    single_kite_options['model.integration.method'] = integration_method
    single_kite_options['model.integration.include_integration_test'] = True

    trial = awe_trial.Trial(single_kite_options, trial_name)
    trial.build()
    trial.optimize(final_homotopy_step='initial')

    return trial

def test_integral_outputs_integration(epsilon=1.e-3):

    trial = build_and_solve_integration_test_nlp(integration_method='integral_outputs')
    expected_time = trial.optimization.global_outputs_opt['time_period'].full()[0][0]

    if 'total_time_unscaled' in trial.model.integral_outputs.keys():
        time_unscaled = trial.optimization.integral_outputs_final_si['int_out', -1, 'total_time_unscaled']
    else:
        message = 'total_time_unscaled not in integral_outputs.keys()'
        print_op.log_and_raise_error(message)
    unscaled_integration_works_correctly = ((time_unscaled - expected_time)**2. < epsilon**2.)

    if 'total_time_scaled' in trial.model.integral_outputs.keys():
        time_scaled = trial.optimization.integral_outputs_final_si['int_out', -1, 'total_time_scaled']
    else:
        message = 'total_time_scaled not in integral_outputs.keys()'
        print_op.log_and_raise_error(message)
    scaled_integration_works_correctly = ((time_scaled - expected_time)**2. < epsilon**2.)

    assert(unscaled_integration_works_correctly)
    assert(scaled_integration_works_correctly)

    return None

def test_constraints_integration(epsilon=1.e-3):

    trial = build_and_solve_integration_test_nlp(integration_method='constraints')
    expected_time = trial.optimization.global_outputs_opt['time_period'].full()[0][0]

    if 'total_time_unscaled' in trial.model.variables_dict['x'].keys():
        time_unscaled = trial.optimization.V_final_si['x', -1, 'total_time_unscaled']
    else:
        message = 'total_time_unscaled not in states.keys()'
        print_op.log_and_raise_error(message)
    unscaled_integration_works_correctly = ((time_unscaled - expected_time)**2. < epsilon**2.)

    if 'total_time_scaled' in trial.model.variables_dict['x'].keys():
        time_scaled = trial.optimization.V_final_si['x', -1, 'total_time_scaled']
    else:
        message = 'total_time_scaled not in states.keys()'
        print_op.log_and_raise_error(message)
    scaled_integration_works_correctly = ((time_scaled - expected_time)**2. < epsilon**2.)

    assert(unscaled_integration_works_correctly)
    assert(scaled_integration_works_correctly)

    return None

test_integral_outputs_integration()
test_constraints_integration()
