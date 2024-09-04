#!/usr/bin/python3
"""Test of dae integrator implementation by comparing
against direct collocation solution of NLP

@author: Jochem De Schutter
"""


from awebox.logger.logger import Logger as awelogger
import awebox.opts.kite_data.ampyx_ap2_settings as ampyx_ap2_settings

import awebox as awe
import logging
import awebox.tools.struct_operations as struct_op
from casadi.tools import *
import numpy as np
import awebox.tools.print_operations as print_op
logging.basicConfig(filemode='w', format='%(levelname)s:    %(message)s', level=logging.WARNING)
awelogger.logger.setLevel(10)


def get_integration_test_inputs():
    # ===========================================
    # SET-UP DIRECT COLLOCATION PROBLEM AND SOLVE
    # ===========================================

    # choose a problem that we know solves reliably
    base_options = {}
    base_options['user_options.system_model.architecture'] = {1: 0}
    base_options = ampyx_ap2_settings.set_ampyx_ap2_settings(base_options)

    # # specify direct collocation options
    # # because we need them for struct_op.get_variables_at_time, later on.
    base_options['nlp.n_k'] = 40
    base_options['nlp.discretization'] = 'direct_collocation'
    base_options['nlp.collocation.u_param'] = 'zoh'
    base_options['nlp.collocation.scheme'] = 'radau'
    base_options['nlp.collocation.d'] = 4

    # homotopy tuning
    base_options['solver.linear_solver'] = 'ma57'

    # make trial, build and run
    trial = awe.Trial(name='test', seed=base_options)
    trial.build()
    trial.optimize()

    if not trial.optimization.solve_succeeded:
        message = 'original optimization failed. integrator check cannot possibly be expected to work.'
        raise Exception(message)

    # extract solution data
    V_opt = trial.optimization.V_opt
    P = trial.optimization.p_fix_num
    model_variables = trial.model.variables
    model_parameters = trial.model.parameters
    dae = trial.model.get_dae()

    # build dae variables for t = 0 within first shooting interval
    variables0 = struct_op.get_variables_at_time(trial.options['nlp'], V_opt, None, model_variables, 0)
    parameters = model_parameters(vertcat(P['theta0'], V_opt['phi']))
    x0, z0, p = dae.fill_in_dae_variables(variables0, parameters)

    return base_options, x0, z0, p, trial


def perform_collocation_integrator_test(base_options, x0, z0, p, trial, tolerance):

    dae = trial.model.get_dae()
    Int_outputs = trial.optimization.integral_output_vals['opt']
    V_opt = trial.optimization.V_opt

    # ===================================
    # TEST COLLOCATION INTEGRATOR
    # ===================================

    # set discretization to multiple shooting
    base_options['nlp.discretization'] = 'multiple_shooting'
    base_options['nlp.integrator.type'] = 'collocation'
    base_options['nlp.integrator.collocation_scheme'] = base_options['nlp.collocation.scheme']
    base_options['nlp.integrator.interpolation_order'] = base_options['nlp.collocation.d']
    base_options['nlp.integrator.num_steps_overwrite'] = 1

    # switch off expand to allow for use of integrator in NLP
    base_options['solver.expand_overwrite'] = False

    test_name = base_options['nlp.integrator.type']

    # build collocation trial
    trial_coll = awe.Trial(name='test_' + test_name, seed=base_options)
    trial_coll.build()

    # multiple shooting dae integrator
    F = trial_coll.nlp.Multiple_shooting.F

    # integrate over one interval
    Ff = F(x0=x0, z0=z0, p=p)
    xf = Ff['xf']
    zf = Ff['zf']
    qf = Ff['qf']

    # values should match up to nlp solver accuracy
    test_dict = {'x': {'found': xf, 'expected': V_opt['x', 1]},
                 'z': {'found': dae.z(zf)['z'], 'expected': V_opt['coll_var', 0, -1, 'z']},
                 'q': {'found': qf, 'expected': Int_outputs['int_out', 1]}
                 }
    test_dict = add_max_abs_error_to_dict(test_dict)

    # evaluate integration error
    perform_check(test_dict, test_name, tolerance)


def perform_rk_4_root_integrator_test(base_options, x0, z0, p, trial, tolerance):

    dae = trial.model.get_dae()
    Int_outputs = trial.optimization.integral_outputs_opt
    V_opt = trial.optimization.V_opt

    # ===================================
    # TEST RK4-ROOT INTEGRATOR
    # ===================================

    # switch off expand to allow for use of integrator in NLP
    base_options['solver.expand_overwrite'] = False

    # set discretization to multiple shooting
    base_options['nlp.discretization'] = 'multiple_shooting'
    base_options['nlp.integrator.collocation_scheme'] = base_options['nlp.collocation.scheme']
    base_options['nlp.integrator.interpolation_order'] = base_options['nlp.collocation.d']

    # set discretization to multiple shooting
    base_options['nlp.integrator.type'] = 'rk4root'
    base_options['nlp.integrator.num_steps_overwrite'] = 30

    test_name = base_options['nlp.integrator.type']

    # build MS trial
    trial_rk = awe.Trial(name='test_' + test_name, seed=base_options)
    trial_rk.build()

    # multiple shooting dae integrator
    F = trial_rk.nlp.Multiple_shooting.F

    # integrate over one interval
    Ff = F(x0=x0, z0=z0, p=p)
    xf = Ff['xf']
    zf = Ff['zf']
    qf = Ff['qf']

    # values should match up to nlp solver accuracy
    test_dict = {'x': {'found': xf, 'expected': V_opt['x', 1]},
                 'z': {'found': dae.z(zf)['z'], 'expected': V_opt['coll_var', 0, -1, 'z']},
                 'q': {'found': qf, 'expected': Int_outputs['int_out', 1]}
                 }
    test_dict = add_max_abs_error_to_dict(test_dict)

    # evaluate integration error
    perform_check(test_dict, test_name, tolerance)

    return None


def add_max_abs_error_to_dict(test_dict):
    for test_variable, values in test_dict.items():
        found = values['found']
        expected = values['expected']
        max_abs_error = np.max(np.abs(error(found, expected)))
        test_dict[test_variable]['max_abs_error'] = max_abs_error
    return test_dict


def perform_check(test_dict, test_name, tolerance):
    for test_variable, values in test_dict.items():
        max_abs_error = values['max_abs_error']

        condition = max_abs_error < tolerance
        message = test_name + ' max_abs_error in ' + test_variable + ' exceeds tolerance: ' + str(max_abs_error)

        assert condition, message
    return None


def error(found_dm, expected_dm):
    return np.divide((found_dm - expected_dm), expected_dm).full()


def test_collocation_integrator(tolerance=1e-7):
    base_options, x0, z0, p, trial = get_integration_test_inputs()
    perform_collocation_integrator_test(base_options, x0, z0, p, trial, tolerance)
    return None


def test_rk_4_integrator(tolerance=2e-2):
    base_options, x0, z0, p, trial = get_integration_test_inputs()
    perform_rk_4_root_integrator_test(base_options, x0, z0, p, trial, tolerance)
    return None


if __name__ == "__main__":
    test_collocation_integrator()
    test_rk_4_integrator()