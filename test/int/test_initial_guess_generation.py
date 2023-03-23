#!/usr/bin/python3
"""Test of initial guess generation w.r.t. consistency
@author: Thilo Bronnenmeyer
"""
import pdb

import awebox as awe
import logging
import awebox.opts.kite_data.ampyx_data as ampyx_data
import numpy as np
import awebox.tools.struct_operations as struct_op

logging.basicConfig(filemode='w', format='%(levelname)s:    %(message)s', level=logging.WARNING)


def build_initial_guess_problem(initialization_type='default'):
    options = {}
    options['user_options.system_model.architecture'] = {1: 0, 2: 1, 3: 1, 4: 1, 5: 4, 6: 4}
    options['user_options.system_model.kite_dof'] = 3
    options['user_options.induction_model'] = 'not_in_use'
    options['user_options.kite_standard'] = ampyx_data.data_dict()
    options['solver.initialization.initialization_type'] = initialization_type
    options['model.tether.control_var'] = 'dddl_t'
    options['solver.hippo_strategy'] = False
    options['solver.generate_solvers'] = False

     # make trial, build and run
    trial = awe.Trial(name='test', seed=options)
    trial.build()
    trial.optimize(final_homotopy_step='initial_guess')

    return trial


def perform_initial_guess_generation_test(initialization_type='default'):

    # be advised: ddc will NOT be zero, because derivative values are NOT included
    # in nlp.V, and thus are NOT set in initialization
    highest_order_derivative = 1

    trial = build_initial_guess_problem(initialization_type=initialization_type)

    architecture = trial.model.architecture
    number_of_nodes = architecture.number_of_nodes
    parent_map = architecture.parent_map

    threshold = 1.e-8

    outputs_init = trial.optimization.outputs_init

    # check that the holonomic-constraints are actually satisfied at the initial guess!
    for node in range(1, number_of_nodes):
        parent = parent_map[node]

        for deriv_order in range(highest_order_derivative + 1):
            differentiated_name = (deriv_order * 'd') + 'c' + str(node) + str(parent)
            index = struct_op.find_output_idx(trial.model.outputs, 'invariants', differentiated_name, output_dim=0)
            avg_abs_value = np.average(abs(np.array(outputs_init[index, :])))

            assert (avg_abs_value < threshold)

    return None

def test_default_generation():
    perform_initial_guess_generation_test(initialization_type='default')
    return None

def test_modular_generation():
    perform_initial_guess_generation_test(initialization_type='modular')
    return None

# test_modular_generation()
# test_default_generation()