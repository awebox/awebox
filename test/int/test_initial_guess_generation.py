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


def test_initial_guess_generation():
    # ===========================================
    # SET-UP DUMMY PROBLEM
    # ===========================================

    # choose simplest model
    options = {}
    options['user_options.system_model.architecture'] = {1:0, 2:1, 3:1, 4:1, 5:4, 6:4}
    options['user_options.system_model.kite_dof'] = 3
    options['user_options.induction_model'] = 'not_in_use'
    options['user_options.kite_standard'] = ampyx_data.data_dict()
    options['solver.initialization.initialization_type'] = 'modular'
    options['model.tether.control_var'] = 'dddl_t'

    # make trial, build and run
    trial = awe.Trial(name='test', seed=options)
    trial.build()
    trial.optimize(final_homotopy_step='initial_guess')

    architecture = trial.model.architecture
    number_of_nodes = architecture.number_of_nodes
    parent_map = architecture.parent_map

    threshold = 1.e-8

    outputs_opt = trial.optimization.outputs_opt

    # check that the invariants of the initial guess are actually zero!
    for node in range(1,number_of_nodes):
        parent = parent_map[node]

        for deriv_order in range(3):
            deriv_name = (deriv_order * 'd') + 'c' + str(node) + str(parent)
            index = struct_op.find_output_idx(trial.model.outputs, 'invariants', deriv_name, output_dim=0)
            avg_mean_value = np.average(abs(np.array(outputs_opt[index, :])))

            assert(avg_mean_value < threshold)