#!/usr/bin/python3
"""Test of initial guess generation w.r.t. consistency
@author: Thilo Bronnenmeyer
"""

import awebox as awe
import logging
import awebox.opts.kite_data.ampyx_data as ampyx_data
import numpy as np

logging.basicConfig(filemode='w', format='%(levelname)s:    %(message)s', level=logging.WARNING)


def test_initial_guess_generation():
    # ===========================================
    # SET-UP DUMMY PROBLEM
    # ===========================================

    # make default options object
    options = awe.Options(True)  # True refers to internal access switch

    # choose simplest model
    options['user_options']['system_model']['architecture'] = {1:0, 2:1, 3:1, 4:1, 5:4, 6:4}
    options['user_options']['system_model']['kite_dof'] = 3
    options['user_options']['tether_drag_model'] = 'split'
    options['user_options']['trajectory']['lift_mode']['phase_fix'] = True
    options['user_options']['induction_model'] = 'not_in_use'
    options['user_options']['kite_standard'] = ampyx_data.data_dict()
    options['solver']['initialization']['initialization_type'] = 'modular'
    options['model']['tether']['control_var'] = 'dddl_t'

    # make trial, build and run
    trial = awe.Trial(name='test', seed=options)
    trial.build()
    trial.optimize(final_homotopy_step='initial_guess')
    architecture = trial.model.architecture
    number_of_nodes = architecture.number_of_nodes
    parent_map = architecture.parent_map
    for node in range(1,number_of_nodes):
        parent = parent_map[node]
        c_avg = np.average(abs(np.array(trial.optimization.output_vals[0]['coll_outputs', :, :, 'tether_length','c' + str(node) + str(parent)])))
        dc_avg = np.average(abs(np.array(trial.optimization.output_vals[0]['coll_outputs', :, :, 'tether_length','dc' + str(node) + str(parent)])))
        ddc_avg = np.average(abs(np.array(trial.optimization.output_vals[0]['coll_outputs', :, :, 'tether_length','ddc' + str(node) + str(parent)])))
        assert(c_avg < 1e-8)
        assert(dc_avg < 1e-8)
        assert(ddc_avg < 1)