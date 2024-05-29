#!/usr/bin/python3
"""Test write to csv functionality

@author: Thilo Bronnenmeyer, kiteswarms 2018
@edit: Rachel Leuthold, ALUF, 2024
"""

import os
import pdb

import awebox.opts.options as awe_options
import awebox.trial as awe_trial
import awebox.sweep as awe_sweep
import awebox.opts.kite_data.ampyx_data as ampyx_data
import logging
import awebox.tools.save_operations as save_op

logging.basicConfig(filemode='w',format='%(message)s', level=logging.DEBUG)

def build_test_trial():

    # basic options
    options = {}
    options['user_options.system_model.architecture'] = {1:0}
    options['user_options.trajectory.lift_mode.windings'] = 1
    options['user_options.kite_standard'] = ampyx_data.data_dict()
    options['user_options.trajectory.type'] = 'power_cycle'
    options['user_options.system_model.kite_dof'] = 3
    options['user_options.induction_model'] = 'not_in_use'
    options['nlp.n_k'] = 2
    options['solver.max_iter'] = 0

    # build trial and optimize
    trial = awe_trial.Trial(options, 'trial1')
    trial.build()
    trial.optimize(final_homotopy_step='initial')
    return trial

def build_test_sweep():
    sweep_opts = [('user_options.wind.u_ref', [5., 5.5])]  # parametric sweep

    # set-up trial options
    options = {}
    options['user_options.system_model.architecture'] = {1:0}
    options['user_options.system_model.kite_dof'] = 3
    options['user_options.kite_standard'] = ampyx_data.data_dict()
    options['user_options.trajectory.lift_mode.windings'] = 1
    options['user_options.induction_model'] = 'not_in_use'
    options['nlp.n_k'] = 2
    options['solver.max_iter'] = 0

    # build, run and save sweep
    sweep = awe_sweep.Sweep(name='sweep1', options=options, seed=sweep_opts)
    sweep.build()
    sweep.run(final_homotopy_step='initial')
    return sweep

def test_trial_save():
    trial = build_test_trial()

    save_dict = save_op.get_dict_of_saveable_objects_and_extensions(trial_or_sweep='trial')

    objects_to_save = list(save_dict.keys())
    file_extensions = list(save_dict.values())

    saveable_methods = objects_to_save + file_extensions
    for saving_method in saveable_methods:
        trial.save(saving_method=saving_method)

    for ext in file_extensions:
        os.remove(trial.name + '.' + ext)

    return None

def test_sweep_save():
    sweep = build_test_sweep()

    save_dict = save_op.get_dict_of_saveable_objects_and_extensions(trial_or_sweep='sweep')

    objects_to_save = list(save_dict.keys())
    file_extensions = list(save_dict.values())

    saveable_methods = objects_to_save + file_extensions
    for saving_method in saveable_methods:
        sweep.save(saving_method=saving_method)

    for ext in file_extensions:
        os.remove(sweep.name + '.' + ext)

    return None

def test_write_to_csv():

    trial = build_test_trial()
    trial.write_to_csv()

    # clean up
    os.remove('trial1.csv')

    return None
#
# test_trial_save()
# test_sweep_save()
# test_write_to_csv()