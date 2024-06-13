#!/usr/bin/python3
"""Test whether trial and sweep objects can be pickled and loaded
whithout compromising functionality.

@author: Jochem De Schutter, alu-freiburg 2018
"""

import os

import awebox as awe
import logging
import pickle
logging.basicConfig(filemode='w',format='%(levelname)s:    %(message)s', level=logging.WARNING)

def build_optimize_and_save_trial_for_serialization(nlp_discretization='direct_collocation'):

    trial_name = 'serialization_trial_' + nlp_discretization

    # set-up trial options
    if nlp_discretization == 'direct_collocation':
        n_k = 2
    elif nlp_discretization == 'multiple_shooting':
        n_k = 10
    else:
        message = 'unfamiliar nlp_discretization selected (' + nlp_discretization + ')'
        raise Exception(message)

    options = {}
    options['user_options.system_model.architecture'] = {1:0}
    options['user_options.system_model.kite_dof'] = 3
    options['user_options.kite_standard'] = awe.ampyx_data.data_dict()
    options['user_options.trajectory.lift_mode.windings'] = 1
    options['user_options.induction_model'] = 'not_in_use'
    options['nlp.discretization'] = nlp_discretization
    options['nlp.n_k'] = n_k
    options['solver.max_iter'] = 0

    # build, optimize and save trial
    trial = awe.Trial(name=trial_name, seed=options)
    trial.build()
    trial.optimize(final_homotopy_step='initial')
    trial.save('dict')

    return trial_name

def load_trial_and_plot(trial_name, load_type):

    filename = trial_name + '.dict'

    if load_type == 'filename':
        # load and test sweep
        trial_test = awe.Trial(filename)

    elif load_type == 'dict':
        # load and test sweep
        filehandler = open(filename, 'rb')
        dict_test = pickle.load(filehandler)

        trial_test = awe.Trial(dict_test)
        filehandler.close()

    os.remove(filename)

    trial_test.plot('all')

    return None

def perform_trial_serial(nlp_discretization, load_type):
    trial_name = build_optimize_and_save_trial_for_serialization(nlp_discretization=nlp_discretization)
    load_trial_and_plot(trial_name, load_type)
    return None


def test_trial_serial_direct_collocation():
    nlp_discretization = 'direct_collocation'
    load_type = 'dict'
    perform_trial_serial(nlp_discretization, load_type)
    return None


def test_trial_serial_multiple_shooting():
    nlp_discretization = 'multiple_shooting'
    load_type = 'dict'
    perform_trial_serial(nlp_discretization, load_type)
    return None


def test_trial_serial_loading_from_filename():
    nlp_discretization = 'direct_collocation'
    load_type = 'filename'
    perform_trial_serial(nlp_discretization, load_type)
    return None

#########
# sweep #
#########


def build_optimize_and_save_sweep_for_serialization(sweep_type='parametric'):

    sweep_name = 'serialization_sweep_' + sweep_type

    # set-up sweep options
    if sweep_type == 'parametric':
        sweep_opts = [('user_options.wind.u_ref', [5., 5.5])] # parametric sweep
        n_k = 2
    elif sweep_type == 'trial':
        sweep_opts = [('nlp.discretization', ['direct_collocation', 'multiple_shooting'])] # trial sweep
        n_k = 8
    else:
        message = 'unfamiliar sweep_type selected (' + sweep_type + ')'
        raise Exception(message)

    # set-up trial options
    options = {}
    options['user_options.system_model.architecture'] = {1:0}
    options['user_options.system_model.kite_dof'] = 3
    options['user_options.kite_standard'] = awe.ampyx_data.data_dict()
    options['user_options.trajectory.lift_mode.windings'] = 1
    options['user_options.induction_model'] = 'not_in_use'
    options['nlp.n_k'] = n_k
    options['solver.max_iter'] = 0

    # build, run and save sweep
    sweep = awe.Sweep(name=sweep_name, options=options, seed=sweep_opts)
    sweep.build()
    sweep.run(final_homotopy_step='initial')
    sweep.save('dict')

    return sweep_name


def load_sweep_and_plot(sweep_name, load_type):

    filename = sweep_name + '.dict'

    if load_type == 'filename':
        # load and test sweep
        trial_sweep = awe.Sweep(filename)

    elif load_type == 'dict':
        # load and test sweep
        filehandler = open(filename, 'rb')
        dict_test = pickle.load(filehandler)

        trial_sweep = awe.Sweep(dict_test)
        filehandler.close()

    os.remove(filename)

    trial_sweep.plot(['all', 'comp_all'])

    return None


def perform_sweep_serial(sweep_type, load_type):
    sweep_name = build_optimize_and_save_sweep_for_serialization(sweep_type=sweep_type)
    load_sweep_and_plot(sweep_name, load_type)
    return None


def test_sweep_serial_trial():
    sweep_type = 'trial'
    load_type = 'dict'
    perform_sweep_serial(sweep_type, load_type)
    return None


def test_sweep_serial_parametric():
    sweep_type = 'parametric'
    load_type = 'dict'
    perform_sweep_serial(sweep_type, load_type)


def test_sweep_serial_loading_from_filename():
    sweep_type = 'parametric'
    load_type = 'filename'
    perform_sweep_serial(sweep_type, load_type)



if __name__ == "__main__":
    test_trial_serial_direct_collocation()
    test_trial_serial_multiple_shooting()
    test_trial_serial_loading_from_filename()
    test_sweep_serial_parametric()
    test_sweep_serial_trial()
    test_sweep_serial_loading_from_filename()
