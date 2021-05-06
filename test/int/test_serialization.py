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

def test_trial_serial():

    # set-up trial options
    options = awe.Options(True) # True refers to internal access switch
    options['user_options']['system_model']['architecture'] = {1:0}
    options['user_options']['system_model']['kite_dof'] = 3
    options['user_options']['kite_standard'] = awe.ampyx_data.data_dict()
    options['user_options']['tether_drag_model'] = 'split'
    options['user_options']['trajectory']['lift_mode']['windings'] = 1
    options['user_options']['induction_model'] = 'not_in_use'
    options['nlp']['n_k'] = 2
    options['solver']['max_iter'] = 0

    # build collocation trial
    trial = awe.Trial(name = 'serial_test', seed = options)
    trial.build()
    trial.optimize(final_homotopy_step = 'initial')
    trial.save('dict')

    # load and test collocation trial
    file_pi = open('serial_test.dict','rb')
    dict_test = pickle.load(file_pi)
    trial_test = awe.Trial(dict_test)
    file_pi.close()
    os.remove("serial_test.dict")

    trial_test.plot('all')

    # set-up ms trial options
    options['nlp']['discretization'] = 'multiple_shooting'
    options['nlp']['n_k'] = 10

    # build multiple shooting trial
    trialMS = awe.Trial(name = 'serial_test_MS', seed = options)
    trialMS.build()
    trialMS.optimize(final_homotopy_step='initial')
    trialMS.save('dict')

    # load and test multiple shooting trial
    file_pi_serial = open('serial_test_MS.dict','rb')
    dict_testMS = pickle.load(file_pi_serial)
    trial_testMS = awe.Trial(dict_testMS)

    file_pi_serial.close()
    os.remove("serial_test_MS.dict")

    trial_testMS.plot('all')

def test_sweep_serial():

    # set-up trial options
    options = awe.Options(True) # True refers to internal access switch
    options['user_options']['system_model']['architecture'] = {1:0}
    options['user_options']['system_model']['kite_dof'] = 3
    options['user_options']['kite_standard'] = awe.ampyx_data.data_dict()
    options['user_options']['tether_drag_model'] = 'split'
    options['user_options']['trajectory']['lift_mode']['windings'] = 1
    options['user_options']['induction_model'] = 'not_in_use'
    options['nlp']['n_k'] = 2
    options['solver']['max_iter'] = 0

    # set-up sweep options
    sweep_opts = [(['nlp','discretization'], ['direct_collocation','multiple_shooting'])] # trial sweep
    sweep_opts = [(['user_options','wind','u_ref'], [5.,5.5])] # parametric sweep

    # build, run and save sweep
    sweep = awe.Sweep(name = 'serial_test', options = options, seed = sweep_opts)
    sweep.build()
    sweep.run(final_homotopy_step = 'initial')
    sweep.save('dict')

    # load and test sweep
    file_pi = open('serial_test.dict','rb')
    dict_test = pickle.load(file_pi)
    sweep_test = awe.Sweep(dict_test)
    file_pi.close()
    os.remove("serial_test.dict")

    sweep_test.plot(['all','comp_all'])