#!/usr/bin/python3
"""Test to check whether any plot function produces an error.

@author: Thilo Bronnenmeyer
"""

import os
import awebox as awe
import logging
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})

logging.basicConfig(filemode='w',format='%(message)s', level=logging.DEBUG)

def test_visualization():

    options = awe.Options(True)

    # basic options
    options['user_options']['system_model']['architecture'] = {1:0}
    options['user_options']['trajectory']['lift_mode']['windings'] = 1
    options['user_options']['kite_standard'] = awe.ampyx_data.data_dict()
    options['user_options']['trajectory']['type'] = 'lift_mode'
    options['user_options']['system_model']['kite_dof'] = 3
    options['user_options']['induction_model'] = 'not_in_use'
    options['user_options']['tether_drag_model'] = 'split'
    options['nlp']['n_k'] = 2
    options['solver']['max_iter'] = 0
    options['visualization']['cosmetics']['plot_ref'] = True

    # build trial and optimize
    trial = awe.Trial(options, 'trial1')
    trial.build()
    trial.optimize(final_homotopy_step='initial', debug_flags='all')

    # set flags and plot
    trial_flags = ['all']
    trial.plot(trial_flags)

    # build sweep and run
    sweep_opts = [(['user_options','wind','u_ref'], [5.,5.5])]
    sweep = awe.Sweep(name = 'sweep_viz_test', options = options, seed = sweep_opts)
    sweep.build()
    sweep.run(final_homotopy_step='initial', debug_flags='all')

    # set flags and plot
    sweep_flags = ['all', 'comp_all', 'outputs:tether_length', 'comp_outputs:tether_length']
    sweep.plot(sweep_flags)

def test_animation():

    options = awe.Options(True)

    # basic options
    options['user_options']['system_model']['architecture'] = {1:0}
    options['user_options']['trajectory']['lift_mode']['windings'] = 1
    options['user_options']['kite_standard'] = awe.ampyx_data.data_dict()
    options['user_options']['trajectory']['type'] = 'power_cycle'
    options['user_options']['trajectory']['system_type'] = 'lift_mode'
    options['user_options']['system_model']['kite_dof'] = 3
    options['user_options']['induction_model'] = 'not_in_use'
    options['user_options']['tether_drag_model'] = 'split'
    options['nlp']['n_k'] = 2
    options['solver']['max_iter'] = 0

    options['visualization']['cosmetics']['trajectory']['kite_bodies'] = True
    options['visualization']['cosmetics']['interpolation']['N'] = 2

    # build trial and optimize
    trial = awe.Trial(options, 'trial1')
    trial.build()
    trial.optimize(final_homotopy_step='initial')

    trial.plot('animation')
    os.remove('trial1.mp4')