#!/usr/bin/python3
"""Test write to csv functionality

@author: Thilo Bronnenmeyer, kiteswarms 2018
"""

import os
import awebox.opts.options as awe_options
import awebox.trial as awe_trial
import awebox.opts.kite_data.ampyx_data as ampyx_data
import logging

logging.basicConfig(filemode='w',format='%(message)s', level=logging.DEBUG)

def test_write_to_csv():

    options = awe_options.Options(True)

    # basic options
    options['user_options']['system_model']['architecture'] = {1:0}
    options['user_options']['trajectory']['lift_mode']['windings'] = 1
    options['user_options']['kite_standard'] = ampyx_data.data_dict()
    options['user_options']['trajectory']['type'] = 'power_cycle'
    options['user_options']['system_model']['kite_dof'] = 3
    options['user_options']['induction_model'] = 'not_in_use'
    options['user_options']['tether_drag_model'] = 'split'
    options['nlp']['n_k'] = 2
    options['solver']['max_iter'] = 0


    # build trial and optimize
    trial = awe_trial.Trial(options, 'trial1')
    trial.build()
    trial.optimize(final_homotopy_step='initial')
    trial.write_to_csv()

    # clean up
    os.remove('trial1.csv')

    return None
