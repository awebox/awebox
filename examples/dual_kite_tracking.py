#!/usr/bin/python3

import awebox as awe
import matplotlib.pyplot as plt


########################
# SET-UP TRIAL OPTIONS #
########################

# make default options object
options = awe.Options(True)

# ddual kite with 6 DOF model
options['user_options']['system_model']['architecture'] = {1:0, 2:1, 3:1}
options['user_options']['system_model']['kite_dof'] = 6
options['user_options']['kite_standard'] = awe.ampyx_data.data_dict()

# trajectory should be a single looping
options['user_options']['trajectory']['type'] = 'tracking'
options['user_options']['trajectory']['system_type'] = 'lift_mode'
options['user_options']['trajectory']['lift_mode']['windings'] = 1

# don't include induction effects, use simple tether drag
options['user_options']['induction_model'] = 'not_in_use'
options['user_options']['tether_drag_model'] = 'split'

# keep main tether length constant
options['user_options']['trajectory']['tracking']['fix_tether_length'] = True

# less discretization points necessary because only one winding
options['nlp']['n_k'] = 10

##################
# OPTIMIZE TRIAL #
##################

# initialize and optimize trial
trial = awe.Trial(options, 'dual_kite_tracking')
trial.build()
trial.optimize()
trial.plot('level_3')
plt.show()
