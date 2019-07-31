#!/usr/bin/python3

import awebox as awe
import matplotlib.pyplot as plt
import numpy as np


########################
# SET-UP TRIAL OPTIONS #
########################

# make default options object
options = awe.Options()

# single kite with point-mass model
options['user_options']['system_model']['architecture'] = {1:0, 2:1, 3:1}
options['user_options']['system_model']['kite_dof'] = 6
options['user_options']['kite_standard'] = awe.ampyx_data.data_dict()
# trajectory should be a single pumping cycle with initial number of five windings
options['user_options']['trajectory']['type'] = 'power_cycle'
options['user_options']['trajectory']['system_type'] = 'lift_mode'
options['user_options']['trajectory']['lift_mode']['windings'] = 5

# don't include induction effects, use simple tether drag
options['user_options']['wind']['u_ref'] = 5.0 # m/s
options['user_options']['induction_model'] = 'not_in_use'
options['user_options']['tether_drag_model'] = 'trivial'

trial = awe.Trial(seed = options, name = 'opt_design')
trial.build()
trial.optimize()

# fix params for wind speed sweep
fixed_params = {}
for name in list(trial.model.variables_dict['theta'].keys()):
    if name != 't_f':
        fixed_params[name] = trial.optimization.V_final['theta',name].full()

options['user_options']['trajectory']['fixed_params'] = fixed_params

########################
# SET-UP SWEEP OPTIONS #
########################
sweep_opts = [(['user_options', 'wind', 'u_ref'], np.linspace(3,15,5, endpoint=True))]

##################
# OPTIMIZE SWEEP #
##################

sweep = awe.Sweep(name = 'dual_kites_power_curve', options = options, seed = sweep_opts)
sweep.run()
sweep.plot('comp_stats')
plt.show()
