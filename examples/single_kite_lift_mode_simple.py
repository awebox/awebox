#!/usr/bin/python3

import awebox as awe
import matplotlib.pyplot as plt

# single kite with point-mass model
options = {}
options['user_options.system_model.architecture'] = {1:0}
options['user_options.system_model.kite_dof'] = 3
options['user_options.kite_standard'] = awe.ampyx_data.data_dict()

# trajectory should be a single pumping cycle with initial number of five windings
options['user_options.trajectory.type'] = 'power_cycle'
options['user_options.trajectory.system_type'] = 'lift_mode'
options['user_options.trajectory.lift_mode.windings'] = 5

# don't include induction effects
options['user_options.induction_model'] = 'not_in_use'

# initialization
options['solver.initialization.groundspeed'] = 45.
options['solver.initialization.inclination_deg'] = 20.

# optimize trial
trial = awe.Trial(options, 'single_kite_lift_mode')
trial.build()
trial.optimize()
trial.plot('level_3')
plt.show()
