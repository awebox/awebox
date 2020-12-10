#!/usr/bin/python3

import awebox as awe
import matplotlib.pyplot as plt


from awebox.logger.logger import Logger as awelogger
awelogger.logger.setLevel(10)

# make default options object
options = awe.Options(True)

# single kite with point-mass model
options['user_options']['system_model']['architecture'] = {1:0, 2:1, 3:1}
options['user_options']['system_model']['kite_dof'] = 3
options['user_options']['kite_standard'] = awe.ampyx_data.data_dict()

# pick drag-mode trajectory
options['user_options']['trajectory']['type'] = 'power_cycle'
options['user_options']['trajectory']['system_type'] = 'drag_mode'

# bounds on dkappa: be aware to check if these values make sense
# from a physical point of view your specific system
options['model']['system_bounds']['u']['dkappa'] = [-1.0, 1.0]

# don't include induction effects, use trivial tether drag
options['user_options']['induction_model'] = 'not_in_use'
options['user_options']['tether_drag_model'] = 'split'

# bounds on tether length
options['model']['system_bounds']['xd']['l_t'] = [1.0e-2, 1.0e3]

# choose coarser grid (single-loop trajectory)
# options['nlp']['n_k'] = 20

# initialize and optimize trial
trial = awe.Trial(options, 'dual_kite_drag_mode')
trial.build()
trial.optimize()
trial.plot(['isometric','states','controls','constraints'])
plt.show()
