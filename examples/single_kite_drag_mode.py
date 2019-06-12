#!/usr/bin/python3

import awebox as awe
import logging
import matplotlib.pyplot as plt
logging.basicConfig(filemode='w',format='%(levelname)s:    %(message)s', level=logging.DEBUG)

# make default options object
options = awe.Options(True)

# single kite with point-mass model
options['user_options']['system_model']['architecture'] = {1:0}
options['user_options']['system_model']['kite_dof'] = 3
options['user_options']['kite_standard'] = awe.ampyx_data.data_dict()

# pick drag-mode trajectory
options['user_options']['trajectory']['type'] = 'drag_mode'

# don't include induction effects, use trivial tether drag
options['user_options']['induction_model'] = 'not_in_use'
options['user_options']['tether_drag_model'] = 'trivial'

# choose coarser grid (single-loop trajectory)
options['nlp']['n_k'] = 20

# initialize and optimize trial
trial = awe.Trial(options, 'single_kite_lift_mode')
trial.build()
trial.optimize()
trial.plot(['isometric','states','controls','constraints'])
