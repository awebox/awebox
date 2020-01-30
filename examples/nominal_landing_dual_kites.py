#!/usr/bin/python3

import awebox as awe
import matplotlib.pyplot as plt
import copy



##########################
# GENERATE TRIAL OPTIONS #
##########################

## PUMPING TRIAL

# make pumping options object
pumping_options = awe.Options(True)

# dual kite with  point-mass model
pumping_options['user_options']['system_model']['architecture'] = {1:0, 2:1, 3:1}
pumping_options['user_options']['system_model']['kite_dof'] = 3
pumping_options['user_options']['kite_standard'] = awe.ampyx_data.data_dict()

# trajectory should be a single pumping cycle with initial number of five windings
pumping_options['user_options']['trajectory']['type'] = 'power_cycle'
pumping_options['user_options']['trajectory']['system_type'] = 'lift_mode'
pumping_options['user_options']['trajectory']['lift_mode']['windings'] = 5

# don't include induction effects, use simple tether drag
pumping_options['user_options']['induction_model'] = 'not_in_use'
pumping_options['user_options']['tether_drag_model'] = 'single'

## NOMINAL LANDING TRIAL

# make landing options object on the basis of pumping options
nominal_landing_options = copy.deepcopy(pumping_options)

# change options to landing trajectory
nominal_landing_options['user_options']['trajectory']['type'] = 'nominal_landing'

# change initial guess generation to modular
nominal_landing_options['solver']['initialization']['initialization_type'] = 'modular'

###################
# OPTIMIZE TRIALS #
###################

# initialize and optimize pumping trial
pumping_trial = awe.Trial(pumping_options, 'dual_kite_lift_mode')
pumping_trial.build()
pumping_trial.optimize()

# set optimized pumping trial as prameterized initial condition for landing
nominal_landing_options['user_options']['trajectory']['transition']['initial_trajectory'] = pumping_trial

# intialize and optimize nominal landing trial
nominal_landing_trial = awe.Trial(nominal_landing_options, 'dual_kite_nominal_landing')
nominal_landing_trial.build()
nominal_landing_trial.optimize()

################
# PLOT RESULTS #
################

pumping_trial.plot('isometric')
nominal_landing_trial.plot('isometric')
plt.show()
