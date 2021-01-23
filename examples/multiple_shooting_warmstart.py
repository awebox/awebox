#!/usr/bin/python3

import awebox as awe
import copy
import matplotlib.pyplot as plt

# ================
# SET-UP AND SOLVE
# ================

# SET INTEGRATOR TO BE TESTED
integrator = 'rk4root'
final_step = 'final'

# make default options object
options = awe.Options(True) # True refers to internal access switch
options['user_options']['system_model']['architecture'] = {1:0}
options['user_options']['system_model']['kite_dof'] = 3
options['user_options']['kite_standard'] = awe.ampyx_data.data_dict()

options['user_options']['tether_drag_model'] = 'split'
options['user_options']['trajectory']['lift_mode']['windings'] = 1
options['user_options']['induction_model'] = 'not_in_use'

# direct collocation options
options['nlp']['n_k'] = 40
options['nlp']['collocation']['d'] = 4
options['nlp']['discretization'] = 'direct_collocation'
options['nlp']['cost']['output_quadrature'] = False

# solver options
options['solver']['mu_hippo'] = 1e-4
options['solver']['tol_hippo'] = 1e-4

# make trial, run and save
trial = awe.Trial(name = 'direct_coll', seed = options)
trial.build()
trial.optimize(final_homotopy_step = final_step)

# set integrator
options_ms = copy.deepcopy(options)
options_ms['nlp']['discretization'] = 'multiple_shooting'
options_ms['nlp']['integrator']['type'] = integrator

# update solver options
if final_step == 'final':
    options_ms['solver']['mu_hippo'] = 1e-9

# build and solve
trial2 = awe.Trial(name = 'multiple_shooting', seed = options_ms)
trial2.build()
trial2.optimize(final_homotopy_step = final_step, warmstart_file = trial)
trial2.plot(['level_1'])
plt.show()
