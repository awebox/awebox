#!/usr/bin/python3
"""
Pumping trajectory of a 2x2 stacked multi-drone configuration with Ampyx AP2 aircraft.
Induction is modeled via an actuator-based model.
Model and constraints as in:

"Performance assessment of a rigid wing Airborne Wind Energy pumping system",
G. Licitra, J. Koenemann, A. Bürger, P. Williams, R. Ruiterkamp, M. Diehl
Energy, Vol.173, pp. 569-585, 2019.

:author: Jochem De Schutter
"""

from ampyx_ap2_settings import set_ampyx_ap2_settings
import awebox as awe
import matplotlib.pyplot as plt

# single kite with point-mass model
options = {}
options['user_options.system_model.architecture'] = {1:0, 2:1, 3:1, 4:1, 5:4, 6:4}
options = set_ampyx_ap2_settings(options)

# pick drag-mode trajectory
options['user_options.trajectory.type'] = 'power_cycle'
options['user_options.trajectory.system_type'] = 'drag_mode'

# bounds on dkappa: be aware to check if these values make sense
# from a physical point of view your specific system
options['model.system_bounds.u.dkappa'] = [-1.0, 1.0]

# bounds on tether length
options['model.system_bounds.x.l_t'] = [1.0e-2, 1.0e3]
options['solver.initialization.theta.l_i'] = 100.0 # stacks-connecting tether
options['solver.initialization.x.l_t'] = 1000.0 # stacks-connecting tether

# wind model
options['params.wind.z_ref'] = 10.0
options['params.wind.power_wind.exp_ref'] = 0.15
options['user_options.wind.model'] = 'power'
options['user_options.wind.u_ref'] = 10.

# induction model
options['user_options.induction_model'] = 'actuator'

# coarse discretization
options['nlp.n_k'] = 20

# initialize and optimize trial
trial = awe.Trial(options, 'dual_kite_drag_mode')
trial.build()
trial.optimize()
trial.plot(['quad','states','controls','constraints'])
plt.show()
