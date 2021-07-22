#!/usr/bin/python3

import awebox as awe
import matplotlib.pyplot as plt
import numpy as np
import casadi as ca
from awebox.logger.logger import Logger as awelogger
awelogger.logger.setLevel('DEBUG')

# single kite with 6DOF Ampyx AP2 model
options = {}
options['user_options.system_model.architecture'] = {1:0}
options['user_options.system_model.kite_dof'] = 6
options['user_options.kite_standard'] = awe.ampyx_data.data_dict()

# trajectory should be a single pumping cycle with initial number of five windings
options['user_options.trajectory.type'] = 'power_cycle'
options['user_options.trajectory.system_type'] = 'lift_mode'
options['user_options.trajectory.lift_mode.windings'] = 1

# tether parameters
options['params.tether.cd'] = 1.2
options['params.tether.rho'] = 0.0046*4/(np.pi*0.002**2)
options['user_options.trajectory.fixed_params'] = {'diam_t': 2e-3}
options['model.tether.use_wound_tether'] = False

# tether drag model
options['user_options.tether_drag_model'] = 'multi'
options['model.tether.aero_elements'] = 5

# wind model
options['params.wind.z_ref'] = 10.0
options['params.wind.power_wind.exp_ref'] = 0.15
options['user_options.wind.model'] = 'power'
options['user_options.wind.u_ref'] = 10.

# phase fix: enforce single reel-out, single reel-in phase
options['user_options.trajectory.lift_mode.phase_fix'] = 'simple'

# don't model generator
options['model.model_bounds.wound_tether_length.include'] = False

# tether force limit
options['model.model_bounds.tether_stress.include'] = False
options['model.model_bounds.tether_force.include'] = True
options['params.model_bounds.tether_force_limits'] = np.array([0.1, 180000.0])

# flight envelope
options['model.model_bounds.airspeed.include'] = True
options['params.model_bounds.airspeed_limits'] = np.array([0.1, 320.0])
options['model.model_bounds.aero_validity.include'] = True
options['user_options.kite_standard.aero_validity.beta_max_deg'] = 20.
options['user_options.kite_standard.aero_validity.beta_min_deg'] = -20.
options['user_options.kite_standard.aero_validity.alpha_max_deg'] = 10.0#9.
options['user_options.kite_standard.aero_validity.alpha_min_deg'] = -10.0#-6.

# acceleration constraint
options['model.model_bounds.acceleration.include'] = False

# aircraft-tether anticollision
options['model.model_bounds.rotation.include'] = True
options['model.model_bounds.rotation.type'] = 'yaw'
options['params.model_bounds.rot_angles'] = np.array([80.0*np.pi/180., 80.0*np.pi/180., 50.0*np.pi/180.0])

# variable bounds
options['model.system_bounds.x.l_t'] =  [10.0, 700.0] # [m]
options['model.system_bounds.x.dl_t'] =  [-15.0, 20.0] # [m/s]
options['model.system_bounds.x.ddl_t'] =  [-20.3, 20.4] # [m/s^2]
options['model.system_bounds.x.q'] =  [np.array([-ca.inf, -ca.inf, 10.0]), np.array([ca.inf, ca.inf, ca.inf])]
# options['model.system_bounds.theta.t_f'] =  [20.0, 70.0] # [s]
options['model.system_bounds.z.lambda'] =  [0., ca.inf] # [N/m]
options['user_options.kite_standard.geometry.delta_max'] = np.array([20., 30., 30.]) * np.pi / 180.
options['user_options.kite_standard.geometry.ddelta_max'] = np.array([2., 2., 2.])

# don't include induction effects
options['user_options.induction_model'] = 'not_in_use'

# nlp discretization
options['nlp.n_k'] = 10
options['nlp.collocation.u_param'] = 'zoh'

# Pmax
options['nlp.cost.P_max'] = False
options['solver.cost.P_max.0'] = 0.9

# initialization
options['solver.initialization.groundspeed'] = 30.
options['solver.initialization.inclination_deg'] = 45.
options['solver.initialization.l_t'] = 400.0
options['solver.initialization.winding_period'] = 15.0

# optimize trial
trial = awe.Trial(options, 'Ampyx_AP2')
trial.build()
trial.optimize()
trial.plot(['states', 'controls', 'constraints','quad','outputs:local_performance'])
plt.show()
