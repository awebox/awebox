#!/usr/bin/python3

import awebox as awe
import matplotlib.pyplot as plt
import numpy as np
import casadi as ca
from awebox.logger.logger import Logger as awelogger
awelogger.logger.setLevel('DEBUG')

# single pumping soft-kite with kitepower point-mass model
options = {}
options['user_options.system_model.architecture'] = {1:0}
options['user_options.system_model.kite_dof'] = 3
options['user_options.system_model.kite_type'] = 'soft'
options['user_options.trajectory.system_type'] = 'lift_mode'
options['user_options.kite_standard'] = awe.kitepower_data.data_dict()

# other model options
options['model.tether.use_wound_tether'] = False
options['user_options.induction_model'] = 'not_in_use'
options['params.tether.rho'] = 724.0 # kg/m^3
options['params.tether.cd'] = 1.1 
options['model.tether.control_var'] = 'dddl_t' # tether jerk control
options['user_options.wind.model'] = 'log_wind'
options['user_options.wind.u_ref'] = 9.0
options['params.wind.z_ref'] = 10. # reference height [m]
options['params.wind.log_wind.z0_air'] = 0.1 # surface roughness [m]

# system bounds
options['model.system_bounds.x.ddl_t'] = [-2.0, 2.0] # m/s^2
options['model.ground_station.ddl_t_max'] = 2.0 # TODO get rid of redundant option
options['model.ground_station.dddl_t_max'] = 50.0
options['model.system_bounds.x.dl_t'] = [-10.0, 10.0] # m/s
options['model.system_bounds.x.pitch'] = [0.0, np.pi/6] # rad
options['model.system_bounds.u.dpitch'] = [-5., 5.] # rad
options['model.system_bounds.u.dyaw'] = [-3, 3] # rad/s
options['model.system_bounds.x.q'] = [
    np.array([-ca.inf, -ca.inf, 30.0]), # q_z > 30 m
    np.array([ca.inf, ca.inf, ca.inf])
    ] 

# model bounds
options['model.model_bounds.tether_stress.include']= True
options['params.tether.f_max'] = 3.0 # tether stress safety factor
options['model.model_bounds.acceleration.include'] = False
options['model.model_bounds.aero_validity.include'] = True # bounds on alpha, beta
options['model.model_bounds.tether_force.include'] = False
options['params.model_bounds.tether_force_limits'] = np.array([1e3, 5e3])

# trajectory options
options['user_options.trajectory.type'] = 'power_cycle'
options['user_options.trajectory.lift_mode.windings'] = 6
options['user_options.trajectory.lift_mode.phase_fix'] = 'single_reelout'
options['nlp.phase_fix_reelout'] = 0.7 # only applicable for 'single_reelout' phase fix option

# NLP options
options['nlp.n_k'] = 60
options['nlp.collocation.u_param'] = 'poly'
options['solver.linear_solver'] = 'ma57' # 'mumps'

# initialization
options['solver.initialization.shape'] = 'lemniscate'
options['solver.initialization.lemniscate.az_width'] = 20*np.pi/180.
options['solver.initialization.lemniscate.el_width'] = 8*np.pi/180.
options['solver.initialization.inclination_deg'] = 20.
options['solver.initialization.groundspeed'] = 20.
options['solver.initialization.winding_period'] = 30.
options['solver.initialization.theta.diam_t'] = 5e-3
options['solver.initialization.l_t'] = 200.0

# visualization options
options['visualization.cosmetics.plot_ref'] = False
options['visualization.cosmetics.plot_bounds'] = True
options['visualization.cosmetics.interpolation.N']  = 250

# optimize trial
trial = awe.Trial(options, 'single_kite_lift_mode')
trial.build()
trial.optimize(final_homotopy_step='final')
trial.plot(['states', 'controls', 'quad'])

outputs = trial.visualization.plot_dict['outputs']
aero_out = outputs['aerodynamics']
plt.figure()
plt.plot(180.0/np.pi*aero_out['alpha1'][0])
plt.title('alpha [deg]')
plt.grid(True)

plt.figure()
plt.plot(180.0/np.pi*aero_out['beta1'][0])
plt.title('beta [deg]')
plt.grid(True)

plt.figure()
plt.plot(aero_out['airspeed1'][0])
plt.grid(True)
plt.title('airspeed [m/s]')
plt.show()
