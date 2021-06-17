#!/usr/bin/python3

import awebox as awe
import matplotlib.pyplot as plt
import numpy as np
import casadi as ca

# single kite with point-mass model
options = {}
options['user_options.trajectory.type'] = 'power_cycle'
options['user_options.trajectory.system_type'] = 'lift_mode'
options['user_options.trajectory.lift_mode.windings'] = 1

options['user_options.system_model.architecture'] = {1:0}
options['user_options.system_model.kite_dof'] = 3
options['user_options.system_model.kite_type'] = 'soft'
options['model.tether.use_wound_tether'] = False
options['user_options.induction_model'] = 'not_in_use'
options['user_options.kite_standard'] = awe.kitepower_data.data_dict()
options['params.tether.rho'] = 724.0
options['params.tether.cd'] = 1.1
options['model.tether.control_var'] = 'dddl_t'
options['model.system_bounds.x.ddl_t'] = [-2.0, 2.0]
options['model.system_bounds.x.dl_t'] = [-10.0, 10.0]
options['model.system_bounds.u.pitch'] = [0.0, np.pi/6]
options['model.system_bounds.u.dyaw'] = [-1.5, 1.5]

options['model.model_bounds.acceleration.include'] = False
options['nlp.n_k'] = 20
options['solver.initialization.groundspeed'] = 15.
options['solver.initialization.winding_period'] = 30.
options['solver.initialization.theta.diam_t'] = 5e-3
options['solver.initialization.l_t'] = 200.0
options['params.tether.f_max'] = 3.0
options['solver.initialization.shape'] = 'lemniscate'
options['visualization.cosmetics.plot_ref'] = True
options['visualization.cosmetics.plot_bounds'] = True
options['user_options.wind.model'] = 'uniform'
options['user_options.wind.u_ref'] = 5.0
options['solver.weights.yaw'] = 1.0
options['model.model_bounds.aero_validity.include'] = True

# optimize trial
trial = awe.Trial(options, 'single_kite_lift_mode')
trial.build()
trial.optimize(final_homotopy_step='initial')
trial.plot(['constraints','states', 'controls'])

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
plt.plot(aero_out['f_aero_earth1'][0])
plt.plot(aero_out['f_aero_earth1'][1])
plt.plot(aero_out['f_aero_earth1'][2])
plt.grid(True)
plt.title('f_aero_earth [N]')

kite_dcm  = []
ortho = []
v_app = []
for k in range(len(aero_out['ehat_chord1'][0])):
    ehat_chord = ca.vertcat(
        aero_out['ehat_chord1'][0][k],
        aero_out['ehat_chord1'][1][k],
        aero_out['ehat_chord1'][2][k],
    )
    ehat_span = ca.vertcat(
        aero_out['ehat_span1'][0][k],
        aero_out['ehat_span1'][1][k],
        aero_out['ehat_span1'][2][k],
    )
    ehat_up = ca.vertcat(
        aero_out['ehat_up1'][0][k],
        aero_out['ehat_up1'][1][k],
        aero_out['ehat_up1'][2][k],
    )
    ua = ca.vertcat(
        aero_out['air_velocity1'][0][k],
        aero_out['air_velocity1'][1][k],
        aero_out['air_velocity1'][2][k]
    )
    dcm = ca.horzcat(ehat_chord, ehat_span, ehat_up)
    kite_dcm.append(dcm)
    ortho.append(ca.mtimes(dcm.T, dcm) - np.eye(3))
    v_app.append(ua)

plt.show()
