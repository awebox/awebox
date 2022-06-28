#!usr/bin/python3
import awebox as awe
import numpy as np
import casadi as ca

def set_megawes_settings(options):

    # 6DOF Ampyx Ap2 model
    options['user_options.system_model.kite_dof'] = 6
    options['user_options.kite_standard'] = awe.megawes_data.data_dict()

    # tether parameters
    options['params.tether.cd'] = 1.2
    diam_t = 0.0297
    options['params.tether.rho'] = 0.6729*4/(np.pi*0.0297**2)
    options['user_options.trajectory.fixed_params'] = {'diam_t': 0.0297}
    options['model.tether.use_wound_tether'] = False # don't model generator inertia
    options['model.tether.control_var'] = 'ddl_t' # tether acceleration control

    # tether drag model (more accurate than the Argatov model in Licitra2019)
    options['user_options.tether_drag_model'] = 'multi' 
    options['model.tether.aero_elements'] = 5


    # don't model generator
    options['model.model_bounds.wound_tether_length.include'] = False

    # tether force limit
    options['model.model_bounds.tether_stress.include'] = False
    options['model.model_bounds.tether_force.include'] = True
    options['params.model_bounds.tether_force_limits'] = np.array([50, 1.7e6])

    # flight envelope
    options['model.model_bounds.airspeed.include'] = True
    options['params.model_bounds.airspeed_limits'] = np.array([10, 90.0])
    options['model.model_bounds.aero_validity.include'] = True
    options['user_options.kite_standard.aero_validity.beta_max_deg'] = 20.
    options['user_options.kite_standard.aero_validity.beta_min_deg'] = -20.
    options['user_options.kite_standard.aero_validity.alpha_max_deg'] = 4.2
    options['user_options.kite_standard.aero_validity.alpha_min_deg'] = -14.5

    # acceleration constraint
    options['model.model_bounds.acceleration.include'] = False

    # aircraft-tether anticollision
    options['model.model_bounds.rotation.include'] = True
    options['model.model_bounds.rotation.type'] = 'yaw'
    options['params.model_bounds.rot_angles'] = np.array([80.0*np.pi/180., 80.0*np.pi/180., 70.0*np.pi/180.0])

    # variable bounds
    options['model.system_bounds.x.l_t'] =  [10.0, 10e3] # [m]
    options['model.system_bounds.x.dl_t'] =  [-30.0, 30.0] # [m/s]
    options['model.ground_station.ddl_t_max'] = 2.4 # [m/s^2]
    options['model.system_bounds.x.q'] =  [np.array([-ca.inf, -ca.inf, 100.0]), np.array([ca.inf, ca.inf, ca.inf])]
    options['model.system_bounds.theta.t_f'] =  [1.0, 1e3] # [s]
    options['model.system_bounds.z.lambda'] =  [0., ca.inf] # [N/m]
    # omega_bound = 50.0*np.pi/180.0
    # options['model.system_bounds.x.omega'] = [np.array(3*[-omega_bound]), np.array(3*[omega_bound])]
    options['user_options.kite_standard.geometry.delta_max'] = np.array([20., 30., 30.]) * np.pi / 180.
    options['user_options.kite_standard.geometry.ddelta_max'] = np.array([2., 2., 2.])

    # don't include induction effects
    options['user_options.induction_model'] = 'not_in_use'

    # initialization
    options['solver.initialization.groundspeed'] = 60.
    options['solver.initialization.inclination_deg'] = 40.
    options['solver.initialization.cone_deg'] = 25.
    options['solver.initialization.l_t'] = 600.

    return options