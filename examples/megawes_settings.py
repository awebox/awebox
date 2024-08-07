#!usr/bin/python3
import copy

import awebox as awe
import numpy as np
import casadi as ca
import csv

def set_megawes_path_generation_settings(aero_model, options):

    # --------------------------- Default aircraft/tether settings --------------------------- #
    # 6DOF megAWES model
    options['user_options.system_model.kite_dof'] = 6
    options['user_options.kite_standard'] = awe.megawes_data.data_dict(aero_model)

    # tether parameters (incl. tether drag model)
    options['params.tether.cd'] = 1.2
    diam_t = 0.0297
    options['params.tether.rho'] = 0.6729*4/(np.pi*diam_t**2)
    options['user_options.trajectory.fixed_params'] = {'diam_t': diam_t}
    options['model.tether.use_wound_tether'] = False # don't model generator inertia
    options['model.tether.control_var'] = 'ddl_t' # tether acceleration control
    options['user_options.tether_drag_model'] = 'multi' 
    options['model.tether.aero_elements'] = 5

    # --------------------------- State and control bounds --------------------------- #
    # state variables bounds
    b = round(options['user_options.kite_standard']['geometry']['b_ref'], 1)
    options['model.system_bounds.x.q'] = [np.array([0, -ca.inf, 2*b]), np.array([ca.inf, ca.inf, ca.inf])] # Spatial footprint [m]
    options['model.system_bounds.x.omega'] = [np.array([-10, -40, -25])*np.pi/180, np.array([10, 40, 25])*np.pi/180] # Angular rates [deg/s]
    options['user_options.kite_standard.geometry.delta_max'] = 0.75 * np.array([20, 10, 10])*np.pi/180 # Surface deflections [deg]
    options['model.system_bounds.x.l_t'] = [10.0, 1e3] # Tether length [m]
    options['model.system_bounds.x.dl_t'] = [-12.0, 12.0] # Tether speed [m/s]

    # control variable bounds
    options['user_options.kite_standard.geometry.ddelta_max'] = np.array(3*[25])*np.pi/180 # Deflection rates [deg/s]
    options['model.ground_station.ddl_t_max'] = 2.5 # Tether acceleration [m/s^2]

    # --------------------------- Operational constraints --------------------------- #
    # validitiy of aerodynamic model
    options['model.model_bounds.aero_validity.include'] = True
    options['user_options.kite_standard.aero_validity.beta_max_deg'] = 5.0
    options['user_options.kite_standard.aero_validity.beta_min_deg'] = -5.0
    options['user_options.kite_standard.aero_validity.alpha_max_deg'] = 4.
    options['user_options.kite_standard.aero_validity.alpha_min_deg'] = -12.

    # airspeed limitation
    options['model.model_bounds.airspeed.include'] = True
    options['params.model_bounds.airspeed_limits'] = np.array([10., 120.]) 

    # tether force limit
    options['model.model_bounds.tether_stress.include'] = False
    options['model.model_bounds.tether_force.include'] = True
    options['params.model_bounds.tether_force_limits'] = np.array([50, 1.7e6]) #[Eijkelhof2022]

    # peak power limit
    options['nlp.cost.P_max'] = True
    options['model.system_bounds.theta.P_max'] = [2.5e6, 2.5e6]
    options['solver.cost.P_max.0'] = 1.0

    # acceleration constraint
    options['model.model_bounds.acceleration.include'] = True 
    options['model.model_bounds.acceleration.acc_max'] = 3. #[g]

    # generator is not modelled
    options['model.model_bounds.wound_tether_length.include'] = False # default: True

    # --------------------------- Initialization --------------------------- #
    # initialization
    options['solver.initialization.groundspeed'] = 80. 
    options['solver.initialization.inclination_deg'] = 60. #45. 
    options['solver.initialization.cone_deg'] = 40. #25. 
    options['solver.initialization.l_t'] = 400. #600.

    return options

def set_megawes_path_tracking_settings(aero_model, options):

    # --------------------------- Default aircraft/tether settings --------------------------- #
    # 6DOF megAWES model
    options['user_options.system_model.kite_dof'] = 6
    options['user_options.kite_standard'] = awe.megawes_data.data_dict(aero_model)

    # tether parameters (incl. tether drag model)
    options['params.tether.cd'] = 1.2
    diam_t = 0.0297
    options['params.tether.rho'] = 0.6729*4/(np.pi*diam_t**2)
    options['user_options.trajectory.fixed_params'] = {'diam_t': diam_t}
    options['model.tether.use_wound_tether'] = False # don't model generator inertia
    options['model.tether.control_var'] = 'ddl_t' # tether acceleration control
    options['user_options.tether_drag_model'] = 'multi' 
    options['model.tether.aero_elements'] = 5

    # --------------------------- State and control bounds --------------------------- #
    # state variables bounds
    b = round(options['user_options.kite_standard']['geometry']['b_ref'], 1)
    options['model.system_bounds.x.q'] = [np.array([0, -ca.inf, 1*b]), np.array([ca.inf, ca.inf, ca.inf])] # Spatial footprint [m]
    options['model.system_bounds.x.omega'] = [np.array(3*[-50])*np.pi/180, np.array(3*[50])*np.pi/180] # Angular rates [deg/s]
    options['user_options.kite_standard.geometry.delta_max'] = np.array([20, 10, 10])*np.pi/180 # Surface deflections [deg]
    options['model.system_bounds.x.l_t'] = [10.0, 1e3] # Tether length [m]
    options['model.system_bounds.x.dl_t'] = [-15.0, 15.0] # Tether speed [m/s]

    # control variable bounds
    options['user_options.kite_standard.geometry.ddelta_max'] = np.array(3*[50])*np.pi/180 # Deflection rates [deg/s]
    options['model.ground_station.ddl_t_max'] = 5. # Tether acceleration [m/s^2]

    # --------------------------- Operational constraints --------------------------- #
    # validitiy of aerodynamic model
    options['model.model_bounds.aero_validity.include'] = True
    options['user_options.kite_standard.aero_validity.beta_max_deg'] = 10.0
    options['user_options.kite_standard.aero_validity.beta_min_deg'] = -10.0
    options['user_options.kite_standard.aero_validity.alpha_max_deg'] = 5.
    options['user_options.kite_standard.aero_validity.alpha_min_deg'] = -15.

    # airspeed limitation
    options['model.model_bounds.airspeed.include'] = True
    options['params.model_bounds.airspeed_limits'] = np.array([10., 120.]) 

    # tether force limit
    options['model.model_bounds.tether_stress.include'] = False
    options['model.model_bounds.tether_force.include'] = True
    options['params.model_bounds.tether_force_limits'] = np.array([50, 1.7e6]) #[Eijkelhof2022]

    # peak power limit
    options['nlp.cost.P_max'] = True
    options['model.system_bounds.theta.P_max'] = [3e6, 3e6]
    options['solver.cost.P_max.0'] = 1.0

    # acceleration constraint
    options['model.model_bounds.acceleration.include'] = True 
    options['model.model_bounds.acceleration.acc_max'] = 4. #[g]

    # generator is not modelled
    options['model.model_bounds.wound_tether_length.include'] = False # default: True

    # --------------------------- Initialization --------------------------- #
    # initialization
    options['solver.initialization.groundspeed'] = 80. 
    options['solver.initialization.inclination_deg'] = 45. 
    options['solver.initialization.cone_deg'] = 25. 
    options['solver.initialization.l_t'] = 600.

    return options

