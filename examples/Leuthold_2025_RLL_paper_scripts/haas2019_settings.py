#!usr/bin/python3

import awebox.tools.print_operations as print_op
import awebox.tools.vector_operations as vect_op

import awebox as awe
import casadi
import numpy as np
import casadi as cas


def data_dict():
    data_dict = {}
    data_dict['name'] = 'haas2019'
    data_dict['geometry'] = geometry()

    stab_derivs, aero_validity = aero()
    data_dict['stab_derivs'] = stab_derivs  # stability derivatives
    data_dict['aero_validity'] = aero_validity

    return data_dict


def geometry():
    geometry = {}

    geometry['m_k'] = 8e3  # [kg]
    geometry['s_ref'] = 138.5  # [m^2]
    geometry['b_ref'] = 60.  # [m]
    geometry['c_ref'] = geometry['s_ref'] / geometry['b_ref']  # [m]

    geometry['ar'] = geometry['b_ref'] / geometry['c_ref']

    # only for plotting
    geometry['length'] = geometry['b_ref']  # only for plotting
    geometry['height'] = geometry['b_ref'] / 5.  # only for plotting
    geometry['c_root'] = 1.4 * geometry['c_ref']
    geometry['c_tip'] = 2. * geometry['c_ref'] - geometry['c_root']

    geometry['fuselage'] = True
    geometry['wing'] = True
    geometry['tail'] = True
    geometry['wing_profile'] = None

    # tether attachment point
    geometry['r_tether'] = np.zeros((3, 1))
    
    print('geometry values...')
    print_op.print_dict_as_table(geometry)

    return geometry


def aero():
    stab_derivs = {}

    stab_derivs['frame'] = {}
    stab_derivs['frame']['force'] = 'wind'
    stab_derivs['frame']['moment'] = 'control'

    stab_derivs['CL'] = {}

    stab_derivs['CS'] = {}

    stab_derivs['CD'] = {}
    stab_derivs['CD']['0'] = [0.0054]  # thesis, page 48

    aero_validity = {}

    alpha_crit = 17.7  # deg, PhD thesis, page 48
    alpha_max = 12. # by inspection from, Fig. 3.8, Thesis, page 48
    alpha_max = 12.9 # from datafile in wake_single_kite3.py
    alpha_min = -10.  # by inspection from, Fig. 3.8, Thesis, page 48
    alpha_min = -8.8 # from datafile in wake_single_kite3.py

    aero_validity['alpha_max_deg'] = alpha_max #20.
    aero_validity['alpha_min_deg'] = alpha_min #-20.
    aero_validity['beta_max_deg'] = 15.
    aero_validity['beta_min_deg'] = -15.0

    return stab_derivs, aero_validity


def set_settings(options, inputs={}):

    ### 3DOF kite model
    options['user_options.system_model.kite_dof'] = 3
    options['user_options.kite_standard'] = data_dict()
    b_ref = options['user_options.kite_standard']['geometry']['b_ref']
    AR = options['user_options.kite_standard']['geometry']['ar']
    
    ### problem basics
    options['user_options.trajectory.system_type'] = 'lift_mode'
    options['user_options.trajectory.lift_mode.windings'] = 5  # Figure B.4, thesis, page 131, also awebox default in 2016

    for name, val in inputs.items():
        options[name] = val
    windings = options['user_options.trajectory.lift_mode.windings'] # gives the option to decrease the number of windings through inputs, for faster-but-small-scale testing.

    n_k_per_winding = 22
    n_k = n_k_per_winding * windings
    options['nlp.n_k'] = n_k # number of control intervals. We've increased this from the value given in Haas2019 (80) because in the current awebox (though not in the 2019 awebox) the inequality constraints are only applied at the control nodes, to avoid LICQ. 22 n_k/windings uses roughly 85% of 128GB RAM.
    options['model.tether.control_var'] = 'dddl_t'  # tether jerk control, Haas2019, page 4
    options['user_options.induction_model'] = 'not_in_use' # don't include induction effects

    ### wind model
    options['user_options.wind.u_ref'] = 10. # values from Thesis page 51
    options['params.wind.z_ref'] = 100. # values from Thesis page 51
    options['params.wind.log_wind.z0_air'] = 0.0002 # value from Haas2019, page 7

    ### tether model
    options['params.tether.cd'] = 0.0501 # in Haas2019, Table 1
    options['model.tether.aero_elements'] = 10 # 10 was the default value in the awebox c. 2019, the value of '13' is after tuning to get 7.5MW power output
    options['user_options.tether_drag_model'] = 'equivalent_buggy' # Haas2019 seems to suggest that the model 'kite_only' is being used, but the results are much closer with the 'equivalent_buggy' tether model, and this was the awebox default in 2019.
    options['params.tether.f_max'] = cas.inf # There is no maximum reel-out factor being applied


    ### path constraints
    options['model.model_bounds.acceleration.include'] = True
    options['model.model_bounds.acceleration.acc_max'] = 12. # Figure B.4 of Thesis, page 131
    options['model.model_bounds.tether_stress.include'] = True
    options['params.tether.max_stress'] = 3.09e9 # value in paper
    options['params.tether.stress_safety_factor'] = 3.
    options['model.model_bounds.airspeed.include'] = False
    # options['params.model_bounds.airspeed_limits'] = np.array([1., 150]) # awebox 2019 default value
    options['model.model_bounds.aero_validity.include'] = False # 3 DOF models don't include these, anyways
    options['model.model_bounds.anticollision.include'] = False
    options['model.model_bounds.rotation.include'] = False
    options['model.model_bounds.ellipsoidal_flight_region.include'] = False

    ### variable bounds
    CL_min = -0.5 # PhD thesis, page 47, and Figure B.3 page 130
    CL_max = 1.675 # PhD thesis, page 47, and Figure B.3 page 130
    psi_min = -80. * np.pi / 180.  # PhD thesis, page 47, and Figure B.3 page 130
    psi_max = 80. * np.pi / 180.  # PhD thesis, page 47, and Figure B.3 page 130
    options['model.system_bounds.x.coeff'] = [np.array([CL_min, psi_min]), np.array([CL_max, psi_max])]
    
    dCL_min = -5. #Figure B.4 page 130
    dCL_max = 5. #Figure B.4 page 130
    dpsi_min = -285. * np.pi / 180. #Figure B.4 page 130
    dpsi_max = 285. * np.pi / 180. #Figure B.4 page 130
    options['model.system_bounds.u.dcoeff'] = [np.array([dCL_min, dpsi_min]), np.array([dCL_max, dpsi_max])]

    q_z_lb = 2. * b_ref
    options['model.system_bounds.x.q'] = [np.array([-cas.inf, -cas.inf, 120.]), np.array([cas.inf, cas.inf, 405.])]
    options['model.system_bounds.x.l_t'] = [10.0, 1.e3]  # [m] # in PhD thesis
    options['model.system_bounds.x.dl_t'] = [-cas.inf, cas.inf] # [m/s] # In the PhD thesis, page 49
    options['model.system_bounds.x.ddl_t'] = [-10., 10.]  # [m/s^2], these would be the values [judging by units] meant in Table 3.1 of the Thesis, and in Figure B.3 of the appendix
    options['model.system_bounds.u.dddl_t'] = [-100., 100.]  # [m/s^3], bounds shown in Figure B.4 of the appendix
    options['model.system_bounds.z.lambda'] = [0., cas.inf]  # [N/m], tether must be in tension
    options['visualization.cosmetics.plot_bounds'] = True

     ### initialization
    tf_target = 43.8178 * (float(windings)/5.) # digitized from Fig. 2 of Haas2019, the scaling by windings/5 was to make testing more convenient and should reach unity if windings are indeed 5.
    t_switch = 32.4886 * (float(windings)/5.)  # digitized from Fig. 2 of Haas2019
    diam_t = 9.55e-2 #m # from PhD thesis page 50 (this seems to be the solution)
    lt_init = 500. # 2019-era initial guess default
    # Notice that the coordinates of the kite trajectory center given in Haas 2019, are mostly because the trajectory is shifted within the virtual windtunnel in the x- and y- directions.
    z_center_init = 260.
    x_center = (lt_init**2. - z_center_init**2.)**0.5 * vect_op.xhat() + z_center_init * vect_op.zhat()  # determine the center's axial position, given the tether length and center altitude
    
    ### pre-process initialization
    z_center_init = float(x_center[2])
    radius_init = float(z_center_init - q_z_lb)    
    inclination_deg = np.arcsin(z_center_init / lt_init) * 180. / np.pi
    cone_deg = np.arcsin(radius_init / lt_init) * 180. / np.pi
    groundspeed_init = 2. * np.pi * radius_init / (tf_target / windings)

    initialization_options = {
        'l_t': lt_init,
        'theta.diam_t': diam_t,
        'groundspeed': groundspeed_init,
        'inclination_deg': inclination_deg,
        'cone_deg': cone_deg,
    }
    print('initialization values are...')
    print_op.print_dict_as_table(initialization_options)
    for name, val in initialization_options.items():
        options['solver.initialization.' + name] = val

    options['solver.initialization.check_reference'] = True # double-check that we haven't started with an infeasible reference

    ### time and phase-fixing 
    phase_fix_reelout = t_switch / tf_target
    options['user_options.trajectory.fixed_params'] = {'diam_t': diam_t, 't_f': tf_target} # match the tether diameter and the final time exactly to those shown in Haas2019


    ### problem scaling
    options['model.scaling.other.flight_radius_estimate'] = 'cone'
    options['model.scaling.other.position_scaling_method'] = 'altitude_and_radius'
    # Thesis (page 50) says to expect tension ~ 2.5e6 N, = 2.5 MW
    # Haas2019 says to expect power ~ 7.5e6 W = 7.5e3 kW = 7.5 MW
    options['model.scaling.other.force_scaling_method'] = 'tension'
    options['model.scaling.other.tension_estimate'] = 'max_stress'
    options['model.scaling.other.print_help_with_scaling'] = True # use this to choose the above problem scalings

    #
    # ### objective function
    options['solver.cost.tracking.0'] = 1e3
    options['solver.cost.tracking.1'] = 0. # just so that we can use the cost report at the end of solution to tune any weights where necessary.

    #just to be sure
    for name, val in inputs.items():
        options[name] = val

    return options
