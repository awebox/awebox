#
#    This file is part of awebox.
#
#    awebox -- A modeling and optimization framework for multi-kite AWE systems.
#    Copyright (C) 2017-2020 Jochem De Schutter, Rachel Leuthold, Moritz Diehl,
#                            ALU Freiburg.
#    Copyright (C) 2018-2020 Thilo Bronnenmeyer, Kiteswarms Ltd.
#    Copyright (C) 2016      Elena Malz, Sebastien Gros, Chalmers UT.
#
#    awebox is free software; you can redistribute it and/or
#    modify it under the terms of the GNU Lesser General Public
#    License as published by the Free Software Foundation; either
#    version 3 of the License, or (at your option) any later version.
#
#    awebox is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#    Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with awebox; if not, write to the Free Software Foundation,
#    Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
#
#
#################################################
# Method sets all user options to a default value
#################################################

import numpy as np
from . import funcs
import casadi as cas

def set_default_user_options(internal_access = False):

    ## notation for dict tree:
    ## (category, sub_categroy, sub_sub_category, parameter name, default value, (tooltip, tooltip list), sweep_type)

    default_user_options_tree = [

        ## user options
        ('user_options',    'trajectory',  None,        'type',                  'power_cycle',      ('possible options', ['power_cycle', 'transition','mpc']), 't'),
        ('user_options',    'trajectory',  None,        'system_type',           'lift_mode',        ('possible options', ['lift_mode','drag_mode']), 't'),
        ('user_options',    'trajectory',  'lift_mode', 'windings',              3,                  ('number of windings [int]', None),'s'),
        ('user_options',    'trajectory',  'lift_mode', 'phase_fix',             'single_reelout',   ('pumping_mode phase fix option', ['single_reelout', 'simple']),'x'),
        ('user_options',    'trajectory',  'lift_mode', 'max_l_t',               None,               ('set maximum main tether length', None),'s'),
        ('user_options',    'trajectory',  'lift_mode', 'pumping_range',         None,               ('set predefined pumping range (only in comb. w. phase-fix)', None),'x'),
        ('user_options',    'trajectory',  'transition','initial_trajectory',    None,               ('relative path to pickled initial trajectory', None),'x'),
        ('user_options',    'trajectory',  'transition','terminal_trajectory',   None,               ('relative path to pickled terminal trajectory', None),'x'),
        ('user_options',    'trajectory',  'compromised_landing','emergency_scenario', ('broken_lift',2),  ('type of emergency scenario as tuple, with (SCENARIO, KITE_NODE)', None),'x'),
        ('user_options',    'trajectory',  'compromised_landing','xi_0_initial',   0.00,             ('starting position on initial trajectory between 0 and 1', None),'s'),
        ('user_options',    'trajectory',  'tracking',  'fix_tether_length',     False,              ('fixing tether length for the trajectory', [True, False]),'s'),
        ('user_options',    'trajectory',  None,        'fixed_params',          {},                 ('give dict of fixed system parameters and their values',None),'s'),
        ('user_options',    'system_model',None,        'kite_dof',              6,                  ('give the number of states that designate each kites position [int]: 3 (implies roll-control), 6 (implies DCM rotation)',[3,6]),'t'),
        ('user_options',    'system_model',None,        'surface_control',       1,                  ('which derivative of the control-surface-deflection is controlled? [int]: 0 (control of deflections), 1 (control of deflection rates)', [0, 1]),'x'),
        ('user_options',    'system_model',None,        'architecture',          {1:0, 2:1, 3:1},    ('choose tuple (layers,siblings)', None),'t'),
        ('user_options',    'system_model',None,        'cross_tether',          False,              ('enable cross_tether', [True, False]),'t'),
        ('user_options',    'wind',        None,        'model',                 'log_wind',         ('possible options', ['log_wind', 'power', 'uniform', 'datafile']),'x'),
        ('user_options',    'wind',        None,        'u_ref',                 5.,                 ('reference wind speed [m/s]', None),'s'),
        ('user_options',    'wind',        None,        'atmosphere_heightsdata', None,              ('data for the heights at this time instant', None),'s'),
        ('user_options',    'wind',        None,        'atmosphere_featuresdata',None,              ('data for the wind features at this time instant', None),'s'),
        ('user_options',    None,          None,        'induction_model',       'actuator',         ('possible options', ['not_in_use', 'actuator']),'x'),
        ('user_options',    None,          None,        'kite_standard',         None,               ('possible options',None),'x'),
        ('user_options',    None,          None,        'atmosphere',            'isa',              ('possible options', ['isa', 'uniform']),'x'),
        ('user_options',    None,          None,        'tether_model',          'default',          ('possible options',['default']),'x'),
        ('user_options',    None,          None,        'tether_drag_model',     'multi',            ('possible options: split drag equally between nodes, get equivalent forces from multiple elements, or apply drag only to tether segments with kite end-nodes', ['split', 'multi', 'kite_only', 'not_in_use']),'t'),
        ('user_options',    None,          None,        'internal_access',       internal_access,    ('Only set internal parameters/options if you know what you are doing', [True, False]),'x'),
    ]

    default_user_options, help_options = funcs.assemble_options_tree(default_user_options_tree, {}, {})

    return default_user_options, help_options

def set_default_options(default_user_options, help_options):

    kite_colors = ['b', 'g', 'r', 'm', 'c'] * 3
    dim_colors = ['b', 'g', 'r', 'm', 'c', 'y', 'darkorange', 'darkkhaki', 'darkviolet']

    default_options_tree = [

        ## atmosphere model
        ('params',  'atmosphere', None, 'g',        9.81,     ('gravitational acceleration [m/s^2]', None),'s'),
        ('params',  'atmosphere', None, 'gamma',    1.4,      ('polytropic exponent of air [-]', None),'s'),
        ('params',  'atmosphere', None, 'r',        287.053,  ('universal gas constant [J/kg/K]', None),'s'),
        ('params',  'atmosphere', None, 't_ref',    288.15,   ('reference temperature [K]', None),'s'),
        ('params',  'atmosphere', None, 'p_ref',    101325.,    ('reference pressure [Pa]', None),'s'),
        ('params',  'atmosphere', None, 'rho_ref',  1.225,      ('reference air density [kg/m^3]', None),'s'),
        ('params',  'atmosphere', None, 'gamma_air',6.5e-3,     ('temperature gradient [K/m]', None),'s'),
        ('params',  'atmosphere', None, 'mu_ref',   1.789e-5,   ('dynamic viscosity of air kg/m/s', None),'s'),
        ('params',  'atmosphere', None, 'c_sutherland',   120., ('sutherland constant relating dynamic viscosity to air temperature [K]', None),'s'),

        ## wind mode
        ('params', 'wind', None,        'z_ref',    10.,    ('reference height [m]', None),'s'),
        ('params', 'wind', 'log_wind',  'z0_air',   0.1,    ('surface roughness length of log-wind profile [m], (0.1: roughness farm land with wind breaks more than 1km apart)', None),'s'),
        ('params', 'wind', 'power_wind','exp_ref',  0.15,   ('terrain-specific exponent for power law wind-profile [-], (0.1: smooth hard ground, calm water, 0.15: tall grass on level ground, 0.2: high crops, hedges and shrubs, 0.3: small town with trees and shrubs, 0.4: large city with tall buildings. see Masters2013.', None), 's'),

        ## aero model
        ('model', 'aero', None,         'aero_coeff_ref_velocity',     'eff',           ('specifies which velocity is used to define the stability derivatives: the APParent velocity (as for wind-tunnel or computer generated derivatives), or the EFFective velocity (as for free-flight measurements using a Pitot-tube)', ['app', 'eff']), 'x'),
        ('model', 'aero', 'three_dof',  'coeff_max',    [2., 80.0 * np.pi / 180.],      ('maximum coefficients in roll-control model', None),'x'),
        ('model', 'aero', 'three_dof',  'coeff_min',    [0., -80.0 * np.pi / 180.],     ('minimum coefficients in roll-control model', None),'x'),
        ('model', 'aero', 'three_dof',  'dcoeff_max',   [5., 80. * np.pi / 180],        ('include a bound on dcoeff', None), 'x'),
        ('model', 'aero', 'three_dof',  'dcoeff_min',   [-5., -80. * np.pi / 180],      ('include a bound on dcoeff', None), 'x'),
        ('params', 'model_bounds', None, 'coeff_compromised_max', np.array([1.5, 60 * np.pi / 180.]), ('include a bound on dcoeff', None), 's'),
        ('params', 'model_bounds', None, 'coeff_compromised_min', np.array([0., -60 * np.pi / 180.]), ('include a bound on dcoeff', None), 's'),
        ('model', 'aero', 'three_dof', 'dcoeff_compromised_factor', 1., ('???', None), 's'),
        ('model', 'aero', None,         'lift_aero_force',      True,        ('lift the aero force into the decision variables', [True, False]), 'x'),
        ('params','aero', None,         'turbine_efficiency',   0.75,        ('combined drag-mode propeller and generator efficiency', None), 's'),

        ('model', 'aero', None,         'induction_comparison',     [],     ('which induction models should we include for comparison', ['act', 'vor']), 'x'),

        ('model', 'aero', 'actuator',   'a_ref',        1./3.,              ('reference value for the induction factors in actuator-disk model. takes values between 0. and 0.4', None),'x'),
        ('model', 'aero', 'actuator',   'a_range',      [-0.5, 0.5],        ('allowed range for induction factors', None),'x'),
        ('model', 'aero', 'actuator',   'scaling',      1.,                 ('scaling factor for the actuator-disk residual', None),'x'),
        ('model', 'aero', 'actuator',   'varrho_ref',   6.,                 ('approximation of the relative orbit radius, for normalization of the actuator disk equations', None),'x'),
        ('model', 'aero', 'actuator',   'varrho_range', [0., cas.inf],      ('allowed range for the relative orbit radius, for normalization of the actuator disk equations', None), 'x'),
        ('model', 'aero', 'actuator',   'steadyness',   'quasi-steady',     ('selection of steady vs unsteady actuator disk model', ['quasi-steady', 'unsteady']),'x'),
        ('model', 'aero', 'actuator',   'symmetry',     'axisymmetric',     ('selection of axisymmetric vs asymmetric actuator disk model', ['axisymmetric', 'asymmetric']), 'x'),
	    ('model', 'aero', 'actuator', 	'steadyness_comparison', [],        ('which steady models should we include for comparison', ['q', 'u']), 'x'),
	    ('model', 'aero', 'actuator', 	'symmetry_comparison', 	 [],        ('which symmetry models should we include for comparison', ['axi', 'asym']), 'x'),
        ('model', 'aero', 'actuator',   'actuator_skew',        'simple',   ('which actuator-skew angle correction to apply', ['not_in_use', 'glauert', 'coleman', 'simple']), 'x'),
        ('model', 'aero', 'actuator',   'wake_skew',            'coleman',  ('which wake-skew angle approximation to apply', ['not_in_use', 'jimenez', 'coleman', 'equal']), 'x'),
        ('model', 'aero', 'actuator',   'gamma_range',  [-80. * np.pi / 180., 80. * np.pi / 180.],  ('range of skew angles [rad] allowed in skew correction', None), 'x'),
        ('model', 'aero', 'actuator',   'normal_vector_model',  'default',  ('selection of estimation method for normal vector', ['default', 'least_squares', 'tether_parallel', 'binormal', 'xhat']), 'x'),
        ('model', 'aero', 'actuator',   'allow_azimuth_jumping', False,     ('put a limit on the azimuthal angle time-derivative to prevent solutions from jumping', None), 'x'),

        ('model', 'aero', 'vortex',     'representation',       'alg',      ('are the wake node positions included as states or algebraic variables', ['alg', 'state']), 'x'),
        ('model', 'aero', 'vortex',     'wake_nodes',           5,          ('number of wake nodes per kite per wingtip', None), 'x'),
        ('model', 'aero', 'vortex',     'far_convection_time',  120.,       ('the time [s] that the infinitely far away vortex nodes have been convected', None), 'x'),
        ('model', 'aero', 'vortex',     'core_to_chord_ratio',  0.1,        ('the ratio between the vortex core radius and the airfoil chord, [-]', None), 'x'),
        ('model', 'aero', 'vortex',     'use_linearization',    False,      ('use an iterative solution procedure, which linearizes the Biot-Savart expression', [True, False]), 'x'),
        ('model', 'aero', 'vortex',     'force_zero',           False,      ('force the induced velocity to remain zero, while maintaining all other constraint structures. Suggested for use in warmstarting only.', [True, False]), 'x'),
        ('model', 'aero', 'vortex',     'verification_test',    False,      ('compare vortex model to Haas2017 LES in outputs', [True, False]), 'x'),
        ('model', 'aero', 'vortex',     'verification_points',  20,         ('the number of observation points to distribute evenly radially, as well as azimuthally', [True, False]), 'x'),
        ('model', 'aero', 'vortex',     'verification_uniform_distribution',   False,   ('distribute the observation points uniformly or sinusoidally', [True, False]), 'x'),

        ('model', 'aero', 'overwrite',  'f_lift_earth',         None,       ('3-component lift force in the earth-fixed-frame, to over-write stability-derivative force in case of verification/validation tests', None), 'x'),

        # geometry (to be loaded!)
        ('model',  'geometry', 'overwrite', 'm_k',         None,     ('geometrical parameter', None),'s'),
        ('model',  'geometry', 'overwrite', 's_ref',       None,     ('geometrical parameter', None),'s'),
        ('model',  'geometry', 'overwrite', 'b_ref',       None,     ('geometrical parameter', None),'s'),
        ('model',  'geometry', 'overwrite', 'c_ref',       None,     ('geometrical parameter', None),'s'),
        ('model',  'geometry', 'overwrite', 'ar',          None,     ('geometrical parameter', None),'s'),
        ('model',  'geometry', 'overwrite', 'j',           None,     ('geometrical parameter', None),'s'),
        ('model',  'geometry', 'overwrite', 'length',      None,     ('geometrical parameter', None),'x'),
        ('model',  'geometry', 'overwrite', 'height',      None,     ('geometrical parameter', None),'x'),
        ('model',  'geometry', 'overwrite', 'delta_max',   None,     ('geometrical parameter', None),'t'),
        ('model',  'geometry', 'overwrite', 'ddelta_max',  None,     ('geometrical parameter', None),'t'),
        ('model',  'geometry', 'overwrite', 'c_root',      None,     ('geometrical parameter', None),'x'),
        ('model',  'geometry', 'overwrite', 'c_tip',       None,     ('geometrical parameter', None),'x'),
        ('model',  'geometry', 'overwrite', 'fuselage',    None,     ('geometrical parameter', None),'x'),
        ('model',  'geometry', 'overwrite', 'wing',        None,     ('geometrical parameter', None),'x'),
        ('model',  'geometry', 'overwrite', 'tail',        None,     ('geometrical parameter', None),'x'),
        ('model',  'geometry', 'overwrite', 'wing_profile',None,     ('geometrical parameter', None),'x'),
        ('model',  'geometry', 'overwrite', 'r_tether',    None,     ('geometrical parameter', None),'s'),

        ('model',  'aero', 'overwrite', 'alpha_max_deg', None,    ('aerodynamic parameter', None),'t'),
        ('model',  'aero', 'overwrite', 'alpha_min_deg', None,    ('aerodynamic parameter', None),'t'),
        ('model',  'aero', 'overwrite', 'beta_max_deg', None,     ('aerodynamic parameter', None),'t'),
        ('model',  'aero', 'overwrite', 'beta_min_deg', None,     ('aerodynamic parameter', None),'t'),

        ## kite model
        #### tether properties
        ('params',  'tether', None,         'kappa',                10.,        ('Baumgarte stabilization constant for constraint formulation[-]', None),'s'),
        ('params',  'tether', None,         'rho',                  970.,       ('tether material density [kg/m^3]', None),'s'),
        ('params',  'tether', None,         'cd',                   1.,         ('drag coefficient [-]', None),'s'),
        ('params',  'tether', None,         'f_max',                5.,         ('max. reel-out factor [-]', None),'s'),
        ('params',  'tether', None,         'max_stress',           3.6e9,      ('maximum material tether stress [Pa]', None),'s'),
        ('params',  'tether', None,         'stress_safety_factor', 10.,        ('tether stress safety factor [-]', None),'x'),
        ('model',   'tether', None,         'control_var',          'dddl_t',    ('tether control variable', ['ddl_t', 'dddl_t']),'x'),
        ('model',   'tether', None,         'aero_elements',        5,         ('number of discretizations made in approximating the tether drag. int greater than 1. [-]', None),'x'),
        ('model',   'tether', None,         'reynolds_smoothing',   1e-1,       ('smoothing width of the heaviside approximation in the cd vs. reynolds polynomial [-]', None),'x'),
        ('model',   'tether', None,         'cd_model',             'constant', ('how to calculate the tether drag coefficient: piecewise interpolation, polyfit interpolation, constant', ['piecewise', 'polyfit', 'constant']),'x'),
        ('model',   'tether', None,         'attachment',           'com',      ('tether attachment mode', ['com', 'stick']),'x'),
        ('model',   'tether', 'cross_tether', 'attachment',         'com',      ('tether attachment mode', ['com', 'stick', 'wing_tip']),'x'),
        ('model',   'tether', None,         'use_wound_tether',     True,       ('include the mass of the wound tether in the system energy calculation', [True, False]),'x'),
        ('model',   'tether', None,         'wound_tether_safety_factor', 1.1,  ('wound tether safety factor', None),'x'),
        ('model',   'tether', None,         'top_mass_alloc_frac',  0.5,        ('where to make a cut on a tether segment, in order to allocate tether mass to neighbor nodes, as fraction of segment length, measured from top', None), 'x'),
        ('model',   'tether', None,         'lift_tether_force',    True,       ('lift the tether force into the decision variables', [True, False]), 'x'),

        #### system bounds and limits (physical)
        ('model',  'system_bounds', 'theta',       'diam_t',       [1.0e-3, 1.0e-1],                                                                ('main tether diameter bounds [m]', None),'x'),
        ('model',  'system_bounds', 'theta',       'diam_s',       [1.0e-3, 1.0e-1],                                                  ('secondary tether diameter bounds [m]', None),'x'),
        ('model',  'system_bounds', 'theta',       'diam_c',       [1.0e-3, 1.0e-1],                                                  ('cross-tether diameter bounds [m]', None),'x'),
        ('model',  'system_bounds', 'xd',          'l_t',          [1.0e-2, 1.0e3],                                                   ('main tether length bounds [m]', None),'x'),
        ('model',  'system_bounds', 'theta',       'l_s',          [1.0e-2, 1.0e3],                                                                 ('secondary tether length bounds [m]', None),'x'),
        ('model',  'system_bounds', 'theta',       'l_i',          [1.0e2, 1.0e2],                                                                  ('intermediate tether length bounds [m]', None),'x'),
        ('model',  'system_bounds', 'theta',       'l_c',          [1.0e-2, 1.0e3],                                                                 ('cross-tether length bounds [m]', None),'x'),
        ('model',  'system_bounds', 'xd',          'q',            [np.array([-cas.inf, -cas.inf, 10.0]), np.array([cas.inf, cas.inf, cas.inf])],   ('kite position bounds [m]', None),'x'),
        ('model',  'system_bounds', 'xd',          'omega',        [np.array([-50.0, -50.0, -50.0]), np.array([50.0, 50.0, 50.0])],   ('kite angular velocity bounds [rad/s]', None),'s'),
        ('model',  'system_bounds', 'xd',          'wz_ext',       [5.0, cas.inf],                                                                  ('wake node position (exterior wing-tips) bounds [m]', None), 'x'),
        ('model',  'system_bounds', 'xd',          'wz_int',       [5.0, cas.inf],                                                                  ('wake node position (interior wing-tips) bounds [m]', None), 'x'),
        ('model',  'system_bounds', 'theta',       't_f',          [1e-3, 500.0],                                                                   ('main tether max acceleration [m/s^2]', None),'x'),
        ('model',  'system_bounds', 'xa',          'lambda',       [0., cas.inf],                                                                   ('multiplier bounds', None),'x'),
        ('model',  'system_bounds', 'u',           'dkappa',       [-1000.0, 1000.0],                                                               ('generator braking constant [kg/m/s]', None),'x'),

        #### model bounds (range of validity)
        ('model',   'model_bounds', 'wound_tether_length', 'include',        True,      ('include constraint that total main tether length include the unrolled main tether length in constraints', [True, False]), 'x'),
        ('model',   'model_bounds', 'tether_stress', 'include',              True,      ('include tether stress inequality in constraints', [True, False]),'x'),
        ('model',   'model_bounds', 'tether_stress', 'scaling',              1.,        ('tightness scaling for tether stress inequality', None),'x'),
        ('model',   'model_bounds', 'tether_force',  'include',              False,     ('include tether force inequality in constraints', [True, False]),'x'),
        ('params',  'model_bounds',  None,           'tether_force_limits',  np.array([1e0, 2e3]),  ('tether force limits [N]', None),'s'),
        ('model',   'model_bounds', 'airspeed',      'include',             False,      ('include airspeed inequality for kites in constraints', [True, False]),'x'),
        ('params',  'model_bounds',  None,           'airspeed_limits',     np.array([13., 32.]),  ('airspeed limits [m/s]', None),'s'),
        ('model',   'model_bounds', 'aero_validity', 'include',              True,       ('include orientation bounds on alpha and beta (not possible in 3dof mode)', [True, False]),'x'),
        ('model',   'model_bounds', 'aero_validity', 'scaling',              1.,         ('tightness scaling for aero_validity inequalities', None),'x'),
        ('model',   'model_bounds', 'aero_validity', 'CD_min',               0.,         ('minimum allowed drag coefficient - included in aero validity constraints', None), 'x'),
        ('model',   'model_bounds', 'anticollision', 'safety_factor',        5.,         ('safety margin for anticollision constraint [m]', None),'x'),
        ('model',   'model_bounds', 'anticollision', 'include',              True,       ('include a minimum distance anticollision inequality in constraints', [True, False]),'x'),
        ('model',   'model_bounds', 'acceleration',  'include',              True,       ('include a hardware limit on node acceleration', [True, False]),'x'),
        ('model',   'model_bounds', 'acceleration',  'acc_max',              12.,        ('maximum acceleration, as measured in multiples of g [-]', None),'x'),
        ('model',   'model_bounds', 'rotation',     'include',               True,      ('include constraints on roll and pitch motion', None), 't'),
        ('model',   'model_bounds', 'rotation',     'type',                 'yaw',      ('rotation constraint type', ['yaw','roll_pitch']), 't'),
        ('params',  'model_bounds', None,           'rot_angles',            np.array([80.0*np.pi/180., 80.0*np.pi/180., 160.0*np.pi/180.0]), ('[roll, pitch, yaw] - [rad]', None), 's'),
        ('params',  'model_bounds', None,           'span_angle',            45.0*np.pi/180., ('[max. angle between span and wing-tip cross-tether] - [rad]', None), 's'),
        ('model',   'model_bounds', 'dcoeff_actuation', 'include',          True,       ('include a bound on dcoeff', None), 'x'),
        ('model',   'model_bounds', 'coeff_actuation',  'include',          True,       ('include a bound on coeff', None), 'x'),

        #### scaling
        ('model',  'scaling', 'xd',     'l_t',      500.,     ('main tether natural length [m]', None),'x'),
        ('model',  'scaling', 'theta',  'l_i',      100.,     ('intermediate tether natural length [m]', None),'x'),
        ('model',  'scaling', 'theta',  'l_s',      50.,      ('secondary tether natural length [m]', None),'x'),
        ('model',  'scaling', 'theta',  'l_c',      100.,     ('cross-tether natural length [m]', None),'x'),
        ('model',  'scaling', 'theta',  'diam_t',   5e-3,     ('main tether natural diameter [m]', None),'x'),
        ('model',  'scaling', 'theta',  'diam_s',   5e-3,     ('secondary tether natural diameter [m]', None),'x'),
        ('model',  'scaling', 'theta',  'diam_c',   5e-3,     ('cross-tether natural diameter [m]', None),'x'),
        ('model',  'scaling', 'xl',     'a',        1.0,      ('induction factor [-]', None),'x'),
        ('model',  'scaling', 'other',  'g',	    9.81,     ('acceleration to use for scaling [m/s^2]', None), 'x'),
        ('model',  'scaling', 'xd',     'kappa',    1e1,      ('generator braking parameter [m]', None),'x'),

        ('model',   'scaling_overwrite',    'lambda_tree', 'include',           True,   ('specific scaling of tether tension per length', None),'t'),
        ('model',   'scaling_overwrite',    None,           'lambda_factor',    1.,     ('factor applied in the scaling of the tether tension-per-unit-length [-]', None),'t'),
        ('model',   'scaling_overwrite',    None,           'energy_factor',    1.,     ('factor applied in the scaling of the energy [-]', None),'t'),

        ('model',  'jit_code_gen',     None, 'include',              False,                  ('generate code with jit for model functions'),'t'),
        ('model',  'jit_code_gen',     None, 'compiler',             'clang',                ('compiler for generated code'),'t'),

        ('params',   None,       None,   'kappa_r',  1.,         ('baumgarte stabilization constant for dcm dynamics', None),'x'),

        #### ground_station
        ('params', 'ground_station', None, 'r_gen',            0.25,   ('winch generator drum radius [m]',None),'x'),
        ('params', 'ground_station', None, 'm_gen',            50.,   ('effective mass of generator [kg], guessed',None),'x'),
        ('model', 'ground_station', None, 'ddl_t_max',        10.,    ('reel-in/out acceleration limit on the tether [m/s^2]', None),'x'),
        ('model', 'ground_station', None, 'dddl_t_max',       100.,    ('reel-in/out jerk limit on the tether [m/s^2]', None), 'x'),

        #### emergency landing
        ('formulation', 'nominal_landing', None, 'main_node_radius', 40.,   ('???', None), 'x'),
        ('formulation', 'nominal_landing', None, 'kite_node_radius', 80.,   ('???', None), 'x'),
        ('formulation', 'nominal_landing', None, 'position_weight',  0.,    ('weight given to landing position in objective', None), 'x'),
        ('formulation', 'nominal_landing', None, 'velocity_weight',  10.,   ('weight given to landing velocity in objective', None), 'x'),

        #### battery parameters
        # todo: some of these parameters have nothing to do with the battery.
        ('formulation', 'compromised_landing', 'battery', 'flap_length', None, ('???', None), 'x'),
        ('formulation', 'compromised_landing', 'battery', 'flap_width', None, ('???', None), 'x'),
        ('formulation', 'compromised_landing', 'battery', 'max_flap_defl', None, ('???', None), 'x'),
        ('formulation', 'compromised_landing', 'battery', 'min_flap_defl', None, ('???', None), 'x'),
        ('formulation', 'compromised_landing', 'battery', 'c_dl', None, ('???', None), 'x'),
        ('formulation', 'compromised_landing', 'battery', 'c_dphi', None, ('???', None), 'x'),
        ('formulation', 'compromised_landing', 'battery', 'defl_lift_0', None, ('???', None), 'x'),
        ('formulation', 'compromised_landing', 'battery', 'defl_roll_0', None, ('???', None), 'x'),
        ('formulation', 'compromised_landing', 'battery', 'voltage', None, ('???', None), 'x'),
        ('formulation', 'compromised_landing', 'battery', 'mAh', None, ('???', None), 'x'),
        ('formulation', 'compromised_landing', 'battery', 'charge', None, ('???', None), 'x'),
        ('formulation', 'compromised_landing', 'battery', 'number_of_cells', None, ('???', None), 'x'),
        ('formulation', 'compromised_landing', 'battery', 'conversion_efficiency', None, ('???', None), 'x'),
        ('formulation', 'compromised_landing', 'battery', 'power_controller', None, ('???', None), 'x'),
        ('formulation', 'compromised_landing', 'battery', 'power_electronics', None, ('???', None), 'x'),
        ('formulation', 'compromised_landing', 'battery', 'charge_fraction', None, ('???', None), 'x'),

        ## numerics
        #### NLP options
        ('nlp',  None,               None, 'n_k',                  40,                     ('control discretization [int]', None),'t'),
        ('nlp',  None,               None,  'discretization',      'direct_collocation',   ('possible options', ['direct_collocation']),'x'),
        ('nlp',  'collocation',      None, 'd',                    4,                      ('degree of lagrange polynomials inside collocation interval [int]', None),'t'),
        ('nlp',  'collocation',      None, 'scheme',               'radau',                ('collocation scheme', ['radau','legendre']),'x'),
        ('nlp',  'collocation',      None, 'u_param',              'zoh',                 ('control parameterization in collocation interval', ['poly','zoh']),'x'),
        ('nlp',  None,               None, 'phase_fix_reelout',    0.7,                    ('time fraction of reel-out phase', None),'x'),
        ('nlp',  None,               None, 'pumping_range',        [None, None],           ('set predefined pumping range (only in comb. w. phase-fix)', None),'x'),
        ('nlp',  'cost',             None, 'output_quadrature',    True,                   ('use quadrature for integral system outputs in cost function', (True, False)),'t'),
        ('nlp',  'parallelization',  None, 'type',                 'openmp',               ('parallellization type', None),'t'),
        ('nlp',  None,               None, 'slack_constraints',    False,                  ('slack path constraints', (True, False)),'t'),
        ('nlp',  None,               None, 'constraint_scale',     1.,                      ('value with which to scale all constraints, to improve kkt matrix conditioning', None), 't'),


        ### Multiple shooting integrator options
        ('nlp',  'integrator',       None, 'type',                 'collocation',          ('integrator type', ('idas', 'collocation')),'t'),
        ('nlp',  'integrator',       None, 'jit_coll',             False,                  ('code-generate coll integrator', (True, False)),'t'),
        ('nlp',  'integrator',       None, 'num_steps_coll',       1,                      ('number of steps within coll integrator', None),'t'),
        ('nlp',  'integrator',       None, 'jit_idas',             False,                  ('code-generate idas integrator', (True, False)),'t'),
        ('nlp',  'integrator',       None, 'num_steps_rk4root',    20,                     ('number of steps within rk4rootintegrator', None),'t'),
        ('nlp',  'integrator',       None, 'jit_overwrite',        None,                   ('code-generate integrator', (True, False)),'t'),
        ('nlp',  'integrator',       None, 'num_steps_overwrite',  None,                   ('number of steps within integrator', None),'t'),
        ('nlp',  'integrator',       None, 'collocation_scheme',   'radau',                ('scheme of collocation integrator', None),'t'),
        ('nlp',  'integrator',       None, 'interpolation_order',  3,                      ('order of interpolating polynomial', None),'t'),

        ### solver options
        # todo: embed other solvers
        ('solver',  None,   None,   'linear_solver',        'ma57',     ('which linear solver to use', ['mumps', 'ma57']),'x'),
        ('solver',  None,   None,   'hessian_approximation',False,      ('use a limited-memory hessian approximation instead of the exact Newton hessian', [True, False]),'x'),
        ('solver',  None,   None,   'max_iter',             2000,       ('maximum ipopt iterations [int]', None),'x'),
        ('solver',  None,   None,   'max_cpu_time',         1.e4,       ('maximum cpu time (seconds) ipopt can spend in one stage of the homotopy', None), 'x'),
        ('solver',  None,   None,   'mu_target',            0.,         ('target for interior point homotopy parameter in ipopt [float]', None),'x'),
        ('solver',  None,   None,   'mu_init',              1.,         ('start value for interior point homotopy parameter in ipopt [float]', None),'x'),
        ('solver',  None,   None,   'tol',                  1e-8,       ('ipopt solution tolerance [float]', None),'x'),
        ('solver',  None,   None,   'callback',             False,      ('plot intermediate solutions', [True,False]),'x'),
        ('solver',  None,   None,   'callback_step',        10,         ('callback interval [int]', None),'x'),
        ('solver',  None,   None,   'jit',                  False,      ('callback interval [int]', None),'t'),
        ('solver',  None,   None,   'compiler',            'clang',     ('callback interval [int]', None),'x'),
        ('solver',  None,   None,   'jit_flags',           '-O0',       ('flags to be passed to jit compiler', None),'t'),
        ('solver',  None,   None,   'expand_overwrite',     None,       ('expand MX --> SX [int]', None),'t'),

        ('solver',  None,   None,   'homotopy_method',      'penalty',  ('homotopy method used', ['penalty', 'classic']), 's'),
        ('solver',  None,   None,   'homotopy_step',        0.1,        ('classical continuation homotopy parameter step',None), 's'),
        ('solver',  None,   None,   'hippo_strategy',       True,       ('enable hippo strategy to increase homotopy speed', [True, False]),'x'),
        ('solver',  None,   None,   'mu_hippo',             1e-2,       ('target for interior point homotop parameter for hippo strategy [float]', None),'x'),
        ('solver',  None,   None,   'tol_hippo',            1e-4,       ('ipopt solution tolerance for hippo strategy [float]', None),'x'),
        ('solver',  None,   None,   'acceptable_iter_hippo',5,          ('number of iterations below tolerance for ipopt to consider the solution converged [int]', None),'x'),

        ('solver',  'initialization', None, 'initialization_type',  'default',  ('set initialization type', None), 't'),
        ('solver',  'initialization', None, 'interpolation_scheme', 's_curve',  ('interpolation scheme used for initial guess generation', ['s_curve', 'poly']), 'x'),
        ('solver',  'initialization', None, 'fix_tether_length',    False,      ('fix tether length for trajectory', [True, False]), 'x'),
        ('solver',  'initialization', None, 'groundspeed',          60.,        ('initial guess of kite speed (magnitude) as measured by earth-fixed observer [m/s]', None),'x'),
        ('solver',  'initialization', None, 'winding_period',       10.,        ('initial guess of reasonable period for one winding [s]', None), 'x'),
        ('solver',  'initialization', None, 'inclination_deg',      15.,        ('initial tether inclination angle [deg]', None),'x'),
        ('solver',  'initialization', None, 'min_rel_radius',       2.,         ('minimum allowed radius to span ratio allowed in initial guess [-]', None), 'x'),
        ('solver',  'initialization', None, 'psi0_rad',             0.,         ('azimuthal angle at time 0 [rad]', None), 'x'),
        ('solver',  'initialization', None, 'l_t',                  500.,       ('initial main tether length [m]', None), 'x'),
        ('solver',  'initialization', None, 'max_cone_angle_multi', 80.,        ('maximum allowed cone angle allowed in initial guess, for multi-kite scenarios [deg]', None),'x'),
        ('solver',  'initialization', None, 'max_cone_angle_single',10.,        ('maximum allowed cone angle allowed in initial guess, for single-kite scenarios [deg]', None),'x'),
        ('solver',  'initialization', None, 'landing_velocity',     22.,        ('initial guess for average reel in velocity during the landing [m/s]', None),'x'),
        ('solver',  'initialization', None, 'clockwise_rotation_about_xhat', True,    ('True: if the kites rotate clockwise about xhat, False: if the kites rotate counter-clockwise about xhat', [True, False]), 'x'),

        ('solver',   'tracking',       None,   'stagger_distance',      0.1,       ('distance between tracking trajectory and initial guess [m]', None),'x'),
        ('solver',   'cost_factor',    None,   'power',                 1.,       ('factor used in generating the power cost [-]', None), 'x'),

        ('solver',   'weights',        None,   'dq',                    1e-1,       ('optimization weight for all dq variables [-]', None),'x'),
        ('solver',   'weights',        None,   'l_t',                   1e-3,       ('optimization weight for all l_t variables [-]', None), 'x'),
        ('solver',   'weights',        None,   'q',                     1e-1,       ('optimization weight for all q variables [-]', None),'x'),
        ('solver',   'weights',        None,   'w',                     1e-10,      ('optimization weight for all vortex variables [-]', None), 'x'),
        ('solver',   'weights',        None,   'omega',                 1e-1,       ('optimization weight for all omega variables [-]', None),'x'),
        ('solver',   'weights',        None,   'r',                     1e1,        ('optimization weight for all r variables [-]', None),'x'),
        ('solver',   'weights',        None,   'delta',                 1e-10,      ('optimization weight for all delta variables [-]', None),'x'),
        ('solver',   'weights',        None,   'ddelta',                1e-10,      ('optimization weight for all ddelta variables [-]', None),'x'),
        ('solver',   'weights',        None,   'lambda',                1.,         ('optimization weight for all lambda variables [-]', None),'x'),
        ('solver',   'weights',        None,   'a',                     1e-3,       ('optimization weight for lifted variable a [-]', None),'x'),
        ('solver',   'weights',        None,   'dkappa',                1e1,        ('optimization weight for control variable dkappa [-]', None),'s'),

        ('solver',   'weights_overwrite', None,   'dddl_t',         None,       ('optimization weight for control variable dddl_t [-]', None),'s'),
        ('solver',   'weights_overwrite', None,   'ddl_t',          None,       ('optimization weight for control variable ddl_t [-]', None), 's'),

        ('solver',  'cost',             'tracking',             0,  1e-1,       ('starting cost for tracking', None),'x'),
        ('solver',  'cost',             'u_regularisation',     0,  1e-4,       ('starting cost for u_regularisation', None),'s'),
        ('solver',  'cost',             'slack',                0,  1e-2,       ('starting cost for slack penalization', None), 's'),
        ('solver',  'cost',             'ddq_regularisation',   0,  0,          ('starting cost for ddq_regularisation', None),'s'),
        ('solver',  'cost',             'theta_regularisation', 0,  1e-2,       ('starting cost for theta', None), 'x'),

        ('solver',  'cost',             'gamma',            0,      0.,         ('starting cost for gamma', None),'x'),
        ('solver',  'cost',             'iota',             0,      0.,         ('starting cost for iota', None),'x'),
        ('solver',  'cost',             'psi',              0,      0.,         ('starting cost for psi', None),'x'),
        ('solver',  'cost',             'tau',              0,      0.,         ('starting cost for tau', None),'x'),
        ('solver',  'cost',             'eta',              0,      0.,         ('starting cost for tau', None),'x'),
        ('solver',  'cost',             'nu',               0,      0.,         ('starting cost for nu', None),'x'),
        ('solver',  'cost',             'upsilon',          0,      0.,         ('starting cost for upsilon', None),'x'),

        ('solver',  'cost',             'fictitious',       0,      1e-4,       ('starting cost for fictitious', None),'x'),
        ('solver',  'cost',             'power',            0,      0.,         ('starting cost for power', None),'x'),
        ('solver',  'cost',             't_f',              0,      1e-2,       ('starting cost for final time', None),'x'),
        ('solver',  'cost',             'nominal_landing',  0,      0,          ('starting cost for nominal_landing', None),'x'),
        ('solver',  'cost',             'compromised_battery',  0,  0,          ('starting cost for compromised_battery', None),'x'),
        ('solver',  'cost',             'transition',       0,      0,          ('starting cost for transition', None),'x'),

        ('solver',  'cost',             'tracking',         1,      1.e-6,         ('update cost for tracking', None),'x'),
        ('solver',  'cost',             'gamma',            1,      1e3,        ('update cost for gamma', None),'s'),
        ('solver',  'cost',             'iota',             1,      1e3,        ('update cost for iota', None),'x'),
        ('solver',  'cost',             'psi',              1,      1e3,        ('update cost for psi', None),'x'),
        ('solver',  'cost',             'tau',              1,      1e3,        ('update cost for tau', None),'x'),
        ('solver',  'cost',             'eta',              1,      1e3,        ('update cost for eta', None),'x'),
        ('solver',  'cost',             'nu',               1,      1e3,        ('update cost for nu', None),'x'),
        ('solver',  'cost',             'upsilon',          1,      1e3,        ('update cost for upsilon', None),'x'),

        ('solver',  'cost',             'fictitious',       1,      1e3,        ('update cost for fictitious', None),'x'),
        ('solver',  'cost',             'nominal_landing',  1,      1e-2,       ('update cost for nominal_landing', None),'x'),
        ('solver',  'cost',             'compromised_battery',  1,  1e1,        ('update cost for compromised_battery', None),'x'),
        ('solver',  'cost',             'transition',       1,      1e-1,        ('update cost for transition', None), 'x'),

        ('solver',  'cost',             'fictitious',           2,  1.e0,       ('second update cost for fictitious', None), 'x'),
        ('solver',  'cost',             'compromised_battery',  2,  0,          ('second update cost for compromised_battery', None),'x'),
        ('solver',  'cost',             'tracking',             2,  1.e-6,          ('second update cost for tracking', None),'x'),

        ('solver',    None,          None,        'save_trial',            False,              ('Automatically save trial after solving', [True, False]),'x'),
        ('solver',    None,          None,        'save_format',    'dict',     ('trial save format', ['awe', 'dict']), 'x'),

        ### problem health diagnostics options
        ('solver',  'health_check',     'when',     'autorun',                  False,  ('run a health-check after every homotopy step. CAUTION: VERY SLOW!', [True, False]),'x'),
        ('solver',  'health_check',     'when',     'failure',                  False,  ('run a health-check when a homotopy step fails. CAUTION: SLOW!', [True, False]),'x'),
        ('solver',  'health_check',     'when',     'final',                    False,  ('run a health-check after final homotopy step. CAUTION: SLOW!', [True, False]), 'x'),
        ('solver',  'health_check',     'thresh',   'active',                   1e0,    ('threshold for a constraint to be considered active (smallest ratio between lambda and g). should be larger than 1', None), 'x'),
        ('solver',  'health_check',     'thresh',   'reduced_hessian_eig',      1e-6,   ('minimum value of eigenvalues of the reduced hessian, allowed for positive-definiteness', None), 'x'),
        ('solver',  'health_check',     'thresh',   'condition_number',         1e7,    ('problem ill-conditioning test threshold - largest problem condition number (ratio between max/min singular values) [-]', None), 'x'),
        ('solver',  'health_check',     'tol',      'reduced_hessian_null',     1e-8,   ('tolerance of null-space computation on reduced hessian', None), 'x'),
        ('solver',  'health_check',     'tol',      'constraint_jacobian_rank', 1e-8,   ('tolerance of rank compution for constraint jacobian', None), 'x'),
        ('solver',  'health_check',     'tol',      'linear_dependence_ratio',  1e-2,   ('tolerance of rough linear dependence identifier', None), 'x'),
        ('solver',  'health_check',     None,       'spy_matrices',           False,    ('make spy plot of KKT matrix - requires manual closing', None), 'x'),

        ### simulation options
        ('sim', None,  None,    'number_of_finite_elements',  20,                 ('Integrator steps in one sampling interval', None), 'x'),
        ('sim', None,  None,    'sys_params',                 None,               ('system parameters dict', None), 'x'),

        ### mpc options
        ('mpc', None,  None,    'N',            10,                 ('MPC horizon', None), 'x'),
        ('mpc', None,  None,    'scheme',      'radau',             ('NLP collocation scheme', ['legendre','radau']), 'x'),
        ('mpc', None,  None,    'd',            4,                  ('NLP collocation polynomial order', None), 'x'),
        ('mpc', None,  None,    'jit',          False,              ('MPC solver jitting', None), 'x'),
        ('mpc', None,  None,    'expand',       True,               ('expand NLP expressions', None), 'x'),
        ('mpc', None,  None,    'cost_type',    'tracking',         ('MPC cost function type', ['tracking','economic']), 'x'),
        ('mpc', None,  None,    'linear_solver','ma57',             ('MPC cost function type', None), 'x'),
        ('mpc', None,  None,    'max_iter',     1000,               ('MPC solver max iterations', None), 'x'),
        ('mpc', None,  None,    'max_cpu_time', 2000,               ('MPC solver max cpu time', None), 'x'),
        ('mpc', None,  None,    'plot_flag',    False,              ('MPC plot solution for each step', None), 'x'),
        ('mpc', None,  None,    'ref_interpolator','spline',        ('periodic reference interpolation method', None), 'x'),

        ### visualization options
        ('visualization', 'cosmetics', 'trajectory', 'colors',      kite_colors,    ('list of colors for trajectory', None), 'x'),
        ('visualization', 'cosmetics', 'trajectory', 'axisfont',    {'size': '20'}, ('???', None), 'x'),
        ('visualization', 'cosmetics', 'trajectory', 'ylabelsize',  15,             ('???', None), 'x'),
        ('visualization', 'cosmetics', 'trajectory', 'kite_bodies', False,          ('choose whether kite bodies should be plotted or not', [True, False]), 'x'),
        ('visualization', 'cosmetics', 'trajectory', 'body_cross_sections_per_meter', 3,       ('discretization level of kite body visualization', None), 'x'),
        ('visualization', 'cosmetics', 'trajectory', 'wake_nodes',  False,          ('draw wake nodes into instantaneous plots', [True, False]), 'x'),
        ('visualization', 'cosmetics', 'trajectory', 'alpha',       0.3,            ('transparency of trajectories in animation', None), 'x'),
        ('visualization', 'cosmetics', 'trajectory', 'margin',      0.05,           ('trajectory figure margins', None), 'x'),
        ('visualization', 'cosmetics', None,         'save_figs',   False,          ('save the figures', [True, False]), 'x'),
        ('visualization', 'cosmetics', None,         'plot_coll',   True,           ('plot the collocation variables', [True, False]), 'x'),
        ('visualization', 'cosmetics', None,         'plot_ref',    False,          ('plot the tracking reference trajectory', [True, False]), 'x'),
        ('visualization', 'cosmetics', None,         'plot_bounds', False,          ('plot the variable bounds', [True, False]), 'x'),
        ('visualization', 'cosmetics', 'interpolation', 'include',  True,           ('???', None), 'x'),
        ('visualization', 'cosmetics', 'interpolation', 'type',     'poly',         ('???', None), 'x'),
        ('visualization', 'cosmetics', 'interpolation', 'N',        100,            ('???', None), 'x'),
        ('visualization', 'cosmetics', 'states',      'colors',     dim_colors,     ('list of colors for states', None), 'x'),
        ('visualization', 'cosmetics', 'states',      'axisfont',   {'size': '20'}, ('???', None), 'x'),
        ('visualization', 'cosmetics', 'states',      'ylabelsize', 15,             ('???', None), 'x'),
        ('visualization', 'cosmetics', 'controls',    'colors',     dim_colors,     ('list of colors for controls', None), 'x'),
        ('visualization', 'cosmetics', 'controls',    'axisfont',   {'size': '20'}, ('???', None), 'x'),
        ('visualization', 'cosmetics', 'controls',    'ylabelsize', 15,             ('???', None), 'x'),
        ('visualization', 'cosmetics', 'invariants',  'colors',     dim_colors,     ('list of colors for invariants', None), 'x'),
        ('visualization', 'cosmetics', 'invariants',  'axisfont',   {'size': '20'}, ('???', None), 'x'),
        ('visualization', 'cosmetics', 'invariants',  'ylabelsize', 15,             ('???', None), 'x'),
        ('visualization', 'cosmetics', 'algebraic_variables', 'colors', dim_colors, ('list of colors for algebraic variables', None), 'x'),
        ('visualization', 'cosmetics', 'algebraic_variables', 'axisfont', {'size': '20'}, ('???', None), 'x'),
        ('visualization', 'cosmetics', 'algebraic_variables', 'ylabelsize', 15,     ('???', None), 'x'),
        ('visualization', 'cosmetics', 'diagnostics', 'colors',     dim_colors,     ('list of colors for algebraic variables', None), 'x'),
        ('visualization', 'cosmetics', 'diagnostics', 'axisfont',   {'size': '20'}, ('???', None), 'x'),
        ('visualization', 'cosmetics', 'diagnostics', 'ylabelsize', 15,             ('???', None), 'x'),
        ('visualization', 'cosmetics', 'animation',   'snapshot_index', 0,          ('???', None), 'x'),
        ('visualization', 'cosmetics', None,          'show_when_ready', False,             ('display plots as soon as they are ready', [True, False]), 'x'),

        # quality check options
        ('quality', 'test_param', None, 'c_max', 1e0,                       ('maximum invariant test parameter', None), 'x'),
        ('quality', 'test_param', None, 'dc_max', 1e1,                      ('maximum invariant test parameter', None), 'x'),
        ('quality', 'test_param', None, 'ddc_max', 5e1,                     ('maximum invariant test parameter', None), 'x'),
        ('quality', 'test_param', None, 'max_loyd_factor', 30,              ('maximum loyd factor test parameter', None), 'x'),
        ('quality', 'test_param', None, 'max_power_harvesting_factor', 100, ('maximum power harvesting factor test parameter', None), 'x'),
        ('quality', 'test_param', None, 'max_tension', 1e6,                 ('maximum max main tether tension test parameter', None), 'x'),
        ('quality', 'test_param', None, 'max_velocity', 100.,               ('maximum kite velocity test parameter', None), 'x'),
        ('quality', 'test_param', None, 't_f_min', 5.,                      ('minimum final time test parameter', None), 'x'),
        ('quality', 'test_param', None, 'power_balance_thresh', 5e-2,       ('power balance threshold test parameter', None), 'x'),
        ('quality', 'test_param', None, 'slacks_thresh', 1.e-6,             ('threshold value for slacked equality constraints being satisfied', None), 'x'),
        ('quality', 'test_param', None, 'max_control_interval', 10.,        ('max control interval test parameter', None), 'x'),
        ('quality', 'test_param', None, 'last_vortex_ind_factor_thresh', 0.01,('maximum ratio between induced velocity from last vortex rings and wind speed', None), 'x'),
        ('quality', 'test_param', None, 'check_energy_summation', False,    ('check that no kinetic or potential energy source has gotten lost', None), 'x'),
        ('quality', 'test_param', None, 'energy_summation_thresh', 1.e-10,  ('maximum lost kinetic or potential energy from different calculations', None), 'x'),
    ]

    default_options_tree = add_available_aerodynamic_stability_derivative_overwrites(default_options_tree)

    default_options, help_options = funcs.assemble_options_tree(default_options_tree, default_user_options, help_options)

    return default_options, help_options


def add_available_aerodynamic_stability_derivative_overwrites(default_options_tree):

    associated_force_coeffs = {
        'control': ['CX', 'CY', 'CZ'],
        'earth': ['Cx', 'Cy', 'Cz'],
        'body': ['CA', 'CY', 'CN'],
        'wind': ['CD', 'CS', 'CL']
    }
    associated_moment_coeffs = {
        'control': ['Cl', 'Cm', 'Cn']
    }
    available_coeffs = []
    for coeffs in [associated_force_coeffs, associated_moment_coeffs]:
        for frame_name in coeffs.keys():
            available_coeffs += coeffs[frame_name]

    available_inputs = ['0', 'alpha', 'beta', 'p', 'q', 'r', 'deltaa', 'deltae', 'deltar']

    for coeff in available_coeffs:
        for input in available_inputs:
            coeff_and_input = coeff + input
            default_options_tree.append(('model', 'aero', 'overwrite', coeff_and_input, None, ('aerodynamic parameter', None), 's'))

    return default_options_tree
