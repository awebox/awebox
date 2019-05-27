#
#    This file is part of awebox.
#
#    awebox -- A modeling and optimization framework for multi-kite AWE systems.
#    Copyright (C) 2017-2019 Jochem De Schutter, Rachel Leuthold, Moritz Diehl,
#                            ALU Freiburg.
#    Copyright (C) 2018-2019 Thilo Bronnenmeyer, Kiteswarms Ltd.
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
import numpy as np
import awebox as awe
import casadi as cas
import copy
import logging
import pickle

import awebox.tools.struct_operations as struct_op

def build_options_tree(options_tree, options, help_options):

    for branch in options_tree: #build tree
        # initialize category field if necessary
        if branch[0] not in list(options.keys()):
            options[branch[0]] = {}
            help_options[branch[0]] = {}

        if branch[1] is None:
            options[branch[0]][branch[3]] = branch[4]
            help_options[branch[0]][branch[3]] = [branch[5], branch[6]]

        elif branch[2] is None:
            # initialize sub-category field if necessary
            if branch[1] not in list(options[branch[0]].keys()):
                options[branch[0]][branch[1]] = {}
                help_options[branch[0]][branch[1]] = {}

            options[branch[0]][branch[1]][branch[3]] = branch[4]
            help_options[branch[0]][branch[1]][branch[3]] = [branch[5], branch[6]]
        else:
            # initialize sub-category field if necessary
            if branch[1] not in list(options[branch[0]].keys()):
                options[branch[0]][branch[1]] = {}
                help_options[branch[0]][branch[1]] = {}
            # initialize sub-sub-category field if necessary
            if branch[2] not in list(options[branch[0]][branch[1]].keys()):
                options[branch[0]][branch[1]][branch[2]] = {}
                help_options[branch[0]][branch[1]][branch[2]] = {}

            options[branch[0]][branch[1]][branch[2]][branch[3]] = branch[4]
            help_options[branch[0]][branch[1]][branch[2]][branch[3]] = [branch[5], branch[6]]

    return options, help_options

def build_model_options(options, help_options, user_options, options_tree, architecture):

    ### geometry
    geometry = load_kite_geometry(options['user_options']['kite_standard'])
    geometry = build_geometry(options['model']['geometry']['overwrite'], geometry)
    for name in list(geometry.keys()):
        if help_options['model']['geometry']['overwrite'][name][1] == 's':
            dict_type = 'params'
        else:
            dict_type = 'model'
        options_tree.append((dict_type, 'geometry', None, name,geometry[name], ('???', None),'x'))

    ### system bounds
    if int(user_options['system_model']['kite_dof']) == 3:
        # do not include rotation constraints (only for 6dof)
        options_tree.append(('model', 'model_bounds', 'rotation', 'include', False, ('include constraints on roll and ptich motion', None),'t'))
    elif int(user_options['system_model']['kite_dof']) == 6:
        delta_max = geometry['delta_max']
        ddelta_max = geometry['ddelta_max']
        options_tree.append(('model', 'system_bounds', 'xd', 'delta', [-1. * delta_max, delta_max], ('control surface deflection bounds', None),'x'))
        options_tree.append(('model', 'system_bounds', 'u', 'ddelta', [-1. * ddelta_max, ddelta_max],
                             ('control surface deflection rate bounds', None),'x'))
    else:
        raise ValueError('Invalid kite DOF chosen.')


    options_tree.append(('model', 'compromised_landing', None, 'emergency_scenario', user_options['trajectory']['compromised_landing']['emergency_scenario'], ('type of emergency scenario', ['broken_roll','broken_lift']),'x'))

    ## orientation
    options_tree.append(('model', None, None, 'kite_dof', user_options['system_model']['kite_dof'],('give the number of states that designate each kites position: 3 (implies roll-control), 6 (implies DCM rotation)',[3,6]),'x')),
    options_tree.append(('model', None, None, 'surface_control', user_options['system_model']['surface_control'],('which derivative of the control-surface-deflection is controlled?: 0 (control of deflections), 1 (control of deflection rates)', [0, 1]),'x')),

    ## system outputs
    options_tree.append(('model', None, None, 'integral_outputs', options['nlp']['cost']['output_quadrature'], ('do not include integral outputs as system states',[True,False]),'x'))

    ## aerodynamics
    options_tree = share_aerodynamics_options(options, options_tree, help_options)

    ## wind
    options_tree.append(('model', 'wind', None, 'model', user_options['wind']['model'],('wind model', None),'x'))
    options_tree.append(('params', 'wind', None, 'u_ref', user_options['wind']['u_ref'],('reference wind speed [m/s]', None),'x'))
    options_tree.append(('model', 'wind', None, 'atmosphere_heightsdata', user_options['wind']['atmosphere_heightsdata'],('data for the heights at this time instant', None),'x'))
    options_tree.append(('model', 'wind', None, 'atmosphere_featuresdata', user_options['wind']['atmosphere_featuresdata'],('data for the features at this time instant', None),'x'))

    ## atmosphere
    options_tree.append(('model',  'atmosphere', None, 'model', user_options['atmosphere'], ('atmosphere model', None),'x'))
    options_tree.append(('params',  'atmosphere', None, 'q_ref', 0.5*options['params']['atmosphere']['rho_ref']*user_options['wind']['u_ref']**2, ('aerodynamic pressure [bar]', None),'x'))

    ### model bounds
    if int(user_options['system_model']['kite_dof']) == 3:
        compromised_factor = options['model']['model_bounds']['dcoeff_compromised_factor']
        options_tree.append(('model', 'model_bounds','aero_validity','include',False,('do not include aero validity for roll control',None),'x'))
        options_tree.append(('model', 'model_bounds','dcoeff_actuation','include',True,('include dcoeff bound for roll control',None),'x'))
        options_tree.append(('model', 'model_bounds','coeff_actuation','include',True,('include dcoeff bound for roll control',None),'x'))
        options_tree.append(('params', 'model_bounds',None,'dcoeff_compromised_max',np.array([5*compromised_factor,5]),('include dcoeff bound for roll control',None),'x'))
        options_tree.append(('params', 'model_bounds',None,'dcoeff_compromised_min',np.array([-5*compromised_factor,-5]),('include dcoeff bound for roll control',None),'x'))

    ua_ref = options['solver']['initialization']['ua_norm']
    options_tree.append(('model', 'model_bounds', 'anticollision_radius', 'num_ref', ua_ref ** 2., ('an estimate of the square of the apparent velocity, for normalization of the anticollision inequality', None),'x'))
    options_tree.append(('model', 'model_bounds', 'aero_validity', 'num_ref', ua_ref, ('an estimate of the apparent velocity, for normalization of the aero_validity orientation inequality', None),'x'))

    if architecture.number_of_kites == 1:
        options_tree.append(('model', 'model_bounds', 'anticollision', 'include', False, ('anticollision inequality', (True,False)),'x'))

    options_tree.append(('model', 'model_constr', None, 'include', False, None,'x'))

    # map single tether power interval constraint to min and max constraint
    if options['model']['model_bounds']['tether_force']['include'] == True:
        options_tree.append(('model', 'model_bounds', 'tether_force_max', 'include', True, None,'x'))
        options_tree.append(('model', 'model_bounds', 'tether_force_min', 'include', True, None,'x'))
        tether_force_include = True
    else:
        tether_force_include = False

    # check which tether force/stress constraints to enforce on which node
    tether_constraint_includes = {'force': [], 'stress': []}
    diameter = None
    if (options['model']['model_bounds']['tether_stress']['include'] and \
        tether_force_include):

        for node in range(1, architecture.number_of_nodes):
            if node in architecture.kite_nodes:
                if node == 1:
                    if 'diam_t' in user_options['trajectory']['fixed_params']:
                        diameter = user_options['trajectory']['fixed_params']['diam_t']
                else:
                    if 'diam_s' in user_options['trajectory']['fixed_params']:
                        diameter = user_options['trajectory']['fixed_params']['diam_s']
                if diameter != None:
                    cross_section = np.pi * 2 / 4 * diameter
                    if diameter * cross_section <= options['params']['model_bounds']['tether_force_max']:
                        tether_constraint_includes['stress'] += [node]
                    else:
                        tether_constraint_includes['force'] += [node]
                else:
                    tether_constraint_includes['stress'] += [node]
                    tether_constraint_includes['force'] += [node]

            else:
                tether_constraint_includes['stress'] += [node]
    
    else:
        if options['model']['model_bounds']['tether_stress']['include']:
            tether_constraint_includes['stress'] = range(1, architecture.number_of_nodes)
        if options['model']['model_bounds']['tether_stress']['include']:
            tether_constraint_includes['force'] = architecture.kite_nodes

    options_tree.append(('model', 'model_bounds', 'tether', 'tether_constraint_includes', tether_constraint_includes, ('logic deciding which tether constraints to enforce', None), 'x'))

    # map single airspeed interval constraint to min/max constraints
    if options['model']['model_bounds']['airspeed']['include']:
        options_tree.append(('model', 'model_bounds', 'airspeed_max', 'include', True,   ('include max airspeed constraint', None),'x'))
        options_tree.append(('model', 'model_bounds', 'airspeed_min', 'include', True,   ('include min airspeed constraint', None),'x'))

    ddl_t_max = options['model']['ground_station']['ddl_t_max']

    if options['model']['tether']['control_var'] == 'ddl_t':
        options_tree.append(('model', 'system_bounds', 'u', 'ddl_t', [-1. * ddl_t_max, ddl_t_max],   ('main tether max acceleration [m/s^2]', None),'x'))
    elif options['model']['tether']['control_var'] == 'dddl_t':
        options_tree.append(('model', 'system_bounds', 'xd', 'ddl_t', [-1. * ddl_t_max, ddl_t_max],   ('main tether max acceleration [m/s^2]', None),'x'))
        options_tree.append(('model', 'system_bounds', 'u', 'dddl_t', [-10. * ddl_t_max, 10. * ddl_t_max],   ('main tether max jerk [m/s^3]', None),'x'))
    else:
        raise ValueError('invalid tether control variable chosen')

    if user_options['trajectory']['type'] not in ['nominal_landing', 'transitions', 'compromised_landing', 'launch']:
        fixed_params = user_options['trajectory']['fixed_params']
    else:
        if user_options['trajectory']['type'] == 'launch':
            initial_or_terminal = 'terminal'
        else:
            initial_or_terminal = 'initial'
        parameterized_trajectory = user_options['trajectory']['transition'][initial_or_terminal + '_trajectory']
        if type(parameterized_trajectory) == awe.trial.Trial:
            parameterized_trial = parameterized_trajectory
            V_pickle = parameterized_trial.optimization.V_final
        elif type(parameterized_trajectory) == str:
            relative_path = copy.deepcopy(parameterized_trajectory)
            parameterized_trial = pickle.load(open(parameterized_trajectory, 'rb'))
            if relative_path[-4:] == ".awe":
                V_pickle = parameterized_trial.optimization.V_final
            elif relative_path[-5:] == ".dict":
                V_pickle = parameterized_trial['solution_dict']['V_final']
        fixed_params = {}
        for theta in struct_op.subkeys(V_pickle, 'theta'):
            if theta not in ['t_f']:
                fixed_params[theta] = V_pickle['theta', theta]
    for theta in list(fixed_params.keys()):
        options_tree.append(('model', 'system_bounds', 'theta', theta, [fixed_params[theta]]*2,  ('user input for fixed bounds on theta', None),'x'))

    ### scaling
    gravity = options['model']['scaling']['other']['g']
    f_fict_scaling = 0.5 * options['model']['model_bounds']['acceleration']['acc_max'] * geometry['m_k'] * gravity
    m_fict_scaling = 0.5 * options['model']['model_bounds']['acceleration']['acc_max'] * geometry['m_k'] * gravity * geometry['b_ref'] / 2.
    options_tree.append(('model', 'scaling', 'u', 'f_fict', f_fict_scaling, ('scaling of fictitious homotopy forces', None),'x'))
    options_tree.append(('model', 'scaling', 'u', 'm_fict', m_fict_scaling, ('scaling of fictitious homotopy moments', None),'x'))

    lambda_scaling_overwrite = options['model']['scaling_overwrite']['xa']['lambda']
    e_scaling_overwrite = options['model']['scaling_overwrite']['xd']['e']

    [lambda_scaling, energy_scaling, power_cost] = get_suggested_lambda_energy_power_scaling(options, architecture)

    if not lambda_scaling_overwrite == None:
        lambda_scaling = lambda_scaling_overwrite

    if not e_scaling_overwrite == None:
        energy_scaling = e_scaling_overwrite

    if options['model']['scaling_overwrite']['lambda_tree']['include']:
        options_tree = generate_lambda_scaling_tree(options= options, options_tree= options_tree, lambda_scaling= lambda_scaling, architecture = architecture)
    else:
        options_tree.append(('model', 'scaling', 'xa', 'lambda', lambda_scaling, ('scaling of tether tension per length', None),'x'))

    options_tree.append(('model', 'scaling', 'xd', 'e', energy_scaling, ('scaling of the energy', None),'x'))

    return options_tree, fixed_params

def build_nlp_options(options, help_options, user_options, options_tree, architecture):

    ### switch off phase fixing for landing/transition trajectories
    if user_options['trajectory']['type'] in ['nominal_landing', 'compromised_landing', 'transition', 'mpc']:
        phase_fix = False
    else:
        phase_fix = user_options['trajectory']['lift_mode']['phase_fix']
    options_tree.append(('nlp', None, None, 'phase_fix', phase_fix,  ('lift-mode phase fix', (True, False)),'x'))

    n_k = options['nlp']['n_k']
    options_tree.append(('nlp', 'cost', 'normalization', 'tracking',             n_k,             ('tracking cost normalization', None),'x'))
    options_tree.append(('nlp', 'cost', 'normalization', 'regularisation',       n_k,             ('regularisation cost normalization', None),'x'))
    options_tree.append(('nlp', 'cost', 'normalization', 'ddq_regularisation',   n_k,             ('ddq_regularisation cost normalization', None),'x'))
    options_tree.append(('nlp', 'cost', 'normalization', 'fictitious',           n_k,             ('fictitious cost normalization', None),'x'))

    options_tree.append(('nlp', 'landing', None, 'emergency_scenario', user_options['trajectory']['compromised_landing']['emergency_scenario'], ('type of emergency scenario', ['broken_roll','broken_lift']),'x'))
    options_tree.append(('nlp', 'landing', None, 'xi_0_initial', user_options['trajectory']['compromised_landing']['xi_0_initial'], ('starting position on initial trajectory between 0 and 1', None),'x'))
    options_tree.append(('solver', 'initialization', 'compromised_landing', 'xi_0_initial', user_options['trajectory']['compromised_landing']['xi_0_initial'], ('starting position on initial trajectory between 0 and 1', None),'x'))
    options_tree.append(('nlp', 'system_model', None, 'kite_dof', user_options['system_model']['kite_dof'], ('???', None),'x'))

    # parallelize function evaluations in NLP
    if options['nlp']['parallelization']['overwrite'] is not None:
        parallelize = options['nlp']['parallelization']['overwrite']
    else:
        if architecture.layers == 1:
            parallelize = False
        else:
            # parallelization starts to become effective from two layers on
            parallelize = True

    # integrator options
    if options['nlp']['integrator']['jit_overwrite'] is not None:
        options_tree.append(('nlp', 'integrator', None, 'jit', options['nlp']['integrator']['jit_overwrite'],  ('jit integrator', (True, False)),'x'))
    elif options['nlp']['integrator']['type'] == 'collocation':
        options_tree.append(('nlp', 'integrator', None, 'jit', options['nlp']['integrator']['jit_coll'],  ('jit integrator', (True, False)),'x'))
    elif options['nlp']['integrator']['type'] in ['idas', 'rk4root']:
        options_tree.append(('nlp', 'integrator', None, 'jit', options['nlp']['integrator']['jit_idas'],  ('jit integrator', (True, False)),'x'))

    if options['nlp']['integrator']['num_steps_overwrite'] is not None:
        options_tree.append(('nlp', 'integrator', None, 'num_steps', options['nlp']['integrator']['num_steps_overwrite'],  ('number of internal integrator steps', (True, False)),'x'))
    elif options['nlp']['integrator']['type'] == 'collocation':
        options_tree.append(('nlp', 'integrator', None, 'num_steps', options['nlp']['integrator']['num_steps_coll'],  ('number of internal integrator steps', (True, False)),'x'))
    elif options['nlp']['integrator']['type'] == 'rk4root':
        options_tree.append(('nlp', 'integrator', None, 'num_steps', options['nlp']['integrator']['num_steps_rk4root'],  ('number of internal integrator steps', (True, False)),'x'))

    options_tree.append(('nlp', 'parallelization', None, 'include', parallelize,  ('parallelize functions in nlp', (True, False)),'x'))


    return options_tree, phase_fix

def build_solver_options(options, help_options, user_options, options_tree, architecture, fixed_params, phase_fix):

    if user_options['trajectory']['type'] in ['nominal_landing','compromised_landing']:
        options_tree.append(('solver', 'cost', 'ddq_regularisation', 0,       1e-1,        ('starting cost for ddq_regularisation', None),'x'))
        options_tree.append(('solver', None, None, 'mu_hippo',       1e-5,        ('target for interior point homotop parameter for hippo strategy [float]', None),'x'))
        options_tree.append(('solver', None, None, 'tol_hippo',       1e-4,        ('tolerance for interior point homotop parameter for hippo strategy [float]', None),'x'))

    if user_options['trajectory']['type'] in ['transition']:
        options_tree.append(('solver', 'cost', 'ddq_regularisation', 0,       1e-3,        ('starting cost for ddq_regularisation', None),'x'))
        options_tree.append(('solver', None, None, 'mu_hippo',       1e-5,        ('target for interior point homotop parameter for hippo strategy [float]', None),'x'))
        options_tree.append(('solver', None, None, 'tol_hippo',       1e-4,        ('tolerance for interior point homotop parameter for hippo strategy [float]', None),'x'))

    # initialize theta params with standard scaling length or fixed length
    initialization_theta = options['model']['scaling']['theta']
    for param in list(user_options['trajectory']['fixed_params'].keys()):
        initialization_theta[param] = user_options['trajectory']['fixed_params'][param]
    for param in list(fixed_params.keys()):
        initialization_theta[param] = fixed_params[param]
    for param in list(initialization_theta.keys()):
        options_tree.append(('solver', 'initialization', 'theta', param, initialization_theta[param], ('initial guess for parameter ' + param, None), 'x'))

    options_tree.append(('solver', 'initialization', 'xd', 'l_t', 500.0, ('secondary tether natural length [m]', None),'x'))
    options_tree.append(('solver', 'initialization', 'model','architecture', user_options['system_model']['architecture'],('secondary  tether natural diameter [m]', None),'x'))

    # solver weights:
    if options['solver']['weights_overwrite']['dddl_t'] is None:
        jerk_weight = 1e1*options['model']['scaling']['xd']['l_t']**2 # make independent of tether length scaling
    else:
        jerk_weight = options['solver']['weights_overwrite']['dddl_t']
    options_tree.append(('solver', 'weights', None, 'dddl_t', jerk_weight,('optimization weight for control variable dddl_t [-]', None),'s'))

    # expand MX -> SX in solver
    expand = True
    if options['solver']['expand_overwrite'] is not None:
        expand = options['solver']['expand_overwrite']
    else:
        if options['nlp']['discretization'] == 'multiple_shooting':
            # integrators / rootfinder do not support eval_sx
            expand = False
        if user_options['trajectory']['type'] in ['transition','nominal_landing','compromised_landing','launch']:
            expand = False

    options_tree.append(('solver', None, None,'expand', expand, ('choose True or False', [True, False]),'x'))

    acc_max = options['model']['model_bounds']['acceleration']['acc_max'] * options['model']['scaling']['other']['g']
    options_tree.append(('solver', 'initialization', None, 'acc_max', acc_max, ('maximum acceleration allowed within hardware constraints [m/s^2]', None),'x'))

    options_tree.append(('solver', 'initialization',  None, 'windings', user_options['trajectory']['lift_mode']['windings'], ('number of windings [int]', None),'x'))
    options_tree.append(('solver', 'homotopy', None, 'phase_fix_reelout', options['nlp']['phase_fix_reelout'], ('time fraction of reel-out phase', None),'x'))
    options_tree.append(('solver', 'homotopy', None, 'phase_fix', phase_fix,  ('lift-mode phase fix', (True, False)),'x'))

    if user_options['trajectory']['type'] == 'aero_test':
        options_tree.append(('solver', None, None, 'fixed_q_r_values', True,
                             ('fix the positions and rotations to their initial guess values', [True, False]),'x'))
    else:
        options_tree.append(('solver', None, None, 'fixed_q_r_values', False,
                             ('fix the positions and rotations to their initial guess values', [True, False]),'x'))

    if user_options['trajectory']['type'] == 'tracking' and user_options['trajectory']['tracking']['fix_tether_length']:
        options['solver']['initialization']['fix_tether_length'] = True

    [lambda_scaling, energy_scaling, power_cost] = get_suggested_lambda_energy_power_scaling(options, architecture)
    power_cost_overwrite = options['solver']['cost_overwrite']['power'][1]
    if not power_cost_overwrite == None:
        power_cost = power_cost_overwrite

    options_tree.append(('solver', 'cost', 'power', 1, power_cost, ('update cost for power', None),'x'))


    return options_tree

def build_formulation_options(options, help_options, user_options, options_tree, architecture):

    options_tree.append(('formulation', 'landing', None, 'xi_0_initial', user_options['trajectory']['compromised_landing']['xi_0_initial'], ('starting position on initial trajectory between 0 and 1', None),'x'))
    options_tree.append(('formulation', 'compromised_landing', None, 'emergency_scenario', user_options['trajectory']['compromised_landing']['emergency_scenario'], ('???', None),'x'))
    options_tree.append(('formulation', None, None, 'n_k', options['nlp']['n_k'], ('???', None),'x'))
    options_tree.append(('formulation', 'collocation', None, 'd', options['nlp']['collocation']['d'], ('???', None),'x'))
    if int(user_options['system_model']['kite_dof']) == 3:
        coeff_max = np.array(options['model']['aero']['three_dof']['coeff_max'])
        coeff_min = np.array(options['model']['aero']['three_dof']['coeff_min'])
        battery_model_parameters = load_battery_parameters(options['user_options']['kite_standard'], coeff_max, coeff_min)
        for name in list(battery_model_parameters.keys()):
            if options['formulation']['compromised_landing']['battery'][name] is None:
                options_tree.append(('formulation', 'compromised_landing', 'battery', name, battery_model_parameters[name], ('???', None),'t'))

    return options_tree

def build_options_dict(options, help_options, architecture):

    # single out user options
    user_options = options['user_options']

    # check for unsupported settings
    if user_options['trajectory']['type'] in ['nominal_landing', 'compromised_landing', 'transition']:
        logging.error('Error: ' + user_options['trajectory']['type'] + ' is not supported for current release. Build the newest casADi from source and check out the awebox develop branch to use nominal_landing, compromised_landing or transition.')

    # initialize additional options tree
    options_tree = []

    options_tree = share_trajectory_type(options, options_tree)

    options_tree, fixed_params = build_model_options(options, help_options, user_options, options_tree, architecture)
    options_tree, phase_fix = build_nlp_options(options, help_options, user_options, options_tree, architecture)
    options_tree = build_solver_options(options, help_options, user_options, options_tree, architecture, fixed_params, phase_fix)
    options_tree = build_formulation_options(options, help_options, user_options, options_tree, architecture)

    # BUILD OPTIONS
    options, help_options = build_options_tree(options_tree, options, help_options)
    options, help_options = build_system_parameter_dict(options, help_options)

    return options, help_options

def build_system_parameter_dict(options, help_options):

    options['model']['params'] = options['params']
    options['solver']['initialization']['sys_params_num'] = options['params']

    return options, help_options


def generate_lambda_scaling_tree(options, options_tree, lambda_scaling, architecture):

    # set lambda_scaling
    options_tree.append(('model', 'scaling', 'xa', 'lambda10', lambda_scaling, ('scaling of tether tension per length', None),'x'))

    # extract architecure options
    layers = architecture.layers

    # extract length scaling information
    l_s_scaling = options['model']['scaling']['theta']['l_s']
    l_t_scaling = options['model']['scaling']['xd']['l_t']
    l_i_scaling = options['model']['scaling']['theta']['l_i']

    #  secondary tether scaling
    lambda_s_scaling = lambda_scaling*l_t_scaling/(l_s_scaling*architecture.number_of_kites)
    # assign scaling according to tree structure
    layer_count = 1
    for node in range(2,architecture.number_of_nodes):
        if node in architecture.kite_nodes:
            options_tree.append(('model', 'scaling', 'xa', 'lambda'+str(node)+str(architecture.parent_map[node]), lambda_s_scaling, ('scaling of tether tension per length', None),'x'))
        else:
            lambda_i_scaling = (layers - layer_count)/(float(layers))*lambda_scaling*l_t_scaling/l_i_scaling
            options_tree.append(('model', 'scaling', 'xa', 'lambda'+str(node)+str(architecture.parent_map[node]), lambda_i_scaling, ('scaling of tether tension per length', None),'x'))
            layer_count += 1

    return options_tree

def get_suggested_lambda_energy_power_scaling(options, architecture):

    # single out user options
    user_options = options['user_options']

    user_levels = architecture.layers
    user_children = architecture.children[architecture.layer_nodes[0]]
    user_kite_dof = user_options['system_model']['kite_dof']
    user_induction = user_options['induction_model']
    user_kite = user_options['kite_standard']['name']

    lambda_scaling = 1e3
    energy_scaling = 1e4
    power_cost = 1e-1

    kite_poss = ['ampyx', 'boeing747', 'bubble']
    induction_poss = ['not_in_use', 'actuator']
    kite_dof_poss = [3, 6]
    children_poss = [1, 2, 3, 4, 5]
    levels_poss = [1, 2, 3]

    lam_scale_dict = {}
    for level in levels_poss:

        if not level in list(lam_scale_dict.keys()):
            lam_scale_dict[level] = {}

        for children in children_poss:

            if not children in list(lam_scale_dict[level].keys()):
                lam_scale_dict[level][children] = {}

            for kite_dof in kite_dof_poss:

                if not kite_dof in list(lam_scale_dict[level][children].keys()):
                    lam_scale_dict[level][children][kite_dof] = {}

                for induction in induction_poss:

                    if not induction in list(lam_scale_dict[level][children][kite_dof].keys()):
                        lam_scale_dict[level][children][kite_dof][induction] = {}

                    for kite in kite_poss:
                        lam_scale_dict[level][children][kite_dof][induction][kite] = None

    # layer - children - dof - induction - kite
    # lam_scale_dict[1][1][6]['actuator']['ampyx'] = [1., 1e3, 1.]
    lam_scale_dict[1][1][6]['actuator']['ampyx'] = [1e3, 1e3, 0.1]
    lam_scale_dict[1][1][6]['actuator']['bubble'] = [1e3, 100., 1.]
    # lam_scale_dict[1][2][6]['actuator']['ampyx'] = [1., 1e3, 0.1]
    # lam_scale_dict[1][2][6]['actuator']['ampyx'] = [1e3, 10., 10.]
    lam_scale_dict[1][2][6]['actuator']['ampyx'] = [1e2, 1e2, 0.01]
    lam_scale_dict[1][2][6]['actuator']['bubble'] = [1e3, 10., 1e-2]
    lam_scale_dict[1][1][3]['actuator']['ampyx'] = [1e3, 1e3, 1]
    lam_scale_dict[1][1][3]['actuator']['bubble'] = [1e3, 1e3, 1]
    lam_scale_dict[1][2][3]['actuator']['ampyx'] = [1., 1e5, 0.1]
    lam_scale_dict[1][2][3]['actuator']['bubble'] = [1., 1e5, 0.1]
    lam_scale_dict[1][2][6]['not_in_use']['ampyx'] = [10., 1e4, 0.1]
    lam_scale_dict[1][2][6]['not_in_use']['bubble'] = [10., 1e4, 0.1]

    if user_kite in kite_poss:
        if not lam_scale_dict[user_levels][user_children][user_kite_dof][user_induction][user_kite] == None:
            given_scaling = lam_scale_dict[user_levels][user_children][user_kite_dof][user_induction][user_kite]
            lambda_scaling = given_scaling[0]
            energy_scaling = given_scaling[1]
            power_cost = given_scaling[2]

    else:
        logging.warning('Warning: no scalings match the chosen kite data. Default values are used.')

    if user_options['trajectory']['type'] == 'nominal_landing':
        power_cost = 1e-4
        lambda_scaling = 1
        energy_scaling = 1e5

    return lambda_scaling, energy_scaling, power_cost

def build_geometry(geometry_options, geometry_data):

    basic_options_params = extract_basic_geometry_params(geometry_options, geometry_data)
    geometry = build_geometry_params(basic_options_params, geometry_options, geometry_data)

    return geometry

def extract_basic_geometry_params(geometry_options, geometry_data):

    basic_params = ['s_ref','b_ref','c_ref','ar']
    basic_options_params = {}
    for name in list(geometry_options.keys()):
        if name in basic_params and geometry_options[name]:
            basic_options_params[name] = geometry_options[name]

    return basic_options_params

def build_geometry_params(basic_options_params, geometry_options, geometry_data):

    basic_params = ['s_ref','b_ref','c_ref','ar']
    dependent_params = ['s_ref','b_ref','c_ref','ar','m_k','j','c_root','c_tip','length','height']

    # initialize geometry
    geometry = {}

    # check if geometry if overdetermined
    if len(list(basic_options_params.keys())) > 2:
        raise ValueError("Geometry overdetermined, possibly inconsistent!")

    # check if basic geometry is being overwritten
    if len(list(basic_options_params.keys())) > 0:
        geometry = build_basic_params(geometry, basic_options_params, geometry_data)
        geometry =  build_dependent_params(geometry, geometry_data)

    # check if independent or dependent geometry parameters are being overwritten
    overwrite_set = set(geometry_options.keys())
    for name in overwrite_set:
        if geometry_options[name] is None:
            32.0
        else:
            geometry[name] = geometry_options[name]

    # fill in remaining geometry data with user-provided data
    for name in list(geometry_data.keys()):
        if name not in list(geometry.keys()):
            geometry[name] = geometry_data[name]

    return geometry

def build_basic_params(geometry, basic_options_params,geometry_data):

    if 's_ref' in list(basic_options_params.keys()):
        geometry['s_ref'] = basic_options_params['s_ref']
        if 'b_ref' in list(basic_options_params.keys()):
            geometry['b_ref'] = basic_options_params['b_ref']
            geometry['c_ref'] = geometry['s_ref']/geometry['b_ref']
            geometry['ar'] = geometry['b_ref']/geometry['c_ref']
        elif 'c_ref' in list(basic_options_params.keys()):
            geometry['c_ref'] = basic_options_params['c_ref']
            geometry['b_ref'] = geometry['s_ref']/geometry['c_ref']
            geometry['ar'] = geometry['b_ref']/geometry['c_ref']
        elif 'ar' in list(basic_options_params.keys()):
            geometry['ar'] = basic_options_params['ar']
            geometry['c_ref'] = np.sqrt(geometry['s_ref']/geometry['ar'])
            geometry['b_ref'] = geometry['s_ref']/geometry['c_ref']
        else:
            geometry['ar'] = geometry_data['ar']
            geometry['c_ref'] = np.sqrt(geometry['s_ref']/geometry['ar'])
            geometry['b_ref'] = geometry['s_ref']/geometry['c_ref']
    elif 'b_ref' in list(basic_options_params.keys()):
        geometry['b_ref'] = basic_options_params['b_ref']
        if 'c_ref' in list(basic_options_params.keys()):
            geometry['c_ref'] = basic_options_params['c_ref']
            geometry['s_ref'] = geometry['b_ref']*geometry['c_ref']
            geometry['ar'] = geometry['b_ref']/geometry['c_ref']
        elif 'ar' in list(basic_options_params.keys()):
            geometry['ar'] = basic_options_params['ar']
            geometry['c_ref'] = geometry['b_ref']/geometry['ar']
            geometry['s_ref'] = geometry['b_ref']*geometry['c_ref']
        else:
            geometry['ar'] = geometry_data['ar']
            geometry['c_ref'] = geometry['b_ref']/geometry['ar']
            geometry['s_ref'] = geometry['b_ref']*geometry['c_ref']
    elif 'c_ref' in list(basic_options_params.keys()):
        geometry['c_ref'] = basic_options_params['c_ref']
        if 'ar' in list(basic_options_params.keys()):
            geometry['ar'] = basic_options_params['ar']
            geometry['b_ref'] = geometry['c_ref']*geometry['ar']
            geometry['s_ref'] = geometry['b_ref']*geometry['c_ref']
        else:
            geometry['ar'] = geometry_data['ar']
            geometry['b_ref'] = geometry['c_ref']*geometry['ar']
            geometry['s_ref'] = geometry['b_ref']*geometry['c_ref']
    elif 'ar' in list(basic_options_params.keys()):
        geometry['s_ref'] = geometry_data['s_ref']
        geometry['c_ref'] = np.sqrt(geometry['s_ref']/geometry['ar'])
        geometry['b_ref'] = geometry['s_ref']/geometry['c_ref']

    return geometry

def build_dependent_params(geometry, geometry_data):

    geometry['m_k'] = geometry['s_ref']/geometry_data['s_ref'] * geometry_data['m_k']  # [kg]

    geometry['j'] = geometry_data['j'] * geometry['m_k']/geometry_data['m_k'] # bad scaling appoximation..
    geometry['length'] = geometry['b_ref']  # only for plotting
    geometry['height'] = geometry['b_ref'] / 5.  # only for plotting

    geometry['c_root'] = 1.4 * geometry['c_ref']
    geometry['c_tip'] = 2. * geometry['c_ref'] - geometry['c_root']

    return geometry

def share_aerodynamics_options(options, options_tree, help_options):

    user_options = options['user_options']

    ## stability derivatives
    aero = load_stability_derivatives(options['user_options']['kite_standard'])
    for name in list(aero.keys()):
        if help_options['model']['aero']['overwrite'][name][1] == 's':
            dict_type = 'params'
        else:
            dict_type = 'model'
        if options['model']['aero']['overwrite'][name]:
            options_tree.append((dict_type, 'aero', None, name,options['model']['aero']['overwrite'][name], ('???', None),'x'))
        else:
            options_tree.append((dict_type, 'aero', None, name,aero[name], ('???', None),'t'))

    ## induction
    induction_model_descript = ('model to approximate induction from wake', ['not_in_use', 'actuator'])
    options_tree.append(('model', None, None, 'induction_model', user_options['induction_model'], induction_model_descript,'x'))
    options_tree.append(('formulation', None, None, 'induction_model', user_options['induction_model'], induction_model_descript,'x'))
    options_tree.append(('nlp', None, None, 'induction_model', user_options['induction_model'], induction_model_descript,'x'))
    options_tree.append(('solver', 'initialization', 'model', 'induction_model', user_options['induction_model'], induction_model_descript,'x'))

    induction_steadyness = options['model']['aero']['actuator']['steadyness']
    options_tree.append(('solver', 'initialization', 'model', 'induction_steadyness', induction_steadyness, ('????', None), 'x')),
    induction_varrho_ref = options['model']['aero']['actuator']['varrho_ref']
    options_tree.append(('solver', 'initialization', 'model', 'induction_varrho_ref', induction_varrho_ref, ('????', None), 'x')),
    induction_correct_tilt = options['model']['aero']['actuator']['correct_tilt']
    options_tree.append(('solver', 'initialization', 'model', 'induction_correct_tilt', induction_correct_tilt, ('????', None), 'x')),

    ## actuator-disk induction
    a_ref = options['model']['aero']['actuator']['a_ref']
    a_range = options['model']['aero']['actuator']['a_range']

    if induction_model_descript == 'not_in_use':
        a_ref = None

    options_tree.append(('model', 'aero', None, 'a_ref', a_ref, ('reference value for the induction factors. takes values between 0. and 0.4', None),'x'))

    if options['model']['aero']['actuator']['steadyness'] == 'steady':
        a_var_type = 'xl'
    elif options['model']['aero']['actuator']['steadyness'] == 'unsteady':
        a_var_type = 'xd'

    options_tree.append(('solver', 'initialization', a_var_type, 'a', a_ref, ('induction factor [-]', None),'x'))
    options_tree.append(('model', 'system_bounds', a_var_type, 'a', a_range, ('induction factor bounds [-]', None),'x'))
    options_tree.append(('model', 'system_bounds', 'xl', 'varrho', [0.,cas.inf], ('relative radius bounds [-]', None), 'x'))
    options_tree.append(('model', 'system_bounds', 'xl', 'cosgamma', [0., 1.], ('tilt angle cosine bounds [-]', None), 'x')),
    options_tree.append(('model', 'system_bounds', 'xl', 'fnorm', [0., cas.inf], ('normalization factor for normal vector [-]', None), 'x')),
    options_tree.append(('model', 'system_bounds', 'xl', 'nhat', [np.array([0., -cas.inf, -cas.inf]), np.array([cas.inf, cas.inf, cas.inf])],   ('normal vector [-]', None), 'x')),

    if options['model']['aero']['actuator']['normal_vector_model'] in ['default','tether_parallel']:
        options_tree.append(('solver', 'initialization', None, 'fnorm', 'tether_length', ('induction factor [-]', None),'x'))
    else:
        options_tree.append(('solver', 'initialization', None, 'fnorm', 'unit_length', ('induction factor [-]', None),'x'))

    ## tether drag
    tether_drag_descript =  ('model to approximate the tether drag on the tether nodes', ['trivial', 'simple', 'equivalence', 'not_in_use'])
    options_tree.append(('model', 'tether', 'tether_drag', 'model_type', user_options['tether_drag_model'], tether_drag_descript,'x'))
    options_tree.append(('formulation', None, None, 'tether_drag_model', user_options['tether_drag_model'], tether_drag_descript,'x'))

    return options_tree

def share_trajectory_type(options, options_tree=[]):

    user_options = options['user_options']

    trajectory_type = user_options['trajectory']['type']
    descript = ('type of trajectory to optimize', ['lift_mode', 'transition', 'aero_test'])

    options_tree.append(('nlp', None, None, 'type', trajectory_type, descript,'x'))
    options_tree.append(('formulation', 'trajectory', None, 'type', trajectory_type, descript,'x'))
    options_tree.append(('solver','initialization',None,'type', trajectory_type, descript,'x'))
    options_tree.append(('model', 'trajectory', None, 'type', trajectory_type, descript,'x'))
    options_tree.append(('formulation', 'trajectory', 'tracking', 'fix_tether_length', user_options['trajectory']['tracking']['fix_tether_length'], descript,'x'))

    if trajectory_type == 'lift_mode' or trajectory_type == 'tracking':
        if (user_options['trajectory']['lift_mode']['max_l_t'] != None):

            options_tree.append(('model', 'system_bounds', 'xd', 'l_t', [options['model']['system_bounds']['xd']['l_t'][0],
                         user_options['trajectory']['lift_mode']['max_l_t']],
                         ('user input for maximum main tether length', None),'x'))

        if user_options['trajectory']['lift_mode']['pumping_range']:
            pumping_range = user_options['trajectory']['lift_mode']['pumping_range']
            for i in range(len(pumping_range)):
                if pumping_range[i]:
                    pumping_range[i] = pumping_range[i]/options['model']['scaling']['xd']['l_t']
            options_tree.append(('nlp',None,None,'pumping_range', pumping_range, ('set predefined pumping range (only in comb. w. phase-fix)', None),'x'))

    if trajectory_type in ['transition','nominal_landing','compromised_landing']:
        options_tree.append(('formulation', 'trajectory', 'transition', 'initial_trajectory', user_options['trajectory']['transition']['initial_trajectory'], ('possible options', ['lift_mode', 'transition']),'x'))
    if trajectory_type in ['transition','launch']:
        options_tree.append(('formulation', 'trajectory', 'transition', 'terminal_trajectory', user_options['trajectory']['transition']['terminal_trajectory'], ('possible options', ['lift_mode', 'transition']),'x'))

    if trajectory_type == 'aero_test':

        phi0_val = user_options['trajectory']['aero_test']['phi_0']
        phi0_descript = ('pitch angle amplitude for pitch-plunge test [rad]', None)
        h0_val = user_options['trajectory']['aero_test']['h_0']
        h0_descript = ('plunge amplitude for pitch-plunge test [m]', None)
        omega_val = user_options['trajectory']['aero_test']['omega']
        omega_descript = ('frequency of pitching/plunging motion for pitch-plunge test [rad/s]', None)

        options_tree.append(('formulation', 'trajectory', 'aero_test', 'phi_0', phi0_val, phi0_descript,'x'))
        options_tree.append(('formulation', 'trajectory', 'aero_test', 'h_0', h0_val, h0_descript,'x'))
        options_tree.append(('formulation', 'trajectory', 'aero_test', 'omega', omega_val, omega_descript,'x'))

        options_tree.append(('solver', 'initialization', 'aero_test', 'phi_0', phi0_val, phi0_descript,'x'))
        options_tree.append(('solver', 'initialization', 'aero_test', 'h_0', h0_val, h0_descript,'x'))
        options_tree.append(('solver', 'initialization', 'aero_test', 'omega', omega_val, omega_descript,'x'))

        options_tree.append(('solver', 'initialization', 'aero_test', 'total_periods',
                             options['formulation']['trajectory']['aero_test']['total_periods'],
                             ('total number of oscillations of the wing', None),'x'))

    return options_tree

def make_fictitious_bounds_update(user_options, architecture):

    entries = []

    for kite in architecture.kite_nodes:
        parent = architecture.parent_map[kite]

        entries += [('solver', 'bounds', 'f_fict' + str(kite) + str(parent), 1, ['ub', 'u', 0],
                             ('first update for fictitious controls', None))]
        entries += [('solver', 'bounds', 'm_fict' + str(kite) + str(parent), 1, ['ub', 'u', 0],
                             ('first update for fictitious controls', None))]

        entries += [('solver', 'bounds', 'f_fict' + str(kite) + str(parent), 2, ['lb', 'u', 0],
                             ('second update for fictitious controls', None))]
        entries += [('solver', 'bounds', 'm_fict' + str(kite) + str(parent), 2, ['lb', 'u', 0],
                             ('second update for fictitious controls', None))]

    return entries


def load_kite_geometry(kite_standard):

    if kite_standard is None:
        raise ValueError("No kite data provided")
    else:
        geometry = kite_standard['geometry']

    return geometry


def load_stability_derivatives(kite_standard):

    if kite_standard is None:
        raise ValueError("No kite data provided")
    else:
        aero_deriv = kite_standard['aero_deriv']

    return aero_deriv

def load_battery_parameters(kite_standard, coeff_max, coeff_min):

    if kite_standard is None:
        raise ValueError("No kite data provided")
    else:
        battery = kite_standard['battery']

    return battery
