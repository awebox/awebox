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

import numpy as np
import copy
from awebox.logger.logger import Logger as awelogger
import awebox.opts.model_funcs as model_funcs
import awebox.tools.print_operations as print_op


def build_options_dict(options, help_options, architecture):

    # single out user options
    user_options = options['user_options']

    # check for unsupported settings
    if user_options['trajectory']['type'] in ['nominal_landing', 'compromised_landing', 'transition']:
        awelogger.logger.error('Error: ' + user_options['trajectory']['type'] + ' is not supported for current release. Build the newest casADi from source and check out the awebox develop branch to use nominal_landing, compromised_landing or transition.')

    # initialize additional options tree
    options_tree = []

    # share the trajectory information among all headers that need the info
    options_tree = share_trajectory_type(options, options_tree)

    # build the options for each of the awebox headers
    fixed_params = {}
    options_tree, fixed_params = model_funcs.build_model_options(options, help_options, user_options, options_tree, fixed_params, architecture)
    options_tree, phase_fix = build_nlp_options(options, help_options, user_options, options_tree, architecture)
    options_tree = build_solver_options(options, help_options, user_options, options_tree, architecture, fixed_params, phase_fix)
    options_tree = build_formulation_options(options, help_options, user_options, options_tree, architecture)

    # assemble all of the options into a complete options tree
    options, help_options = assemble_options_tree(options_tree, options, help_options)
    options, help_options = assemble_system_parameter_dict(options, help_options)

    return options, help_options



###### assemble the options information

def assemble_options_tree(options_tree, options, help_options):

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

def assemble_system_parameter_dict(options, help_options):

    options['model']['params'] = options['params']
    options['solver']['initialization']['sys_params_num'] = options['params']

    return options, help_options



###### share the trajectory type information with all headers that need this info

def share_trajectory_type(options, options_tree=[]):

    user_options = options['user_options']

    trajectory_type = user_options['trajectory']['type']
    system_type = user_options['trajectory']['system_type']
    descript = ('type of trajectory to optimize', ['power_cycle', 'transition', 'mpc'])

    options_tree.append(('nlp', None, None, 'type', trajectory_type, descript,'x'))
    options_tree.append(('formulation', 'trajectory', None, 'type', trajectory_type, descript,'x'))
    options_tree.append(('solver','initialization',None,'type', trajectory_type, descript,'x'))
    options_tree.append(('solver','initialization',None,'system_type', system_type, descript,'x'))
    options_tree.append(('model', 'trajectory', None, 'type', trajectory_type, descript,'x'))
    options_tree.append(('model', 'trajectory', None, 'system_type', system_type, descript,'x'))

    options_tree.append(('formulation', 'trajectory', 'tracking', 'fix_tether_length', user_options['trajectory']['tracking']['fix_tether_length'], descript,'x'))

    if trajectory_type in ['power_cycle', 'tracking']:
        if (user_options['trajectory']['lift_mode']['max_l_t'] != None):

            options_tree.append(('model', 'system_bounds', 'xd', 'l_t', [options['model']['system_bounds']['xd']['l_t'][0],
                         user_options['trajectory']['lift_mode']['max_l_t']],
                         ('user input for maximum main tether length', None),'x'))

        if user_options['trajectory']['lift_mode']['pumping_range']:
            pumping_range = copy.copy(user_options['trajectory']['lift_mode']['pumping_range'])
            for i in range(len(pumping_range)):
                if pumping_range[i]:
                    pumping_range[i] = pumping_range[i]/options['model']['scaling']['xd']['l_t']
            options_tree.append(('nlp',None,None,'pumping_range', pumping_range, ('set predefined pumping range (only in comb. w. phase-fix)', None),'x'))

    if trajectory_type in ['transition','nominal_landing','compromised_landing']:
        options_tree.append(('formulation', 'trajectory', 'transition', 'initial_trajectory', user_options['trajectory']['transition']['initial_trajectory'], ('possible options', ['lift_mode', 'transition']),'x'))
    if trajectory_type in ['transition','launch']:
        options_tree.append(('formulation', 'trajectory', 'transition', 'terminal_trajectory', user_options['trajectory']['transition']['terminal_trajectory'], ('possible options', ['lift_mode', 'transition']),'x'))

    return options_tree



###### build options for each options header

def build_nlp_options(options, help_options, user_options, options_tree, architecture):

    ### switch off phase fixing for landing/transition trajectories
    if user_options['trajectory']['type'] in ['nominal_landing', 'compromised_landing', 'transition', 'mpc']:
        phase_fix = False
    else:
        if user_options['trajectory']['system_type'] == 'lift_mode':
            phase_fix = user_options['trajectory']['lift_mode']['phase_fix']
        else:
            phase_fix = False
    options_tree.append(('nlp', None, None, 'phase_fix', phase_fix,  ('lift-mode phase fix', (True, False)),'x'))
    options_tree.append(('nlp', None, None, 'system_type', user_options['trajectory']['system_type'],  ('AWE system type', ('lift_mode', 'drag_mode')),'x'))

    n_k = options['nlp']['n_k']
    options_tree.append(('nlp', 'cost', 'normalization', 'tracking',             n_k,             ('tracking cost normalization', None),'x'))
    options_tree.append(('nlp', 'cost', 'normalization', 'u_regularisation',     n_k,             ('regularisation cost normalization', None),'x'))
    options_tree.append(('nlp', 'cost', 'normalization', 'theta_regularisation', n_k,             ('regularisation cost normalization', None), 'x'))
    options_tree.append(('nlp', 'cost', 'normalization', 'ddq_regularisation',   n_k,             ('ddq_regularisation cost normalization', None),'x'))
    options_tree.append(('nlp', 'cost', 'normalization', 'fictitious',           n_k,             ('fictitious cost normalization', None),'x'))
    options_tree.append(('nlp', 'cost', 'normalization', 'slack',                n_k,             ('regularisation cost normalization', None),'x'))

    options_tree.append(('nlp', None, None, 'kite_dof', user_options['system_model']['kite_dof'], ('give the number of states that designate each kites position: 3 (implies roll-control), 6 (implies DCM rotation)', [3, 6]), 'x')),

    options_tree.append(('nlp', 'landing', 'cost', 'position_weight', options['formulation']['nominal_landing']['position_weight'], ('???', None),'x'))
    options_tree.append(('nlp', 'landing', 'cost', 'velocity_weight', options['formulation']['nominal_landing']['velocity_weight'], ('???', None),'x'))

    options_tree.append(('nlp', 'landing', None, 'emergency_scenario', user_options['trajectory']['compromised_landing']['emergency_scenario'], ('type of emergency scenario', ['broken_roll','broken_lift']),'x'))
    options_tree.append(('nlp', 'landing', None, 'xi_0_initial', user_options['trajectory']['compromised_landing']['xi_0_initial'], ('starting position on initial trajectory between 0 and 1', None),'x'))
    options_tree.append(('solver', 'initialization', 'compromised_landing', 'xi_0_initial', user_options['trajectory']['compromised_landing']['xi_0_initial'], ('starting position on initial trajectory between 0 and 1', None),'x'))
    options_tree.append(('nlp', 'system_model', None, 'kite_dof', user_options['system_model']['kite_dof'], ('???', None),'x'))

    options_tree.append(('nlp', 'jit_code_gen', None, 'include', options['model']['jit_code_gen']['include'],  ('????', None),'x'))
    options_tree.append(('nlp', 'jit_code_gen', None, 'compiler', options['model']['jit_code_gen']['compiler'], ('????', None), 'x'))

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

    options_tree.append(('solver', 'initialization', 'model','architecture', user_options['system_model']['architecture'],('secondary  tether natural diameter [m]', None),'x'))

    ## cross-tether
    options_tree.append(('solver', 'initialization', None, 'cross_tether', user_options['system_model']['cross_tether'], ('enable cross-tether',[True,False]),'x'))
    options_tree.append(('solver', 'initialization', None, 'cross_tether_attachment', options['model']['tether']['cross_tether']['attachment'], ('cross-tether attachment',[True,False]),'x'))
    rotation_bounds = options['params']['model_bounds']['rot_angles'][0]
    options_tree.append(('solver', 'initialization', None, 'rotation_bounds', np.pi/2-rotation_bounds, ('enable cross-tether',[True,False]),'x'))

    # solver weights:
    if options['solver']['weights_overwrite']['dddl_t'] is None:
        jerk_scale = 1.e-3 #1.e1
        jerk_weight = jerk_scale * options['model']['scaling']['xd']['l_t']**2 # make independent of tether length scaling
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

    options_tree.append(('solver',  'initialization', 'xd', 'l_t', options['solver']['initialization']['l_t'],      ('initial guess main tether length', [True, False]), 'x'))

    options_tree.append(('solver', None, None,'expand', expand, ('choose True or False', [True, False]),'x'))

    acc_max = options['model']['model_bounds']['acceleration']['acc_max'] * options['model']['scaling']['other']['g']
    options_tree.append(('solver', 'initialization', None, 'acc_max', acc_max, ('maximum acceleration allowed within hardware constraints [m/s^2]', None),'x'))

    if user_options['trajectory']['system_type'] == 'drag_mode':
        windings = 1
    else:
        windings = user_options['trajectory']['lift_mode']['windings']

    options_tree.append(('solver', 'initialization',  None, 'windings', windings, ('number of windings [int]', None),'x'))
    options_tree.append(('solver', 'homotopy', None, 'phase_fix_reelout', options['nlp']['phase_fix_reelout'], ('time fraction of reel-out phase', None),'x'))
    options_tree.append(('solver', 'homotopy', None, 'phase_fix', phase_fix,  ('lift-mode phase fix', (True, False)),'x'))

    options_tree.append(('solver', None, None, 'fixed_q_r_values', False,
                         ('fix the positions and rotations to their initial guess values', [True, False]),'x'))

    if user_options['trajectory']['type'] == 'tracking' and user_options['trajectory']['tracking']['fix_tether_length']:
        options['solver']['initialization']['fix_tether_length'] = True

    if options['solver']['homotopy_method'] not in ['penalty', 'classic']:
        awelogger.logger.error('Error: homotopy method "' + options['solver']['homotopy_method'] + '" unknown!')

    return options_tree



def build_formulation_options(options, help_options, user_options, options_tree, architecture):

    options_tree.append(('formulation', 'landing', None, 'xi_0_initial', user_options['trajectory']['compromised_landing']['xi_0_initial'], ('starting position on initial trajectory between 0 and 1', None),'x'))
    options_tree.append(('formulation', 'compromised_landing', None, 'emergency_scenario', user_options['trajectory']['compromised_landing']['emergency_scenario'], ('???', None),'x'))
    options_tree.append(('formulation', None, None, 'n_k', options['nlp']['n_k'], ('???', None),'x'))
    options_tree.append(('formulation', None, None, 'phase_fix', options['user_options']['trajectory']['lift_mode']['phase_fix'], ('???', None), 'x'))
    options_tree.append(('formulation', 'collocation', None, 'd', options['nlp']['collocation']['d'], ('???', None),'x'))
    options_tree.append(('formulation', None, None, 'phase_fix_reelout', options['nlp']['phase_fix_reelout'], ('???', None),'x'))
    options_tree.append(
        ('formulation', None, None, 'discretization', options['nlp']['discretization'], ('???', None), 'x'))
    options_tree.append(
        ('formulation', 'collocation', None, 'u_param', options['nlp']['collocation']['u_param'], ('???', None), 'x'))

    options_tree.append(
        ('formulation', 'collocation', None, 'scheme', options['nlp']['collocation']['scheme'], ('???', None), 'x'))
    if int(user_options['system_model']['kite_dof']) == 3:
        coeff_max = np.array(options['model']['aero']['three_dof']['coeff_max'])
        coeff_min = np.array(options['model']['aero']['three_dof']['coeff_min'])
        battery_model_parameters = load_battery_parameters(options['user_options']['kite_standard'], coeff_max, coeff_min)
        for name in list(battery_model_parameters.keys()):
            if options['formulation']['compromised_landing']['battery'][name] is None:
                options_tree.append(('formulation', 'compromised_landing', 'battery', name, battery_model_parameters[name], ('???', None),'t'))

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



def load_battery_parameters(kite_standard, coeff_max, coeff_min):

    if kite_standard is None:
        raise ValueError("No kite data provided")
    else:
        battery = kite_standard['battery']

    return battery

