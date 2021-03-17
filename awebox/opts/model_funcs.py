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
'''
options_tree extension functions for options initially related to heading 'model'
_python-3.5 / casadi-3.4.5
- author: jochem de scutter, rachel leuthold, thilo bronnenmeyer, alu-fr/kiteswarms 2017-20
'''

import numpy as np
import awebox as awe
import casadi as cas
import copy
from awebox.logger.logger import Logger as awelogger
import pickle
import awebox.tools.struct_operations as struct_op
import awebox.tools.performance_operations as perf_op
import awebox.tools.print_operations as print_op
import awebox.mdl.wind as wind
import awebox.tools.vector_operations as vect_op

def build_model_options(options, help_options, user_options, options_tree, fixed_params, architecture):

    # kite
    options_tree, fixed_params = build_geometry_options(options, help_options, options_tree, fixed_params)
    options_tree, fixed_params = build_kite_dof_options(options, options_tree, fixed_params)

    # problem specifics
    options_tree, fixed_params = build_constraint_applicablity_options(options, options_tree, fixed_params, architecture)
    options_tree, fixed_params = build_trajectory_options(options, options_tree, fixed_params, architecture)
    options_tree, fixed_params = build_integral_options(options, options_tree, fixed_params)

    # aerodynamics
    options_tree, fixed_params = build_stability_derivative_options(options, help_options, options_tree, fixed_params)
    options_tree, fixed_params = build_induction_options(options, help_options, options_tree, fixed_params, architecture)
    options_tree, fixed_params = build_actuator_options(options, options_tree, fixed_params)
    options_tree, fixed_params = build_vortex_options(options, options_tree, fixed_params, architecture)

    # tether
    options_tree, fixed_params = build_tether_drag_options(options, options_tree, fixed_params)
    options_tree, fixed_params = build_wound_tether_length_options(options, options_tree, fixed_params)
    options_tree, fixed_params = build_tether_stress_options(options, options_tree, fixed_params, architecture)
    options_tree, fixed_params = build_tether_control_options(options, options_tree, fixed_params)

    # environment
    options_tree, fixed_params = build_wind_options(options, options_tree, fixed_params)
    options_tree, fixed_params = build_atmosphere_options(options, options_tree, fixed_params)

    # scaling
    options_tree, fixed_params = build_fict_scaling_options(options, options_tree, fixed_params)
    options_tree, fixed_params = build_lambda_e_power_scaling(options, options_tree, fixed_params, architecture)

    return options_tree, fixed_params


####### geometry

def build_geometry_options(options, help_options, options_tree, fixed_params):

    geometry = get_geometry(options)
    for name in list(geometry.keys()):
        if help_options['model']['geometry']['overwrite'][name][1] == 's':
            dict_type = 'params'
        else:
            dict_type = 'model'
        options_tree.append((dict_type, 'geometry', None, name,geometry[name], ('???', None),'x'))

    return options_tree, fixed_params

def get_geometry(options):

    standard_geometry = load_kite_geometry(options['user_options']['kite_standard'])
    overwrite_options = options['model']['geometry']['overwrite']

    basic_options_params = extract_basic_geometry_params(overwrite_options, standard_geometry)
    geometry = get_geometry_params(basic_options_params, overwrite_options, standard_geometry)

    return geometry

def load_kite_geometry(kite_standard):

    if kite_standard is None:
        raise ValueError("No kite data provided")
    else:
        geometry = kite_standard['geometry']

    return geometry

def extract_basic_geometry_params(geometry_options, geometry_data):

    basic_params = ['s_ref','b_ref','c_ref','ar']
    basic_options_params = {}
    for name in list(geometry_options.keys()):
        if name in basic_params and geometry_options[name]:
            basic_options_params[name] = geometry_options[name]

    return basic_options_params

def get_geometry_params(basic_options_params, geometry_options, geometry_data):

    basic_params = ['s_ref','b_ref','c_ref','ar']
    dependent_params = ['s_ref','b_ref','c_ref','ar','m_k','j','c_root','c_tip','length','height']

    # initialize geometry
    geometry = {}

    # check if geometry if overdetermined
    if len(list(basic_options_params.keys())) > 2:
        raise ValueError("Geometry overdetermined, possibly inconsistent!")

    # check if basic geometry is being overwritten
    if len(list(basic_options_params.keys())) > 0:
        geometry = get_basic_params(geometry, basic_options_params, geometry_data)
        geometry = get_dependent_params(geometry, geometry_data)

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

def get_basic_params(geometry, basic_options_params,geometry_data):

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

def get_dependent_params(geometry, geometry_data):

    geometry['m_k'] = geometry['s_ref']/geometry_data['s_ref'] * geometry_data['m_k']  # [kg]

    geometry['j'] = geometry_data['j'] * geometry['m_k']/geometry_data['m_k'] # bad scaling appoximation..
    geometry['length'] = geometry['b_ref']  # only for plotting
    geometry['height'] = geometry['b_ref'] / 5.  # only for plotting

    geometry['c_root'] = 1.4 * geometry['c_ref']
    geometry['c_tip'] = 2. * geometry['c_ref'] - geometry['c_root']

    return geometry


##### kite dof

def build_kite_dof_options(options, options_tree, fixed_params):

    user_options = options['user_options']

    kite_dof = get_kite_dof(user_options)

    options_tree.append(('model', None, None, 'kite_dof', kite_dof, ('give the number of states that designate each kites position: 3 (implies roll-control), 6 (implies DCM rotation)',[3,6]),'x')),
    options_tree.append(('model', None, None, 'surface_control', user_options['system_model']['surface_control'], ('which derivative of the control-surface-deflection is controlled?: 0 (control of deflections), 1 (control of deflection rates)', [0, 1]),'x')),

    if (not int(kite_dof) == 6) and (not int(kite_dof) == 3):
        raise ValueError('Invalid kite DOF chosen.')

    elif int(kite_dof) == 6:
        geometry = get_geometry(options)
        delta_max = geometry['delta_max']
        ddelta_max = geometry['ddelta_max']
        options_tree.append(('model', 'system_bounds', 'xd', 'delta', [-1. * delta_max, delta_max], ('control surface deflection bounds', None),'x'))
        options_tree.append(('model', 'system_bounds', 'u', 'ddelta', [-1. * ddelta_max, ddelta_max],
                             ('control surface deflection rate bounds', None),'x'))

        options_tree.append(('model', 'scaling', 'xd', 'delta', 1, ('???', None), 'x'))
        options_tree.append(('model', 'scaling', 'xd', 'omega', 1, ('???', None), 'x'))
        options_tree.append(('model', 'scaling', 'xd', 'r', 1, ('descript', None), 'x'))


    return options_tree, fixed_params

def get_kite_dof(user_options):
    kite_dof = user_options['system_model']['kite_dof']
    return kite_dof


###### constraint applicability

def build_constraint_applicablity_options(options, options_tree, fixed_params, architecture):

    user_options = options['user_options']

    kite_dof = get_kite_dof(user_options)
    kite_has_3_dof = (int(kite_dof) == 3)
    if kite_has_3_dof:

        # do not include rotation constraints (only for 6dof)
        options_tree.append(('model', 'model_bounds', 'rotation', 'include', False, ('include constraints on roll and ptich motion', None),'t'))

        coeff_scaling = 0.1
        options_tree.append(('model', 'scaling', 'xd', 'coeff', coeff_scaling, ('???', None), 'x'))

        options_tree.append(('model', 'model_bounds','aero_validity','include',False,('do not include aero validity for roll control',None),'x'))

        compromised_factor = options['model']['aero']['three_dof']['dcoeff_compromised_factor']
        dcoeff_compromised_max = np.array([5*compromised_factor,5])
        options_tree.append(('params', 'model_bounds', None, 'dcoeff_compromised_max', dcoeff_compromised_max, ('????', None), 'x'))
        options_tree.append(('params', 'model_bounds', None, 'dcoeff_compromised_min', -1. * dcoeff_compromised_max, ('?????', None), 'x'))

    else:
        options_tree.append(('model', 'model_bounds', 'coeff_actuation', 'include', False, ('???', None), 'x'))
        options_tree.append(('model', 'model_bounds', 'dcoeff_actuation', 'include', False, ('???', None), 'x'))

    groundspeed = options['solver']['initialization']['groundspeed']
    options_tree.append(('model', 'model_bounds', 'anticollision_radius', 'num_ref', groundspeed ** 2., ('an estimate of the square of the kite speed, for normalization of the anticollision inequality', None),'x'))

    u_ref = get_u_ref(options['user_options'])
    airspeed_ref = cas.sqrt(groundspeed**2 + u_ref**2)
    options_tree.append(('model', 'model_bounds', 'aero_validity', 'airspeed_ref', airspeed_ref, ('an estimate of the kite speed, for normalization of the aero_validity orientation inequality', None),'x'))

    airspeed_limits = options['params']['model_bounds']['airspeed_limits']
    airspeed_include = options['model']['model_bounds']['airspeed']['include']
    options_tree.append(('solver', 'initialization', None, 'airspeed_limits', airspeed_limits, ('airspeed limits [m/s]', None), 's'))
    options_tree.append(('solver', 'initialization', None, 'airspeed_include', airspeed_include, ('apply airspeed limits [m/s]', None), 's'))


    options_tree.append(('model', None, None, 'cross_tether', user_options['system_model']['cross_tether'], ('enable cross-tether',[True,False]),'x'))
    if architecture.number_of_kites == 1 or user_options['system_model']['cross_tether']:
        options_tree.append(('model', 'model_bounds', 'anticollision', 'include', False, ('anticollision inequality', (True,False)),'x'))

    # map single airspeed interval constraint to min/max constraints
    if options['model']['model_bounds']['airspeed']['include']:
        options_tree.append(('model', 'model_bounds', 'airspeed_max', 'include', True,   ('include max airspeed constraint', None),'x'))
        options_tree.append(('model', 'model_bounds', 'airspeed_min', 'include', True,   ('include min airspeed constraint', None),'x'))

    return options_tree, fixed_params


####### trajectory specifics

def build_trajectory_options(options, options_tree, fixed_params, architecture):

    user_options = options['user_options']

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

        for theta in struct_op.subkeys(V_pickle, 'theta'):
            if theta not in ['t_f']:
                fixed_params[theta] = V_pickle['theta', theta]
    for theta in list(fixed_params.keys()):
        options_tree.append(('model', 'system_bounds', 'theta', theta, [fixed_params[theta]]*2,  ('user input for fixed bounds on theta', None),'x'))

    scenario, broken_kite = user_options['trajectory']['compromised_landing']['emergency_scenario']
    if not broken_kite in architecture.kite_nodes:
        broken_kite = architecture.kite_nodes[0]

    options_tree.append(('model', 'compromised_landing', None, 'emergency_scenario', [scenario, broken_kite], ('type of emergency scenario', ['broken_roll','broken_lift']),'x'))
    options_tree.append(('nlp', 'trajectory', None, 'type', user_options['trajectory']['type'], ('??', None), 'x'))

    t_f_guess = estimate_time_period(options, architecture)
    options_tree.append(('nlp', 'normalization', None, 't_f', t_f_guess, ('??', None), 'x'))


    return options_tree, fixed_params


def get_windings(user_options):
    if user_options['trajectory']['system_type'] == 'drag_mode':
        windings = 1
    else:
        windings = user_options['trajectory']['lift_mode']['windings']
    return windings

###### integral_outputs

def build_integral_options(options, options_tree, fixed_params):

    options_tree.append(('model', None, None, 'integral_outputs', options['nlp']['cost']['output_quadrature'], ('do not include integral outputs as system states',[True,False]),'x'))

    check_energy_summation = options['quality']['test_param']['check_energy_summation']
    options_tree.append(('model', 'test', None, 'check_energy_summation', check_energy_summation, ('check that no kinetic or potential energy source has gotten lost', None), 'x'))

    return options_tree, fixed_params




####### stability derivatives

def build_stability_derivative_options(options, help_options, options_tree, fixed_params):

    stab_derivs, aero_validity = load_stability_derivatives(options['user_options']['kite_standard'])
    for deriv_name in list(stab_derivs.keys()):

        if deriv_name == 'frame':
            for frame_type in stab_derivs[deriv_name].keys():

                specified_frame = stab_derivs[deriv_name][frame_type]
                options_tree.append(('model', 'aero', 'stab_derivs', frame_type + '_frame', specified_frame, ('???', None), 't'))

        else:
            for input_name in stab_derivs[deriv_name].keys():
                local_vals = stab_derivs[deriv_name][input_name]

                combi_name = deriv_name + input_name

                if help_options['model']['aero']['overwrite'][combi_name][1] == 's':
                    dict_type = 'params'
                else:
                    dict_type = 'model'

                overwrite_vals = options['model']['aero']['overwrite'][combi_name]
                if not overwrite_vals == None:
                    local_vals = overwrite_vals

                local_vals = cas.DM(local_vals)

                options_tree.append((dict_type, 'aero', deriv_name, input_name, local_vals, ('???', None),'x'))

    for bound_name in aero_validity.keys():
        local_vals = aero_validity[bound_name]

        overwrite_vals = options['model']['aero']['overwrite'][bound_name]
        if not overwrite_vals == None:
            local_vals = overwrite_vals

        options_tree.append(
            ('model', 'aero', None, bound_name, local_vals, ('???', None), 'x'))

    return options_tree, fixed_params

def load_stability_derivatives(kite_standard):

    if kite_standard is None:
        raise ValueError("No kite data provided")
    else:
        aero_deriv = kite_standard['stab_derivs']
        aero_validity = kite_standard['aero_validity']

    return aero_deriv, aero_validity


######## general induction

def build_induction_options(options, help_options, options_tree, fixed_params, architecture):

    user_options = options['user_options']

    options_tree.append(('model', None, None, 'induction_model', user_options['induction_model'], ('????', None), 'x')),
    options_tree.append(('formulation', 'induction', None, 'induction_model', user_options['induction_model'], ('????', None), 'x')),
    options_tree.append(('nlp', 'induction', None, 'induction_model', user_options['induction_model'], ('????', None), 'x')),
    options_tree.append(('solver', 'initialization', 'model', 'induction_model', user_options['induction_model'], ('????', None), 'x')),

    options_tree.append(('model', 'system_bounds', 'xl', 'n_vec_length', [0., cas.inf],
                         ('normalization factor for normal vector [-]', None), 'x')),
    options_tree.append(('model', 'system_bounds', 'xl', 'z_vec_length', [0.1, 2.],
                         ('normalization factor for normal vector [-]', None), 'x')),

    if options['model']['aero']['actuator']['normal_vector_model'] in ['default','tether_parallel']:
        options_tree.append(('solver', 'initialization', None, 'n_factor', 'tether_length', ('induction factor [-]', None),'x'))
    else:
        options_tree.append(('solver', 'initialization', None, 'n_factor', 'unit_length', ('induction factor [-]', None),'x'))

    allow_azimuth_jumping = options['model']['aero']['actuator']['allow_azimuth_jumping']
    if not allow_azimuth_jumping:
        dpsi_max_rate= -np.pi / 4.
        options_tree.append(('model', 'system_bounds', 'xd', 'dpsi', [-1. * dpsi_max_rate, dpsi_max_rate], ('azimuth-jumping bounds on the azimuthal angle derivative', None), 'x'))

    return options_tree, fixed_params



######## actuator induction

def build_actuator_options(options, options_tree, fixed_params):

    user_options = options['user_options']

    actuator_symmetry = options['model']['aero']['actuator']['symmetry']
    actuator_steadyness = options['model']['aero']['actuator']['steadyness']
    options_tree.append(
        ('solver', 'initialization', 'model', 'actuator_steadyness', actuator_steadyness, ('????', None), 'x')),

    comparison_labels = get_comparison_labels(options, user_options)
    options_tree.append(('model', 'aero', 'induction', 'comparison_labels', comparison_labels, ('????', None), 'x')),
    options_tree.append(('formulation', 'induction', None, 'comparison_labels', comparison_labels, ('????', None), 'x')),
    options_tree.append(('nlp', 'induction', None, 'comparison_labels', comparison_labels, ('????', None), 'x')),
    options_tree.append(('solver', 'initialization', 'model', 'comparison_labels', comparison_labels, ('????', None), 'x')),

    induction_varrho_ref = options['model']['aero']['actuator']['varrho_ref']
    options_tree.append(('solver', 'initialization', 'model', 'induction_varrho_ref', induction_varrho_ref, ('????', None), 'x')),

    options_tree.append(('formulation', 'induction', None, 'steadyness', actuator_steadyness, ('actuator steadyness', None), 'x')),
    options_tree.append(('formulation', 'induction', None, 'symmetry',   actuator_symmetry, ('actuator symmetry', None), 'x')),

    options_tree.append(('nlp', 'induction', None, 'steadyness', actuator_steadyness, ('actuator steadyness', None), 'x')),
    options_tree.append(('nlp', 'induction', None, 'symmetry',   actuator_symmetry, ('actuator symmetry', None), 'x')),

    options_tree.append(('model', 'system_bounds', 'xd', 'dpsi', [-2., 2.], ('forwards-only', None), 'x')),  # no jumps... smoothen out psi

    local_label = get_local_actuator_label(actuator_steadyness, actuator_symmetry)
    options_tree.append(('model', 'system_bounds', 'xl', 'chi_' + local_label, [-np.pi/2., np.pi/2.], ('chi limit', None), 'x')),

    ## actuator-disk induction
    a_ref = options['model']['aero']['actuator']['a_ref']
    a_range = options['model']['aero']['actuator']['a_range']
    if (a_ref > a_range[1]) or (a_ref < a_range[0]):
        a_ref = a_range[1] / 2.

    # if actuator_symmetry == 'asymmetric':
    options_tree.append(('model', 'system_bounds', 'xd', 'local_a', a_range, ('local induction factor', None), 'x')),

    if user_options['induction_model'] == 'not_in_use':
        a_ref = None

    options_tree.append(('model', 'aero', None, 'a_ref', a_ref, ('reference value for the induction factors. takes values between 0. and 0.4', None),'x'))

    a_var_type = 'xd'
    options_tree.append(('solver', 'initialization', a_var_type, 'a', a_ref, ('induction factor [-]', None),'x'))

    options_tree.append(('model', 'system_bounds', 'xl', 'varrho', [0., cas.inf], ('relative radius bounds [-]', None), 'x'))
    options_tree.append(('model', 'system_bounds', 'xl', 'corr', [0., cas.inf], ('square root sign bounds on glauert correction [-]', None), 'x'))

    gamma_range = options['model']['aero']['actuator']['gamma_range']
    options_tree.append(('model', 'system_bounds', 'xl', 'gamma', gamma_range, ('tilt angle bounds [rad]', None), 'x')),

    options_tree.append(('model', 'system_bounds', 'xl', 'g_vec_length', [0.1, 2.],
                         ('normalization factor for normal vector [-]', None), 'x')),
    options_tree.append(('model', 'system_bounds', 'xl', 'u_vec_length', [0.1, cas.inf],
                         ('normalization factor for normal vector [-]', None), 'x')),

    options_tree.append(('model', 'system_bounds', 'xl', 'LLinv', [-100., 100.], ('relative radius bounds [-]', None), 'x'))

    return options_tree, fixed_params


def get_comparison_labels(options, user_options):
    induction_model = user_options['induction_model']
    induction_comparison = options['model']['aero']['induction_comparison']

    if (induction_model[:3] not in induction_comparison) and (not induction_model == 'not_in_use'):
        induction_comparison += [induction_model[:3]]

    comparison_labels = []
    if 'vor' in induction_comparison:
        comparison_labels += ['vor']

    if 'act' in induction_comparison:

        actuator_steadyness = options['model']['aero']['actuator']['steadyness']
        actuator_symmetry = options['model']['aero']['actuator']['symmetry']

        steadyness_comparison = options['model']['aero']['actuator']['steadyness_comparison']
        symmetry_comparison = options['model']['aero']['actuator']['symmetry_comparison']

        if (actuator_steadyness == 'quasi-steady' or actuator_steadyness == 'steady') and 'q' not in steadyness_comparison:
            steadyness_comparison += ['q']
        if actuator_steadyness == 'unsteady' and 'u' not in steadyness_comparison:
            steadyness_comparison += ['u']
        if actuator_symmetry == 'axisymmetric' and 'axi' not in symmetry_comparison:
            symmetry_comparison += ['axi']
        if actuator_symmetry == 'asymmetric' and 'asym' not in symmetry_comparison:
            symmetry_comparison += ['asym']

        for steadyness_label in steadyness_comparison:
            for symmetry_label in symmetry_comparison:
                new_label = 'act_' + steadyness_label + symmetry_label
                comparison_labels += [new_label]

    return comparison_labels

def get_local_actuator_label(actuator_steadyness, actuator_symmetry):
    local_label = ''
    if (actuator_steadyness == 'quasi-steady' or actuator_steadyness == 'steady'):
        if actuator_symmetry == 'axisymmetric':
            local_label = 'qaxi'

        elif actuator_symmetry == 'asymmetric':
            local_label = 'qasym'

    elif actuator_steadyness == 'unsteady':
        if actuator_symmetry == 'axisymmetric':
            local_label = 'uaxi'

        elif actuator_symmetry == 'asymmetric':
            local_label = 'uasym'

    return local_label


###### vortex induction

def build_vortex_options(options, options_tree, fixed_params, architecture):

    n_k = options['nlp']['n_k']
    d = options['nlp']['collocation']['d']
    options_tree.append(('model', 'aero', 'vortex', 'n_k', n_k, ('how many nodes to track over one period: n_k', None), 'x')),
    options_tree.append(('model', 'aero', 'vortex', 'd', d, ('how many nodes to track over one period: d', None), 'x')),

    wake_nodes = options['model']['aero']['vortex']['wake_nodes']
    options_tree.append(('solver', 'initialization', 'model', 'vortex_wake_nodes', wake_nodes, ('????', None), 'x')),
    options_tree.append(('model', 'induction', None, 'vortex_wake_nodes', wake_nodes, ('????', None), 'x')),
    options_tree.append(('formulation', 'induction', None, 'vortex_wake_nodes', wake_nodes, ('????', None), 'x')),
    options_tree.append(('nlp', 'induction', None, 'vortex_wake_nodes', wake_nodes, ('????', None), 'x')),

    u_ref = get_u_ref(options['user_options'])
    vortex_u_ref = u_ref
    options_tree.append(('solver', 'initialization', 'model', 'vortex_u_ref', vortex_u_ref, ('????', None), 'x')),
    options_tree.append(('model', 'induction', None, 'vortex_u_ref', vortex_u_ref, ('????', None), 'x')),
    options_tree.append(('formulation', 'induction', None, 'vortex_u_ref', vortex_u_ref, ('????', None), 'x')),
    options_tree.append(('nlp', 'induction', None, 'vortex_u_ref', vortex_u_ref, ('????', None), 'x')),

    far_convection_time = options['model']['aero']['vortex']['far_convection_time']
    # options_tree.append(('solver', 'initialization', 'model', 'vortex_far_convection_time', far_convection_time, ('????', None), 'x')),
    options_tree.append(('model', 'induction', None, 'vortex_far_convection_time', far_convection_time, ('????', None), 'x')),
    options_tree.append(('formulation', 'induction', None, 'vortex_far_convection_time', far_convection_time, ('????', None), 'x')),
    options_tree.append(('nlp', 'induction', None, 'vortex_far_convection_time', far_convection_time, ('????', None), 'x')),

    vortex_representation = options['model']['aero']['vortex']['representation']
    options_tree.append(('model', 'induction', None, 'vortex_representation', vortex_representation, ('????', None), 'x')),
    options_tree.append(('formulation', 'induction', None, 'vortex_representation', vortex_representation, ('????', None), 'x')),
    options_tree.append(('nlp', 'induction', None, 'vortex_representation', vortex_representation, ('????', None), 'x')),
    options_tree.append(('solver', 'initialization', 'induction', 'vortex_representation', vortex_representation, ('????', None), 'x')),

    geometry = get_geometry(options)
    c_ref = geometry['c_ref']
    r_core = options['model']['aero']['vortex']['core_to_chord_ratio'] * c_ref

    options_tree.append(('model', 'induction', None, 'vortex_core_radius', r_core, ('????', None), 'x')),
    options_tree.append(('formulation', 'induction', None, 'vortex_core_radius', r_core, ('????', None), 'x')),
    options_tree.append(('nlp', 'induction', None, 'vortex_core_radius', r_core, ('????', None), 'x')),

    CL = estimate_CL(options)
    b_ref = geometry['b_ref']

    groundspeed = options['solver']['initialization']['groundspeed']
    airspeed_ref = cas.sqrt(groundspeed**2 + u_ref**2)

    rings = wake_nodes - 1
    filaments = wake_nodes * 3 * len(architecture.kite_nodes)
    wingtips = ['ext', 'int']

    gamma_scale = 0.5 * CL * airspeed_ref * c_ref
    for kite in architecture.kite_nodes:
        for ring in range(rings):
            gamma_name = 'wg_' + str(kite) + '_' + str(ring)
            options_tree.append(('model', 'scaling', 'xl', gamma_name, gamma_scale, ('descript', None), 'x'))
    options_tree.append(('solver', 'initialization', 'induction', 'vortex_gamma_scale', gamma_scale, ('????', None), 'x')),

    u_ref = get_u_ref(options['user_options'])
    vortex_position_scale = 2. * u_ref
    for kite in architecture.kite_nodes:
        for wake_node in range(wake_nodes):
            for tip in wingtips:
                coord_name = 'wx_' + str(kite) + '_' + tip + '_' + str(wake_node)
                options_tree.append(('model', 'scaling', 'xd', coord_name, vortex_position_scale, ('descript', None), 'x'))
                options_tree.append(('model', 'scaling', 'xd', 'd' + coord_name, vortex_position_scale, ('descript', None), 'x'))
                options_tree.append(('model', 'scaling', 'xd', 'dd' + coord_name, vortex_position_scale, ('descript', None), 'x'))

    a_ref = options['model']['aero']['actuator']['a_ref']
    u_ind = a_ref * u_ref

    for kite_obs in architecture.kite_nodes:
        for fdx in range(filaments):
            ind_name = 'wu_fil_' + str(fdx) + '_' + str(kite_obs)
            options_tree.append(('model', 'scaling', 'xl', ind_name, u_ind / float(filaments), ('descript', None), 'x'))

        ind_name = 'wu_ind_' + str(kite_obs)
        options_tree.append(('model', 'scaling', 'xl', ind_name, u_ind, ('descript', None), 'x'))




    return options_tree, fixed_params


####### tether drag

def build_tether_drag_options(options, options_tree, fixed_params):

    tether_drag_descript =  ('model to approximate the tether drag on the tether nodes', ['split', 'single', 'multi', 'not_in_use'])
    options_tree.append(('model', 'tether', 'tether_drag', 'model_type', options['user_options']['tether_drag_model'], tether_drag_descript,'x'))
    options_tree.append(('formulation', None, None, 'tether_drag_model', options['user_options']['tether_drag_model'], tether_drag_descript,'x'))

    return options_tree, fixed_params


###### tether stress

def build_tether_stress_options(options, options_tree, fixed_params, architecture):

    user_options = options['user_options']

    fix_diam_t = None
    fix_diam_s = None
    if 'diam_t' in user_options['trajectory']['fixed_params']:
        fix_diam_t = user_options['trajectory']['fixed_params']['diam_t']
    if 'diam_s' in user_options['trajectory']['fixed_params']:
        fix_diam_s = user_options['trajectory']['fixed_params']['diam_s']

    tether_force_limits = options['params']['model_bounds']['tether_force_limits']
    max_tether_force = tether_force_limits[1]

    max_stress = options['params']['tether']['max_stress']
    stress_safety_factor = options['params']['tether']['stress_safety_factor']
    max_tether_stress = max_stress / stress_safety_factor

    # map single tether power interval constraint to min and max constraint
    if options['model']['model_bounds']['tether_force']['include'] == True:
        options_tree.append(('model', 'model_bounds', 'tether_force_max', 'include', True, None,'x'))
        options_tree.append(('model', 'model_bounds', 'tether_force_min', 'include', True, None,'x'))
        tether_force_include = True
    else:
        tether_force_include = False

    tether_stress_include = options['model']['model_bounds']['tether_stress']['include']

    # check which tether force/stress constraints to enforce on which node
    tether_constraint_includes = {'force': [], 'stress': []}

    if tether_force_include and tether_stress_include:

        for node in range(1, architecture.number_of_nodes):
            if node in architecture.kite_nodes:

                if node == 1:
                    fix_diam = fix_diam_t
                else:
                    fix_diam = fix_diam_s

                diameter_is_fixed = not (fix_diam == None)
                if diameter_is_fixed:
                    awelogger.logger.warning(
                        'Both tether force and stress constraints are enabled, while tether diameter is restricted ' + \
                        'for tether segment with upper node ' + str(node) + '. To avoid LICQ violations, tightest bound is selected.')

                    cross_section = np.pi * (fix_diam / 2.)**2.
                    force_equivalent_to_stress = max_tether_stress * cross_section
                    if force_equivalent_to_stress <= max_tether_force:
                        tether_constraint_includes['stress'] += [node]
                    else:
                        tether_constraint_includes['force'] += [node]

                else:
                    tether_constraint_includes['stress'] += [node]
                    tether_constraint_includes['force'] += [node]

            else:
                tether_constraint_includes['stress'] += [node]


    elif tether_force_include:
        tether_constraint_includes['force'] = architecture.kite_nodes

    elif tether_stress_include:
        tether_constraint_includes['stress'] = range(1, architecture.number_of_nodes)

    options_tree.append(('model', 'model_bounds', 'tether', 'tether_constraint_includes', tether_constraint_includes, ('logic deciding which tether constraints to enforce', None), 'x'))

    return options_tree, fixed_params

def build_wound_tether_length_options(options, options_tree, fixed_params):
    system_type = options['user_options']['trajectory']['system_type']
    if system_type == 'drag_mode':
        options['model']['tether']['use_wound_tether'] = False
        options['params']['ground_station']['m_gen'] = 0.

    use_wound_tether = options['model']['tether']['use_wound_tether']

    if use_wound_tether:
        l_t_bounds = options['model']['system_bounds']['xd']['l_t']

        # prevent licq errors.
        options_tree.append(('model', 'system_bounds', 'xd', 'l_t', [l_t_bounds[0], cas.inf], ('???', None), 'x'))

        l_t_scaling = np.max([options['model']['scaling']['xd']['l_t'], l_t_bounds[0]])
        options_tree.append(('model', 'scaling', 'theta', 'l_t_full', l_t_scaling,
                             ('length of the main tether when unrolled [m]', None), 'x'))
        options_tree.append(('model', 'system_bounds', 'theta', 'l_t_full', l_t_bounds, ('length of the unrolled main tether bounds [m]', None), 'x'))
        options_tree.append(('solver', 'initialization', 'theta', 'l_t_full', l_t_scaling, ('length of the main tether when unrolled [m]', None), 'x'))

    if not use_wound_tether:
        options['model']['model_bounds']['wound_tether_length']['include'] = False

    # note: lagrangian dynamics is not yet set up for scaled q's
    q_scaling = 1.
    options_tree.append(('model', 'scaling', 'xd', 'q', q_scaling, ('descript', None), 'x'))

    options_tree.append(('model', 'scaling', 'theta', 't_f', cas.DM(1.0), ('descript', None), 'x'))


    return options_tree, fixed_params


######## tether control

def build_tether_control_options(options, options_tree, fixed_params):

    user_options = options['user_options']
    in_drag_mode_operation = user_options['trajectory']['system_type'] == 'drag_mode'

    ddl_t_max = options['model']['ground_station']['ddl_t_max']
    dddl_t_max = options['model']['ground_station']['dddl_t_max']

    control_name = options['model']['tether']['control_var']

    if in_drag_mode_operation:
        options_tree.append(('model', 'system_bounds', 'u', control_name, [0.0, 0.0], ('main tether reel-out acceleration', None), 'x'))

    else:
        if control_name == 'ddl_t':
            options_tree.append(('model', 'system_bounds', 'u', 'ddl_t', [-1. * ddl_t_max, ddl_t_max],   ('main tether max acceleration [m/s^2]', None),'x'))

        elif control_name == 'dddl_t':
            options_tree.append(('model', 'system_bounds', 'xd', 'ddl_t', [-1. * ddl_t_max, ddl_t_max],   ('main tether max acceleration [m/s^2]', None),'x'))
            options_tree.append(('model', 'system_bounds', 'u', 'dddl_t', [-1. * dddl_t_max, dddl_t_max],   ('main tether max jerk [m/s^3]', None),'x'))
        else:
            raise ValueError('invalid tether control variable chosen')

    return options_tree, fixed_params

######## wind

def build_wind_options(options, options_tree, fixed_params):

    u_ref = get_u_ref(options['user_options'])
    options_tree.append(('model', 'wind', None, 'model', options['user_options']['wind']['model'],('wind model', None),'x'))
    options_tree.append(('params', 'wind', None, 'u_ref', u_ref, ('reference wind speed [m/s]', None),'x'))
    options_tree.append(('model', 'wind', None, 'u_ref', u_ref, ('reference wind speed [m/s]', None),'x'))
    options_tree.append(('model', 'wind', None, 'atmosphere_heightsdata', options['user_options']['wind']['atmosphere_heightsdata'],('data for the heights at this time instant', None),'x'))
    options_tree.append(('model', 'wind', None, 'atmosphere_featuresdata', options['user_options']['wind']['atmosphere_featuresdata'],('data for the features at this time instant', None),'x'))

    z_ref = options['params']['wind']['z_ref']
    z0_air = options['params']['wind']['log_wind']['z0_air']
    exp_ref = options['params']['wind']['power_wind']['exp_ref']
    options_tree.append(('model', 'wind', None, 'z_ref', z_ref, ('?????', None), 'x'))
    options_tree.append(('model', 'wind', 'log_wind', 'z0_air', z0_air, ('?????', None), 'x'))
    options_tree.append(('model', 'wind', 'power_wind', 'exp_ref', exp_ref, ('?????', None), 'x'))

    options_tree.append(('solver', 'initialization', 'model', 'wind_u_ref', u_ref, ('reference wind speed [m/s]', None),'x'))
    options_tree.append(('solver', 'initialization', 'model', 'wind_model', options['user_options']['wind']['model'], ('???', None), 'x'))
    options_tree.append(('solver', 'initialization', 'model', 'wind_z_ref', options['params']['wind']['z_ref'],
         ('?????', None), 'x'))
    options_tree.append(('solver', 'initialization', 'model', 'wind_z0_air', options['params']['wind']['log_wind']['z0_air'],
                         ('?????', None), 'x'))
    options_tree.append(('solver', 'initialization', 'model', 'wind_exp_ref', options['params']['wind']['power_wind']['exp_ref'],
                         ('?????', None), 'x'))

    return options_tree, fixed_params

def get_u_ref(user_options):

    u_ref = user_options['wind']['u_ref']

    return u_ref

def get_u_at_altitude(options, zz):

    model = options['user_options']['wind']['model']
    u_ref = get_u_ref(options['user_options'])
    z_ref = options['params']['wind']['z_ref']
    z0_air = options['params']['wind']['log_wind']['z0_air']
    exp_ref = options['params']['wind']['power_wind']['exp_ref']
    u = wind.get_speed(model, u_ref, z_ref, z0_air, exp_ref, zz)

    return u

######## atmosphere

def build_atmosphere_options(options, options_tree, fixed_params):

    options_tree.append(('model',  'atmosphere', None, 'model', options['user_options']['atmosphere'], ('atmosphere model', None),'x'))
    q_ref = get_q_ref(options)
    options_tree.append(('params',  'atmosphere', None, 'q_ref', q_ref, ('aerodynamic dynamic pressure [Pa]', None),'x'))

    return options_tree, fixed_params

def get_q_ref(options):

    u_ref = get_u_ref(options['user_options'])
    q_ref = 0.5*options['params']['atmosphere']['rho_ref'] * u_ref**2

    return q_ref

def get_q_at_altitude(options, zz):

    u = get_u_at_altitude(options, zz)
    q = 0.5 * options['params']['atmosphere']['rho_ref'] * u ** 2

    return q


####### scaling

def build_fict_scaling_options(options, options_tree, fixed_params):

    gravity = get_gravity_ref(options)
    geometry = get_geometry(options)
    m_k = geometry['m_k']
    b_ref = geometry['b_ref']

    acc_max = options['model']['model_bounds']['acceleration']['acc_max']

    f_scaling = m_k * gravity * acc_max
    m_scaling = m_k * gravity * b_ref / 2.
    options_tree.append(('model', 'scaling', 'u', 'f_fict', f_scaling, ('scaling of fictitious homotopy forces', None),'x'))
    options_tree.append(('model', 'scaling', 'u', 'm_fict', m_scaling, ('scaling of fictitious homotopy moments', None),'x'))

    options_tree.append(('model', 'scaling', 'xl', 'f_aero', f_scaling, ('scaling of aerodynamic forces', None),'x'))
    options_tree.append(('model', 'scaling', 'xl', 'm_aero', m_scaling, ('scaling of aerodynamic forces', None),'x'))

    # q_ref = get_q_ref(options)
    # l_t_scaling = options['model']['scaling']['xd']['l_t']
    # diam_t_scaling = options['model']['scaling']['theta']['diam_t']
    # cd = 1.
    # sin_loss = np.sin(options['solver']['initialization']['inclination_deg'] * np.pi / 180.)
    #
    # f_tether_scaling = cd * q_ref * l_t_scaling * diam_t_scaling * sin_loss
    options_tree.append(('model', 'scaling', 'xl', 'f_tether', f_scaling, ('scaling of tether drag forces', None),'x'))

    return options_tree, fixed_params

def get_gravity_ref(options):

    gravity = options['model']['scaling']['other']['g']

    return gravity



####### lambda, energy, power scaling

def build_lambda_e_power_scaling(options, options_tree, fixed_params, architecture):

    lambda_scaling, energy_scaling, power_cost = get_suggested_lambda_energy_power_scaling(options, architecture)

    if options['model']['scaling_overwrite']['lambda_tree']['include']:
        options_tree = generate_lambda_scaling_tree(options= options, options_tree= options_tree, lambda_scaling= lambda_scaling, architecture = architecture)
    else:
        options_tree.append(('model', 'scaling', 'xa', 'lambda', lambda_scaling, ('scaling of tether tension per length', None),'x'))

    options_tree.append(('model', 'scaling', 'xd', 'e', energy_scaling, ('scaling of the energy', None),'x'))

    options_tree.append(('solver', 'cost', 'power', 1, power_cost, ('update cost for power', None),'x'))

    return options_tree, fixed_params

def generate_lambda_scaling_tree(options, options_tree, lambda_scaling, architecture):

    description = ('scaling of tether tension per length', None)

    # set lambda_scaling
    options_tree.append(('model', 'scaling', 'xa', 'lambda10', lambda_scaling, description,'x'))

    # extract architecure options
    layers = architecture.layers

    # extract length scaling information
    l_s_scaling = options['model']['scaling']['theta']['l_s']
    l_t_scaling = options['model']['scaling']['xd']['l_t']
    l_i_scaling = options['model']['scaling']['theta']['l_i']

    #  secondary tether scaling
    tension_main = lambda_scaling * l_t_scaling
    tension_secondary = tension_main / architecture.number_of_kites
    lambda_s_scaling = tension_secondary / l_s_scaling

    # tension in the intermediate tethers is not constant
    lambda_i_max = tension_main / l_i_scaling

    # assign scaling according to tree structure
    layer_count = 1
    for node in range(2,architecture.number_of_nodes):
        label = 'lambda'+str(node)+str(architecture.parent_map[node])

        if node in architecture.kite_nodes:
            options_tree.append(('model', 'scaling', 'xa', label, lambda_s_scaling, description,'x'))

        else:
            # if there are no kites here, we must be at an intermediate, layer node

            # the tension should decrease as we move to higher layers, because there are fewer kites pulling on the nodes
            linear_factor = (layers - layer_count) / (float(layers))
            lambda_i_scaling = linear_factor * lambda_i_max
            options_tree.append(('model', 'scaling', 'xa', label, lambda_i_scaling, description,'x'))
            layer_count += 1

    return options_tree


def get_suggested_lambda_energy_power_scaling(options, architecture):

    if options['user_options']['trajectory']['type'] == 'nominal_landing':
        power_cost = 1e-4
        lambda_scaling = 1
        energy_scaling = 1e5
    else:

        # this will scale the multiplier on the main tether, from 'si'
        lam = estimate_tether_lambda(options, architecture)
        lambda_factor = options['model']['scaling_overwrite']['lambda_factor']
        lambda_scaling = lambda_factor * lam

        # this will scale the energy 'si'. see dynamics.make_dynamics
        energy = estimate_energy(options, architecture)
        energy_factor = options['model']['scaling_overwrite']['energy_factor']
        energy_scaling = energy_factor * energy

        # this will be used to weight the scaled power (energy / time) cost
        # so: for clarity of what is physically happening, I've written this in terms of the
        # power and energy scaling values.
        # but, what's actually happening depends ONLY on the tuning factor and on the estimated time period.
        # so, if this scaling leads to bad convergence in final solution step of homotopy, then check the
        # estimate time period function (below) FIRST.
        power = estimate_power(options, architecture)
        scaled_power = power / energy_scaling # yes, this = (1 / time_period_estimate)
        power_cost_factor = options['solver']['cost_factor']['power']
        power_cost = power_cost_factor * (1. / scaled_power)  # yes, this = pcf * time_period_estimate

    return lambda_scaling, energy_scaling, power_cost


def estimate_power(options, architecture):

    zz = estimate_alitude(options)
    uu = get_u_at_altitude(options, zz)
    qq = get_q_at_altitude(options, zz)
    power_density = uu * qq

    geometry = get_geometry(options)
    s_ref = geometry['s_ref']

    elevation_angle = options['solver']['initialization']['inclination_deg'] * np.pi / 180.

    CL = estimate_CL(options)
    CD = estimate_CD(options)
    p_loyd = perf_op.get_loyd_power(power_density, CL, CD, s_ref, elevation_angle)

    induction_model = options['user_options']['induction_model']
    if induction_model == 'not_in_use':
        induction_efficiency = 1.
    else:
        induction_efficiency = 0.5

    number_of_kites = architecture.number_of_kites

    estimate_1 = number_of_kites * p_loyd * induction_efficiency

    power = estimate_1

    return power

def estimate_reelout_speed(options):
    zz = estimate_alitude(options)
    uu = get_u_at_altitude(options, zz)
    loyd_factor = 1. / 3.
    reelout_speed = loyd_factor * uu

    return reelout_speed

def estimate_CL(options):

    kite_standard = options['user_options']['kite_standard']
    aero_deriv, aero_validity = load_stability_derivatives(kite_standard)

    alpha = aero_validity['alpha_max_deg'] * np.pi / 180.
    cos = cas.cos(alpha)
    sin = cas.sin(alpha)

    if 'CL' in aero_deriv.keys():
        CL = aero_deriv['CL']['0'][0] + aero_deriv['CL']['alpha'][0] * alpha
    elif 'CZ' in aero_deriv.keys():
        CX = aero_deriv['CX']['0'][0] + aero_deriv['CX']['alpha'][0] * alpha
        CZ = aero_deriv['CZ']['0'][0] + aero_deriv['CZ']['alpha'][0] * alpha
        xhat = cas.vertcat(-1. * cos, sin)
        zhat = cas.vertcat(-1. * sin, -1. * cos)
        rot = CX * xhat + CZ * zhat
        CL = rot[1]
    elif 'CN' in aero_deriv.keys():
        CA = aero_deriv['CA']['0'][0] + aero_deriv['CA']['alpha'][0] * alpha
        CN = aero_deriv['CN']['0'][0] + aero_deriv['CN']['alpha'][0] * alpha
        ahat = cas.vertcat(cos, -1. * sin)
        nhat = cas.vertcat(sin, cos)
        rot = CA * ahat + CN * nhat
        CL = rot[1]

    return CL

def estimate_CD(options):

    kite_standard = options['user_options']['kite_standard']
    aero_deriv, aero_validity = load_stability_derivatives(kite_standard)

    alpha = aero_validity['alpha_max_deg'] * np.pi / 180.
    cos = cas.cos(alpha)
    sin = cas.sin(alpha)

    if 'CD' in aero_deriv.keys():
        CD = aero_deriv['CD']['0'][0] + aero_deriv['CD']['alpha'][0] * alpha
    elif 'CZ' in aero_deriv.keys():
        CX = aero_deriv['CX']['0'][0] + aero_deriv['CX']['alpha'][0] * alpha
        CZ = aero_deriv['CZ']['0'][0] + aero_deriv['CZ']['alpha'][0] * alpha
        xhat = cas.vertcat(-1. * cos, sin)
        zhat = cas.vertcat(-1. * sin, -1. * cos)
        rot = CX * xhat + CZ * zhat
        CD = rot[0]
    elif 'CN' in aero_deriv.keys():
        CA = aero_deriv['CA']['0'][0] + aero_deriv['CA']['alpha'][0] * alpha
        CN = aero_deriv['CN']['0'][0] + aero_deriv['CN']['alpha'][0] * alpha
        ahat = cas.vertcat(cos, -1. * sin)
        nhat = cas.vertcat(sin, cos)
        rot = CA * ahat + CN * nhat
        CD = rot[0]
    return CD


def estimate_alitude(options):
    elevation_angle = options['solver']['initialization']['inclination_deg'] * np.pi / 180.
    length = options['model']['scaling']['xd']['l_t']
    zz = length * np.sin(elevation_angle)

    return zz

def estimate_tether_lambda(options, architecture):

    power = estimate_power(options, architecture)

    reelout_speed = estimate_reelout_speed(options)

    length = options['model']['scaling']['xd']['l_t']

    tension = power / reelout_speed
    multiplier = tension / length

    return multiplier

def estimate_energy(options, architecture):

    power = estimate_power(options, architecture)

    time_period = estimate_time_period(options, architecture)
    energy = power * time_period

    energy = energy

    return energy

def estimate_time_period(options, architecture):

    windings = float(options['user_options']['trajectory']['lift_mode']['windings'])
    winding_period = float(options['solver']['initialization']['winding_period'])

    estimate_1 = windings * winding_period

    number_of_kites = architecture.number_of_kites
    if number_of_kites == 1:
        cone_angle = options['solver']['initialization']['max_cone_angle_single'] * np.pi / 180.
        length = options['model']['scaling']['xd']['l_t']
    else:
        cone_angle = options['solver']['initialization']['max_cone_angle_multi'] * np.pi / 180.
        length = options['model']['scaling']['theta']['l_s']
    radius = length * np.sin(cone_angle)
    acc_max = options['model']['model_bounds']['acceleration']['acc_max'] * options['model']['scaling']['other']['g']

    estimate_2 = (2. * np.pi * windings) / np.sqrt( acc_max / radius)

    time_period = (estimate_1 + estimate_2) / 2.

    return time_period