#
#    This file is part of awebox.
#
#    awebox -- A modeling and optimization framework for multi-kite AWE systems.
#    Copyright (C) 2017-2021 Jochem De Schutter, Rachel Leuthold, Moritz Diehl,
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
import pdb

import numpy as np
import awebox as awe
import casadi as cas
import copy
import pickle
from awebox.logger.logger import Logger as awelogger

import awebox.tools.struct_operations as struct_op
import awebox.tools.performance_operations as perf_op
import awebox.tools.print_operations as print_op
import awebox.tools.vector_operations as vect_op

import awebox.mdl.aero.induction_dir.actuator_dir.flow as actuator_flow
import awebox.mdl.aero.induction_dir.vortex_dir.alg_repr_dir.scaling as vortex_alg_repr_scaling

import awebox.mdl.wind as wind


def build_model_options(options, help_options, user_options, options_tree, fixed_params, architecture):

    # kite
    options_tree, fixed_params = build_geometry_options(options, help_options, options_tree, fixed_params)
    options_tree, fixed_params = build_kite_dof_options(options, options_tree, fixed_params, architecture)

    options_tree, fixed_params = build_scaling_options(options, options_tree, fixed_params, architecture)

    # problem specifics
    options_tree, fixed_params = build_constraint_applicablity_options(options, options_tree, fixed_params, architecture)
    options_tree, fixed_params = build_trajectory_options(options, options_tree, fixed_params, architecture)
    options_tree, fixed_params = build_integral_options(options, options_tree, fixed_params)

    # aerodynamics
    options_tree, fixed_params = build_stability_derivative_options(options, help_options, options_tree, fixed_params)
    options, options_tree, fixed_params = build_induction_options(options, help_options, options_tree, fixed_params, architecture)
    options_tree, fixed_params = build_actuator_options(options, options_tree, fixed_params, architecture)
    options_tree, fixed_params = build_vortex_options(options, options_tree, fixed_params, architecture)

    # tether
    options_tree, fixed_params = build_tether_drag_options(options, options_tree, fixed_params)
    options_tree, fixed_params = build_tether_stress_options(options, options_tree, fixed_params, architecture)
    options_tree, fixed_params = build_tether_control_options(options, options_tree, fixed_params)

    # environment
    options_tree, fixed_params = build_wind_options(options, options_tree, fixed_params)
    options_tree, fixed_params = build_atmosphere_options(options, options_tree, fixed_params)

    # scaling
    options_tree, fixed_params = build_fict_scaling_options(options, options_tree, fixed_params, architecture)
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

    # basic_params = ['s_ref', 'b_ref', 'c_ref', 'ar']
    # dependent_params = ['s_ref', 'b_ref', 'c_ref', 'ar', 'm_k', 'j', 'c_root', 'c_tip', 'length', 'height']

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
            pass
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


def get_position_scaling(options, architecture):

    position = estimate_position_of_main_tether_end(options)
    flight_radius = estimate_flight_radius(options, architecture)
    geometry = get_geometry(options)
    b_ref = geometry['b_ref']

    position_scaling_method = options['model']['scaling']['other']['position_scaling_method']
    if position_scaling_method == 'radius':
        q_scaling = flight_radius * cas.DM.ones((3, 1))
    elif position_scaling_method == 'altitude':
        q_scaling = position[2] * cas.DM.ones((3, 1))
    elif position_scaling_method == 'b_ref':
        q_scaling = b_ref * cas.DM.ones((3, 1))
    elif position_scaling_method == 'radius_and_tether':
        q_scaling = cas.vertcat(position[0], flight_radius, flight_radius)
    elif ('radius' in position_scaling_method) and ('altitude' in position_scaling_method):
        q_scaling = cas.vertcat(position[0], flight_radius, position[2])
    else:
        message = 'unexpected position scaling source (' + position_scaling_method + ')'
        print_op.log_and_raise_error(message)

    return q_scaling


def build_scaling_options(options, options_tree, fixed_params, architecture):

    length = options['solver']['initialization']['l_t']
    length_scaling = length
    options_tree.append(('model', 'scaling', 'x', 'l_t', length_scaling, ('???', None), 'x'))
    options_tree.append(('model', 'scaling', 'theta', 'l_t', length_scaling, ('???', None), 'x'))

    q_scaling = get_position_scaling(options, architecture)
    options_tree.append(('model', 'scaling', 'x', 'q', q_scaling, ('???', None),'x'))

    u_altitude = get_u_at_altitude(options, estimate_altitude(options))
    dq_scaling = u_altitude
    options_tree.append(('model', 'scaling', 'x', 'dq', dq_scaling, ('???', None), 'x'))

    dl_t_scaling = u_altitude
    options_tree.append(('model', 'scaling', 'x', 'dl_t', dl_t_scaling, ('???', None), 'x'))

    kappa_scaling = options['model']['scaling']['x']['kappa']
    options_tree.append(('model', 'scaling', 'u', 'dkappa', kappa_scaling, ('???', None), 'x'))

    initialization_theta = options['solver']['initialization']['theta']
    for param in initialization_theta.keys():
        options_tree.append(('model', 'scaling', 'theta', param, options['solver']['initialization']['theta'][param], ('descript', None), 'x'))
    options_tree.append(('model', 'scaling', 'theta', 't_f', cas.DM(1.0), ('descript', None), 'x'))

    return options_tree, fixed_params

##### kite dof

def build_kite_dof_options(options, options_tree, fixed_params, architecture):

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

        t_f_guess = estimate_time_period(options, architecture)
        windings = options['user_options']['trajectory']['lift_mode']['windings']
        omega_guess = 2. * np.pi / (t_f_guess / float(windings))

        options_tree.append(('model', 'system_bounds', 'x', 'delta', [-1. * delta_max, delta_max], ('control surface deflection bounds', None),'x'))
        options_tree.append(('model', 'system_bounds', 'u', 'ddelta', [-1. * ddelta_max, ddelta_max],
                             ('control surface deflection rate bounds', None),'x'))

        options_tree.append(('model', 'scaling', 'x', 'delta', cas.DM(delta_max)/2., ('???', None), 'x'))
        options_tree.append(('model', 'scaling', 'u', 'ddelta', cas.DM(ddelta_max)/2., ('???', None), 'x'))
        options_tree.append(('model', 'scaling', 'x', 'omega', omega_guess, ('???', None), 'x'))
        options_tree.append(('model', 'scaling', 'x', 'r', cas.DM.ones((9, 1)), ('descript', None), 'x'))

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

        coeff_max = cas.DM(options['model']['system_bounds']['x']['coeff'][1])
        coeff_scaling = coeff_max
        options_tree.append(('model', 'scaling', 'x', 'coeff', coeff_scaling, ('???', None), 'x'))

        dcoeff_max = cas.DM(options['model']['system_bounds']['u']['dcoeff'][1])
        dcoeff_scaling = dcoeff_max
        options_tree.append(('model', 'scaling', 'u', 'dcoeff', dcoeff_scaling, ('???', None), 'x'))

        options_tree.append(('model', 'model_bounds', 'aero_validity', 'include', False,
                             ('do not include aero validity for roll control', None), 'x'))

        compromised_factor = options['model']['aero']['three_dof']['dcoeff_compromised_factor']
        dcoeff_compromised_max = np.array([5 * compromised_factor, 5])

        options_tree.append(('params', 'model_bounds', None, 'dcoeff_compromised_max', dcoeff_compromised_max, ('????', None), 'x'))
        options_tree.append(('params', 'model_bounds', None, 'dcoeff_compromised_min', -1. * dcoeff_compromised_max, ('?????', None), 'x'))

    else:
        options_tree.append(('model', 'model_bounds', 'coeff_actuation', 'include', False, ('???', None), 'x'))
        options_tree.append(('model', 'model_bounds', 'dcoeff_actuation', 'include', False, ('???', None), 'x'))

    groundspeed = options['solver']['initialization']['groundspeed']
    options_tree.append(('model', 'model_bounds', 'anticollision_radius', 'num_ref', groundspeed ** 2., ('an estimate of the square of the kite speed, for normalization of the anticollision inequality', None),'x'))

    include_acceleration_constraint = options['model']['model_bounds']['acceleration']['include']
    options_tree.append(('solver', 'initialization', None, 'include_acceleration_constraint', include_acceleration_constraint, ('??', None), 'x'))

    u_altitude = get_u_at_altitude(options, estimate_altitude(options))
    pythagorean_speed = (groundspeed ** 2. + u_altitude ** 2.) ** 0.5
    airspeed_ref = pythagorean_speed

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

    integration_method = options['model']['integration']['method']

    if integration_method not in ['integral_outputs', 'constraints']:
        message = 'unexpected model integration method specified (' + integration_method + ')'
        print_op.log_and_raise_error(message)

    use_integral_outputs = (integration_method == 'integral_outputs')

    options_tree.append(('nlp', 'cost', None, 'output_quadrature', use_integral_outputs, ('use quadrature for integral system outputs in cost function', (True, False)), 't'))
    options_tree.append(('model', None, None, 'integral_outputs', use_integral_outputs, ('do not include integral outputs as system states',[True,False]),'x'))

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

    options_tree.append(
        ('solver', 'initialization', 'induction', 'dynamic_pressure', get_q_at_altitude(options, estimate_altitude(options)), ('????', None), 'x')),

    options_tree.append(('model', 'system_bounds', 'z', 'n_vec_length', [0., cas.inf], ('positive-direction parallel for actuator orientation [-]', None), 'x')),
    options_tree.append(('model', 'system_bounds', 'z', 'u_vec_length', [0., cas.inf], ('positive-direction parallel for actuator orientation [-]', None), 'x')),
    options_tree.append(('model', 'system_bounds', 'z', 'z_vec_length', [0., cas.inf], ('positive-direction parallel for actuator orientation [-]', None), 'x')),
    options_tree.append(('model', 'system_bounds', 'z', 'g_vec_length', [0., cas.inf], ('positive-direction parallel for actuator orientation [-]', None), 'x')),

    options_tree.append(('model', 'scaling', 'z', 'act_dcm', 1., ('descript', None), 'x'))
    options_tree.append(('model', 'scaling', 'z', 'wind_dcm', 1., ('descript', None), 'x'))

    u_vec_length_ref = get_u_at_altitude(options, estimate_altitude(options))
    options_tree.append(('model', 'scaling', 'z', 'u_vec_length', u_vec_length_ref, ('descript', None), 'x'))

    if options['model']['aero']['actuator']['support_only']:
        if (user_options['induction_model'] == 'actuator') and (not options['model']['aero']['induction']['force_zero']):
            message = 'model.aero.actuator.support_only is true, while the actuator induction model is selected.' \
                      ' this implies that model.aero.induction.force_zero must also be true.' \
                      ' proceeding with force_zero option reset to true.'
            print_op.base_print(message, level='warning')
            options['model']['aero']['induction']['force_zero'] = True

    normal_vector_model = options['model']['aero']['actuator']['normal_vector_model']
    number_of_kites = architecture.number_of_kites
    if normal_vector_model == 'least_squares':
        length = options['solver']['initialization']['theta']['l_s']
        n_vec_length_ref = length**2.
    elif normal_vector_model == 'binormal':
        length = options['solver']['initialization']['l_t']
        n_vec_length_ref = number_of_kites * length**2.
    elif normal_vector_model == 'tether_parallel':
        n_vec_length_ref = 1.
    else:  # normal_vector_model == 'xhat':
        n_vec_length_ref = 1.
    options_tree.append(('model', 'scaling', 'z', 'n_vec_length', n_vec_length_ref, ('descript', None), 'x'))
    options_tree.append(
        ('solver', 'initialization', 'induction', 'n_vec_length', n_vec_length_ref, ('descript', None), 'x'))

    options_tree.append(
        ('solver', 'initialization', 'induction', 'normal_vector_model', normal_vector_model, ('descript', None), 'x'))


    g_vec_length_ref = get_u_ref(user_options)
    options_tree.append(('model', 'scaling', 'z', 'g_vec_length', g_vec_length_ref, ('descript', None), 'x'))
    options_tree.append(
        ('solver', 'initialization', 'induction', 'g_vec_length', g_vec_length_ref, ('descript', None), 'x'))

    options_tree.append(('model', 'scaling', 'z', 'z_vec_length', 1., ('descript', None), 'x'))
    options_tree.append(
        ('solver', 'initialization', 'induction', 'z_vec_length', 1., ('descript', None), 'x'))

    psi_scale = 2. * np.pi
    options_tree.append(('model', 'scaling', 'z', 'psi', psi_scale, ('descript', None), 'x'))
    options_tree.append(('model', 'scaling', 'z', 'cospsi', 0.5, ('descript', None), 'x'))
    options_tree.append(('model', 'scaling', 'z', 'sinpsi', 0.5, ('descript', None), 'x'))

    psi_epsilon = np.pi
    options_tree.append(('model', 'system_bounds', 'z', 'psi', [0. - psi_epsilon, 2. * np.pi + psi_epsilon], ('azimuth-jumping bounds on the azimuthal angle derivative', None), 'x'))

    if options['model']['aero']['actuator']['geometry_overwrite'] is not None:
        geometry_type = options['model']['aero']['actuator']['geometry_overwrite']
    elif architecture.number_of_kites > 1:
        geometry_type = 'averaged'
    elif (architecture.number_of_kites == 1) and (architecture.parent_map[architecture.kite_nodes[0]] == 0):
        geometry_type = 'frenet'
    else:
        geometry_type = 'parent'

    options_tree.append(('model', 'aero', None, 'geometry_type', geometry_type, ('descript', None), 'x'))

    return options, options_tree, fixed_params



######## actuator induction

def build_actuator_options(options, options_tree, fixed_params, architecture):

    # todo: ensure that system bounds don't get enforced when actuator is only comparison against vortex model
    if 'actuator' in options['user_options']['induction_model']:
        message = 'current problem tunings may not be optimally set for actuator-model induction problems. the fix is currently in progress! please stay tuned for the update!'
        print_op.base_print(message, level='warning')

    user_options = options['user_options']

    actuator_symmetry = options['model']['aero']['actuator']['symmetry']
    actuator_steadyness = options['model']['aero']['actuator']['steadyness']
    options_tree.append(
        ('solver', 'initialization', 'model', 'actuator_steadyness', actuator_steadyness, ('????', None), 'x')),
    options_tree.append(('model', 'induction', None, 'steadyness', actuator_steadyness, ('actuator steadyness', None), 'x')),
    options_tree.append(('model', 'induction', None, 'symmetry',   actuator_symmetry, ('actuator symmetry', None), 'x')),

    comparison_labels = get_comparison_labels(options, user_options)
    options_tree.append(('model', 'aero', 'induction', 'comparison_labels', comparison_labels, ('????', None), 'x')),
    options_tree.append(('formulation', 'induction', None, 'comparison_labels', comparison_labels, ('????', None), 'x')),
    options_tree.append(('nlp', 'induction', None, 'comparison_labels', comparison_labels, ('????', None), 'x')),
    options_tree.append(('solver', 'initialization', 'induction', 'comparison_labels', comparison_labels, ('????', None), 'x')),

    flight_radius = estimate_flight_radius(options, architecture)
    geometry = get_geometry(options)
    b_ref = geometry['b_ref']
    induction_varrho_ref = flight_radius / b_ref
    options_tree.append(('model', 'aero', 'actuator', 'varrho_ref', induction_varrho_ref, ('descript', None), 'x'))
    options_tree.append(('model', 'scaling', 'z', 'varrho', induction_varrho_ref, ('descript', None), 'x'))
    options_tree.append(('model', 'scaling', 'z', 'bar_varrho', induction_varrho_ref, ('descript', None), 'x'))
    options_tree.append(('model', 'system_bounds', 'z', 'varrho', [0., cas.inf], ('relative radius bounds [-]', None), 'x'))
    options_tree.append(('model', 'scaling', 'z', 'area', 2. * np.pi * flight_radius * b_ref, ('descript', None), 'x'))

    act_q = estimate_altitude(options)
    act_dq = estimate_reelout_speed(options)
    options_tree.append(('model', 'scaling', 'z', 'act_q', act_q, ('descript', None), 'x'))
    options_tree.append(('model', 'scaling', 'z', 'act_dq', act_dq, ('descript', None), 'x'))

    options_tree.append(('formulation', 'induction', None, 'steadyness', actuator_steadyness, ('actuator steadyness', None), 'x')),
    options_tree.append(('formulation', 'induction', None, 'symmetry',   actuator_symmetry, ('actuator symmetry', None), 'x')),

    options_tree.append(('nlp', 'induction', None, 'steadyness', actuator_steadyness, ('actuator steadyness', None), 'x')),
    options_tree.append(('nlp', 'induction', None, 'symmetry',   actuator_symmetry, ('actuator symmetry', None), 'x')),

    ## actuator-disk induction
    a_ref = options['model']['aero']['actuator']['a_ref']
    a_range = options['model']['aero']['actuator']['a_range']
    a_fourier_range = options['model']['aero']['actuator']['a_fourier_range']
    if (a_ref < a_range[0]) or (a_ref > a_range[1]):
        a_ref_new = a_range[1] / 2.
        message = 'reference induction factor (' + str(a_ref) + ') is outside of the allowed range of ' + str(a_range) + '. proceeding with reference value of ' + str(a_ref_new)
        awelogger.logger.warning(message)
        a_ref = a_ref_new

    a_labels_dict = {'qaxi': 'z', 'qasym': 'z', 'uaxi': 'x', 'uasym' : 'x'}
    for label in a_labels_dict.keys():
        for a_name in ['a', 'acos', 'asin']:
            options_tree.append(('model', 'scaling', a_labels_dict[label], a_name + '_' + label, a_ref, ('descript', None), 'x'))
    options_tree.append(('model', 'scaling', 'z', 'local_a', a_ref, ('???', None), 'x')),
    options_tree.append(('solver', 'initialization', 'z', 'a', a_ref, ('???', None), 'x')),

    local_label = actuator_flow.get_label({'induction': {'steadyness': actuator_steadyness, 'symmetry': actuator_symmetry}})
    options_tree.append(('model', 'system_bounds', a_labels_dict[local_label], 'a_' + local_label, a_range,
                         ('local induction factor', None), 'x')),
    for a_name in ['acos', 'asin']:
        options_tree.append(('model', 'system_bounds', a_labels_dict[local_label], a_name + '_' + local_label, a_fourier_range, ('??', None), 'x')),

    gamma_range = options['model']['aero']['actuator']['gamma_range']
    options_tree.append(('model', 'system_bounds', 'z', 'gamma', gamma_range, ('tilt angle bounds [rad]', None), 'x')),
    gamma_ref = gamma_range[1] * 0.8
    options_tree.append(('model', 'scaling', 'z', 'gamma', gamma_ref, ('tilt angle bounds [rad]', None), 'x')),
    options_tree.append(('model', 'scaling', 'z', 'cosgamma', 0.5, ('tilt angle bounds [rad]', None), 'x')),
    options_tree.append(('model', 'scaling', 'z', 'singamma', 0.5, ('tilt angle bounds [rad]', None), 'x')),

    return options_tree, fixed_params


def get_comparison_labels(options, user_options):
    induction_model = user_options['induction_model']
    induction_comparison = options['model']['aero']['induction']['comparison']

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


###### vortex induction

def build_vortex_options(options, options_tree, fixed_params, architecture):

    n_k = options['nlp']['n_k']
    d = options['nlp']['collocation']['d']
    options_tree.append(('model', 'aero', 'vortex', 'n_k', n_k, ('how many nodes to track over one period: n_k', None), 'x')),
    options_tree.append(('model', 'aero', 'vortex', 'd', d, ('how many nodes to track over one period: d', None), 'x')),

    wake_nodes = options['model']['aero']['vortex']['wake_nodes']
    options_tree = share_among_induction_subaddresses(options, options_tree, ('model', 'aero', 'vortex', 'wake_nodes'), 'vortex_wake_nodes')


    u_ref = get_u_ref(options['user_options'])
    vortex_u_ref = u_ref
    vec_u_ref = u_ref * vect_op.xhat_np()
    options_tree.append(('solver', 'initialization', 'induction', 'vortex_u_ref', vortex_u_ref, ('????', None), 'x')),
    options_tree.append(('model', 'induction', None, 'vortex_u_ref', vortex_u_ref, ('????', None), 'x')),
    options_tree.append(('formulation', 'induction', None, 'vortex_u_ref', vortex_u_ref, ('????', None), 'x')),
    options_tree.append(('nlp', 'induction', None, 'vortex_u_ref', vortex_u_ref, ('????', None), 'x')),
    options_tree.append(('visualization', 'cosmetics', 'trajectory', 'vortex_vec_u_ref', vec_u_ref, ('???? of trajectories in animation', None), 'x')),

    t_f_guess = estimate_time_period(options, architecture)
    near_wake_unit_length = t_f_guess / n_k * u_ref
    far_wake_l_start = (wake_nodes - 1) * near_wake_unit_length

    options_tree.append(('model', 'aero', 'vortex', 'near_wake_unit_length', near_wake_unit_length, ('????', None), 'x')),
    options_tree.append(('model', 'aero', 'vortex', 'far_wake_l_start', far_wake_l_start, ('????', None), 'x')),


    far_wake_convection_time = options['model']['aero']['vortex']['far_wake_convection_time']
    options_tree = share_among_induction_subaddresses(options, options_tree, ('model', 'aero', 'vortex', 'far_wake_convection_time'), 'vortex_far_wake_convection_time')
    options_tree.append(('visualization', 'cosmetics', 'trajectory', 'vortex_far_wake_convection_time', far_wake_convection_time, ('???? of trajectories in animation', None), 'x')),

    for vortex_name in ['degree_of_induced_velocity_lifting', 'far_wake_element_type', 'epsilon_m', 'epsilon_r', 'representation']:
        options_tree = share_among_induction_subaddresses(options, options_tree, ('model', 'aero', 'vortex', vortex_name), 'vortex_' + vortex_name)
    options_tree = share_among_induction_subaddresses(options, options_tree, ('solver', 'initialization', 'inclination_deg'), 'inclination_ref_deg')

    geometry = get_geometry(options)
    c_ref = geometry['c_ref']
    r_core = options['model']['aero']['vortex']['core_to_chord_ratio'] * c_ref

    options_tree.append(('model', 'induction', None, 'vortex_core_radius', r_core, ('????', None), 'x')),
    options_tree.append(('formulation', 'induction', None, 'vortex_core_radius', r_core, ('????', None), 'x')),
    options_tree.append(('nlp', 'induction', None, 'vortex_core_radius', r_core, ('????', None), 'x')),

    rings = wake_nodes
    options_tree.append(('solver', 'initialization', 'induction', 'vortex_rings', rings, ('????', None), 'x')),
    options_tree.append(('model', 'induction', None, 'vortex_rings', rings, ('????', None), 'x')),
    options_tree.append(('model', 'aero', 'vortex', 'rings', rings, ('????', None), 'x')),
    options_tree.append(('formulation', 'induction', None, 'vortex_rings', rings, ('????', None), 'x')),
    options_tree.append(('nlp', 'induction', None, 'vortex_rings', rings, ('????', None), 'x')),

    flight_radius = estimate_flight_radius(options, architecture)
    b_ref = geometry['b_ref']
    varrho_ref = flight_radius / b_ref
    t_f_guess = estimate_time_period(options, architecture)
    windings = options['user_options']['trajectory']['lift_mode']['windings']
    winding_period = t_f_guess / float(windings)

    CL = estimate_CL(options)

    integrated_circulation = 1.
    for kite in architecture.kite_nodes:
        options_tree.append(('model', 'scaling', 'other', 'integrated_circulation' + str(kite), integrated_circulation, ('????', None), 'x')),
        options_tree.append(('nlp', 'induction', None, 'integrated_circulation' + str(kite), integrated_circulation, ('????', None), 'x')),
        options_tree.append(('solver', 'initialization', 'induction', 'integrated_circulation' + str(kite), integrated_circulation, ('????', None), 'x')),

    q_scaling = get_position_scaling(options, architecture)
    u_altitude = get_u_at_altitude(options, estimate_altitude(options))
    options_tree = vortex_alg_repr_scaling.append_scaling_to_options_tree(options, geometry, options_tree, architecture, q_scaling, u_altitude, CL, varrho_ref, winding_period)

    a_ref = options['model']['aero']['actuator']['a_ref']
    u_ref = get_u_ref(options['user_options'])
    u_ind = a_ref * u_ref

    clockwise_rotation_about_xhat = options['solver']['initialization']['clockwise_rotation_about_xhat']
    options_tree.append(('model', 'aero', 'vortex', 'clockwise_rotation_about_xhat', clockwise_rotation_about_xhat, ('descript', None), 'x'))

    options_tree.append(('model', 'scaling', 'z', 'ui', u_ind, ('descript', None), 'x'))

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


######## tether control

def build_tether_control_options(options, options_tree, fixed_params):

    user_options = options['user_options']
    in_drag_mode_operation = user_options['trajectory']['system_type'] == 'drag_mode'

    ddl_t_bounds = options['model']['system_bounds']['x']['ddl_t']
    dddl_t_bounds = options['model']['system_bounds']['u']['dddl_t']

    control_name = options['model']['tether']['control_var']

    if in_drag_mode_operation:
        options_tree.append(('model', 'system_bounds', 'u', control_name, [0.0, 0.0], ('main tether reel-out acceleration', None), 'x'))

    else:
        if control_name == 'ddl_t':
            options_tree.append(('model', 'system_bounds', 'u', 'ddl_t', ddl_t_bounds,   ('main tether max acceleration [m/s^2]', None),'x'))
            options_tree.append(('model', 'scaling', 'u', 'ddl_t', np.max(np.array(ddl_t_bounds)) / 2., ('???', None), 'x'))

        elif control_name == 'dddl_t':
            options_tree.append(('model', 'system_bounds', 'x', 'ddl_t', ddl_t_bounds,   ('main tether max acceleration [m/s^2]', None),'x'))
            options_tree.append(('model', 'system_bounds', 'u', 'dddl_t', dddl_t_bounds,   ('main tether max jerk [m/s^3]', None),'x'))
            options_tree.append(('model', 'scaling', 'x', 'ddl_t', np.max(np.array(ddl_t_bounds))/2., ('???', None), 'x'))
            options_tree.append(('model', 'scaling', 'u', 'dddl_t', np.max(np.array(dddl_t_bounds))/2., ('???', None), 'x'))

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

def build_fict_scaling_options(options, options_tree, fixed_params, architecture):

    geometry = get_geometry(options)
    b_ref = geometry['b_ref']

    q_altitude = get_q_at_altitude(options, estimate_altitude(options))

    centripetal_force = float(estimate_centripetal_force(options, architecture))

    gravity = options['model']['scaling']['other']['g']
    mass_kite = geometry['m_k']
    acc_max = options['model']['model_bounds']['acceleration']['acc_max']
    max_acceleration_force = float(mass_kite * acc_max * gravity)

    aero_force = float(estimate_aero_force(options, architecture))

    total_mass = estimate_total_mass(options, architecture)
    gravity_force = total_mass * gravity / float(architecture.number_of_kites)

    tension_per_unit_length = estimate_main_tether_tension_per_unit_length(options, architecture)
    length = options['solver']['initialization']['l_t']
    tension = tension_per_unit_length * length

    available_estimates = [max_acceleration_force, tension, gravity_force, centripetal_force, aero_force]
    synthesized_force = vect_op.synthesize_estimate_from_a_list_of_positive_scalar_floats(available_estimates)

    force_scaling_method = options['model']['scaling']['other']['force_scaling_method']
    if force_scaling_method == 'max_acceleration':
        f_scaling = max_acceleration_force
    elif force_scaling_method == 'tension':
        f_scaling = tension
    elif force_scaling_method == 'gravity':
        f_scaling = gravity_force
    elif force_scaling_method == 'centripetal':
        f_scaling = centripetal_force
    elif force_scaling_method == 'aero':
        f_scaling = aero_force
    elif force_scaling_method == 'synthesized':
        f_scaling = synthesized_force
    else:
        message = 'unknown force_scaling_method (' + force_scaling_method + ')'
        print_op.log_and_raise_error(message)

    moment_scaling_factor = b_ref / 2.

    options_tree.append(('model', 'scaling', 'u', 'f_fict', f_scaling, ('scaling of fictitious homotopy forces', None),'x'))
    options_tree.append(('model', 'scaling', 'u', 'm_fict', f_scaling * moment_scaling_factor, ('scaling of fictitious homotopy moments', None),'x'))
    options_tree.append(('model', 'scaling', 'z', 'f_aero', f_scaling, ('scaling of aerodynamic forces', None),'x'))
    options_tree.append(('model', 'scaling', 'z', 'm_aero', f_scaling * moment_scaling_factor, ('scaling of aerodynamic moments', None),'x'))

    area = 2. * np.pi * estimate_flight_radius(options, architecture) * b_ref
    q_infty = get_q_at_altitude(options, estimate_altitude(options))
    a_ref = options['model']['aero']['actuator']['a_ref']
    actuator_thrust = 4. * a_ref * (1. - a_ref) * area * q_infty

    options_tree.append(('model', 'scaling', 'z', 'thrust', actuator_thrust, ('scaling of aerodynamic forces', None), 'x'))

    CD_tether = options['params']['tether']['cd']
    diam_t = options['solver']['initialization']['theta']['diam_t']
    length = options['solver']['initialization']['l_t']

    tether_drag_force = 0.5 * CD_tether * (0.25 * q_altitude) * diam_t * length
    options_tree.append(('model', 'scaling', 'z', 'f_tether', tether_drag_force, ('scaling of tether drag forces', None),'x'))

    return options_tree, fixed_params

def get_gravity_ref(options):

    gravity = options['model']['scaling']['other']['g']

    return gravity



####### lambda, energy, power scaling

def build_lambda_e_power_scaling(options, options_tree, fixed_params, architecture):

    lambda_scaling, energy_scaling, power_cost, power = get_suggested_lambda_energy_power_scaling(options, architecture)

    if options['model']['scaling_overwrite']['lambda_tree']['include']:
        options_tree = generate_lambda_scaling_tree(options=options, options_tree=options_tree,
                                                    lambda_scaling=lambda_scaling, architecture=architecture)
    else:
        options_tree.append(('model', 'scaling', 'z', 'lambda', lambda_scaling, ('scaling of tether tension per length', None),'x'))

    options_tree.append(('model', 'scaling', 'x', 'e', energy_scaling, ('scaling of the energy', None),'x'))
    options_tree.append(('nlp', 'scaling', 'x', 'e', energy_scaling, ('scaling of the energy', None),'x'))
    options_tree.append(('solver', 'cost', 'power', 1, power_cost, ('update cost for power', None),'x'))

    options_tree.append(('model', 'scaling', 'theta', 'P_max', power, ('Max. power scaling factor', None),'x'))
    options_tree.append(('solver', 'initialization', 'theta', 'P_max', power, ('Max. power initialization', None),'x'))

    if options['model']['integration']['include_integration_test']:
        arbitrary_integration_scaling = 7283.  # some large prime number
        options_tree.append(('model', 'scaling', 'x', 'total_time_unscaled', 1., ('???', None), 'x'))
        options_tree.append(('model', 'scaling', 'x', 'total_time_scaled', arbitrary_integration_scaling, ('???', None), 'x'))

    return options_tree, fixed_params

def generate_lambda_scaling_tree(options, options_tree, lambda_scaling, architecture):

    description = ('scaling of tether tension per length', None)

    # set lambda_scaling
    options_tree.append(('model', 'scaling', 'z', 'lambda10', lambda_scaling, description, 'x'))

    # extract architecture options
    layers = architecture.layers

    # extract length scaling information
    l_s_scaling = options['solver']['initialization']['theta']['l_s']
    l_t_scaling = options['solver']['initialization']['l_t']
    l_i_scaling = options['solver']['initialization']['theta']['l_i']

    # it's tempting to put a cosine correction in here, but then using the
    # resulting scaling values to set the initialization will lead to the
    # max-tension-force path constraints being violated right-away. so: don't do it.
    cone_angle_correction = 1.

    #  secondary tether scaling
    tension_main = lambda_scaling * l_t_scaling
    tension_secondary = tension_main / architecture.number_of_kites * cone_angle_correction
    lambda_s_scaling = tension_secondary / l_s_scaling

    # tension in the intermediate tethers is not constant
    lambda_i_max = tension_main / l_i_scaling

    # assign scaling according to tree structure
    layer_count = 1
    for node in range(2,architecture.number_of_nodes):
        label = 'lambda'+str(node)+str(architecture.parent_map[node])

        if node in architecture.kite_nodes:
            options_tree.append(('model', 'scaling', 'z', label, lambda_s_scaling, description,'x'))

        else:
            # if there are no kites here, we must be at an intermediate, layer node

            # the tension should decrease as we move to higher layers, because there are fewer kites pulling on the nodes
            linear_factor = (layers - layer_count) / (float(layers))
            lambda_i_scaling = linear_factor * lambda_i_max
            options_tree.append(('model', 'scaling', 'z', label, lambda_i_scaling, description,'x'))
            layer_count += 1

    return options_tree


def get_suggested_lambda_energy_power_scaling(options, architecture):

    if options['user_options']['trajectory']['type'] == 'nominal_landing':
        power_cost = 1e-4
        lambda_scaling = 1
        corrected_estimated_energy = 1e5
    else:

        # this will scale the multiplier on the main tether, from 'si'
        lam = estimate_main_tether_tension_per_unit_length(options, architecture)
        lambda_factor = options['model']['scaling_overwrite']['lambda_factor']
        lambda_scaling = lambda_factor * lam

        # this will scale the energy 'si'. see dynamics.make_dynamics
        energy = estimate_energy(options, architecture)
        energy_factor = options['model']['scaling_overwrite']['energy_factor']
        corrected_estimated_energy = energy_factor * energy

        # this will be used to weight the scaled power (energy / time) cost
        # so: for clarity of what is physically happening, I've written this in terms of the
        # power and energy scaling values.
        # but, what's actually happening depends ONLY on the tuning factor and on the estimated time period*.
        # so, if this scaling leads to bad convergence in final solution step of homotopy, then check the
        # estimate time period function (below) FIRST.
        #
        # *: Because the integral E = \int_0^T {p dt} actually integrates the power-scaled-by-the-characteristic-energy,
        # as in integral_output = \int_0^T {p/\char{E} dt}, which means that the term in the power cost which we
        # normally think of as (1/T) \int_0^T {d pT}, ie, [(1/s)(kg m^2/s^2)] is actually being implemented as [(1/s)(-)],
        # and we should normalize/nondimensionalize that above output by (1/\char{T}) to get a completely nondimensional
        # power term, ie: multiply by 1/(1/\char{T}) -> multiply by \char{T}.
        # That is, the term in the objective is actually equivalent to
        # (1/T) \int_0^T {p dt} * (\char{T}/\char{E}) = (\average{p}/\char{p})
        #
        # see model.dynamics get_dictionary_of_derivatives and manage_alongside_integration for implementation

        estimated_average_power = estimate_power(options, architecture)
        estimated_inverse_time_period = estimated_average_power / corrected_estimated_energy  # yes, this = (1 / time_period_estimate)
        power_cost_factor = options['solver']['cost_factor']['power']
        power_cost = power_cost_factor * (1. / estimated_inverse_time_period)  # yes, this = pcf * time_period_estimate

    return lambda_scaling, corrected_estimated_energy, power_cost, estimated_average_power

def estimate_flight_radius(options, architecture):

    b_ref = get_geometry(options)['b_ref']
    anticollision_radius = b_ref * options['model']['model_bounds']['anticollision']['safety_factor']

    acc_max = options['model']['model_bounds']['acceleration']['acc_max']
    gravity = options['model']['scaling']['other']['g']
    groundspeed = options['solver']['initialization']['groundspeed']
    centripetal_radius = groundspeed**2. / (acc_max * gravity)

    cone_angle = float(options['solver']['initialization']['cone_deg']) * np.pi / 180.0
    if architecture.number_of_kites == 1:
        length = options['solver']['initialization']['l_t']
    else:
        length = options['solver']['initialization']['theta']['l_s']
    cone_radius = float(length * np.sin(cone_angle))

    available_estimates = [anticollision_radius, centripetal_radius, cone_radius]
    synthesized_radius = vect_op.synthesize_estimate_from_a_list_of_positive_scalar_floats(available_estimates)

    flight_radius_estimate = options['model']['scaling']['other']['flight_radius_estimate']
    if flight_radius_estimate == 'anticollision':
        return anticollision_radius
    elif flight_radius_estimate == 'centripetal':
        return centripetal_radius
    elif flight_radius_estimate == 'cone':
        return cone_radius
    elif flight_radius_estimate == 'synthesized':
        return synthesized_radius
    else:
        message = 'unknown flight radius scaling method (' + flight_radius_estimate + ')'
        print_op.log_and_raise_error(message)

    return None


def estimate_aero_force(options, architecture):
    geometry = get_geometry(options)

    CL = estimate_CL(options)
    q_altitude = get_q_at_altitude(options, estimate_altitude(options))
    s_ref = geometry['s_ref']

    aero_force = CL * q_altitude * s_ref
    return aero_force

def estimate_centripetal_force(options, architecture):

    geometry = get_geometry(options)
    m_k = geometry['m_k']
    groundspeed = options['solver']['initialization']['groundspeed']
    radius = estimate_flight_radius(options, architecture)

    centripetal_force = m_k * groundspeed**2. / radius
    return centripetal_force


def estimate_power(options, architecture):

    zz = estimate_altitude(options)
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

    kite_dof = get_kite_dof(options['user_options'])
    if kite_dof == 3:
        dof_efficiency = 1.
    elif kite_dof == 6:
        dof_efficiency = 0.5
    else:
        message = 'something went wrong with the number of kite degrees of freedom (' + str(kite_dof) + ')'
        print_op.log_and_raise_error(message)

    number_of_kites = architecture.number_of_kites

    loyd_estimate = number_of_kites * p_loyd
    power = loyd_estimate * induction_efficiency * dof_efficiency

    return power

def estimate_reelout_speed(options):
    zz = estimate_altitude(options)
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

    kite_dof = get_kite_dof(options['user_options'])
    if kite_dof == 3:
        coeff_bounds = options['model']['system_bounds']['x']['coeff']
        CL = coeff_bounds[1][0]
    else:
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


def estimate_position_of_main_tether_end(options):
    elevation_angle = options['solver']['initialization']['inclination_deg'] * np.pi / 180.
    length = options['solver']['initialization']['l_t']
    q_t = length * (cas.cos(elevation_angle) * vect_op.xhat_dm() + cas.sin(elevation_angle) * vect_op.zhat_dm())
    return q_t


def estimate_altitude(options):
    q_t = estimate_position_of_main_tether_end(options)
    return q_t[2]


def estimate_main_tether_tension_per_unit_length(options, architecture):

    power = estimate_power(options, architecture)
    reelout_speed = estimate_reelout_speed(options)
    tension_estimate_via_power = float(power/reelout_speed)

    aero_force_per_kite = estimate_aero_force(options, architecture)
    cone_angle_rad = options['solver']['initialization']['cone_deg'] * np.pi / 180.
    aero_force_per_kite_in_main_tether_direction = aero_force_per_kite * np.cos(cone_angle_rad)
    aero_force_projected_and_summed = aero_force_per_kite_in_main_tether_direction * architecture.number_of_kites

    total_mass = estimate_total_mass(options, architecture)
    gravity = options['model']['scaling']['other']['g']
    inclination_angle = options['solver']['initialization']['inclination_deg'] * np.pi / 180.
    gravity_force_projected_and_summed = total_mass * gravity * np.sin(inclination_angle)

    tension_estimate_via_force_summation = np.abs(float(aero_force_projected_and_summed - gravity_force_projected_and_summed))

    arbitrary_margin_from_max = 0.5
    max_stress = options['params']['tether']['max_stress'] / options['params']['tether']['stress_safety_factor']
    diam_t = options['solver']['initialization']['theta']['diam_t']
    cross_sectional_area_t = np.pi * (diam_t / 2.) ** 2.
    tension_estimate_via_max_stress = arbitrary_margin_from_max * max_stress * cross_sectional_area_t

    tension_estimate_via_min_force = options['params']['model_bounds']['tether_force_limits'][0]
    tension_estimate_via_max_force = options['params']['model_bounds']['tether_force_limits'][1]
    tension_estimate_via_average_force = (tension_estimate_via_min_force + tension_estimate_via_max_force)/2.

    available_estimates = [tension_estimate_via_power, tension_estimate_via_max_stress, tension_estimate_via_average_force, tension_estimate_via_force_summation]
    tension_estimate_via_synthesis = vect_op.synthesize_estimate_from_a_list_of_positive_scalar_floats(available_estimates)

    tension_estimate = options['model']['scaling']['other']['tension_estimate']
    if tension_estimate == 'power':
        tension = tension_estimate_via_power
    elif tension_estimate == 'max_stress':
        tension = tension_estimate_via_max_stress
    elif tension_estimate == 'average_force':
        tension = tension_estimate_via_average_force
    elif tension_estimate == 'force_summation':
        tension = tension_estimate_via_force_summation
    elif tension_estimate == 'synthesized':
        tension = tension_estimate_via_synthesis
    else:
        message = 'unknown tension estimation method (' + tension_estimate + ')'
        print_op.log_and_raise_error(message)

    length = options['solver']['initialization']['l_t']
    multiplier = tension / length
    return multiplier


def estimate_total_mass(options, architecture):

    mass_of_all_kites = get_geometry(options)['m_k'] * architecture.number_of_kites

    diam_t = options['solver']['initialization']['theta']['diam_t']
    rho_tether = options['params']['tether']['rho']
    cross_sectional_area_t = np.pi * (diam_t / 2.) ** 2.
    length = options['solver']['initialization']['l_t']
    mass_of_main_tether = cross_sectional_area_t * length * rho_tether

    if architecture.number_of_kites > 1:
        diam_s = options['solver']['initialization']['theta']['diam_s']
        cross_sectional_area_s = np.pi * (diam_s / 2.) ** 2.
        length_s = options['solver']['initialization']['theta']['l_s']
        mass_of_secondary_tether = cross_sectional_area_s * length_s * rho_tether * architecture.number_of_kites
    else:
        mass_of_secondary_tether = 0.

    number_of_intermediate_tethers = architecture.number_of_nodes - 1 - architecture.number_of_kites
    if number_of_intermediate_tethers > 0:
        diam_i = options['solver']['initialization']['theta']['diam_i']
        cross_sectional_area_i = np.pi * (diam_i / 2.) ** 2.
        length_i = options['solver']['initialization']['theta']['l_i']
        mass_of_intermediate_tether = cross_sectional_area_i * length_i * rho_tether * number_of_intermediate_tethers
    else:
        mass_of_intermediate_tether = 0.

    total_mass = mass_of_all_kites + mass_of_main_tether + mass_of_secondary_tether + mass_of_intermediate_tether
    return total_mass

def estimate_energy(options, architecture):
    power = estimate_power(options, architecture)
    time_period = estimate_time_period(options, architecture)
    energy = power * time_period
    return energy

def estimate_time_period(options, architecture):

    if 't_f' in options['user_options']['trajectory']['fixed_params']:
        return options['user_options']['trajectory']['fixed_params']['t_f']

    windings = options['user_options']['trajectory']['lift_mode']['windings']
    groundspeed = options['solver']['initialization']['groundspeed']
    radius = estimate_flight_radius(options, architecture)

    time_period = float((2. * np.pi * windings * radius) / groundspeed)

    return time_period




def share(options, options_tree, from_tuple, to_tuple):
    if len(from_tuple) == 4:
        value = options[from_tuple[0]][from_tuple[1]][from_tuple[2]][from_tuple[3]]
    elif len(from_tuple) == 3:
        value = options[from_tuple[0]][from_tuple[1]][from_tuple[2]]
    elif len(from_tuple) == 2:
        value = options[from_tuple[0]][from_tuple[1]]
    else:
        message = 'inappropriate_sharing_address (from)'
        print_op.log_and_raise_error(message)

    if len(to_tuple) == 4:
        pass
    elif len(to_tuple) == 3:
        to_tuple = (to_tuple[0], to_tuple[1], None, to_tuple[2])
    elif len(to_tuple) == 2:
        to_tuple = (to_tuple[0], None, None, to_tuple[1])
    else:
        message = 'inappropriate_sharing_address (to)'
        print_op.log_and_raise_error(message)

    options_tree.append(
        (to_tuple[0], to_tuple[1], to_tuple[2], to_tuple[3],
         value,
         ('???', None), 'x'))
    return options_tree


def share_among_induction_subaddresses(options, options_tree, from_tuple, entry_name):
    options_tree = share(options, options_tree, from_tuple, ('solver', 'initialization', 'induction', entry_name))
    options_tree = share(options, options_tree, from_tuple, ('model', 'induction', entry_name))
    options_tree = share(options, options_tree, from_tuple, ('formulation', 'induction', entry_name))
    options_tree = share(options, options_tree, from_tuple, ('nlp', 'induction', entry_name))
    return options_tree
