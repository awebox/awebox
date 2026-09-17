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
- edited: rachel leuthold, 2017-2025
'''
from platform import architecture

import numpy as np
from sympy.assumptions.predicates.order import NonNegativePredicate

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
import awebox.mdl.aero.induction_dir.actuator_dir.system as actuator_system
import awebox.mdl.aero.induction_dir.vortex_dir.alg_repr_dir.scaling as vortex_alg_repr_scaling

import awebox.mdl.wind as wind
from awebox.tools.vector_operations import zhat_np


def build_model_options(options, help_options, user_options, options_tree, fixed_params, architecture):

    # kite
    options_tree, fixed_params = build_geometry_options(options, help_options, options_tree, fixed_params)
    options_tree, fixed_params = build_kite_dof_options(options, options_tree, fixed_params, architecture)

    options_tree, fixed_params = build_scaling_options(options, options_tree, fixed_params, architecture)

    # problem specifics
    options_tree, fixed_params = build_constraint_applicablity_options(options, options_tree, fixed_params, architecture)
    options_tree, fixed_params = build_trajectory_options(options, help_options, options_tree, fixed_params, architecture)
    options_tree, fixed_params = build_integral_options(options, options_tree, fixed_params)

    # aerodynamics
    options_tree, fixed_params = build_stability_derivative_options(options, help_options, options_tree, fixed_params)
    options, options_tree, fixed_params = build_induction_options(options, help_options, options_tree, fixed_params, architecture)
    options_tree, fixed_params = build_actuator_options(options, options_tree, fixed_params, architecture)
    options_tree, fixed_params = build_vortex_options(options, options_tree, fixed_params, architecture)

    # tether
    options_tree, fixed_params = build_tether_drag_options(options, help_options, options_tree, fixed_params)
    options_tree, fixed_params = build_tether_stress_options(options, options_tree, fixed_params, architecture)
    options_tree, fixed_params = build_tether_control_options(options, help_options, options_tree, fixed_params)

    # environment
    options_tree, fixed_params = build_wind_options(options, options_tree, fixed_params)
    options_tree, fixed_params = build_atmosphere_options(options, help_options, options_tree, fixed_params)

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
        options_tree.append((dict_type, 'geometry', None, name, geometry[name], ('???', None),'x'))

    if options['user_options']['trajectory']['type'] not in ['nominal_landing', 'transitions', 'compromised_landing', 'launch']:
        fixed_params = options['user_options']['trajectory']['fixed_params']

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


def get_position_scaling(options, architecture, suppress_help_statement=False, overwrite_method=None):

    thing_estimated = 'position [m]'

    position = estimate_position_of_main_tether_end(options)
    flight_radius = estimate_flight_radius(options, architecture, suppress_help_statement=True)
    geometry = get_geometry(options)
    b_ref = geometry['b_ref']

    scaling_dict = {'radius': flight_radius * cas.DM.ones((3, 1)),
                 'altitude': position[2] * cas.DM.ones((3, 1)),
                 'b_ref': b_ref * cas.DM.ones((3, 1)),
                 'radius_and_tether': cas.vertcat(position[0], flight_radius, flight_radius),
                 'radius_and_altitude': cas.vertcat(position[0], flight_radius, position[2])
                 }
    scaling_dict['altitude_and_radius'] = scaling_dict['radius_and_altitude']

    method_in_options = options['model']['scaling']['other']['position_scaling_method']
    selected_method = select_scaling_method(method_in_options, overwrite_method, scaling_dict, thing_estimated)
    value = scaling_dict[selected_method]

    print_help_with_scaling(options, scaling_dict, selected_method, thing_estimated, suppress_help_statement)

    return value

def select_scaling_method(method_in_options, overwrite_method, scaling_dict, thing_estimated):
    if (overwrite_method is not None) and (overwrite_method in scaling_dict.keys()):
        selected_method = overwrite_method
    elif method_in_options in scaling_dict.keys():
        selected_method = method_in_options
    else:
        message = 'unexpected ' + thing_estimated + ' scaling/estimation method in options (' + method_in_options + ')'
        print_op.log_and_raise_error(message)

    return selected_method

def print_help_with_scaling(options, scaling_dict, selected_method, thing_estimated, suppress_help_statement):
    if options['model']['scaling']['other']['print_help_with_scaling'] and not suppress_help_statement:
        table_name = 'available ' + thing_estimated + ' estimates are:'
        sorted_scaling_dict = dict(sorted(scaling_dict.items(), key=lambda item: float(vect_op.norm(item[1]))))
        print_op.print_dict_as_table(sorted_scaling_dict, level='debug', caption=table_name)
        selection_message = 'currently selected ' + thing_estimated + ' scaling/estimation option: ' + selected_method
        print_op.base_print(selection_message + '\n', level='debug')

    return None

def transfer_synthesization_estimates_to_a_scaling_dictionary(scaling_dict, synthesizing_dict):

    available_estimates = []
    for name, val in synthesizing_dict.items():
        if vect_op.is_numeric_scalar(val):
            available_estimates += [float(val)]
        else:
            message = 'Entry at ' + name + ' of scaling dict is not a numeric scalar. This entry will be skipped while synthesizing estimates.'
            print_op.base_print(message, level='warning')

    number_of_estimates = len(available_estimates)
    # if number_of_estimates > 0:
    averaging_fraction = 1. / float(number_of_estimates)
    product = 1.
    for val in available_estimates:
        product = product * val
    geometric_average = product ** averaging_fraction

    for name, val in synthesizing_dict.items():
        scaling_dict[name] = val

    scaling_dict['synthesized'] = geometric_average

    return scaling_dict

def build_scaling_options(options, options_tree, fixed_params, architecture):

    length = options['solver']['initialization']['l_t']
    length_scaling = length
    options_tree.append(('model', 'scaling', 'x', 'l_t', length_scaling, ('???', None), 'x'))
    options_tree.append(('model', 'scaling', 'theta', 'l_t', length_scaling, ('???', None), 'x'))

    flight_radius = estimate_flight_radius(options, architecture, suppress_help_statement=False) # include this even
    # though radius is not needed here, so that we get a print-out of radius options
    time_period = estimate_time_period(options, architecture, suppress_help_statement=False) # again, print the options
    power = estimate_power(options, architecture, suppress_help_statement=False) # again, print the options

    q_scaling = get_position_scaling(options, architecture)
    options_tree.append(('model', 'scaling', 'x', 'q', q_scaling, ('???', None),'x'))

    u_altitude = get_u_at_altitude(options, estimate_altitude(options))
    groundspeed = options['solver']['initialization']['groundspeed']
    for node in range(1, architecture.number_of_nodes):
        parent = architecture.parent_map[node]
        dq_name = 'dq' + str(node) + str(parent)
        if node in architecture.kite_nodes:
            options_tree.append(('model', 'scaling', 'x', dq_name, groundspeed, ('???', None), 'x'))
        else:
            options_tree.append(('model', 'scaling', 'x', dq_name, u_altitude, ('???', None), 'x'))

    dl_t_scaling = u_altitude / 3. #loyd result
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
    options_tree.append(('model', None, None, 'kite_name', user_options['kite_standard']['name'], ('???', None),'x')),

    system_summary_string = str(architecture.number_of_kites) + "x " + user_options['kite_standard']['name']
    options_tree.append(('model', None, None, 'system_summary_string', system_summary_string, ('???', None, None),'x'))

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

        options_tree.append(('model', 'system_bounds', 'x', 'delta', [-1. * delta_max, delta_max], ('control surface deflection bounds', None, 'rad'),'x'))
        options_tree.append(('model', 'system_bounds', 'u', 'ddelta', [-1. * ddelta_max, ddelta_max],
                             ('control surface deflection rate bounds', None, 'rad/s'),'x'))

        standard_geometry = load_kite_geometry(options['user_options']['kite_standard'])
        delta_scaling = []
        ddelta_scaling = []
        for idx in range(delta_max.shape[0]):
            if vect_op.is_numeric_scalar(delta_max[idx]):
                local_delta = delta_max[idx]/2.
            else:
                local_delta = standard_geometry['delta_max'][idx] / 2.
            delta_scaling = cas.vertcat(delta_scaling, local_delta)

            if vect_op.is_numeric_scalar(ddelta_max[idx]):
                local_ddelta = ddelta_max[idx]/2.
            else:
                local_ddelta = standard_geometry['ddelta_max'][idx] / 2.
            ddelta_scaling = cas.vertcat(ddelta_scaling, local_ddelta)

        options_tree.append(('model', 'scaling', 'x', 'delta', delta_scaling, ('???', None), 'x'))
        options_tree.append(('model', 'scaling', 'u', 'ddelta', ddelta_scaling, ('???', None), 'x'))
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
        if not vect_op.is_numeric_scalar(coeff_scaling[0]):
            coeff_scaling[0] = estimate_CL(options)
        if not vect_op.is_numeric_scalar(coeff_scaling[1]):
            coeff_scaling[1] = 20. * np.pi / 180.
        options_tree.append(('model', 'scaling', 'x', 'coeff', coeff_scaling, ('???', None), 'x'))

        dcoeff_max = cas.DM(options['model']['system_bounds']['u']['dcoeff'][1])
        dcoeff_scaling = dcoeff_max
        if not vect_op.is_numeric_scalar(dcoeff_scaling[0]):
            dcoeff_scaling[0] = 2.5 #todo: currently taken from default values, find better approx.
        if not vect_op.is_numeric_scalar(dcoeff_scaling[1]):
            dcoeff_scaling[1] = 40. #todo: currently taken from default values, find better approx.
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
    # todo: are we using this for anything?
    # options_tree.append(('model', 'model_bounds', 'anticollision_radius', 'num_ref', groundspeed ** 2., ('an estimate of the square of the kite speed, for normalization of the anticollision inequality', None),'x'))

    include_acceleration_constraint = options['model']['model_bounds']['acceleration']['include']
    options_tree.append(('solver', 'initialization', None, 'include_acceleration_constraint', include_acceleration_constraint, ('??', None), 'x'))

    airspeed_ref = get_airspeed_average(options)
    options_tree.append(('model', 'model_bounds', 'aero_validity', 'airspeed_ref', airspeed_ref, ('an estimate of thef kite speed, for normalization of the aero_validity orientation inequality', None),'x'))

    airspeed_include = options['model']['model_bounds']['airspeed']['include']
    airspeed_limits = get_airspeed_limits(options)
    options_tree.append(('solver', 'initialization', None, 'airspeed_limits', airspeed_limits, ('airspeed limits [m/s]', None), 's'))
    options_tree.append(('solver', 'initialization', None, 'airspeed_include', airspeed_include, ('apply airspeed limits [m/s]', None), 's'))

    options_tree.append(('model', None, None, 'cross_tether', user_options['system_model']['cross_tether'], ('enable cross-tether',[True,False]),'x'))
    if architecture.number_of_kites == 1 or user_options['system_model']['cross_tether']:
        options_tree.append(('model', 'model_bounds', 'anticollision', 'include', False, ('anticollision inequality', (True,False)),'x'))

    # map single airspeed interval constraint to min/max constraints
    if options['model']['model_bounds']['airspeed']['include']:
        options_tree.append(('model', 'model_bounds', 'airspeed_min', 'include', vect_op.is_numeric_scalar(airspeed_limits[0]),   ('include min airspeed constraint', None),'x'))
        options_tree.append(('model', 'model_bounds', 'airspeed_max', 'include', vect_op.is_numeric_scalar(airspeed_limits[1]), ('include max airspeed constraint', None), 'x'))

    return options_tree, fixed_params

def get_airspeed_limits(options):
    airspeed_include = options['model']['model_bounds']['airspeed']['include']
    kite_standard = options['user_options']['kite_standard']
    aero_deriv, aero_validity = load_stability_derivatives(kite_standard)
    overwrite_airspeed_limits = options['params']['model_bounds']['airspeed_limits']
    if vect_op.is_numeric_scalar(overwrite_airspeed_limits[0]):
        airspeed_min = overwrite_airspeed_limits[0]
    elif 'airspeed_min' in aero_validity.keys():
        airspeed_min = aero_validity['airspeed_min']
    else:
        airspeed_min = -cas.inf
        if airspeed_include:
            message = 'no airspeed minimum given despite request to include airspeed limits; setting minimum airspeed to -inf'
            print_op.base_print(message, level='warning')

    if vect_op.is_numeric_scalar(overwrite_airspeed_limits[1]):
        airspeed_max = overwrite_airspeed_limits[1]
    elif 'airspeed_max' in aero_validity.keys():
        airspeed_max = aero_validity['airspeed_max']
    else:
        airspeed_max = cas.inf
        if airspeed_include:
            message = 'no airspeed maximum given despite request to include airspeed limits; setting maximum airspeed to +inf'
            print_op.base_print(message, level='warning')

    airspeed_limits = np.array([airspeed_min, airspeed_max])
    return airspeed_limits


####### trajectory specifics

def build_trajectory_options(options, help_dict, options_tree, fixed_params, architecture):

    user_options = options['user_options']

    sys_bound_help_info = help_dict['model']['system_bounds']['theta']
    init_help_info = help_dict['solver']['initialization']['theta']
    dict_of_fixed_units = {}
    for theta_name in fixed_params.keys():
        if theta_name in sys_bound_help_info.keys():
            fixed_param_help_info = sys_bound_help_info[theta_name]
        elif theta_name in init_help_info.keys():
            fixed_param_help_info = init_help_info[theta_name]
        else:
            fixed_param_help_info = None

        if (fixed_param_help_info is not None) and (len(fixed_param_help_info[0]) > 2):
            fixed_param_units = fixed_param_help_info[0][2]
            dict_of_fixed_units[theta_name] = fixed_param_units

    if user_options['trajectory']['type'] not in ['nominal_landing', 'transitions', 'compromised_landing', 'launch']:
        fixed_params = user_options['trajectory']['fixed_params']
        options_tree.append(('model', 'system_bounds_other', None, 'fixed_params', fixed_params,
                         ('user input for fixed bounds on theta', None, list(dict_of_fixed_units.items())), 'x'))

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
        if theta in dict_of_fixed_units.keys():
            options_tree.append(('model', 'system_bounds', 'theta', theta, [fixed_params[theta]] * 2,
                                 ('user input for fixed bounds on theta', None, dict_of_fixed_units[theta]), 'x'))
        else:
            options_tree.append(('model', 'system_bounds', 'theta', theta, [fixed_params[theta]] * 2,
                                 ('user input for fixed bounds on theta', None), 'x'))

    scenario, broken_kite = user_options['trajectory']['compromised_landing']['emergency_scenario']
    if not broken_kite in architecture.kite_nodes:
        broken_kite = architecture.kite_nodes[0]

    options_tree.append(('model', 'compromised_landing', None, 'emergency_scenario', [scenario, broken_kite], ('type of emergency scenario', ['broken_roll','broken_lift']),'x'))
    options_tree.append(('nlp', 'trajectory', None, 'type', user_options['trajectory']['type'], ('??', None), 'x'))

    t_f_guess = estimate_time_period(options, architecture)
    options_tree.append(('nlp', 'normalization', None, 't_f', t_f_guess, ('??', None), 'x'))
    # todo: is this nlp.normalization.t_f option being used for anything?

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
    options_tree.append(('model', None, None, 'beta_cost', options['nlp']['cost']['beta'], ('add beta_cost to integral outputs',[True,False]),'x'))

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

    for bound_name in (set(aero_validity.keys() - set(['airspeed_min', 'airspeed_max']))):
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
    u_at_altitude = get_u_at_altitude(options, estimate_altitude(options))
    options_tree = actuator_system.add_scaling_of_support_variables(options, architecture, u_at_altitude, options_tree)
    options_tree.append(
        ('solver', 'initialization', 'induction', 'u_at_altitude', u_at_altitude, ('????', None), 'x')),


    options_tree = actuator_system.add_system_bounds_of_support_variables(options, help_options, options_tree)


    if options['model']['aero']['actuator']['support_only']:
        if (user_options['induction_model'] == 'actuator') and (not options['model']['aero']['induction']['force_zero']):
            message = 'model.aero.actuator.support_only is true, while the actuator induction model is selected.' \
                      ' this implies that model.aero.induction.force_zero must also be true.' \
                      ' proceeding with force_zero option reset to true.'
            print_op.base_print(message, level='warning')
            options['model']['aero']['induction']['force_zero'] = True

    normal_vector_model = options['model']['aero']['actuator']['normal_vector_model']
    options_tree.append(
        ('solver', 'initialization', 'induction', 'normal_vector_model', normal_vector_model, ('descript', None), 'x'))

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

    options_tree.append(('model', 'aero', 'reduced_frequency', 'n_k', options['nlp']['n_k'], ('????', None), 'x')),

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

    flight_radius = estimate_flight_radius(options, architecture, suppress_help_statement=True)
    geometry = get_geometry(options)
    b_ref = geometry['b_ref']
    induction_varrho_ref = flight_radius / b_ref
    options_tree.append(('model', 'aero', 'actuator', 'varrho_ref', induction_varrho_ref, ('descript', None), 'x'))
    options_tree.append(('model', 'scaling', 'z', 'varrho', induction_varrho_ref, ('descript', None), 'x'))
    options_tree.append(('model', 'scaling', 'z', 'bar_varrho', induction_varrho_ref, ('descript', None), 'x'))
    options_tree.append(('model', 'system_bounds', 'z', 'varrho', [0., cas.inf], ('relative radius bounds [-]', None), 'x'))
    options_tree.append(('model', 'scaling', 'z', 'area', 2. * np.pi * flight_radius * b_ref, ('descript', None), 'x'))

    if options['model']['aero']['actuator']['position_scaling_method'] == 'default':
        overwrite_position_scaling_method = None
    else:
        overwrite_position_scaling_method = options['model']['aero']['actuator']['position_scaling_method']
    act_q = get_position_scaling(options, architecture, suppress_help_statement=True, overwrite_method=overwrite_position_scaling_method)
    act_dq = estimate_reelout_speed(options)
    options_tree.append(('model', 'scaling', 'z', 'act_q', act_q, ('descript', None), 'x'))
    options_tree.append(('model', 'scaling', 'z', 'act_dq', act_dq, ('descript', None), 'x'))
    q_bounds = [np.array([-cas.inf, -cas.inf, 10.0]), np.array([cas.inf, cas.inf, cas.inf])]
    options_tree.append(('model', 'system_bounds', 'z', 'act_q', q_bounds, ('??', None), 'x')),


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
    gamma_ref = gamma_range[1] * 0.5
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

    options_tree = share_among_induction_subaddresses(options, options_tree, ('model', 'aero', 'vortex', 'convection_type'), 'vortex_convection_type')

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

    flight_radius = estimate_flight_radius(options, architecture, suppress_help_statement=True)
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

    q_scaling = get_position_scaling(options, architecture, suppress_help_statement=True)
    u_altitude = get_u_at_altitude(options, estimate_altitude(options))
    airspeed_avg = get_airspeed_average(options)
    options_tree = vortex_alg_repr_scaling.append_scaling_to_options_tree(options, geometry, options_tree, architecture, q_scaling, u_altitude, CL, varrho_ref, winding_period, airspeed_avg)

    a_ref = options['model']['aero']['actuator']['a_ref']
    u_ref = get_u_ref(options['user_options'])
    u_ind = a_ref * u_ref

    clockwise_rotation_about_xhat = options['solver']['initialization']['clockwise_rotation_about_xhat']
    options_tree.append(('model', 'aero', 'vortex', 'clockwise_rotation_about_xhat', clockwise_rotation_about_xhat, ('descript', None), 'x'))

    options_tree.append(('model', 'scaling', 'z', 'wui', u_ind, ('descript', None), 'x'))

    return options_tree, fixed_params


####### tether drag

def build_tether_drag_options(options, help_dict, options_tree, fixed_params):

    tether_drag_descript =  help_dict['user_options']['tether_drag_model'][0]
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

def build_tether_control_options(options, help_dict, options_tree, fixed_params):

    user_options = options['user_options']
    in_drag_mode_operation = user_options['trajectory']['system_type'] == 'drag_mode'

    ddl_t_bounds = options['model']['system_bounds']['x']['ddl_t']
    dddl_t_bounds = options['model']['system_bounds']['u']['dddl_t']

    control_name = options['model']['tether']['control_var']

    gravity = options['model']['scaling']['other']['g']
    ddl_t_scaling = ddl_t_bounds[1] / 2.
    if not vect_op.is_numeric_scalar(ddl_t_scaling):
        ddl_t_scaling = gravity

    dddl_t_scaling = dddl_t_bounds[1] / 2.
    if not vect_op.is_numeric_scalar(dddl_t_scaling):
        dddl_t_scaling = 2. * gravity #todo: this is arbitrary. find something more physics-based.

    if in_drag_mode_operation:
        options_tree.append(('model', 'system_bounds', 'u', control_name, [0.0, 0.0], ('main tether reel-out acceleration', None), 'x'))

    else:
        if control_name == 'ddl_t':
            ddlt_descript = help_dict['model']['system_bounds']['x']['ddl_t'][0]
            options_tree.append(('model', 'system_bounds', 'u', 'ddl_t', ddl_t_bounds, ddlt_descript, 'x'))
            options_tree.append(('model', 'scaling', 'u', 'ddl_t', ddl_t_scaling, ('???', None), 'x'))

        elif control_name == 'dddl_t':
            options_tree.append(('model', 'system_bounds', 'x', 'ddl_t', ddl_t_bounds,   ('main tether max acceleration [m/s^2]', None),'x'))
            options_tree.append(('model', 'system_bounds', 'u', 'dddl_t', dddl_t_bounds,   ('main tether max jerk [m/s^3]', None),'x'))
            options_tree.append(('model', 'scaling', 'x', 'ddl_t', ddl_t_scaling, ('???', None), 'x'))
            options_tree.append(('model', 'scaling', 'u', 'dddl_t', dddl_t_scaling, ('???', None), 'x'))

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
    options_tree.append(('model', 'wind', 'parallelization', 'type', options['model']['construction']['parallelization']['type'], ('?????', None), 'x'))

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

def build_atmosphere_options(options, help_options, options_tree, fixed_params):

    options_tree.append(('model',  'atmosphere', None, 'model', options['user_options']['atmosphere'], help_options['user_options']['atmosphere'][0], 'x'))
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

def build_fict_scaling_options(options, options_tree, fixed_params, architecture, suppress_help_statement=False):

    thing_estimated = 'fictitious force [N]'

    geometry = get_geometry(options)
    b_ref = geometry['b_ref']
    q_altitude = get_q_at_altitude(options, estimate_altitude(options))

    scaling_dict = {}
    synthesizing_dict = {}

    centripetal_force = float(estimate_centripetal_force(options, architecture))
    synthesizing_dict['centripetal'] = centripetal_force

    gravity = options['model']['scaling']['other']['g']
    mass_kite = geometry['m_k']
    acc_max = options['model']['model_bounds']['acceleration']['acc_max']
    max_acceleration_force = float(mass_kite * acc_max * gravity)
    if options['model']['model_bounds']['acceleration']['include']:
        synthesizing_dict['max_acceleration'] = max_acceleration_force
    else:
        scaling_dict['max_acceleration'] = max_acceleration_force

    aero_force = float(estimate_aero_force(options))
    synthesizing_dict['aero'] = aero_force

    total_mass, _ = estimate_total_mass(options, architecture)
    gravity_force = total_mass * gravity / float(architecture.number_of_kites)
    if options['params']['atmosphere']['g'] > 0.1:
        synthesizing_dict['gravity'] = gravity_force
    else:
        scaling_dict['gravity'] = gravity_force

    tension_per_unit_length = estimate_main_tether_tension_per_unit_length(options, architecture, suppress_help_statement=True)
    length = options['solver']['initialization']['l_t']
    tension = tension_per_unit_length * length
    synthesizing_dict['tension'] = tension

    scaling_dict = transfer_synthesization_estimates_to_a_scaling_dictionary(scaling_dict, synthesizing_dict)

    method_in_options = options['model']['scaling']['other']['force_scaling_method']
    overwrite_method = None
    selected_method = select_scaling_method(method_in_options, overwrite_method, scaling_dict, thing_estimated)
    f_scaling = scaling_dict[selected_method]

    print_help_with_scaling(options, scaling_dict, selected_method, thing_estimated, suppress_help_statement)

    moment_scaling_factor = b_ref / 2.

    options_tree.append(('model', 'scaling', 'u', 'f_fict', f_scaling, ('scaling of fictitious homotopy forces', None),'x'))
    options_tree.append(('model', 'scaling', 'u', 'm_fict', f_scaling * moment_scaling_factor, ('scaling of fictitious homotopy moments', None),'x'))
    options_tree.append(('model', 'scaling', 'z', 'f_aero', f_scaling, ('scaling of aerodynamic forces', None),'x'))
    options_tree.append(('model', 'scaling', 'z', 'm_aero', f_scaling * moment_scaling_factor, ('scaling of aerodynamic moments', None),'x'))

    suppress_actuator_thrust_help = (options['user_options']['induction_model'] != 'actuator')
    actuator_thrust = estimate_actuator_thrust(options, architecture, suppress_help_statement=suppress_actuator_thrust_help)
    options_tree.append(('model', 'scaling', 'z', 'thrust', actuator_thrust, ('scaling of aerodynamic forces', None), 'x'))

    CD_tether = options['params']['tether']['cd']
    diam_t = options['solver']['initialization']['theta']['diam_t']
    length = options['solver']['initialization']['l_t']

    tether_drag_force = 0.5 * CD_tether * (0.25 * q_altitude) * diam_t * length
    options_tree.append(('model', 'scaling', 'z', 'f_tether', tether_drag_force, ('scaling of tether drag forces', None),'x'))

    return options_tree, fixed_params

def get_momentum_theory_thrust(options, architecture):
    b_ref = get_geometry(options)['b_ref']
    radius = estimate_flight_radius(options, architecture, suppress_help_statement=True)
    area = 2. * np.pi * radius * b_ref
    q_infty = get_q_at_altitude(options, estimate_altitude(options))
    a_ref = options['model']['aero']['actuator']['a_ref']
    ct_thrust = 4. * a_ref * (1. - a_ref) * area * q_infty
    return ct_thrust

def estimate_actuator_thrust(options, architecture, suppress_help_statement=False):

    thing_estimated = 'actuator thrust [N]'

    ct_thrust = get_momentum_theory_thrust(options, architecture)

    aero_thrust = architecture.number_of_kites * estimate_aero_force(options)

    tension_per_unit_length = estimate_main_tether_tension_per_unit_length(options, architecture, suppress_help_statement=True)
    length = options['solver']['initialization']['l_t']
    tension = tension_per_unit_length * length
    tension_thrust = tension

    synthesizing_dict = {'thrust_coeff': ct_thrust,
                     'aero': aero_thrust,
                     'tension': tension_thrust
                     }
    scaling_dict = {}

    scaling_dict = transfer_synthesization_estimates_to_a_scaling_dictionary(scaling_dict, synthesizing_dict)

    method_in_options = options['model']['aero']['actuator']['thrust_scaling_method']
    overwrite_method = None
    selected_method = select_scaling_method(method_in_options, overwrite_method, scaling_dict, thing_estimated)
    estimate = scaling_dict[selected_method]

    print_help_with_scaling(options, scaling_dict, selected_method, thing_estimated, suppress_help_statement)

    return estimate


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
    options_tree.append(('model', 'scaling', 'x', 'e_without_fictitious', energy_scaling, ('scaling of the energy', None),'x'))
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

    distribution_method = options['model']['scaling_overwrite']['lambda_tree']['distribution_method']
    if distribution_method == 'vector_sum':
        # this method seems to work better in the case that we use intermediate tether segments (that aren't layer
        # nodes) to represent tether sag/lag - ie, the "segmented tether trial"

        tether_vector_tree = get_tether_vector_tree(options, architecture)
        _, tether_mass_tree = estimate_total_mass(options, architecture)
        gravity = options['model']['scaling']['other']['g']

        total_tension = lambda_scaling * l_t_scaling

        tension_fraction = {}
        for kite in architecture.kite_nodes:
            tension_fraction[kite] = 1. / float(architecture.number_of_kites) + (tether_mass_tree[kite] * gravity / total_tension)

        node_list = list(range(1, architecture.number_of_nodes))
        if len(node_list) > 0:
            node_list.reverse()
            for node in node_list:
                if node not in architecture.kite_nodes:
                    sum_of_child_fractions = cas.DM.zeros((3, 1))
                    for child in architecture.children_map[node]:
                        sum_of_child_fractions += tension_fraction[child] * tether_vector_tree[child]
                    redistributed_tension_above = cas.mtimes(tether_vector_tree[node].T, sum_of_child_fractions)
                    gravity_contribution = (tether_mass_tree[node] * gravity / total_tension)
                    # we add a gravity term so that lower intermediate tethers feel more tension than neighboring
                    # upper intermediate tethers
                    # and we add that gravity 'tension' as a scalar, to avoid having different scaling values for the
                    # different secondary tethers, depending on the psi value during the vector-tree generation
                    tension_fraction[node] = redistributed_tension_above + gravity_contribution

        normalization = 1. / tension_fraction[1]
        lambda_dict = {}

        lambda_dict[1] = float(tension_fraction[1] * total_tension * normalization / l_t_scaling)
        for node in range(2, architecture.number_of_nodes):
            label = 'lambda' + str(node) + str(architecture.parent_map[node])
            if node in architecture.kite_nodes:
                lambda_dict[node] = tension_fraction[node] * total_tension * normalization / l_s_scaling
            else:
                lambda_dict[node] = tension_fraction[node] * total_tension * normalization / l_i_scaling
            options_tree.append(('model', 'scaling', 'z', label, lambda_dict[node], description, 'x'))


    elif distribution_method == 'linear_sum':
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
        lambda_dict = {1: lambda_scaling}

        # assign scaling according to tree structure
        layer_count = 1
        for node in range(2,architecture.number_of_nodes):
            label = 'lambda'+str(node)+str(architecture.parent_map[node])

            if node in architecture.kite_nodes:
                options_tree.append(('model', 'scaling', 'z', label, lambda_s_scaling, description,'x'))
                lambda_dict[node] = lambda_s_scaling

            else:
                # if there are no kites here, we must be at an intermediate, layer node
                # the tension should decrease as we move to higher layers, because there are fewer kites pulling on the nodes
                linear_factor = (layers - layer_count) / (float(layers))
                lambda_i_scaling = linear_factor * lambda_i_max
                options_tree.append(('model', 'scaling', 'z', label, lambda_i_scaling, description,'x'))
                lambda_dict[node] = lambda_i_scaling

                layer_count += 1
    else:
        message = 'unfamiliar method of distributing main tether tension among all of the tether elements (' + distribution_method + ')'
        print_op.log_and_raise_error(message)

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

        estimated_average_power = estimate_power(options, architecture, suppress_help_statement=True)
        estimated_inverse_time_period = estimated_average_power / corrected_estimated_energy  # yes, this = (1 / time_period_estimate)
        power_cost_factor = options['solver']['cost_factor']['power']
        power_cost = power_cost_factor * (1. / estimated_inverse_time_period)  # yes, this = pcf * time_period_estimate

    return lambda_scaling, corrected_estimated_energy, power_cost, estimated_average_power

def estimate_flight_radius(options, architecture, suppress_help_statement=False):
    thing_estimated = 'flight radius [m]'

    scaling_dict = {}
    synthesizing_dict = {}

    geometry = get_geometry(options)

    if architecture.number_of_kites == 1:
        length = options['solver']['initialization']['l_t']
        # max_cone_angle = options['solver']['initialization']['max_cone_angle_single']
    else:
        length = options['solver']['initialization']['theta']['l_s']
        # max_cone_angle = options['solver']['initialization']['max_cone_angle_multi']
    # cone_angle_rad = max_cone_angle * np.pi / 180.
    cone_angle_rad = options['solver']['initialization']['cone_deg'] * np.pi / 180.
    cone_radius = float(length * np.sin(cone_angle_rad))
    synthesizing_dict['cone'] = cone_radius

    airspeed = get_airspeed_average(options)
    if not vect_op.is_numeric_scalar(airspeed):
        groundspeed = options['solver']['initialization']['groundspeed']
        airspeed = groundspeed
    kite_standard = options['user_options']['kite_standard']
    aero_deriv, aero_validity = load_stability_derivatives(kite_standard)

    # assuming a level/horizontal turn, with the roll angle = bank angle
    coeff_bounds = options['model']['system_bounds']['x']['coeff']
    roll_angle = coeff_bounds[1][1]
    if not vect_op.is_numeric_scalar(roll_angle):
        roll_angle = 20.0 * np.pi / 180. #arbitrary
    gravity = options['model']['scaling']['other']['g']
    aircraft_3dof_radius = airspeed ** 2 / (gravity * np.tan(roll_angle))

    # assuming constant inflow, constant angle of attack, no sideslip, omega along aerodynamic body-fixed coordinates
    omega_bounds = options['model']['system_bounds']['x']['omega']
    p = omega_bounds[1][0] / 2.
    q = omega_bounds[1][1] / 2.
    r = omega_bounds[1][2] / 2.
    alpha = aero_validity['alpha_max_deg'] * np.pi / 180.
    if not vect_op.is_numeric_scalar(alpha):
        alpha = 0.
    cos = cas.cos(alpha)
    sin = cas.sin(alpha)
    aircraft_6dof_radius = airspeed / (q ** 2. + (r * cos - p * sin) ** 2.) ** 0.5

    kite_dof = get_kite_dof(options['user_options'])
    if int(kite_dof) == 6 and vect_op.is_numeric_scalar(aircraft_6dof_radius):
        synthesizing_dict['aircraft'] = aircraft_6dof_radius
    elif int(kite_dof) == 3 and vect_op.is_numeric_scalar(aircraft_3dof_radius):
        synthesizing_dict['aircraft'] = aircraft_3dof_radius

    b_ref = get_geometry(options)['b_ref']
    manual_anticollision_radius_scaling_factor = 2.
    anticollision_diameter = b_ref * options['model']['model_bounds']['anticollision']['safety_factor']
    anticollision_radius = anticollision_diameter / 2. * manual_anticollision_radius_scaling_factor
    if options['model']['model_bounds']['anticollision']['include']:
        synthesizing_dict['anticollision'] = anticollision_radius
    else:
        scaling_dict['anticollision'] = anticollision_radius

    acc_max = options['model']['model_bounds']['acceleration']['acc_max']
    gravity = options['model']['scaling']['other']['g']
    groundspeed = options['solver']['initialization']['groundspeed']
    centripetal_radius = groundspeed**2. / (acc_max * gravity)
    if options['model']['model_bounds']['acceleration']['include']:
        synthesizing_dict['centripetal'] = centripetal_radius
    else:
        scaling_dict['centripetal'] = centripetal_radius

    # if the loyd power = the momentum theory power, at 0 inclination/elevation angle:
    # P_loyd = (2/27) rho s_kite u^3 CL^3/CD^2
    # P_momentum = 4 a (1 - a)^2 (1/2 rho A_actuator u^3)
    # P_loyd = P_momentum ->  s_kite (2/27)(CL^3/CD^2) = 4a(1-a)^2 (1/2) (2 pi radius wingspan) ->
    # radius = (s_kite / (pi wingspan)) (2/27)(CL^3/CD^2) / (4a(1-a)^2) = (c_ref/pi) loyd/momentum
    CL = estimate_CL(options)
    CD = estimate_CD(options)
    loyd_factor = (2./27.) * (CL**3 / CD**2) * architecture.number_of_kites
    a_ref = options['model']['aero']['actuator']['a_ref']
    c_ref = geometry['c_ref']
    momentum_factor = 4. * a_ref * (1 - a_ref)**2.
    loyd_actuator_radius = (c_ref / np.pi) * loyd_factor / momentum_factor
    if not (options['user_options']['induction_model'] == 'not_in_use'):
        synthesizing_dict['loyd_actuator'] = loyd_actuator_radius
    else:
        scaling_dict['loyd_actuator'] = loyd_actuator_radius

    scaling_dict = transfer_synthesization_estimates_to_a_scaling_dictionary(scaling_dict, synthesizing_dict)

    method_in_options = options['model']['scaling']['other']['flight_radius_estimate']
    overwrite_method = None
    selected_method = select_scaling_method(method_in_options, overwrite_method, scaling_dict, thing_estimated)
    radius = scaling_dict[selected_method]

    print_help_with_scaling(options, scaling_dict, selected_method, thing_estimated, suppress_help_statement)

    return radius


def estimate_aero_force(options):

    overwrite_f_aero = options['model']['aero']['overwrite']['f_aero_rot']
    if overwrite_f_aero is not None:
        return vect_op.norm(overwrite_f_aero)

    geometry = get_geometry(options)
    s_ref = geometry['s_ref']

    CL = estimate_CL(options)

    airspeed_avg = get_airspeed_average(options)
    q_app = 0.5 * options['params']['atmosphere']['rho_ref'] * airspeed_avg ** 2

    aero_force = CL * q_app * s_ref
    return aero_force

def estimate_centripetal_force(options, architecture):

    geometry = get_geometry(options)
    m_k = geometry['m_k']
    groundspeed = options['solver']['initialization']['groundspeed']
    radius = estimate_flight_radius(options, architecture, suppress_help_statement=True)

    centripetal_force = m_k * groundspeed**2. / radius
    return centripetal_force


def get_airspeed_average(options):
    airspeed_limits = get_airspeed_limits(options)
    airspeed_avg = (airspeed_limits[0] * airspeed_limits[1])**0.5
    if not (options['model']['model_bounds']['airspeed']['include'] and vect_op.is_numeric_scalar(airspeed_avg)):
        u_altitude = get_u_at_altitude(options, estimate_altitude(options))
        groundspeed_init = options['solver']['initialization']['groundspeed']
        airspeed_avg = (groundspeed_init ** 2. + u_altitude ** 2.) ** 0.5

    return airspeed_avg


def estimate_power(options, architecture, suppress_help_statement=True):

    thing_estimated = 'power [W]'

    scaling_dict = {}
    synthesizing_dict = {}

    zz = estimate_altitude(options)
    uu = get_u_at_altitude(options, zz)
    qq = get_q_at_altitude(options, zz)
    power_density = uu * qq

    geometry = get_geometry(options)

    s_ref = geometry['s_ref']
    elevation_angle = options['solver']['initialization']['inclination_deg'] * np.pi / 180.
    CL = estimate_CL(options)
    CD = estimate_CD(options)
    p_loyd = perf_op.get_loyd_power(power_density, CL, CD, s_ref, elevation_angle) * architecture.number_of_kites
    synthesizing_dict['loyd'] = p_loyd

    a_ref = options['model']['aero']['actuator']['a_ref']
    thrust = get_momentum_theory_thrust(options, architecture)
    p_actuator = thrust * uu * (1. - a_ref) * float(len(architecture.layer_nodes))
    if not (options['user_options']['induction_model'] == 'not_in_use'):
        synthesizing_dict['actuator'] = p_actuator
    else:
        scaling_dict['actuator'] = p_actuator

    turbine_efficiency = options['params']['aero']['turbine_efficiency']
    kappa = options['model']['scaling']['x']['kappa']
    airspeed_avg = get_airspeed_average(options)
    p_drag_mode = turbine_efficiency * kappa * airspeed_avg**3. * architecture.number_of_kites
    if options['user_options']['trajectory']['system_type'] == 'drag_mode':
        synthesizing_dict['drag_mode'] = p_drag_mode
    else:
        scaling_dict['drag_mode'] = p_drag_mode

    scaling_dict = transfer_synthesization_estimates_to_a_scaling_dictionary(scaling_dict, synthesizing_dict)

    method_in_options = options['model']['scaling']['other']['power_estimate']
    overwrite_method = None
    selected_method = select_scaling_method(method_in_options, overwrite_method, scaling_dict, thing_estimated)
    print_help_with_scaling(options, scaling_dict, selected_method, thing_estimated, suppress_help_statement)

    power = scaling_dict[selected_method]
    #
    # induction_model = options['user_options']['induction_model']
    # if induction_model == 'not_in_use':
    #     induction_efficiency = 1.
    # else:
    #     induction_efficiency = 0.5
    #
    # kite_dof = get_kite_dof(options['user_options'])
    # if kite_dof == 3:
    #     dof_efficiency = 1.
    # elif kite_dof == 6:
    #     dof_efficiency = 0.5
    # else:
    #     message = 'something went wrong with the number of kite degrees of freedom (' + str(kite_dof) + ')'
    #     print_op.log_and_raise_error(message)
    #
    # power = p_loyd * induction_efficiency * dof_efficiency

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

    kite_dof = get_kite_dof(options['user_options'])
    if kite_dof == 3:
        coeff_bounds = options['model']['system_bounds']['x']['coeff']
        CL = coeff_bounds[1][0]
        if not vect_op.is_numeric_scalar(CL):
            CL = 1.5 #todo: this is arbitrary

    else:
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

    AR = kite_standard['geometry']['b_ref'] / kite_standard['geometry']['c_ref']

    kite_dof = get_kite_dof(options['user_options'])
    if kite_dof == 3 and ('CD' in aero_deriv.keys()):
        CL = estimate_CL(options)
        CD0 = aero_deriv['CD']['0'][0]
        return (CD0 + CL**2. / (np.pi * AR))

    alpha = aero_validity['alpha_max_deg'] * np.pi / 180.
    cos = cas.cos(alpha)
    sin = cas.sin(alpha)

    if ('CD' in aero_deriv.keys()):
        CD0 = aero_deriv['CD']['0'][0]
        CD = CD0 + aero_deriv['CD']['alpha'][0] * alpha

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

def get_tether_vector_tree(options, architecture):
    xhat = vect_op.xhat_np()
    zhat = vect_op.zhat_np()

    vector_tree = {}

    inclination_angle_rad = options['solver']['initialization']['inclination_deg'] * np.pi / 180.
    cone_angle_rad = options['solver']['initialization']['cone_deg'] * np.pi / 180.
    vector_tree[1] = np.cos(inclination_angle_rad) * xhat + np.sin(inclination_angle_rad) * zhat

    def continue_straight(node):
        parent = architecture.parent_map[node]
        parent_vector = vector_tree[parent]
        return parent_vector

    def branch_at_cone_angle(node):
        parent = architecture.parent_map[node]
        parent_vector = vector_tree[parent]

        sibling_list = architecture.children_map[architecture.parent_map[node]]
        number_siblings = len(sibling_list)
        sibling_index = sibling_list.index(node)

        psi = 2. * np.pi * float(sibling_index) / float(number_siblings)
        bhat = vect_op.normed_cross(zhat, parent_vector)
        chat = vect_op.normed_cross(parent_vector, bhat)
        rhat = np.cos(psi) * (-1.) * chat + np.sin(psi) * bhat
        ehat = np.cos(cone_angle_rad) * parent_vector + np.sin(cone_angle_rad) * rhat
        return ehat

    for node in range(2, architecture.number_of_nodes):
        number_siblings = architecture.get_number_children(architecture.parent_map[node])
        if number_siblings == 1:
            vector_tree[node] = continue_straight(node)
        else:
            vector_tree[node] = branch_at_cone_angle(node)
    #
    #
    # vector_tree[2] = continue_straight(2)
    # vector_tree[3] = continue_straight(3)
    # vector_tree[4] = branch_at_cone_angle(4)
    # vector_tree[5] = branch_at_cone_angle(5)
    # vector_tree[6] = branch_at_cone_angle(6)
    # vector_tree[7] = branch_at_cone_angle(7)
    # vector_tree[9] = branch_at_cone_angle(9)
    # vector_tree[10] = branch_at_cone_angle(10)

    return vector_tree

def estimate_main_tether_tension_per_unit_length(options, architecture, suppress_help_statement=False):

    thing_estimated = 'main tether tension [N]'

    scaling_dict = {}
    synthesizing_dict = {}
    tension_acts_on = {}

    power = estimate_power(options, architecture, suppress_help_statement=True)
    reelout_speed = estimate_reelout_speed(options)
    tension_estimate_via_power = float(power/reelout_speed)
    synthesizing_dict['power'] = tension_estimate_via_power
    tension_acts_on['power'] = 'ground'

    tether_vector_tree = get_tether_vector_tree(options, architecture)
    aero_force_per_kite = estimate_aero_force(options)
    total_mass, _ = estimate_total_mass(options, architecture)
    gravity = options['model']['scaling']['other']['g']
    total_force_vector = total_mass * gravity * (-1 * vect_op.zhat_np())
    for kite in architecture.kite_nodes:
        total_force_vector = total_force_vector + (aero_force_per_kite * tether_vector_tree[kite])
    tension_estimate_via_force_summation = cas.mtimes(tether_vector_tree[1].T, total_force_vector)
    synthesizing_dict['force_summation'] = np.abs(float(tension_estimate_via_force_summation))
    tension_acts_on['force_summation'] = 'ground'

    ct_thrust = get_momentum_theory_thrust(options, architecture) * float(architecture.layers)
    if not options['user_options']['induction_model'] == 'not_in_use':
        synthesizing_dict['thrust_coeff'] = ct_thrust
    else:
        scaling_dict['thrust_coeff'] = ct_thrust
    tension_acts_on['thrust_coeff'] = 'ground'

    tension_estimate_via_min_force = options['params']['model_bounds']['tether_force_limits'][0]
    tension_estimate_via_max_force = options['params']['model_bounds']['tether_force_limits'][1]
    tension_via_average_force = (tension_estimate_via_min_force + tension_estimate_via_max_force) / 2.
    scaling_dict['average_force'] = tension_via_average_force
    tension_acts_on['average_force'] = 'ground'

    max_stress = options['params']['tether']['max_stress'] / options['params']['tether']['stress_safety_factor']
    diam_t = options['solver']['initialization']['theta']['diam_t']
    cross_sectional_area_t = np.pi * (diam_t / 2.) ** 2.
    tension_via_max_stress = max_stress * cross_sectional_area_t
    scaling_dict['max_stress'] = tension_via_max_stress
    tension_acts_on['max_stress'] = 'ground'

    if options['model']['model_bounds']['tether_force']['include'] == True:
        synthesizing_dict['material_limits'] = tension_via_average_force
    elif options['model']['model_bounds']['tether_stress']['include'] == True:
        synthesizing_dict['material_limits'] = tension_via_max_stress
    tension_acts_on['material_limits'] = 'ground'

    scaling_dict = transfer_synthesization_estimates_to_a_scaling_dictionary(scaling_dict, synthesizing_dict)
    tension_acts_on['synthesized'] = 'ground'

    method_in_options = options['model']['scaling']['other']['tension_estimate']
    overwrite_method = None
    selected_method = select_scaling_method(method_in_options, overwrite_method, scaling_dict, thing_estimated)

    tension = scaling_dict[selected_method]
    length = options['solver']['initialization']['l_t']
    multiplier = tension / length

    print_help_with_scaling(options, scaling_dict, selected_method, thing_estimated, suppress_help_statement)

    if options['model']['scaling']['other']['print_help_with_scaling'] and not suppress_help_statement:
        table_name = thing_estimated + ' estimates correspond to following power estimates:'
        power_estimate_dict = {}
        for name, val in scaling_dict.items():
            power_estimate_dict[name] = val * reelout_speed
        sorted_scaling_dict = dict(sorted(power_estimate_dict.items(), key=lambda item: float(vect_op.norm(item[1]))))
        print_op.print_dict_as_table(sorted_scaling_dict, level='debug', caption=table_name)

    return multiplier


def estimate_total_mass(options, architecture):

    mass_of_all_kites = get_geometry(options)['m_k'] * architecture.number_of_kites

    tether_mass_tree = {}

    diam_t = options['solver']['initialization']['theta']['diam_t']
    rho_tether = options['params']['tether']['rho']
    cross_sectional_area_t = np.pi * (diam_t / 2.) ** 2.
    length = options['solver']['initialization']['l_t']
    mass_of_main_tether = cross_sectional_area_t * length * rho_tether
    tether_mass_tree[1] = mass_of_main_tether

    if architecture.kite_nodes != [1]:
        diam_s = options['solver']['initialization']['theta']['diam_s']
        cross_sectional_area_s = np.pi * (diam_s / 2.) ** 2.
        length_s = options['solver']['initialization']['theta']['l_s']
        mass_of_secondary_tether = cross_sectional_area_s * length_s * rho_tether * architecture.number_of_kites

        for kite in set(architecture.kite_nodes) - set([1]):
            tether_mass_tree[kite] = cross_sectional_area_s * length_s * rho_tether

    else:
        mass_of_secondary_tether = 0.

    number_of_intermediate_tethers = architecture.get_number_intermediate_tethers()
    if number_of_intermediate_tethers > 0:
        diam_i = options['solver']['initialization']['theta']['diam_i']
        cross_sectional_area_i = np.pi * (diam_i / 2.) ** 2.
        length_i = options['solver']['initialization']['theta']['l_i']
        mass_of_intermediate_tether = cross_sectional_area_i * length_i * rho_tether * number_of_intermediate_tethers

        for node in set(range(2, architecture.number_of_nodes)) - set(architecture.kite_nodes):
            tether_mass_tree[node] = cross_sectional_area_i * length_i * rho_tether

    else:
        mass_of_intermediate_tether = 0.

    total_mass = mass_of_all_kites + mass_of_main_tether + mass_of_secondary_tether + mass_of_intermediate_tether

    comparison = mass_of_all_kites + np.sum(np.array([val for val in tether_mass_tree.values()]))
    if np.abs(total_mass - comparison) > 0.001:
        message = 'something went wrong while estimating the total system mass in model_funcs'
        print_op.log_and_raise_error(message)

    return total_mass, tether_mass_tree

def estimate_energy(options, architecture):
    power = estimate_power(options, architecture, suppress_help_statement=True)
    time_period = estimate_time_period(options, architecture, suppress_help_statement=True)
    energy = power * time_period
    return energy

def estimate_time_period(options, architecture, suppress_help_statement=True):

    thing_estimated = "single winding period [s]"

    if 't_f' in options['user_options']['trajectory']['fixed_params']:
        return options['user_options']['trajectory']['fixed_params']['t_f']

    windings = options['user_options']['trajectory']['lift_mode']['windings']
    tf_bounds = options['model']['system_bounds']['theta']['t_f']

    scaling_dict = {}
    synthesizing_dict = {}

    # period from time bounds
    period1_from_tf_bounds = (tf_bounds[0] + tf_bounds[1]) / 2. / windings
    scaling_dict['t_f_bounds'] = period1_from_tf_bounds

    # period from groundspeed initialization
    groundspeed_init = options['solver']['initialization']['groundspeed']
    radius = estimate_flight_radius(options, architecture, suppress_help_statement=True)
    period1_from_groundspeed_initialization = float((2. * np.pi * radius) / groundspeed_init)
    val = period1_from_groundspeed_initialization
    if vect_op.is_numeric_scalar(val) and val > tf_bounds[0] and val < tf_bounds[1]:
        synthesizing_dict['groundspeed_init'] = period1_from_groundspeed_initialization
    else:
        scaling_dict['groundspeed_init'] = period1_from_groundspeed_initialization

    # period_from_groundspeed_bounds
    dq_bounds = options['model']['system_bounds']['x']['dq']
    avg_groundspeed_max = np.average(dq_bounds[1])
    period1_from_groundspeed_bounds = float((2. * np.pi * radius) / avg_groundspeed_max)
    val = period1_from_groundspeed_bounds
    if vect_op.is_numeric_scalar(val) and val > tf_bounds[0] and val < tf_bounds[1]:
        synthesizing_dict['groundspeed_bounds'] = period1_from_groundspeed_bounds

    # period from assuming that the maximum acceleration is in the centripetal direction
    # r omega^2 = acc_max * gravity -> omega^2 = acc_max * gravity / radius
    # and omega = 2 pi / T
    gravity = options['model']['scaling']['other']['g']
    acc_max = options['model']['model_bounds']['acceleration']['acc_max']
    omega_squared = acc_max * gravity / radius
    omega_from_max_acc = omega_squared**0.5
    period1_from_max_acceleration = float(2. * np.pi / omega_from_max_acc)
    if options['model']['model_bounds']['acceleration']['include']:
        synthesizing_dict['max_acceleration'] = period1_from_max_acceleration
    else:
        scaling_dict['max_acceleration'] = period1_from_max_acceleration

    # period from natural frequency of an approximate pendulum made with a rigid rod length of outermost tether
    if architecture.number_of_kites == 1:
        length = options['solver']['initialization']['l_t']
    else:
        length = options['solver']['initialization']['theta']['l_s']
    period1_from_pendulum = float(2. * np.pi * (length / gravity)**0.5)
    scaling_dict['pendulum'] = period1_from_pendulum

    # period for convection distance in one winding to be "large" compared to (2 radius) for hawt "quasi-steady" inflow
    # u_conv * T = strouhal * "diameter"
    strouhal_approx = options['model']['aero']['induction']['strouhal_scaling']
    u_altitude = get_u_at_altitude(options, estimate_altitude(options))
    period1_from_convection = float(2. * radius / u_altitude / strouhal_approx)
    if not options['user_options']['induction_model'] == 'not_in_use':
        synthesizing_dict['convection'] = period1_from_convection
    else:
        scaling_dict['convection'] = period1_from_convection

    # period from angular velocity bounds
    kite_dof = get_kite_dof(options['user_options'])
    omega_bounds = options['model']['system_bounds']['x']['omega']
    omega_about_kite_z_axis = omega_bounds[1][2]
    period1_from_ang_velocity_bounds = float((2. * np.pi) / omega_about_kite_z_axis)
    if int(kite_dof) == 6:
        synthesizing_dict['angular_velocity_bounds'] = period1_from_ang_velocity_bounds
    else:
        scaling_dict['angular_velocity_bounds'] = period1_from_ang_velocity_bounds

    kite_standard = options['user_options']['kite_standard']
    aero_deriv, aero_validity = load_stability_derivatives(kite_standard)
    if options['model']['aero']['overwrite']['beta_max_deg'] is not None:
        beta_max = options['model']['aero']['overwrite']['beta_max_deg'] * np.pi / 180.
    elif aero_validity['beta_max_deg'] is not None:
        beta_max = aero_validity['beta_max_deg'] * np.pi / 180.
    else:
        beta_max = 10. * np.pi / 180.
    inclination_angle_rad = options['solver']['initialization']['inclination_deg'] * np.pi / 180.
    omega_from_beta = u_altitude * np.sin(inclination_angle_rad) / (beta_max * radius)
    period1_from_beta_max = float((2. * np.pi) / omega_from_beta)
    if options['model']['model_bounds']['aero_validity']['include']:
        synthesizing_dict['sideslip_max'] = period1_from_beta_max
    else:
        scaling_dict['sideslip_max'] = period1_from_beta_max

    scaling_dict = transfer_synthesization_estimates_to_a_scaling_dictionary(scaling_dict, synthesizing_dict)

    method_in_options = options['model']['scaling']['other']['period_estimate']
    overwrite_method = None
    selected_method = select_scaling_method(method_in_options, overwrite_method, scaling_dict, thing_estimated)
    value = scaling_dict[selected_method]

    print_help_with_scaling(options, scaling_dict, selected_method, thing_estimated, suppress_help_statement)

    optimization_period = value * windings

    return optimization_period




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
