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
define the structure of an algebtraic-representation vortex-modelled wake
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2022
'''
import copy
import pdb

import casadi.tools as cas
import numpy as np
import matplotlib.pyplot as plt

import awebox.mdl.aero.induction_dir.vortex_dir.tools as vortex_tools

import awebox.mdl.architecture as archi
import awebox.mdl.wind as wind_module
import awebox.tools.struct_operations as struct_op
import awebox.tools.vector_operations as vect_op
import awebox.tools.print_operations as print_op

from awebox.logger.logger import Logger as awelogger


# get the names of the specific variables, for use in model_system and in local wake objects
# remember, that the presence of the wake-node-variables in model_variables is not itself the wake objects,
# the wake objects are constructed *with* those variables.
# which means that each of the vortex elements induced velocities depends on the wake structure,
# but not the positions and strengths; those belong to the wake nodes

def get_define_wake_types():
    return ['bound', 'near', 'far']

def get_wake_node_position_si(variables, kite_shed, tip, wake_node, scaling=None):
    var_name = get_wake_node_position_name(kite_shed, tip, wake_node)
    return vortex_tools.get_variable_si(variables, 'xl', var_name, scaling)

def get_vortex_ring_strength_si(variables, kite_shed, ring, scaling=None):
    var_name = get_vortex_ring_strength_name(kite_shed, ring)
    return vortex_tools.get_variable_si(variables, 'xl', var_name, scaling)

def get_element_induced_velocity_si(variables, wake_type, element_type, element_number, kite_obs, scaling=None):
    var_name = get_element_induced_velocity_name(wake_type, element_type, element_number, kite_obs)
    return vortex_tools.get_variable_si(variables, 'xl', var_name, scaling)


def get_induced_velocity_at_kite_si(variables, kite_obs, scaling=None):
    var_name = get_induced_velocity_at_kite_name(kite_obs)
    return vortex_tools.get_variable_si(variables, 'xl', var_name, scaling)

def get_far_wake_finite_filament_pathwise_convection_velocity_si(variables, kite_shed, scaling=None):
    var_name = get_far_wake_finite_filament_pathwise_convection_velocity_name(kite_shed)
    return vortex_tools.get_variable_si(variables, 'xl', var_name, scaling)

def get_far_wake_cylinder_center_position_si(variables, parent_shed, scaling=None):
    var_name = get_far_wake_cylinder_center_position_name(parent_shed)
    return vortex_tools.get_variable_si(variables, 'xl', var_name, scaling)

def get_far_wake_cylinder_pitch_si(variables, parent_shed, scaling=None):
    var_name = get_far_wake_cylinder_pitch_name(parent_shed)
    return vortex_tools.get_variable_si(variables, 'xl', var_name, scaling)


def get_var_name(abbreviated_var_name, kite_shed_or_parent_shed=None, tip=None, wake_node_or_ring=None):
    if abbreviated_var_name == 'wx':
        return get_wake_node_position_name(kite_shed=kite_shed_or_parent_shed, tip=tip, wake_node=wake_node_or_ring)
    elif abbreviated_var_name == 'wx_center':
        return get_far_wake_cylinder_center_position_name(parent_shed=kite_shed_or_parent_shed)
    elif abbreviated_var_name == 'wg':
        return get_vortex_ring_strength_name(kite_shed=kite_shed_or_parent_shed, ring=wake_node_or_ring)
    elif abbreviated_var_name == 'wh':
        return get_far_wake_cylinder_pitch_name(parent_shed=kite_shed_or_parent_shed)
    else:
        message = 'get_var_name function is not set up for this abbreviation (' + abbreviated_var_name + ') yet.'
        awelogger.logger.error(message)
        raise Exception(message)

def get_wake_node_position_name(kite_shed, tip, wake_node):
    var_name = 'wx_' + str(kite_shed) + '_' + tip + '_' + str(wake_node)
    return var_name

def get_vortex_ring_strength_name(kite_shed, ring):
    var_name = 'wg_' + str(kite_shed) + '_' + str(ring)
    return var_name

def get_element_induced_velocity_name(wake_type, element_type, element_number, kite_obs):
    var_name = 'wu_' + wake_type + '_' + element_type + '_' + str(element_number) + '_' + str(kite_obs)
    return var_name

def get_induced_velocity_at_kite_name(kite_obs):
    var_name = 'wu_ind_' + str(kite_obs)
    return var_name

def get_far_wake_finite_filament_pathwise_convection_velocity_name(kite_shed):
    var_name = 'wu_pathwise_' + str(kite_shed)
    return var_name

def get_far_wake_cylinder_center_position_name(parent_shed):
    var_name = 'wx_center_' + str(parent_shed)
    return var_name

def get_far_wake_cylinder_pitch_name(parent_shed):
    var_name = 'wh_' + str(parent_shed)
    return var_name

def get_expected_number_of_elements_dict_for_wake_types(options, architecture):
    expected_number_of_elements_dict_for_wake_types = {
        'bound': get_expected_number_of_bound_wake_elements_dict(architecture),
        'near': get_expected_number_of_near_wake_elements_dict(options, architecture),
        'far': get_expected_number_of_far_wake_elements_dict(options, architecture)
    }
    return expected_number_of_elements_dict_for_wake_types

def get_expected_number_of_bound_wake_elements_dict(architecture):
    number_of_kites = architecture.number_of_kites
    expected_dict = {'finite_filament': 1 * number_of_kites}
    return expected_dict

def get_expected_number_of_near_wake_elements_dict(options, architecture):
    number_of_kites = architecture.number_of_kites
    wake_nodes = vortex_tools.get_option_from_possible_dicts(options, 'wake_nodes')
    expected_dict = {'finite_filament': 3 * number_of_kites * (wake_nodes - 1)}
    return expected_dict

def get_expected_number_of_far_wake_elements_dict(options, architecture):
    far_wake_element_type = vortex_tools.get_option_from_possible_dicts(options, 'far_wake_element_type')
    if far_wake_element_type == 'finite_filament':
        expected_dict = get_expected_number_of_finite_filament_far_wake_elements(architecture)
    elif far_wake_element_type == 'semi_infinite_filament':
        expected_dict = get_expected_number_of_semi_infinite_filament_far_wake_elements(architecture)
    elif far_wake_element_type == 'semi_infinite_cylinder':
        expected_dict = get_expected_number_of_semi_infinite_cylinder_far_wake_elements(architecture)
    else:
        message = 'unexpected type of far-wake vortex element (' + far_wake_element_type + '). maybe, check your spelling?'
        awelogger.logger.error(message)
        raise Exception(message)

    return expected_dict

def get_expected_number_of_semi_infinite_cylinder_far_wake_elements(architecture):
    number_of_kites = architecture.number_of_kites
    expected_dict = {'semi_infinite_tangential_cylinder': 2 * number_of_kites,
                     'semi_infinite_longitudinal_cylinder': 2 * number_of_kites}
    return expected_dict

def get_expected_number_of_finite_filament_far_wake_elements(architecture):
    number_of_kites = architecture.number_of_kites
    expected_dict = {'finite_filament': 2 * number_of_kites}
    return expected_dict

def get_expected_number_of_semi_infinite_filament_far_wake_elements(architecture):
    number_of_kites = architecture.number_of_kites
    expected_dict = {'semi_infinite_filament': 2 * number_of_kites}
    return expected_dict

def extend_system_variables(options, system_lifted, system_states, architecture):

    # the part that describes the wake nodes and consequent vortex rings
    wingtips = ['ext', 'int']
    wake_nodes = options['aero']['vortex']['wake_nodes']
    rings = options['aero']['vortex']['rings']

    for kite_shed in architecture.kite_nodes:
        for wake_node in range(wake_nodes):
            for tip in wingtips:
                var_name = get_wake_node_position_name(kite_shed, tip, wake_node)
                system_lifted.extend([(var_name, (3, 1))])

        for ring in range(rings):
            var_name = get_vortex_ring_strength_name(kite_shed, ring)
            system_lifted.extend([(var_name, (1, 1))])

    far_wake_element_type = options['aero']['vortex']['far_wake_element_type']
    if (far_wake_element_type == 'semi_infinite_cylinder'):
        for parent_shed in set([architecture.parent_map[kite] for kite in architecture.kite_nodes]):
            var_name = get_far_wake_cylinder_center_position_name(parent_shed=parent_shed)
            system_lifted.extend([(var_name, (3, 1))])

            var_name = get_far_wake_cylinder_pitch_name(parent_shed=parent_shed)
            system_lifted.extend([(var_name, (1, 1))])

    # induced velocity part: the part that depends on the wake types and wake structure
    expected_number_of_elements_dict_for_wake_types = get_expected_number_of_elements_dict_for_wake_types(options, architecture)

    for kite_obs in architecture.kite_nodes:
        var_name = get_induced_velocity_at_kite_name(kite_obs)
        system_lifted.extend([(var_name, (3, 1))])

        for wake_type, local_expected_number_of_elements_dict in expected_number_of_elements_dict_for_wake_types.items():
            for element_type, expected_number in local_expected_number_of_elements_dict.items():
                for element_number in range(expected_number):
                    var_name = get_element_induced_velocity_name(wake_type, element_type, element_number, kite_obs)
                    system_lifted.extend([(var_name, (3, 1))])

    return system_lifted, system_states


def get_total_number_of_vortex_elements(options, architecture):

    expected_number_of_elements_dict_for_wake_types = get_expected_number_of_elements_dict_for_wake_types(options,
                                                                                                          architecture)

    total_number = 0
    for local_dict in expected_number_of_elements_dict_for_wake_types.values():
        for count in local_dict.values():
            total_number += count

    return total_number

def construct_test_model_variable_structures(element_type='finite_filament'):

    options = {}

    options['wind'] = {}
    options['wind']['u_ref'] = 1.
    options['wind']['model'] = 'uniform'
    options['wind']['z_ref'] = -999.
    options['wind']['log_wind'] = {'z0_air': -999}
    options['wind']['power_wind'] = {'exp_ref': -999}

    wake_nodes = 2
    rings = wake_nodes
    options['aero'] = {}
    options['aero']['vortex'] = {}
    options['aero']['vortex']['wake_nodes'] = wake_nodes
    options['aero']['vortex']['rings'] = rings
    options['aero']['vortex']['core_to_chord_ratio'] = 0.1
    options['aero']['vortex']['far_wake_element_type'] = element_type

    options['induction'] = {}
    options['induction']['vortex_wake_nodes'] = wake_nodes
    options['induction']['vortex_rings'] = rings
    options['induction']['vortex_far_wake_convection_time'] = 1.
    options['induction']['vortex_far_wake_element_type'] = element_type
    options['induction']['vortex_representation'] = 'alg'
    options['induction']['vortex_epsilon_m'] = 1.0e-8
    options['induction']['vortex_epsilon_r'] = 1.0e-8

    options['n_k'] = 10
    options['collocation'] = {'d':4, 'scheme':'radau'}
    options['discretization'] = 'direct_collocation'
    options['phase_fix'] = 'single_reelout'
    options['phase_fix_reelout'] = 0.5

    architecture = archi.Architecture({1: 0})

    system_lifted, system_states = extend_system_variables(options, [], [], architecture)

    for kite in architecture.kite_nodes:
        parent = architecture.parent_map[kite]
        system_states.extend([('q' + str(kite) + str(parent), (3, 1))])

    system_variable_list = {}
    system_variable_list['xl'] = system_lifted
    system_variable_list['xd'] = system_states

    var_struct, variables_dict = struct_op.generate_variable_struct(system_variable_list)

    geometry_struct = cas.struct([
        cas.entry("c_ref", shape=(1, 1))
    ])
    wind_struct = cas.struct([
        cas.entry('u_ref', shape=(1, 1)),
        cas.entry('z_ref', shape=(1, 1))
    ])
    theta_struct = cas.struct_symSX([
        cas.entry('geometry', struct=geometry_struct),
        cas.entry('wind', struct=wind_struct)
    ])
    param_struct = cas.struct_symSX([
        cas.entry('theta0', struct=theta_struct),

    ])

    wind_params = param_struct(0.)
    wind_params['theta0', 'wind', 'u_ref'] = options['wind']['u_ref']
    wind_params['theta0', 'wind', 'z_ref'] = options['wind']['z_ref']

    wind = wind_module.Wind(options['wind'], wind_params)

    return options, architecture, wind, var_struct, param_struct, variables_dict

def construct_straight_flight_test_object(element_type='finite_filament'):
    options, architecture, wind, var_struct, param_struct, variables_dict = construct_test_model_variable_structures(
        element_type)
    kite = architecture.kite_nodes[0]

    half_b_span_vec = 0.5 * vect_op.yhat_dm()
    x_kite = cas.DM.zeros((3, 1))

    x_NE = x_kite - half_b_span_vec
    x_PE = x_kite + half_b_span_vec

    variables_si = var_struct(0.)

    variables_si['xl', get_wake_node_position_name(kite, 'ext', 0)] = x_PE
    variables_si['xl', get_wake_node_position_name(kite, 'int', 0)] = x_NE
    variables_si['xl', get_wake_node_position_name(kite, 'ext', 1)] = x_PE + vect_op.xhat_dm()
    variables_si['xl', get_wake_node_position_name(kite, 'int', 1)] = x_NE + vect_op.xhat_dm()
    variables_si['xl', get_vortex_ring_strength_name(kite, 0)] = cas.DM(4.)
    variables_si['xl', get_vortex_ring_strength_name(kite, 1)] = cas.DM(1.)

    parameters = param_struct(0.)
    parameters['theta0', 'geometry', 'c_ref'] = 0.1

    return options, architecture, wind, var_struct, param_struct, variables_dict, variables_si, parameters

def construct_vortex_ring_test_object(element_type='semi_infinite_filament'):
    options, architecture, wind, var_struct, param_struct, variables_dict = construct_test_model_variable_structures(
        element_type)
    kite = architecture.kite_nodes[0]

    half_b_span_vec = 0.5 * vect_op.yhat_dm()
    x_kite = cas.DM.zeros((3, 1))

    x_NE = x_kite - half_b_span_vec
    x_PE = x_kite + half_b_span_vec

    variables_si = var_struct(0.)

    variables_si['xl', get_wake_node_position_name(kite, 'ext', 0)] = x_PE
    variables_si['xl', get_wake_node_position_name(kite, 'int', 0)] = x_NE
    variables_si['xl', get_wake_node_position_name(kite, 'ext', 1)] = x_PE + vect_op.xhat_dm()
    variables_si['xl', get_wake_node_position_name(kite, 'int', 1)] = x_NE + vect_op.xhat_dm()
    variables_si['xl', get_vortex_ring_strength_name(kite, 0)] = cas.DM(4.)
    variables_si['xl', get_vortex_ring_strength_name(kite, 1)] = cas.DM(0.)

    parameters = param_struct(0.)
    parameters['theta0', 'geometry', 'c_ref'] = 0.1

    return options, architecture, wind, var_struct, param_struct, variables_dict, variables_si, parameters


def construct_circular_flight_test_object(element_type='semi_infinite_cylinder'):
    options, architecture, wind, var_struct, param_struct, variables_dict = construct_test_model_variable_structures(
        element_type)
    kite = architecture.kite_nodes[0]
    parent = architecture.parent_map[kite]

    x_center = cas.DM.zeros((3, 1))
    a_hat = vect_op.yhat()
    b_hat = vect_op.zhat()

    vec_u_wind = wind.get_velocity(1.)

    omega = 2. * np.pi * 1.e8
    delta_t = 1.

    pitch = cas.DM(2. * np.pi * vect_op.norm(vec_u_wind) / omega)  # -> 1.e-8

    theta0 = 0.
    theta1 = -1. * omega * delta_t

    b_span = 1.
    radius_kite = 2.
    r_ext = radius_kite + b_span / 2.
    r_int = radius_kite - b_span / 2.

    x_center0 = x_center
    x_center1 = x_center0 + delta_t * vec_u_wind

    x_PE_0 = x_center0 + r_ext * (np.sin(theta0) * a_hat + np.cos(theta0) * b_hat)
    x_PE_1 = x_center1 + r_ext * (np.sin(theta1) * a_hat + np.cos(theta1) * b_hat)

    x_NE_0 = x_center0 + r_int * (np.sin(theta0) * a_hat + np.cos(theta0) * b_hat)
    x_NE_1 = x_center1 + r_int * (np.sin(theta1) * a_hat + np.cos(theta1) * b_hat)

    variables_si = var_struct(0.)

    variables_si['xl', get_wake_node_position_name(kite, 'ext', 0)] = x_PE_0
    variables_si['xl', get_wake_node_position_name(kite, 'int', 0)] = x_NE_0
    variables_si['xl', get_wake_node_position_name(kite, 'ext', 1)] = x_PE_1
    variables_si['xl', get_wake_node_position_name(kite, 'int', 1)] = x_NE_1
    variables_si['xl', get_vortex_ring_strength_name(kite, 0)] = cas.DM(4.)
    variables_si['xl', get_vortex_ring_strength_name(kite, 1)] = cas.DM(1.)

    variables_si['xl', get_far_wake_cylinder_center_position_name(parent)] = x_center
    variables_si['xl', get_far_wake_cylinder_pitch_name(parent)] = pitch

    parameters = param_struct(0.)
    parameters['theta0', 'geometry', 'c_ref'] = 0.1

    return options, architecture, wind, var_struct, param_struct, variables_dict, variables_si, parameters