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
various structural tools for the vortex model
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2019-2021
'''
import pdb

import casadi.tools as cas
import numpy as np
from awebox.logger.logger import Logger as awelogger

import awebox.mdl.aero.induction_dir.general_dir.flow as general_flow
import awebox.mdl.aero.induction_dir.general_dir.tools as general_tools

import awebox.tools.struct_operations as struct_op
import awebox.tools.vector_operations as vect_op
import awebox.tools.print_operations as print_op

def get_list_of_abbreviated_variables(model_options):
    abbreviated_variables = ['wx', 'wg']
    far_wake_element_type = general_tools.get_option_from_possible_dicts(model_options, 'far_wake_element_type',
                                                                         'vortex')
    if far_wake_element_type == 'semi_infinite_right_cylinder':
        abbreviated_variables += ['wx_center', 'wh']

    return abbreviated_variables


def extend_system_variables(model_options, system_lifted, system_states, architecture):

    abbreviated_variables = get_list_of_abbreviated_variables(model_options)

    for abbreviated_var_name in abbreviated_variables:
        system_lifted, system_states = extend_specific_geometric_variable(abbreviated_var_name, model_options, system_lifted, system_states,
                                           architecture)

    system_lifted, system_states = extend_velocity_variables(model_options, system_lifted, system_states, architecture)

    return system_lifted, system_states


def model_is_included_in_comparison(options):
    comparison_labels = general_tools.get_option_from_possible_dicts(options, 'comparison_labels', 'vortex')
    any_vor = any(label[:3] == 'vor' for label in comparison_labels)
    return any_vor

def bound_element_number_corresponds_to_bound_vortex_on_observer_kite(model_options, architecture, kite_obs, element_number):
    element_type = 'finite_filament'
    wake_type = 'bound'
    kite_shed = get_shedding_kite_from_element_number(model_options, wake_type, element_type, element_number,
                                                      architecture)
    if kite_shed == kite_obs:
        return True
    else:
        return False


def model_includes_induced_velocity_from_kite_bound_on_itself(model_options, variables_si, architecture):
    # start with kite_obs
    kite_obs = architecture.kite_nodes[0]

    # figure out what the element number is that corresponds to the bound vortex on that kite
    element_type = 'finite_filament'
    wake_type = 'bound'
    expected_number = get_expected_number_of_bound_wake_elements_dict(architecture)[element_type]
    for element_number in range(expected_number):
        if bound_element_number_corresponds_to_bound_vortex_on_observer_kite(model_options, architecture, kite_obs, element_number):
            element_number_at_kites_own_bound = element_number

    # get the hypothetical variable name for the induced velocity from that element number on the kite_obs
    hypothetical_name = get_element_induced_velocity_name(wake_type, element_type, element_number_at_kites_own_bound, kite_obs)

    # is that variable name in the list of variables
    hypothetical_label = '[,z,' + hypothetical_name + ',0]'
    return hypothetical_label in variables_si.labels()


def get_number_of_algebraic_variables_set_outside_dynamics(nlp_options, model):

    count = 0

    if (nlp_options['induction']['induction_model'] == 'not_in_use'):
        return count
    if not model_is_included_in_comparison(nlp_options):
        return count

    abbreviated_variables = get_list_of_abbreviated_variables(nlp_options)

    for abbreviated_var_name in abbreviated_variables:
        kite_shed_or_parent_shed_list, tip_list, wake_node_or_ring_list = get_kite_or_parent_and_tip_and_node_or_ring_list_for_abbreviated_vars(
            abbreviated_var_name, nlp_options, model.architecture)

        for kite_shed_or_parent_shed in kite_shed_or_parent_shed_list:
            for tip in tip_list:
                for wake_node_or_ring in wake_node_or_ring_list:

                    if abbreviated_var_name[:2] == 'wx':
                        count += 3

                    elif abbreviated_var_name[:2] == 'wg':
                        count += 1

                    elif abbreviated_var_name[:2] == 'wh':
                        count += 1

                    else:
                        message = 'unexpected abbreviated_var_name (' + abbreviated_var_name + ')'
                        print_op.log_and_raise_error(message)
    return count


def extend_specific_geometric_variable(abbreviated_var_name, model_options, system_lifted, system_states, architecture):
    kite_shed_or_parent_shed_list, tip_list, wake_node_or_ring_list = get_kite_or_parent_and_tip_and_node_or_ring_list_for_abbreviated_vars(
        abbreviated_var_name, model_options, architecture)

    if abbreviated_var_name == 'wx':
        var_type = get_wake_node_position_var_type(model_options)
    else:
        var_type = 'z'

    if abbreviated_var_name[:2] == 'wx':
        var_shape = (3, 1)
    else:
        var_shape = (1, 1)

    for kite_shed_or_parent_shed in kite_shed_or_parent_shed_list:
        for tip in tip_list:
            for wake_node_or_ring in wake_node_or_ring_list:

                var_name = get_var_name(abbreviated_var_name, kite_shed_or_parent_shed=kite_shed_or_parent_shed, tip=tip, wake_node_or_ring=wake_node_or_ring)

                if var_type == 'z':
                    system_lifted.extend([(var_name, var_shape)])
                elif var_type == 'x':
                    system_states.extend([(var_name, var_shape)])
                    system_states.extend([('d' + var_name, var_shape)])

                else:
                    message = 'unexpected variable type: (' + var_type + ')'
                    print_op.log_and_raise_error(message)

    return system_lifted, system_states

def extend_velocity_variables(model_options, system_lifted, system_states, architecture):

    # induced velocity part: the part that depends on the wake types and wake structure
    expected_number_of_elements_dict_for_wake_types = get_expected_number_of_elements_dict_for_wake_types(model_options, architecture)
    biot_savart_residual_assembly = model_options['aero']['vortex']['biot_savart_residual_assembly']

    for kite_obs in architecture.kite_nodes:
        var_name = get_induced_velocity_at_kite_name(kite_obs)
        system_lifted.extend([(var_name, (3, 1))])

        for wake_type, local_expected_number_of_elements_dict in expected_number_of_elements_dict_for_wake_types.items():
            for element_type, expected_number in local_expected_number_of_elements_dict.items():
                for element_number in range(expected_number):

                    kite_shed = get_shedding_kite_from_element_number(model_options, wake_type, element_type, element_number, architecture)
                    if not (wake_type == 'bound' and kite_obs == kite_shed):

                        var_name = get_element_induced_velocity_name(wake_type, element_type, element_number, kite_obs)
                        system_lifted.extend([(var_name, (3, 1))])

                        if biot_savart_residual_assembly == 'lifted':
                            var_name = get_element_biot_savart_numerator_name(wake_type, element_type, element_number, kite_obs)
                            system_lifted.extend([(var_name, (3, 1))])

                            var_name = get_element_biot_savart_denominator_name(wake_type, element_type, element_number, kite_obs)
                            system_lifted.extend([(var_name, (1, 1))])

    return system_lifted, system_states


def ordering_of_filaments_in_vortex_horseshoe():
    return {0: get_NE_wingtip_name(),
            1: get_PE_wingtip_name(),
            2: 'closing'}


def get_which_wingtip_shed_this_far_wake_element(element_number):
    elements_per_far_wake_per_kite = 2
    element_number_in_kite_list = np.mod(element_number, elements_per_far_wake_per_kite)

    filament_ordering = ordering_of_filaments_in_vortex_horseshoe()

    if not (element_number_in_kite_list in filament_ordering.keys()):
        message = 'something went wrong when trying to determine which wingtip shed a vortex filament'
        print_op.log_and_raise_error(message)

    tentative_position = filament_ordering[element_number_in_kite_list]
    if tentative_position not in get_wingtip_name_and_strength_direction_dict().keys():
        message = 'tentative position, which wingtip shed a vortex filament, (' + tentative_position + ') is not an acceptable far-wake wingtip'
        print_op.log_and_raise_error(message)

    return tentative_position



def get_position_of_near_wake_element_in_horseshoe(model_options, element_number):
    wake_nodes = model_options['aero']['vortex']['wake_nodes']
    elements_per_ring = 3
    rings_per_kite = (wake_nodes - 1)
    elements_per_kite = elements_per_ring * rings_per_kite
    element_number_in_kite_list = np.mod(element_number, elements_per_kite)
    element_number_in_ring = np.mod(element_number_in_kite_list, elements_per_ring)

    filament_ordering = ordering_of_filaments_in_vortex_horseshoe()

    if not (element_number_in_ring in filament_ordering.keys()):
        message = 'something went wrong when trying to determine the position of a particular near-wake vortex filament'
        print_op.log_and_raise_error(message)
    else:
        return filament_ordering[element_number_in_ring]


def get_shedding_kite_from_element_number(options, wake_type, element_type, element_number, architecture, suppress_error_logging=False):

    expected_number_of_elements_dict_for_wake_types = get_expected_number_of_elements_dict_for_wake_types(
        options,
        architecture)

    if (wake_type in expected_number_of_elements_dict_for_wake_types.keys()) and (element_type in expected_number_of_elements_dict_for_wake_types[wake_type].keys()):
        total = expected_number_of_elements_dict_for_wake_types[wake_type][element_type]
    else:
        message = 'unable to retrieve expected number of elements for wake type (' + wake_type + ') and element type (' + element_type + ')'
        print_op.log_and_raise_error(message)

    if element_number >= total:
        message = 'input element number is greater than total number of elements of this type'
        print_op.log_and_raise_error(message, suppress_error_logging=suppress_error_logging)

    element_number = np.mod(element_number, total)
    total_per_kite = total / float(architecture.number_of_kites)
    remainder = np.mod(element_number, total_per_kite)
    divisible = element_number - remainder

    kite_shed_count = int(divisible / total_per_kite)
    kite_shed = architecture.kite_nodes[kite_shed_count]

    return kite_shed


def get_wake_node_whose_position_relates_to_velocity_element_number(wake_type, element_number, model_options):
    wake_nodes = model_options['aero']['vortex']['wake_nodes']

    if 'near' in wake_type:
        elements_per_ring = 3
        rings_per_kite = (wake_nodes - 1)
        elements_per_kite = elements_per_ring * rings_per_kite
        element_number_in_kite_list = np.mod(element_number, elements_per_kite)
        wake_node_at_most_recently_shed_edge_of_ring = np.floor(float(element_number_in_kite_list) / float(elements_per_ring))

        filament_position = get_position_of_near_wake_element_in_horseshoe(model_options, element_number)
        if filament_position in ['int', 'ext']:
            return wake_node_at_most_recently_shed_edge_of_ring
        elif filament_position == 'closing':
            return wake_node_at_most_recently_shed_edge_of_ring + 1
        else:
            message = 'unfamiliar vortex filament position found (' + filament_position + ')'
            print_op.log_and_raise_error(message)

    elif 'bound' in wake_type:
        return 0

    elif 'far' in wake_type:
        return wake_nodes - 1
    else:
        message = 'unfamiliar wake [substructure] type found (' + wake_type + ')'
        print_op.log_and_raise_error(message)
    return None


def get_expected_time_per_control_interval(options, winding_period):

    expected_number_of_windings = options['user_options']['trajectory']['lift_mode']['windings']
    expected_total_time = winding_period * float(expected_number_of_windings)
    expected_delta_t = expected_total_time / float(options['nlp']['n_k'])
    return expected_delta_t


def get_adjustment_factor_for_trailing_vortices_due_to_interior_and_exterior_circumferences(options, geometry, wingtip, varrho_ref, winding_period):

    type_of_adjustment = 'not_in_use'
    if type_of_adjustment == 'not_in_use':
        local_adjustment_factor = 1.
    else:

        b_ref = geometry['b_ref']
        u_ref = options['user_options']['wind']['u_ref']
        expected_delta_t = get_expected_time_per_control_interval(options, winding_period)
        expected_number_of_windings = options['user_options']['trajectory']['lift_mode']['windings']
        expected_total_angle = 2. * np.pi * float(expected_number_of_windings)
        expected_delta_angle = expected_total_angle / float(options['nlp']['n_k'])

        if wingtip == 'int':
            wingtip_sign = -1.
        elif wingtip == 'ext':
            wingtip_sign = +1.
        else:
            message = 'unfamiliar wingtip abbreviation (' + wingtip + ')'
            print_op.log_and_raise_error(message)

        approx_downstream_distance = expected_delta_t * u_ref

        central_tangential_distance = varrho_ref * expected_delta_angle * b_ref
        approx_tangential_distance = (varrho_ref + wingtip_sign * 0.5) * expected_delta_angle * b_ref

        central_length_squared = approx_downstream_distance**2. + central_tangential_distance**2.
        local_length_squared = approx_downstream_distance**2. + approx_tangential_distance**2.

        if type_of_adjustment == 'squared':
            local_adjustment_factor = local_length_squared / central_length_squared
        elif type_of_adjustment == 'root':
            local_adjustment_factor = np.sqrt(local_length_squared / central_length_squared)

    return local_adjustment_factor


def get_biot_savart_reference_object_properties(model_options, kite_obs_index=0, kite_shed_index=0, parameters=None, geometry=None, inputs={}):
    inputs = compute_biot_savart_reference_object_inputs(model_options, parameters=parameters, geometry=geometry, inputs=inputs)
    properties = compute_biot_savart_reference_object_properties_from_inputs(kite_shed_index, kite_obs_index, inputs)
    return properties


def compute_biot_savart_reference_object_inputs(model_options, parameters=None, geometry=None, inputs={}):

    for local_name in ['b_ref', 'c_ref']:
        if (parameters is not None):
            inputs[local_name] = parameters['theta0', 'geometry', local_name]
        elif (geometry is not None):
            inputs[local_name] = geometry[local_name]
        else:
            message = 'geometry is underdefined. cannot create biot-savart reference object'
            print_op.log_and_raise_error(message)

    for local_name in ['u_ref']:
        in_inputs = local_name in inputs.keys()
        if in_inputs and (parameters is None):
            pass
        elif (not in_inputs) and (parameters is not None):
            inputs[local_name] = parameters['theta0', 'wind', local_name]
        else:
            message = 'no ' + local_name + ' information available for biot-savart reference object'
            print_op.log_and_raise_error(message)

    for local_name in ['filament_strength_ref', 'varrho_ref',  'core_to_chord_ratio', 'wake_nodes', 'number_of_kites', 'winding_period', 'shedding_delta_time']:
        in_vortex_options = local_name in model_options['aero']['vortex'].keys()
        in_actuator_options = local_name in model_options['aero']['actuator'].keys()
        in_inputs = local_name in inputs.keys()

        if in_inputs:
            pass
        elif in_vortex_options:
            inputs[local_name] = model_options['aero']['vortex'][local_name]
        elif in_actuator_options:
            inputs[local_name] = model_options['aero']['actuator'][local_name]
        else:
            message = 'no ' + local_name + ' information available for biot-savart reference object'
            print_op.log_and_raise_error(message)

    return inputs

def compute_biot_savart_reference_object_properties_from_inputs(kite_shed_index, kite_obs_index, inputs):
    properties = {}

    b_ref = inputs['b_ref']
    c_ref = inputs['c_ref']
    properties['b_ref'] = b_ref
    properties['c_ref'] = c_ref

    properties['filament_strength'] = inputs['filament_strength_ref']

    x_center = cas.DM.zeros((3, 1))
    properties['x_center'] = x_center

    a_hat = vect_op.zhat_dm()
    b_hat = -1. * vect_op.yhat_dm()
    properties['a_hat'] = a_hat
    properties['b_hat'] = b_hat

    number_of_kites = inputs['number_of_kites']
    psi_shed = float(kite_shed_index) / float(number_of_kites) * (2. * np.pi)
    psi_obs = float(kite_obs_index) / float(number_of_kites) * (2. * np.pi)
    properties['psi_shed'] = psi_shed
    properties['psi_obs'] = psi_obs

    l_hat = vect_op.xhat_dm()

    r_hat_shed = np.cos(psi_shed) * a_hat + np.sin(psi_shed) * b_hat
    t_hat_shed = vect_op.normed_cross(l_hat, r_hat_shed)

    r_hat_obs = np.cos(psi_obs) * a_hat + np.sin(psi_obs) * b_hat
    t_hat_obs = vect_op.normed_cross(l_hat, r_hat_obs)

    properties['r_hat_shed'] = r_hat_shed
    properties['t_hat_shed'] = t_hat_shed
    properties['r_hat_obs'] = r_hat_obs
    properties['t_hat_obs'] = t_hat_obs
    properties['l_hat'] = l_hat

    ehat_1 = -1. * t_hat_shed
    ehat_2 = r_hat_shed
    ehat_3 = l_hat
    properties['ehat_1'] = ehat_1
    properties['ehat_2'] = ehat_2
    properties['ehat_3'] = ehat_3

    varrho_ref = inputs['varrho_ref']
    radius = varrho_ref * b_ref
    properties['radius'] = radius
    properties['varrho_int'] = varrho_ref - 0.5
    properties['varrho_ext'] = varrho_ref + 0.5

    x_kite_obs = x_center + radius * r_hat_obs
    properties['x_kite_obs'] = x_kite_obs

    x_kite_shed = x_center + radius * r_hat_shed
    x_int_shed = x_kite_shed - (b_ref / 2.) * ehat_2
    x_ext_shed = x_kite_shed + (b_ref / 2.) * ehat_2
    properties['x_int_shed'] = x_int_shed
    properties['x_ext_shed'] = x_ext_shed

    properties['shedding_delta_time'] = inputs['shedding_delta_time']
    properties['u_ref'] = inputs['u_ref']
    properties['omega'] = 2. * np.pi / inputs['winding_period']

    properties['near_wake_unit_length'] = inputs['shedding_delta_time'] * inputs['u_ref']
    properties['far_wake_l_start'] = properties['near_wake_unit_length'] * float(inputs['wake_nodes'] - 1.)
    properties['r_core'] = c_ref * inputs['core_to_chord_ratio']

    return properties


def log_and_raise_unknown_representation_error(vortex_representation):
    message = 'vortex representation (' + vortex_representation + ') is not recognized'
    print_op.log_and_raise_error(message)
    return None


def get_variable_si(variables, var_type, var_name, scaling=None):
    var = struct_op.get_variable_from_model_or_reconstruction(variables, var_type, var_name)
    if scaling is not None:
        return struct_op.var_scaled_to_si(var_type, var_name, var, scaling)
    else:
        return var

def get_define_wake_types():
    return ['bound', 'near', 'far']

def get_wake_node_position_var_type(model_options):
    vortex_representation = general_tools.get_option_from_possible_dicts(model_options, 'representation', 'vortex')
    if vortex_representation == 'alg':
        var_type = 'z'
    elif vortex_representation == 'state':
        var_type = 'x'
    else:
        log_and_raise_unknown_representation_error(vortex_representation)

    return var_type

def get_wake_node_position_si(model_options, variables, kite_shed, tip, wake_node, scaling=None):
    var_name = get_wake_node_position_name(kite_shed, tip, wake_node)
    var_type = get_wake_node_position_var_type(model_options)
    return get_variable_si(variables, var_type, var_name, scaling)


def get_vortex_ring_strength_si(variables, kite_shed, ring, scaling=None):
    var_name = get_vortex_ring_strength_name(kite_shed, ring)
    return get_variable_si(variables, 'z', var_name, scaling)

def get_element_induced_velocity_si(variables, wake_type, element_type, element_number, kite_obs, scaling=None):
    var_name = get_element_induced_velocity_name(wake_type, element_type, element_number, kite_obs)
    return get_variable_si(variables, 'z', var_name, scaling)

def get_element_induced_velocity_numerator_si(variables, wake_type, element_type, element_number, kite_obs, scaling=None):
    var_name = get_element_biot_savart_numerator_name(wake_type, element_type, element_number, kite_obs)
    return get_variable_si(variables, 'z', var_name, scaling)

def get_element_induced_velocity_denominator_si(variables, wake_type, element_type, element_number, kite_obs, scaling=None):
    var_name = get_element_biot_savart_denominator_name(wake_type, element_type, element_number, kite_obs)
    return get_variable_si(variables, 'z', var_name, scaling)

def get_induced_velocity_at_kite_si(variables, kite_obs, scaling=None):
    var_name = get_induced_velocity_at_kite_name(kite_obs)
    return get_variable_si(variables, 'z', var_name, scaling)

def get_far_wake_finite_filament_pathwise_convection_velocity_si(variables, kite_shed, scaling=None):
    var_name = get_far_wake_finite_filament_pathwise_convection_velocity_name(kite_shed)
    return get_variable_si(variables, 'z', var_name, scaling)

def get_far_wake_cylinder_center_position_si(variables, parent_shed, scaling=None):
    var_name = get_far_wake_cylinder_center_position_name(parent_shed)
    return get_variable_si(variables, 'z', var_name, scaling)

def get_far_wake_cylinder_pitch_si(variables, parent_shed, scaling=None):
    var_name = get_far_wake_cylinder_pitch_name(parent_shed)
    return get_variable_si(variables, 'z', var_name, scaling)


def get_kite_or_parent_and_tip_and_node_or_ring_list_for_abbreviated_vars(abbreviated_var_name, nlp_options, architecture):

    wake_nodes = general_tools.get_option_from_possible_dicts(nlp_options, 'wake_nodes', 'vortex')
    rings = general_tools.get_option_from_possible_dicts(nlp_options, 'rings', 'vortex')
    kite_nodes = architecture.kite_nodes
    kite_parents = set([architecture.parent_map[kite] for kite in kite_nodes])
    wingtips = ['ext', 'int']

    if abbreviated_var_name == 'wx':
        kite_shed_or_parent_shed_list = kite_nodes
        tip_list = wingtips
        wake_node_or_ring_list = range(wake_nodes)
    elif abbreviated_var_name == 'wg':
        kite_shed_or_parent_shed_list = kite_nodes
        tip_list = [None]
        wake_node_or_ring_list = range(rings)
    elif abbreviated_var_name == 'wh':
        kite_shed_or_parent_shed_list = kite_parents
        tip_list = [None]
        wake_node_or_ring_list = [wake_nodes - 1]
    elif abbreviated_var_name == 'wx_center':
        kite_shed_or_parent_shed_list = kite_parents
        tip_list = [None]
        wake_node_or_ring_list = [wake_nodes - 1]
    else:
        message = 'get_specific_constraint function is not set up for this abbreviation (' + abbreviated_var_name + ') yet.'
        print_op.log_and_raise_error(message)

    return kite_shed_or_parent_shed_list, tip_list, wake_node_or_ring_list

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
        print_op.log_and_raise_error(message)

def get_wake_node_position_name(kite_shed, tip, wake_node):
    var_name = 'wx_' + str(kite_shed) + '_' + tip + '_' + str(wake_node)
    return var_name

def get_vortex_ring_strength_name(kite_shed, ring):
    var_name = 'wg_' + str(kite_shed) + '_' + str(ring)
    return var_name


def get_element_biot_savart_numerator_name(wake_type, element_type, element_number, kite_obs):
    var_name = 'wu_' + wake_type + '_' + element_type + '_' + str(element_number) + '_' + str(kite_obs) + '_num'
    return var_name

def get_element_biot_savart_denominator_name(wake_type, element_type, element_number, kite_obs):
    var_name = 'wu_' + wake_type + '_' + element_type + '_' + str(element_number) + '_' + str(kite_obs) + '_den'
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


def get_total_number_of_vortex_elements(options, architecture):

    expected_number_of_elements_dict_for_wake_types = get_expected_number_of_elements_dict_for_wake_types(options,
                                                                                                          architecture)
    total_number = 0
    for local_dict in expected_number_of_elements_dict_for_wake_types.values():
        for count in local_dict.values():
            total_number += count

    return total_number

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
    expected_dict = {}

    number_of_kites = architecture.number_of_kites
    wake_nodes = general_tools.get_option_from_possible_dicts(options, 'wake_nodes', 'vortex')

    near_wake_rings = (wake_nodes - 1)
    if near_wake_rings > 0:
        expected_dict['finite_filament'] = 3 * number_of_kites * near_wake_rings

    return expected_dict


def get_expected_number_of_far_wake_elements_dict(options, architecture):
    far_wake_element_type = general_tools.get_option_from_possible_dicts(options, 'far_wake_element_type', 'vortex')
    if far_wake_element_type == 'finite_filament':
        expected_dict = get_expected_number_of_finite_filament_far_wake_elements(architecture)
    elif far_wake_element_type == 'semi_infinite_filament':
        expected_dict = get_expected_number_of_semi_infinite_filament_far_wake_elements(architecture)
    elif far_wake_element_type == 'semi_infinite_right_cylinder':
        expected_dict = get_expected_number_of_semi_infinite_right_cylinder_far_wake_elements(architecture)
    elif far_wake_element_type == 'not_in_use':
        return {}
    else:
        message = 'unexpected type of far-wake vortex element (' + far_wake_element_type + '). maybe, check your spelling?'
        print_op.log_and_raise_error(message)

    return expected_dict

def get_expected_number_of_semi_infinite_right_cylinder_far_wake_elements(architecture):
    number_of_kites = architecture.number_of_kites
    expected_dict = {'semi_infinite_tangential_right_cylinder': 2 * number_of_kites,
                     'semi_infinite_longitudinal_right_cylinder': 2 * number_of_kites}
    return expected_dict

def get_expected_number_of_finite_filament_far_wake_elements(architecture):
    number_of_kites = architecture.number_of_kites
    expected_dict = {'finite_filament': 2 * number_of_kites}
    return expected_dict

def get_expected_number_of_semi_infinite_filament_far_wake_elements(architecture):
    number_of_kites = architecture.number_of_kites
    expected_dict = {'semi_infinite_filament': 2 * number_of_kites}
    return expected_dict


def check_positive_vortex_wake_nodes(options):
    wake_nodes = options['induction']['vortex_wake_nodes']
    if wake_nodes < 0:
        message = 'insufficient wake nodes for creating a filament list: wake_nodes = ' + str(wake_nodes)
        print_op.log_and_raise_error(message)
    return None

def not_bound_and_shed_is_obs(model_options, substructure_type, element_type, element_number, kite_obs, architecture):
    kite_shed = get_shedding_kite_from_element_number(model_options, substructure_type, element_type,
                                                      element_number, architecture)
    if not (substructure_type == 'bound' and kite_obs == kite_shed):
        return True
    else:
        return False


def superpose_induced_velocities_at_kite(model_options, wake, variables_si, kite_obs, architecture, substructure_types = None):

    if substructure_types is None:
        substructure_types = wake.get_initialized_substructure_types()

    vec_u_superposition = cas.DM.zeros((3, 1))
    for substructure_type in substructure_types:
        initialized_elements = wake.get_substructure(substructure_type).get_initialized_element_types()
        for element_type in initialized_elements:
            number_of_elements = wake.get_substructure(substructure_type).get_list(element_type).number_of_elements
            for element_number in range(number_of_elements):

                if not_bound_and_shed_is_obs(model_options, substructure_type, element_type, element_number, kite_obs,
                                             architecture):
                    elem_u_ind_si = get_element_induced_velocity_si(variables_si, substructure_type,
                                                                                 element_type, element_number, kite_obs)
                    vec_u_superposition += elem_u_ind_si

    return vec_u_superposition


def get_induction_factor_normalizing_speed(model_options, wind, kite, parent, variables, architecture):
    induction_factor_normalizing_speed = model_options['aero']['vortex']['induction_factor_normalizing_speed']
    if induction_factor_normalizing_speed == 'u_zero':
        u_vec = general_flow.get_vec_u_zero(model_options, wind, parent, variables, architecture)
    elif induction_factor_normalizing_speed == 'u_inf':
        u_vec = general_flow.get_kite_vec_u_infty(variables, wind, kite, parent)
    elif induction_factor_normalizing_speed == 'u_ref':
        u_vec = wind.get_speed_ref() # * xhat
    else:
        message = 'desired induction_factor_normalizing_speed (' + induction_factor_normalizing_speed + ') is not yet available'
        print_op.log_and_raise_error(message)

    u_normalizing = vect_op.smooth_norm(u_vec)
    return u_normalizing

def evaluate_symbolic_on_segments_and_sum(filament_fun, segment_list):

    n_filaments = segment_list.shape[1]
    filament_map = filament_fun.map(n_filaments, 'openmp')
    all_eval = filament_map(segment_list)

    total = cas.sum2(all_eval)

    return total


def get_epsilon(options, parameters):
    c_ref = parameters['theta0', 'geometry', 'c_ref']
    epsilon = options['aero']['vortex']['epsilon_to_chord_ratio'] * c_ref
    return epsilon


def get_r_core(model_options, parameters=None, geometry=None):

    core_to_chord_ratio = model_options['aero']['vortex']['core_to_chord_ratio']
    if core_to_chord_ratio == 0.:
        r_core = 0.

    else:
        if (parameters is not None) and (geometry is None):
            c_ref = parameters['theta0', 'geometry', 'c_ref']
        elif (geometry is not None) and (parameters is None):
            c_ref = geometry['c_ref']
        else:
            message = 'geometry/parameters are overdefined when computing the vortex core radius'
            print_op.log_and_raise_error(message)

        r_core = core_to_chord_ratio * c_ref

    return r_core


def get_PE_wingtip_name():
    return 'ext'


def get_NE_wingtip_name():
    return 'int'


def get_wingtip_name_and_strength_direction_dict():
    dict = {
        get_NE_wingtip_name(): -1.,
        get_PE_wingtip_name(): +1.
    }
    return dict