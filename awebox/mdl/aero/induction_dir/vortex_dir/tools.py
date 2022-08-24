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


def get_list_of_abbreviated_variables(model_options):
    abbreviated_variables = ['wx', 'wg']
    far_wake_element_type = general_tools.get_option_from_possible_dicts(model_options, 'far_wake_element_type',
                                                                         'vortex')
    if far_wake_element_type == 'semi_infinite_cylinder':
        abbreviated_variables += ['wx_center', 'wh']

    return abbreviated_variables

def extend_system_variables(model_options, system_lifted, system_states, architecture):

    abbreviated_variables = get_list_of_abbreviated_variables(model_options)

    for abbreviated_var_name in abbreviated_variables:
        system_lifted, system_states = extend_specific_geometric_variable(abbreviated_var_name, model_options, system_lifted, system_states,
                                           architecture)

    system_lifted, system_states = extend_velocity_variables(model_options, system_lifted, system_states, architecture)

    return system_lifted, system_states

def extend_specific_geometric_variable(abbreviated_var_name, model_options, system_lifted, system_states, architecture):
    kite_shed_or_parent_shed_list, tip_list, wake_node_or_ring_list = get_kite_or_parent_and_tip_and_node_or_ring_list_for_abbreviated_vars(
        abbreviated_var_name, model_options, architecture)

    if abbreviated_var_name == 'wx':
        var_type = get_wake_node_position_var_type(model_options)
    else:
        var_type = 'xl'

    if abbreviated_var_name[:2] == 'wx':
        var_shape = (3, 1)
    else:
        var_shape = (1, 1)

    for kite_shed_or_parent_shed in kite_shed_or_parent_shed_list:
        for tip in tip_list:
            for wake_node_or_ring in wake_node_or_ring_list:

                var_name = get_var_name(abbreviated_var_name, kite_shed_or_parent_shed=kite_shed_or_parent_shed, tip=tip, wake_node_or_ring=wake_node_or_ring)

                if var_type == 'xl':
                    system_lifted.extend([(var_name, var_shape)])
                elif var_type == 'xd':
                    system_states.extend([(var_name, var_shape)])
                    system_states.extend([('d' + var_name, var_shape)])

                else:
                    message = 'unexpected variable type: (' + var_type + ')'
                    awelogger.logger.error(message)
                    raise Exception(message)

    return system_lifted, system_states

def extend_velocity_variables(model_options, system_lifted, system_states, architecture):

    # induced velocity part: the part that depends on the wake types and wake structure
    expected_number_of_elements_dict_for_wake_types = get_expected_number_of_elements_dict_for_wake_types(model_options, architecture)

    for kite_obs in architecture.kite_nodes:
        var_name = get_induced_velocity_at_kite_name(kite_obs)
        system_lifted.extend([(var_name, (3, 1))])

        for wake_type, local_expected_number_of_elements_dict in expected_number_of_elements_dict_for_wake_types.items():
            for element_type, expected_number in local_expected_number_of_elements_dict.items():
                for element_number in range(expected_number):
                    var_name = get_element_induced_velocity_name(wake_type, element_type, element_number, kite_obs)
                    system_lifted.extend([(var_name, (3, 1))])

    return system_lifted, system_states


def append_scaling_to_options_tree(options, geometry, options_tree, architecture):

    # the part that describes the wake nodes and consequent vortex rings
    wingtips = ['ext', 'int']
    wake_nodes = options['model']['aero']['vortex']['wake_nodes']
    rings = options['model']['aero']['vortex']['wake_nodes']

    CL = 1.5

    c_ref = geometry['c_ref']
    b_ref = geometry['b_ref']
    u_ref = options['user_options']['wind']['u_ref']

    groundspeed = options['solver']['initialization']['groundspeed']
    airspeed_ref = cas.sqrt(groundspeed**2 + u_ref**2)

    wx_scale = 1.
    wg_scale = 0.5 * CL * airspeed_ref * c_ref
    wx_center_scale = b_ref
    wh_scale = b_ref
    wu_ind_scale = u_ref
    wu_ind_element_scale = u_ref

    for kite_shed in architecture.kite_nodes:
        for wake_node in range(wake_nodes):
            for tip in wingtips:
                var_name = get_wake_node_position_name(kite_shed, tip, wake_node)
                var_type = get_wake_node_position_var_type(options['model'])
                options_tree.append(('model', 'scaling', var_type, var_name, wx_scale, ('descript', None), 'x'))

                if var_type == 'xd':
                    options_tree.append(('model', 'scaling', 'xd', 'd' + var_name, wx_scale, ('descript', None), 'x'))
                    options_tree.append(('model', 'scaling', 'xd', 'dd' + var_name, wx_scale, ('descript', None), 'x'))

        for ring in range(rings):
            var_name = get_vortex_ring_strength_name(kite_shed, ring)
            options_tree.append(('model', 'scaling', 'xl', var_name, wg_scale, ('descript', None), 'x'))

    far_wake_element_type = options['model']['aero']['vortex']['far_wake_element_type']
    if (far_wake_element_type == 'semi_infinite_cylinder'):

        if (architecture.number_of_kites == 1):
            message = 'the cylindrical far_wake_model may perform poorly with only one kite. we recommend switching to a filament far_wake_model'
            awelogger.logger.warning(message)

        for parent_shed in set([architecture.parent_map[kite] for kite in architecture.kite_nodes]):
            var_name = get_far_wake_cylinder_center_position_name(parent_shed=parent_shed)
            options_tree.append(('model', 'scaling', 'xl', var_name, wx_center_scale, ('descript', None), 'x'))

            var_name = get_far_wake_cylinder_pitch_name(parent_shed=parent_shed)
            options_tree.append(('model', 'scaling', 'xl', var_name, wh_scale, ('descript', None), 'x')),
            options_tree.append(('model', 'system_bounds', 'xl', var_name, [0.0, cas.inf], ('', None), 'x'))

    # induced velocity part: the part that depends on the wake types and wake structure
    expected_number_of_elements_dict_for_wake_types = get_expected_number_of_elements_dict_for_wake_types(options, architecture)

    for kite_obs in architecture.kite_nodes:
        var_name = get_induced_velocity_at_kite_name(kite_obs)
        options_tree.append(('model', 'scaling', 'xl', var_name, wu_ind_scale, ('descript', None), 'x'))

        for wake_type, local_expected_number_of_elements_dict in expected_number_of_elements_dict_for_wake_types.items():
            for element_type, expected_number in local_expected_number_of_elements_dict.items():
                for element_number in range(expected_number):
                    var_name = get_element_induced_velocity_name(wake_type, element_type, element_number, kite_obs)
                    options_tree.append(('model', 'scaling', 'xl', var_name, wu_ind_element_scale, ('descript', None), 'x'))

    return options_tree




def log_and_raise_unknown_representation_error(vortex_representation):
    message = 'vortex representation (' + vortex_representation + ') is not recognized'
    awelogger.logger.error(message)
    raise Exception(message)
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
        var_type = 'xl'
    elif vortex_representation == 'state':
        var_type = 'xd'
    else:
        log_and_raise_unknown_representation_error(vortex_representation)

    return var_type

def get_wake_node_position_si(model_options, variables, kite_shed, tip, wake_node, scaling=None):
    var_name = get_wake_node_position_name(kite_shed, tip, wake_node)
    var_type = get_wake_node_position_var_type(model_options)
    return get_variable_si(variables, var_type, var_name, scaling)


def get_vortex_ring_strength_si(variables, kite_shed, ring, scaling=None):
    var_name = get_vortex_ring_strength_name(kite_shed, ring)
    return get_variable_si(variables, 'xl', var_name, scaling)

def get_element_induced_velocity_si(variables, wake_type, element_type, element_number, kite_obs, scaling=None):
    var_name = get_element_induced_velocity_name(wake_type, element_type, element_number, kite_obs)
    return get_variable_si(variables, 'xl', var_name, scaling)

def get_induced_velocity_at_kite_si(variables, kite_obs, scaling=None):
    var_name = get_induced_velocity_at_kite_name(kite_obs)
    return get_variable_si(variables, 'xl', var_name, scaling)

def get_far_wake_finite_filament_pathwise_convection_velocity_si(variables, kite_shed, scaling=None):
    var_name = get_far_wake_finite_filament_pathwise_convection_velocity_name(kite_shed)
    return get_variable_si(variables, 'xl', var_name, scaling)

def get_far_wake_cylinder_center_position_si(variables, parent_shed, scaling=None):
    var_name = get_far_wake_cylinder_center_position_name(parent_shed)
    return get_variable_si(variables, 'xl', var_name, scaling)

def get_far_wake_cylinder_pitch_si(variables, parent_shed, scaling=None):
    var_name = get_far_wake_cylinder_pitch_name(parent_shed)
    return get_variable_si(variables, 'xl', var_name, scaling)

def get_lifted_vortex_wake_related_degrees_of_freedom(variables):
    lifted_variables = struct_op.subkeys(variables, 'xl')

    dof = 0
    for var_name in lifted_variables:
        dof += variables['xl', var_name].shape[0]

    return dof

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
        awelogger.logger.error(message)
        raise Exception(message)

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
    elif far_wake_element_type == 'semi_infinite_cylinder':
        expected_dict = get_expected_number_of_semi_infinite_cylinder_far_wake_elements(architecture)
    elif far_wake_element_type == 'not_in_use':
        return {}
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


def check_positive_vortex_wake_nodes(options):
    wake_nodes = options['induction']['vortex_wake_nodes']
    if wake_nodes < 0:
        message = 'insufficient wake nodes for creating a filament list: wake_nodes = ' + str(wake_nodes)
        awelogger.logger.error(message)
        raise Exception(message)
    return None

def superpose_induced_velocities_at_kite(wake, variables_si, kite_obs, substructure_types = None):

    if substructure_types is None:
        substructure_types = wake.get_initialized_substructure_types()

    vec_u_superposition = cas.DM.zeros((3, 1))
    for substructure_type in substructure_types:
        initialized_elements = wake.get_substructure(substructure_type).get_initialized_element_types()
        for element_type in initialized_elements:
            number_of_elements = wake.get_substructure(substructure_type).get_list(element_type).number_of_elements
            for edx in range(number_of_elements):
                elem_u_ind_si = get_element_induced_velocity_si(variables_si, substructure_type,
                                                                             element_type, edx, kite_obs)
                vec_u_superposition += elem_u_ind_si

    return vec_u_superposition


def get_induction_factor_normalizing_speed(model_options, wind, kite, parent, variables, architecture):
    induction_factor_normalizing_speed = model_options['aero']['vortex']['induction_factor_normalizing_speed']
    if induction_factor_normalizing_speed == 'u_zero':
        u_vec = general_flow.get_uzero_vec(model_options, wind, parent, variables, architecture)
    elif induction_factor_normalizing_speed == 'u_inf':
        u_vec = general_flow.get_kite_uinfy_vec(variables, wind, kite, parent)
    elif induction_factor_normalizing_speed == 'u_ref':
        u_vec = wind.get_speed_ref()
    else:
        message = 'desired induction_factor_normalizing_speed (' + induction_factor_normalizing_speed + ') is not yet available'
        awelogger.logger.error(message)
        raise Exception(message)

    u_normalizing = vect_op.smooth_norm(u_vec)
    return u_normalizing

def evaluate_symbolic_on_segments_and_sum(filament_fun, segment_list):

    n_filaments = segment_list.shape[1]
    filament_map = filament_fun.map(n_filaments, 'openmp')
    all = filament_map(segment_list)

    total = cas.sum2(all)

    return total

def get_epsilon(options, parameters):
    c_ref = parameters['theta0','geometry','c_ref']
    epsilon = options['aero']['vortex']['epsilon_to_chord_ratio'] * c_ref
    return epsilon

def get_r_core(options, parameters):

    core_to_chord_ratio = options['aero']['vortex']['core_to_chord_ratio']
    if core_to_chord_ratio == 0.:
        r_core = 0.

    else:
        c_ref = parameters['theta0','geometry','c_ref']
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