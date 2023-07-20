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

import awebox.mdl.aero.induction_dir.vortex_dir.tools as vortex_tools
import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.finite_filament as finite_filament
import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.semi_infinite_filament as semi_infinite_filament

import awebox.tools.struct_operations as struct_op
import awebox.tools.vector_operations as vect_op
import awebox.tools.print_operations as print_op



def append_scaling_to_options_tree(options, geometry, options_tree, architecture, q_scaling, u_altitude, CL, varrho_ref, winding_period):
    options_tree = append_geometric_scaling(options, geometry, options_tree, architecture, q_scaling, u_altitude, CL, varrho_ref, winding_period)
    options_tree = append_induced_velocity_scaling(options, geometry, options_tree, architecture, u_altitude, CL, varrho_ref, winding_period)
    return options_tree


def get_filament_strength(options, geometry, CL):
    c_ref = geometry['c_ref']

    u_ref = options['user_options']['wind']['u_ref']
    a_ref = options['model']['aero']['actuator']['a_ref']
    axial_speed = u_ref * (1. - a_ref)

    # tangential_speed = options['solver']['initialization']['groundspeed']
    # airspeed_ref = cas.sqrt(tangential_speed ** 2 + axial_speed ** 2.)

    print_op.warn_about_temporary_functionality_alteration()
    airspeed_ref = axial_speed

    filament_strength = 0.5 * CL * airspeed_ref * c_ref
    return filament_strength


def append_geometric_scaling(options, geometry, options_tree, architecture, q_scaling, u_altitude, CL, varrho_ref, winding_period):
    # the part that describes the wake nodes and consequent vortex rings
    wingtips = ['ext', 'int']
    wake_nodes = options['model']['aero']['vortex']['wake_nodes']
    rings = options['model']['aero']['vortex']['wake_nodes']

    u_ref = options['user_options']['wind']['u_ref']

    filament_strength = get_filament_strength(options, geometry, CL)

    inputs = get_scaling_inputs(options, geometry, architecture, u_altitude, CL, varrho_ref, winding_period)

    properties_ref = vortex_tools.get_biot_savart_reference_object_properties(options['model'], geometry=geometry, kite_obs_index=0, kite_shed_index=0, inputs=inputs)
    avg_downstream = properties_ref['far_wake_l_start'] / 2.

    position_scaling_method = options['model']['aero']['vortex']['position_scaling_method']
    if position_scaling_method == 'q10':
        position_scaling = q_scaling
    elif position_scaling_method == 'convection':
        position_scaling = q_scaling + avg_downstream * vect_op.xhat_dm()
    elif position_scaling_method in ['radius', 'b_ref', 'c_ref']:
        position_scaling = properties_ref[position_scaling_method]
    else:
        message = 'unexpected vortex-position-variable wx scaling method (' + position_scaling_method + ').'
        print_op.log_and_raise_error(message)

    wx_scale = position_scaling
    wx_center_scale = wx_scale
    wg_scale = filament_strength
    wh_scale = winding_period * u_ref

    options_tree.append(('model', 'aero', 'vortex', 'varrho_ref', varrho_ref, ('descript', None), 'x'))

    for kite_shed in architecture.kite_nodes:
        for wake_node in range(wake_nodes):
            for tip in wingtips:
                var_name = vortex_tools.get_wake_node_position_name(kite_shed, tip, wake_node)
                var_type = vortex_tools.get_wake_node_position_var_type(options['model'])
                options_tree.append(('model', 'scaling', var_type, var_name, wx_scale, ('descript', None), 'x'))

                if var_type == 'x':
                    options_tree.append(('model', 'scaling', 'x', 'd' + var_name, wx_scale, ('descript', None), 'x'))
                    options_tree.append(('model', 'scaling', 'x', 'dd' + var_name, wx_scale, ('descript', None), 'x'))

        for ring in range(rings):
            var_name = vortex_tools.get_vortex_ring_strength_name(kite_shed, ring)
            options_tree.append(('model', 'scaling', 'z', var_name, wg_scale, ('descript', None), 'x'))

    far_wake_element_type = options['model']['aero']['vortex']['far_wake_element_type']
    if 'cylinder' in far_wake_element_type:

        if (architecture.number_of_kites == 1):
            message = 'the cylindrical far_wake_model may perform poorly with only one kite. we recommend switching to a filament far_wake_model'
            awelogger.logger.warning(message)

        for parent_shed in set([architecture.parent_map[kite] for kite in architecture.kite_nodes]):
            var_name = vortex_tools.get_far_wake_cylinder_center_position_name(parent_shed=parent_shed)
            options_tree.append(('model', 'scaling', 'z', var_name, wx_center_scale, ('descript', None), 'x'))

            var_name = vortex_tools.get_far_wake_cylinder_pitch_name(parent_shed=parent_shed)
            options_tree.append(('model', 'scaling', 'z', var_name, wh_scale, ('descript', None), 'x')),
            options_tree.append(('model', 'system_bounds', 'z', var_name, [0.0, cas.inf], ('', None), 'x'))

    return options_tree


def get_scaling_inputs(options, geometry, architecture, u_altitude, CL, varrho_ref, winding_period):
    u_ref = options['user_options']['wind']['u_ref']
    a_ref = options['model']['aero']['actuator']['a_ref']
    wake_nodes = options['model']['aero']['vortex']['wake_nodes']
    filament_strength = get_filament_strength(options, geometry, CL)

    inputs = {
        'u_ref': u_ref * a_ref,
        'wake_nodes': wake_nodes,
        'filament_strength_ref': filament_strength,
        'varrho_ref': varrho_ref,
        'number_of_kites': architecture.number_of_kites,
        'winding_period': winding_period,
        'shedding_delta_time': vortex_tools.get_expected_time_per_control_interval(options, winding_period)
    }
    return inputs


def append_induced_velocity_scaling(options, geometry, options_tree, architecture, u_altitude, CL, varrho_ref, winding_period):

    biot_savart_residual_assembly = options['model']['aero']['vortex']['biot_savart_residual_assembly']

    u_ref = options['user_options']['wind']['u_ref']
    a_ref = options['model']['aero']['actuator']['a_ref']
    wu_ind_scale = u_ref * a_ref
    for kite_obs in architecture.kite_nodes:
        var_name = vortex_tools.get_induced_velocity_at_kite_name(kite_obs)
        options_tree.append(('model', 'scaling', 'z', var_name, wu_ind_scale, ('descript', None), 'x'))

    inputs = get_scaling_inputs(options, geometry, architecture, u_altitude, CL, varrho_ref, winding_period)

    expected_number_of_elements_dict_for_wake_types = vortex_tools.get_expected_number_of_elements_dict_for_wake_types(
        options,
        architecture)

    for kite_obs in architecture.kite_nodes:
        for wake_type, local_expected_number_of_elements_dict in expected_number_of_elements_dict_for_wake_types.items():
            for element_type, expected_number in local_expected_number_of_elements_dict.items():
                for element_number in range(expected_number):
                    kite_shed = vortex_tools.get_shedding_kite_from_element_number(options, wake_type, element_type, element_number, architecture)
                    var_name = vortex_tools.get_element_induced_velocity_name(wake_type, element_type, element_number, kite_obs)

                    kite_shed_index = architecture.kite_nodes.index(kite_shed)
                    kite_obs_index = architecture.kite_nodes.index(kite_obs)

                    if wake_type == 'bound':
                        value, num, den = get_induced_velocity_scaling_for_bound_filament(options['model'], geometry, kite_obs_index=kite_obs_index, kite_shed_index=kite_shed_index, inputs=inputs)
                    elif wake_type == 'far':
                        value, num, den = get_induced_velocity_scaling_for_far_filament(options['model'], geometry, element_number=element_number, kite_obs_index=kite_obs_index, kite_shed_index=kite_shed_index, inputs=inputs)
                    elif wake_type == 'near':
                        value, num, den = get_induced_velocity_scaling_for_near_filament(options['model'], geometry, element_number=element_number, kite_obs_index=kite_obs_index, kite_shed_index=kite_shed_index, inputs=inputs)
                    else:
                        message = 'unfamiliar wake type (' + wake_type + ') when generating induced velocity scaling values'
                        print_op.log_and_raise_error(message)

                    value = vect_op.smooth_norm(value) * cas.DM.ones((3, 1))
                    options_tree.append(
                        ('model', 'scaling', 'z', var_name, value, ('descript', None), 'x'))

                    if biot_savart_residual_assembly == 'lifted':

                        num = vect_op.smooth_norm(num) * cas.DM.ones((3, 1))
                        num_name = vortex_tools.get_element_biot_savart_numerator_name(wake_type, element_type, element_number, kite_obs)
                        options_tree.append(
                            ('model', 'scaling', 'z', num_name, num, ('descript', None), 'x'))

                        den = vect_op.smooth_abs(den)
                        den_name = vortex_tools.get_element_biot_savart_denominator_name(wake_type, element_type, element_number, kite_obs)
                        options_tree.append(
                            ('model', 'scaling', 'z', den_name, den, ('descript', None), 'x'))

    return options_tree


def get_induced_velocity_scaling_for_bound_filament(model_options, geometry, kite_obs_index=0, kite_shed_index=0, inputs={}):

    properties_ref = vortex_tools.get_biot_savart_reference_object_properties(model_options, geometry=geometry, kite_obs_index=kite_obs_index, kite_shed_index=kite_shed_index, inputs=inputs)

    x_kite_obs = properties_ref['x_kite_obs']

    ehat_1 = properties_ref['ehat_1']

    r_core = properties_ref['r_core']
    strength = properties_ref['filament_strength']

    x_int_shed = properties_ref['x_int_shed']
    x_ext_shed = properties_ref['x_ext_shed']

    print_op.warn_about_temporary_functionality_alteration()
    if kite_obs_index == kite_shed_index:

        bound_induction_offset = model_options['aero']['vortex']['bound_induction_offset']
        if bound_induction_offset == 'micro':
            offset = 1.e-6
        elif bound_induction_offset == 'r_core':
            offset = properties_ref['r_core']
        elif bound_induction_offset == 'c_ref':
            offset = properties_ref['c_ref']
        elif bound_induction_offset == 'b_ref':
            offset = properties_ref['b_ref']
        elif bound_induction_offset == 'radius':
            offset = properties_ref['radius']
        else:
            message = 'unexpected bound_induction_offset option (' + bound_induction_offset + ')'
            print_op.log_and_raise_error(message)

    else:
        offset = cas.DM(0.)

    x_start = x_int_shed + offset * ehat_1
    x_end = x_ext_shed + offset * ehat_1

    info_dict = {'x_start': x_start,
                 'x_end': x_end,
                 'r_core': r_core,
                 'strength': strength}

    fil = finite_filament.FiniteFilament(info_dict)
    fil.define_biot_savart_induction_function()

    x_obs = x_kite_obs
    value, num, den = fil.calculate_biot_savart_induction(info_dict, x_obs)

    print_op.warn_about_temporary_functionality_alteration()
    if kite_obs_index == kite_shed_index:
        u_ref = properties_ref['u_ref']
        a_ref = model_options['aero']['actuator']['a_ref']

        bound_induction_scaling_factor = model_options['aero']['vortex']['bound_induction_scaling_factor']
        value = bound_induction_scaling_factor * u_ref * (1. - a_ref) * vect_op.xhat_dm()

        print_op.warn_about_temporary_functionality_alteration()
        den *= 1.e4

        num = value * den

    return value, num, den


def get_induced_velocity_scaling_for_far_filament(model_options, geometry, element_number=0, kite_obs_index=0, kite_shed_index=0, inputs={}):
    wingtip = vortex_tools.get_which_wingtip_shed_this_far_wake_element(element_number)

    properties_ref = vortex_tools.get_biot_savart_reference_object_properties(model_options, geometry=geometry, kite_obs_index=kite_obs_index, kite_shed_index=kite_shed_index, inputs=inputs)

    x_kite_obs = properties_ref['x_kite_obs']

    r_core = properties_ref['r_core']
    strength = properties_ref['filament_strength']

    a_hat = properties_ref['a_hat']
    b_hat = properties_ref['b_hat']

    psi_shed = properties_ref['psi_shed']
    omega = properties_ref['omega']
    time_since_shedding = float(inputs['wake_nodes'] - 1) * properties_ref['shedding_delta_time']
    psi_shed_rewound = psi_shed - omega * time_since_shedding
    r_hat_shed_rewound = np.cos(psi_shed_rewound) * a_hat + np.sin(psi_shed_rewound) * b_hat

    x_center = properties_ref['x_center']

    # radius_shed = properties_ref['radius']
    radius_shed = properties_ref['varrho_' + wingtip] * properties_ref['b_ref']

    wingtip_name_and_strength_direction_dict = vortex_tools.get_wingtip_name_and_strength_direction_dict()
    if wingtip in wingtip_name_and_strength_direction_dict.keys():
        strength *= wingtip_name_and_strength_direction_dict[wingtip]

    l_hat = properties_ref['l_hat']
    far_wake_l_start = properties_ref['far_wake_l_start']
    x_start = x_center + far_wake_l_start * l_hat + radius_shed * r_hat_shed_rewound

    info_dict = {'x_start': x_start,
                 'l_hat': l_hat,
                 'r_core': r_core,
                 'strength': strength}

    fil = semi_infinite_filament.SemiInfiniteFilament(info_dict)
    fil.define_biot_savart_induction_function()

    x_obs = x_kite_obs
    value, num, den = fil.calculate_biot_savart_induction(info_dict, x_obs)
    return value, num, den


def get_induced_velocity_scaling_for_near_filament(model_options, geometry, element_number=0, kite_obs_index=0, kite_shed_index=0, inputs={}):

    position_in_horseshoe = vortex_tools.get_position_of_near_wake_element_in_horseshoe(model_options, element_number)
    associated_wake_node = vortex_tools.get_wake_node_whose_position_relates_to_velocity_element_number('near', element_number, model_options)

    properties_ref = vortex_tools.get_biot_savart_reference_object_properties(model_options, geometry=geometry, kite_obs_index=kite_obs_index, kite_shed_index=kite_shed_index, inputs=inputs)

    x_kite_obs = properties_ref['x_kite_obs']
    x_center = properties_ref['x_center']

    r_core = properties_ref['r_core']

    strength = properties_ref['filament_strength']

    if position_in_horseshoe == 'closing':
        rate_of_change_scaling_factor = model_options['aero']['vortex']['rate_of_change_scaling_factor']
        strength *= rate_of_change_scaling_factor

    wingtip_name_and_strength_direction_dict = vortex_tools.get_wingtip_name_and_strength_direction_dict()
    if position_in_horseshoe in wingtip_name_and_strength_direction_dict.keys():
        strength *= wingtip_name_and_strength_direction_dict[position_in_horseshoe]

    psi_shed = properties_ref['psi_shed']
    omega = properties_ref['omega']
    time_since_most_recently_shed_edge_was_shed = float(associated_wake_node) * properties_ref['shedding_delta_time']
    psi_when_most_recently_shed_edge_was_shed = psi_shed - omega * time_since_most_recently_shed_edge_was_shed
    psi_one_step_before_that = psi_when_most_recently_shed_edge_was_shed - omega * properties_ref['shedding_delta_time']

    a_hat = properties_ref['a_hat']
    b_hat = properties_ref['b_hat']
    r_hat_when_most_recently_shed_edge_was_shed = np.cos(psi_when_most_recently_shed_edge_was_shed) * a_hat + np.sin(psi_when_most_recently_shed_edge_was_shed) * b_hat
    r_hat_one_step_before_that = np.cos(psi_one_step_before_that) * a_hat + np.sin(psi_one_step_before_that) * b_hat

    u_ref = properties_ref['u_ref']
    downstream_dist_at_most_recently_shed_edge = time_since_most_recently_shed_edge_was_shed * u_ref
    l_hat = properties_ref['l_hat']
    center_convected_since_most_recently_shed_edge_was_shed = x_center + downstream_dist_at_most_recently_shed_edge * l_hat
    center_convected_since_one_step_before_that = center_convected_since_most_recently_shed_edge_was_shed - u_ref * properties_ref['shedding_delta_time'] * l_hat

    b_ref = properties_ref['b_ref']
    varrho_int = properties_ref['varrho_int']
    varrho_ext = properties_ref['varrho_ext']
    radius_int = varrho_int * b_ref
    radius_ext = varrho_ext * b_ref

    if position_in_horseshoe == 'closing':
        x_start = center_convected_since_most_recently_shed_edge_was_shed + radius_int * r_hat_when_most_recently_shed_edge_was_shed
        x_end = center_convected_since_most_recently_shed_edge_was_shed + radius_ext * r_hat_when_most_recently_shed_edge_was_shed

    elif position_in_horseshoe == 'int':
        x_start = center_convected_since_most_recently_shed_edge_was_shed + radius_int * r_hat_when_most_recently_shed_edge_was_shed
        x_end = center_convected_since_one_step_before_that + radius_int * r_hat_one_step_before_that

    elif position_in_horseshoe == 'ext':
        x_start = center_convected_since_most_recently_shed_edge_was_shed + radius_ext * r_hat_when_most_recently_shed_edge_was_shed
        x_end = center_convected_since_one_step_before_that + radius_ext * r_hat_one_step_before_that

    else:
        message = 'unexpected_position_in_horseshoe found (' + position_in_horseshoe + '), while setting up near-wake induced velocity scaling'
        print_op.warn_about_temporary_functionality_alteration(message)

    info_dict = {'x_start': x_start,
                 'x_end': x_end,
                 'r_core': r_core,
                 'strength': strength}

    fil = finite_filament.FiniteFilament(info_dict)
    fil.define_biot_savart_induction_function()

    x_obs = x_kite_obs
    value, num, den = fil.calculate_biot_savart_induction(info_dict, x_obs)
    return value, num, den
