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
    inputs = get_scaling_inputs(options, geometry, architecture, u_altitude, CL, varrho_ref, winding_period)

    options_tree = append_geometric_scaling(options, geometry, options_tree, architecture, inputs, q_scaling, varrho_ref, winding_period)
    options_tree = append_induced_velocity_scaling(options, geometry, options_tree, architecture, inputs, u_altitude)
    return options_tree


def get_filament_strength(options, geometry, u_altitude, CL, varrho_ref, winding_period):
    c_ref = geometry['c_ref']
    b_ref = geometry['b_ref']

    a_ref = options['model']['aero']['actuator']['a_ref']

    flight_radius = varrho_ref * b_ref
    rotational_speed = 2. * np.pi * flight_radius / winding_period
    axial_speed = u_altitude * (1 - a_ref)
    airspeed = (rotational_speed**2. + axial_speed**2.)**0.5

    if not (options['model']['aero']['overwrite']['f_aero_rot'] is None):
        # L/b = rho v gamma
        # gamma = L / (b rho v)
        rho_ref = options['params']['atmosphere']['rho_ref']
        gamma = vect_op.norm(options['model']['aero']['overwrite']['f_aero_rot']) / (b_ref * rho_ref * airspeed)
        filament_strength = gamma
    else:
        # gamma = L / (b rho v) = (CL/2) (rho v^2 b c) / (b rho v) = (CL/2) (v c)
        filament_strength = 0.5 * CL * airspeed * c_ref

    strength_dict = {'CL': CL, 'varrho_ref': varrho_ref, 'airspeed': airspeed, 'c_ref': c_ref, 'strength': filament_strength}
    print_op.base_print('initialization values related to filament strength are:')
    print_op.print_dict_as_table(strength_dict)

    return filament_strength


def append_geometric_scaling(options, geometry, options_tree, architecture, inputs, q_scaling, varrho_ref, winding_period):
    # the part that describes the wake nodes and consequent vortex rings
    wingtips = ['ext', 'int']
    wake_nodes = options['model']['aero']['vortex']['wake_nodes']
    rings = options['model']['aero']['vortex']['wake_nodes']

    u_ref = options['user_options']['wind']['u_ref']

    filament_strength = inputs['filament_strength_ref']

    properties_ref = vortex_tools.get_biot_savart_reference_object_properties(options, geometry=geometry, kite_obs_index=0, kite_shed_index=0, inputs=inputs)
    avg_downstream = properties_ref['far_wake_l_start'] / 2.
    near_wake_unit_length = properties_ref['near_wake_unit_length']
    position_scaling_method = options['model']['aero']['vortex']['position_scaling_method']
    if position_scaling_method == 'q10':
        position_scaling = q_scaling
    elif position_scaling_method == 'convection':
        position_scaling = q_scaling
        # add the convection below
    elif position_scaling_method == 'average':
        position_scaling = q_scaling + avg_downstream * vect_op.xhat_dm()
    elif position_scaling_method in ['radius', 'b_ref', 'c_ref']:
        position_scaling = properties_ref[position_scaling_method]
    elif position_scaling_method in ['wingspan', 'span']:
        position_scaling = properties_ref['b_ref']
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

            if position_scaling_method == 'convection':
                wx_scale = q_scaling + near_wake_unit_length * wake_node * vect_op.xhat_dm()

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
    options_tree.append(('solver', 'initialization', 'induction', 'vortex_gamma_scale', wg_scale, ('????', None), 'x')),

    circulation_max_estimate = 2.5 * wg_scale
    options_tree.append(('model', 'aero', 'vortex', 'filament_strength_ref', wg_scale, ('????', None), 'x')),
    options_tree.append(('visualization', 'cosmetics', 'trajectory', 'circulation_max_estimate', circulation_max_estimate, ('????', None), 'x')),

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
    filament_strength = get_filament_strength(options, geometry, u_altitude, CL, varrho_ref, winding_period)

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


def append_induced_velocity_scaling(options, geometry, options_tree, architecture, inputs, u_altitude):

    u_ref = options['user_options']['wind']['u_ref']
    a_ref = options['model']['aero']['actuator']['a_ref']
    expected_number_of_elements_dict_for_wake_types = vortex_tools.get_expected_number_of_elements_dict_for_wake_types(
        options,
        architecture)

    degree_of_induced_velocity_lifting = options['model']['aero']['vortex']['degree_of_induced_velocity_lifting']
    if degree_of_induced_velocity_lifting < 1:
        message = 'unexpected degree_of_induced_velocity_lifting (' + str(degree_of_induced_velocity_lifting) + ').'
        print_op.log_and_raise_error(message)

    wu_ind_scale = u_ref * a_ref
    for kite_obs in architecture.kite_nodes:
        var_name = vortex_tools.get_induced_velocity_at_kite_name(kite_obs)
        options_tree.append(('model', 'scaling', 'z', var_name, wu_ind_scale, ('descript', None), 'x'))

    for kite_obs in architecture.kite_nodes:
        for wake_type, local_expected_number_of_elements_dict in expected_number_of_elements_dict_for_wake_types.items():
            for element_type, expected_number in local_expected_number_of_elements_dict.items():
                for element_number in range(expected_number):

                    kite_shed = vortex_tools.get_shedding_kite_from_element_number(options, wake_type, element_type, element_number, architecture)

                    kite_shed_index = architecture.kite_nodes.index(kite_shed)
                    kite_obs_index = architecture.kite_nodes.index(kite_obs)

                    if wake_type == 'bound':
                        value, num, den = get_induced_velocity_scaling_for_bound_filament(options, architecture, geometry, u_altitude, element_number=element_number, kite_obs_index=kite_obs_index, kite_shed_index=kite_shed_index, inputs=inputs)
                    elif wake_type == 'far':
                        value, num, den = get_induced_velocity_scaling_for_far_filament(options, architecture, geometry, u_altitude, element_number=element_number, kite_obs_index=kite_obs_index, kite_shed_index=kite_shed_index, inputs=inputs)
                    elif wake_type == 'near':
                        value, num, den = get_induced_velocity_scaling_for_near_filament(options, architecture, geometry, u_altitude, element_number=element_number, kite_obs_index=kite_obs_index, kite_shed_index=kite_shed_index, inputs=inputs)
                    else:
                        message = 'unfamiliar wake type (' + wake_type + ') when generating induced velocity scaling values'
                        print_op.log_and_raise_error(message)

                    # this just happens to work. :(
                    num *= 10
                    value *= 10
                    den *= 10

                    num_scaling = vect_op.smooth_norm(num) * cas.DM.ones((3, 1))
                    num_name = vortex_tools.get_element_biot_savart_numerator_name(wake_type, element_type, element_number, kite_obs)
                    options_tree.append(
                        ('model', 'scaling', 'z', num_name, num_scaling, ('descript', None), 'x'))

                    den_scaling = vect_op.smooth_abs(den)
                    den_name = vortex_tools.get_element_biot_savart_denominator_name(wake_type, element_type, element_number, kite_obs)
                    options_tree.append(
                        ('model', 'scaling', 'z', den_name, den_scaling, ('descript', None), 'x'))

                    value_scaling = vect_op.smooth_norm(value) * cas.DM.ones((3, 1))
                    val_name = vortex_tools.get_element_induced_velocity_name(wake_type, element_type, element_number, kite_obs)
                    options_tree.append(
                        ('model', 'scaling', 'z', val_name, value_scaling, ('descript', None), 'x'))

    return options_tree


def get_induced_velocity_scaling_for_bound_filament(options, architecture, geometry, u_altitude, element_number=0, kite_obs_index=0, kite_shed_index=0, inputs={}):

    properties_ref = vortex_tools.get_biot_savart_reference_object_properties(options, geometry=geometry, kite_obs_index=kite_obs_index, kite_shed_index=kite_shed_index, inputs=inputs)

    offset = cas.DM(0.)
    ehat_3 = properties_ref['ehat_3']

    r_core = properties_ref['r_core']
    strength = properties_ref['filament_strength']

    x_kite_int = get_convected_wake_node_position(options, architecture, properties_ref, u_altitude, kite_obs_index,
                                               0, 'int')
    x_kite_ext = get_convected_wake_node_position(options, architecture, properties_ref, u_altitude, kite_obs_index,
                                             0, 'ext')
    x_kite_obs = (x_kite_int + x_kite_ext) / 2.
    x_obs = x_kite_obs + offset * ehat_3

    x_shed_int = get_convected_wake_node_position(options, architecture, properties_ref, u_altitude, kite_shed_index,
                                               0, 'int')
    x_shed_ext = get_convected_wake_node_position(options, architecture, properties_ref, u_altitude, kite_shed_index,
                                             0, 'ext')
    x_kite_shed = (x_shed_int + x_shed_ext) / 2.
    x_start = x_shed_int
    x_end = x_shed_ext

    info_dict = {'x_start': x_start,
                 'x_end': x_end,
                 'r_core': r_core,
                 'strength': strength}

    fil = finite_filament.FiniteFilament(info_dict)
    fil.define_biot_savart_induction_function()

    value, num, den = fil.calculate_biot_savart_induction(info_dict, x_obs)

    # unfortunately, for even numbers of kites-per-layer, there are bound vortices that are co-linear with the kite observation points.
    # so, keep the denominator, but approximate the numerator terms
    distance = vect_op.smooth_norm(x_kite_shed - x_kite_obs)
    # remember, it's 1/r^2 in 3D and 1/r in 2D. and the initialization is circular, therefore planar.
    estimated_magnitude = strength / distance
    value = estimated_magnitude * vect_op.xhat_dm()

    num = value * den

    return value, num, den


def get_induced_velocity_scaling_for_far_filament(options, architecture, geometry, u_altitude, element_number=0, kite_obs_index=0, kite_shed_index=0, inputs={}):
    wingtip = vortex_tools.get_which_wingtip_shed_this_far_wake_element(element_number)

    properties_ref = vortex_tools.get_biot_savart_reference_object_properties(options, geometry=geometry, kite_obs_index=kite_obs_index, kite_shed_index=kite_shed_index, inputs=inputs)

    r_core = properties_ref['r_core']
    strength = properties_ref['filament_strength']

    wingtip_name_and_strength_direction_dict = vortex_tools.get_wingtip_name_and_strength_direction_dict()
    if wingtip in wingtip_name_and_strength_direction_dict.keys():
        strength *= wingtip_name_and_strength_direction_dict[wingtip]

    l_hat = properties_ref['l_hat']

    wake_node = inputs['wake_nodes'] - 1
    x_start = get_convected_wake_node_position(options, architecture, properties_ref, u_altitude, kite_shed_index, wake_node, wingtip)

    info_dict = {'x_start': x_start,
                 'l_hat': l_hat,
                 'r_core': r_core,
                 'strength': strength}

    fil = semi_infinite_filament.SemiInfiniteFilament(info_dict)
    fil.define_biot_savart_induction_function()

    x_kite_int = get_convected_wake_node_position(options, architecture, properties_ref, u_altitude, kite_obs_index,
                                               0, 'int')
    x_kite_ext = get_convected_wake_node_position(options, architecture, properties_ref, u_altitude, kite_obs_index,
                                             0, 'ext')
    x_kite_obs = (x_kite_int + x_kite_ext) / 2.
    x_obs = x_kite_obs

    value, num, den = fil.calculate_biot_savart_induction(info_dict, x_obs)
    return value, num, den


def get_convected_wake_node_position(options, architecture, properties_ref, u_altitude, kite_shed_index, wake_node, tip):
    omega = properties_ref['omega']
    period = properties_ref['winding_period']
    n_k = options['nlp']['n_k']

    number_of_siblings_shed = len(architecture.siblings_map[architecture.kite_nodes[kite_shed_index]])

    psi_zero = 0.
    psi_shift_shed = 2. * np.pi * kite_shed_index / number_of_siblings_shed

    time_current = 0.
    delta_t = period / n_k
    time_when_associate_wake_node_was_shed = time_current - delta_t * wake_node
    psi_when_associate_wake_node_was_shed = psi_zero + psi_shift_shed + time_when_associate_wake_node_was_shed * omega

    a_hat = properties_ref['a_hat']
    b_hat = properties_ref['b_hat']
    r_hat_when_associate_wake_node_was_shed = np.cos(psi_when_associate_wake_node_was_shed) * a_hat + np.sin(psi_when_associate_wake_node_was_shed) * b_hat

    x_zero = cas.DM.zeros((3, 1))
    l_hat = vect_op.xhat_dm()
    x_center_convected_from_when_associate_wake_node_was_shed = x_zero + u_altitude * (time_current - time_when_associate_wake_node_was_shed) * l_hat

    b_ref = properties_ref['b_ref']
    radius = properties_ref['varrho_' + tip] * b_ref

    x_convected = x_center_convected_from_when_associate_wake_node_was_shed + radius * r_hat_when_associate_wake_node_was_shed
    return x_convected

def get_induced_velocity_scaling_for_near_filament(options, architecture, geometry, u_altitude, element_number=0, kite_obs_index=0, kite_shed_index=0, inputs={}):

    model_options = options['model']
    position_in_horseshoe = vortex_tools.get_position_of_near_wake_element_in_horseshoe(model_options, element_number)
    associated_wake_node = vortex_tools.get_wake_node_whose_position_relates_to_velocity_element_number('near', element_number, model_options)

    properties_ref = vortex_tools.get_biot_savart_reference_object_properties(options, geometry=geometry, kite_obs_index=kite_obs_index, kite_shed_index=kite_shed_index, inputs=inputs)

    r_core = properties_ref['r_core']

    strength = properties_ref['filament_strength']
    rate_of_change_scaling_factor = model_options['aero']['vortex']['rate_of_change_scaling_factor']
    wingtip_name_and_strength_direction_dict = vortex_tools.get_wingtip_name_and_strength_direction_dict()

    x_kite_int = get_convected_wake_node_position(options, architecture, properties_ref, u_altitude, kite_obs_index,
                                               0, 'int')
    x_kite_ext = get_convected_wake_node_position(options, architecture, properties_ref, u_altitude, kite_obs_index,
                                             0, 'ext')
    x_kite_obs = (x_kite_int + x_kite_ext) / 2.

    if position_in_horseshoe == 'closing':
        x_start = get_convected_wake_node_position(options, architecture, properties_ref, u_altitude, kite_shed_index,
                                                   associated_wake_node, 'int')
        x_end = get_convected_wake_node_position(options, architecture, properties_ref, u_altitude, kite_shed_index,
                                                   associated_wake_node, 'ext')
        strength *= rate_of_change_scaling_factor

    elif position_in_horseshoe in ['int', 'ext']:
        x_start = get_convected_wake_node_position(options, architecture, properties_ref, u_altitude, kite_shed_index,
                                                   associated_wake_node, position_in_horseshoe)
        x_end = get_convected_wake_node_position(options, architecture, properties_ref, u_altitude, kite_shed_index,
                                                   associated_wake_node + 1, position_in_horseshoe)
        strength *= wingtip_name_and_strength_direction_dict[position_in_horseshoe]

    else:
        message = 'unexpected_position_in_horseshoe found (' + position_in_horseshoe + '), while setting up near-wake induced velocity scaling'
        print_op.log_and_raise_error(message)

    info_dict = {'x_start': x_start,
                 'x_end': x_end,
                 'r_core': r_core,
                 'strength': strength}

    fil = finite_filament.FiniteFilament(info_dict)
    fil.define_biot_savart_induction_function()

    x_obs = x_kite_obs
    value, num, den = fil.calculate_biot_savart_induction(info_dict, x_obs)

    return value, num, den
