#
#    This file is part of awebox.
#
#    awebox -- A modeling and optimization framework for multi-kite AWE systems.
#    Copyright (C) 2017-2025 Jochem De Schutter, Rachel Leuthold, Moritz Diehl,
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
various structural tools for the actuator models
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2019-2025
'''
import copy

import casadi.tools as cas
import numpy as np
from awebox.logger.logger import Logger as awelogger

import awebox.tools.struct_operations as struct_op
import awebox.tools.vector_operations as vect_op
import awebox.tools.print_operations as print_op

def extend_actuator_induction_factors(options, system_lifted, system_states, architecture):

    comparison_labels = options['aero']['induction']['comparison_labels']

    actuator_comp_labels = []
    for label in comparison_labels:
        if label[:3] == 'act':
            actuator_comp_labels += [label[4:]]

    for kite in architecture.kite_nodes:
        parent = architecture.parent_map[kite]
        system_lifted.extend([('local_a' + str(kite) + str(parent), (1, 1))])

    for layer_node in architecture.layer_nodes:
        for label in actuator_comp_labels:
            if label[0] == 'q':
                system_lifted.extend([('a_' + label + str(layer_node), (1, 1))])
            elif label[0] == 'u':
                system_states.extend([('a_' + label + str(layer_node), (1, 1))])

            if label == 'qasym':
                system_lifted.extend([('acos_' + label + str(layer_node), (1, 1))])
                system_lifted.extend([('asin_' + label + str(layer_node), (1, 1))])

            if label == 'uasym':
                system_states.extend([('acos_' + label + str(layer_node), (1, 1))])
                system_states.extend([('asin_' + label + str(layer_node), (1, 1))])
    return system_lifted, system_states

def extend_actuator_support(options, system_lifted, system_states, architecture):
    for kite in architecture.kite_nodes:
        parent = architecture.parent_map[kite]
        system_lifted.extend([('varrho' + str(kite) + str(parent), (1, 1))])
        system_lifted.extend([('psi' + str(kite) + str(parent), (1, 1))])
        # system_lifted.extend([('cospsi' + str(kite) + str(parent), (1, 1))])
        # system_lifted.extend([('sinpsi' + str(kite) + str(parent), (1, 1))])

    for layer_node in architecture.layer_nodes:
        system_lifted.extend([('bar_varrho' + str(layer_node), (1, 1))])
        system_lifted.extend([('area' + str(layer_node), (1, 1))])

        system_lifted.extend([('act_q' + str(layer_node), (3, 1))])
        system_lifted.extend([('act_dq' + str(layer_node), (3, 1))])

        system_lifted.extend([('gamma' + str(layer_node), (1, 1))])
        # system_lifted.extend([(get_actuator_vector_length_name('g', layer_node), (1, 1))])
        # system_lifted.extend([('cosgamma' + str(layer_node), (1, 1))])
        # system_lifted.extend([('singamma' + str(layer_node), (1, 1))])

        for dir in get_list_of_directions():
            system_lifted.extend([(get_actuator_vector_unit_name(dir, layer_node), (3, 1))])
            # system_lifted.extend([(get_actuator_vector_length_name(dir, layer_node), (1, 1))])

        system_lifted.extend([('thrust' + str(layer_node), (1, 1))])

    return system_lifted, system_states

def get_list_of_directions():
    return ['n', 'uzero']

def add_system_bounds_of_support_variables(options, help_options, options_tree):
    for dir in get_list_of_directions() + ['g']:
        options_tree.append(('model', 'system_bounds', 'z', get_actuator_vector_length_name_base(dir), [0., cas.inf], ('positive-direction parallel for actuator orientation [-]', None), 'x')),

    psi_epsilon = np.pi
    options_tree.append(('model', 'system_bounds', 'z', 'psi', [0. - psi_epsilon, 2. * np.pi + psi_epsilon],
                         ('azimuth-jumping bounds on the azimuthal angle derivative', None), 'x'))

    return options_tree


def add_scaling_of_support_variables(options, architecture, u_at_altitude, options_tree):

    scaling_dict = {}

    normal_vector_model = options['model']['aero']['actuator']['normal_vector_model']
    l_t = options['solver']['initialization']['l_t']
    number_of_kites = architecture.number_of_kites
    if normal_vector_model == 'least_squares':
        length = options['solver']['initialization']['theta']['l_s']
        n_vec_length_ref = length**2.
    elif normal_vector_model == 'binormal':
        n_vec_length_ref = number_of_kites * l_t**2.
    elif normal_vector_model == 'tether_parallel':
        n_vec_length_ref = l_t
    elif normal_vector_model == 'xhat':
        n_vec_length_ref = 1.
    else:
        n_vec_length_ref = 1.
    scaling_dict['n'] = n_vec_length_ref
    scaling_dict['z'] = cas.DM(1.)

    scaling_dict['uzero'] = u_at_altitude

    scaling_dict['g'] = cas.DM(1.)

    for dir, val in scaling_dict.items():
        var_name = get_actuator_vector_length_name_base(dir)
        options_tree.append(('model', 'scaling', 'z', var_name, val, ('descript', None), 'x'))
        options_tree.append(('solver', 'initialization', 'induction', var_name, val, ('descript', None), 'x'))

    psi_scale = 2. * np.pi
    options_tree.append(('model', 'scaling', 'z', 'psi', psi_scale, ('descript', None), 'x'))
    options_tree.append(('model', 'scaling', 'z', 'cospsi', 0.5, ('descript', None), 'x'))
    options_tree.append(('model', 'scaling', 'z', 'sinpsi', 0.5, ('descript', None), 'x'))

    return options_tree

def get_actuator_direction_name_base(direction):
    return 'act_' + direction

def get_actuator_vector_unit_name_base(direction):
    return get_actuator_direction_name_base(direction) + '_hat'

def get_actuator_vector_unit_name(direction, layer_node):
    return get_actuator_vector_unit_name_base(direction) + str(layer_node)

def get_actuator_vector_length_name_base(direction):
    return get_actuator_direction_name_base(direction) + '_vec_length'

def get_actuator_vector_length_name(direction, layer_node):
    return get_actuator_vector_length_name_base(direction) + str(layer_node)

def get_actuator_vector_unit_var(variables_si, direction, layer_node):
    var_type = 'z'
    var_name = get_actuator_vector_unit_name(direction, layer_node)
    var_val = struct_op.get_variable_from_model_or_reconstruction(variables_si, var_type, var_name)
    return var_val

def get_actuator_vector_length_var(variables_si, direction, layer_node):
    var_type = 'z'
    var_name = get_actuator_vector_length_name(direction, layer_node)
    var_val = struct_op.get_variable_from_model_or_reconstruction(variables_si, var_type, var_name)
    return var_val