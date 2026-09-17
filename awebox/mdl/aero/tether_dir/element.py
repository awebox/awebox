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
drag on a tether element (a smaller portion of a tether segment)
_python-3.5 / casadi-3.4.5
- edited: rachel leuthold, jochem de schutter alu-fr 2017-20
'''

import awebox.tools.vector_operations as vect_op
import casadi.tools as cas

import awebox.mdl.aero.tether_dir.reynolds as reynolds
import awebox.tools.print_operations as print_op


def get_element_info_column(variables, upper_node, architecture, element, n_elements):
    q_upper, q_lower, dq_upper, dq_lower = get_element_upper_pos_and_vel(variables, upper_node, architecture, element, n_elements)
    diameter = get_element_diameter(variables, upper_node, architecture)
    info_column = columnize_element_info(q_upper=q_upper, q_lower=q_lower, dq_upper=dq_upper, dq_lower=dq_lower, diameter=diameter)
    return info_column

def columnize_element_info(q_upper=None, q_lower=None, dq_upper=None, dq_lower=None, diameter=None, ehat_1=None, ehat_3=None, alpha=None, kite_dynamic_pressure=None, air_velocity=None, unpacked_as_dict=None, kite_only=False):

    column = []

    if kite_only:
        listed_info_names = [(q_upper, 'q_upper'), (q_lower, 'q_lower'), (diameter, 'diameter'), (ehat_1, 'ehat_1'), (ehat_3, 'ehat_3'), (alpha, 'alpha'), (kite_dynamic_pressure, 'kite_dynamic_pressure'), (air_velocity, 'air_velocity')]
    else:
        listed_info_names = [(q_upper, 'q_upper'), (q_lower, 'q_lower'), (dq_upper, 'dq_upper'), (dq_lower, 'dq_lower'), (diameter, 'diameter')]

    for info_tuple in listed_info_names:
        given = info_tuple[0]
        name_in_dict = info_tuple[1]

        if (given is not None):
            column = cas.vertcat(column, given)
        elif name_in_dict in unpacked_as_dict.keys():
            column = cas.vertcat(column, unpacked_as_dict[name_in_dict])
        else:
            message = 'tether element drag input ' + name_in_dict + ' not available.'
            print_op.log_and_raise_error(message)

    return column

def get_size_of_element_info_column(kite_only=False):
    if kite_only:
        return (18, 1)
    else:
        return (13, 1)

def unpack_element_info_column(info_column, kite_only=False):

    if kite_only:
        q_upper = info_column[:3]
        q_lower = info_column[3:6]
        diameter = info_column[6:7]
        ehat_1 = info_column[7:10]
        ehat_3 = info_column[10:13]
        alpha = info_column[13:14]
        kite_dynamic_pressure = info_column[14:15]
        air_velocity = info_column[15:18]
        unpacked_as_dict = {'q_upper': q_upper,
                            'q_lower': q_lower,
                            'diameter': diameter,
                            'ehat_1': ehat_1,
                            'ehat_3': ehat_3,
                            'alpha': alpha,
                            'kite_dynamic_pressure': kite_dynamic_pressure,
                            'air_velocity': air_velocity
                            }
    else:
        q_upper = info_column[:3]
        q_lower = info_column[3:6]
        dq_upper = info_column[6:9]
        dq_lower = info_column[9:12]
        diam = info_column[12:13]
        unpacked_as_dict = {'q_upper': q_upper,
                            'q_lower': q_lower,
                            'dq_upper': dq_upper,
                            'dq_lower': dq_lower,
                            'diameter': diam}
    return unpacked_as_dict




def get_uapp(q_upper, q_lower, dq_upper, dq_lower, wind):
    q_average = (q_upper + q_lower) / 2.
    zz = q_average[2]
    uw_average = wind.get_velocity(zz)
    dq_average = (dq_upper + dq_lower) / 2.
    ua = uw_average - dq_average
    return ua

def get_element_drag_fun(wind, atmos, parameters, cd_tether_fun, reynolds_fun=None):
    info_to_add_to_applied_params_dict = {}
    info_sym = cas.SX.sym('info_sym', (13, 1))

    unpacked = unpack_element_info_column(info_sym)
    q_upper = unpacked['q_upper']
    q_lower = unpacked['q_lower']
    dq_upper = unpacked['dq_upper']
    dq_lower = unpacked['dq_lower']
    diam = unpacked['diameter']

    q_average = (q_upper + q_lower) / 2.
    zz = q_average[2]

    ua = get_uapp(q_upper, q_lower, dq_upper, dq_lower, wind)

    epsilon = 1.e-6

    ua_norm = vect_op.smooth_norm(ua, epsilon)
    ehat_ua = vect_op.smooth_normalize(ua, epsilon)

    tether = q_upper - q_lower

    length_sq = cas.mtimes(tether.T, tether)
    length_parallel_to_wind = cas.mtimes(tether.T, ehat_ua)
    length_perp_to_wind = vect_op.smooth_sqrt(length_sq - length_parallel_to_wind**2., epsilon**2.)

    if reynolds_fun is not None:
        re_number = reynolds_fun(q_average, ua, diam, parameters)
    else:
        re_number = reynolds.get_reynolds_number(atmos, diam=diam, ua_local=ua, q_upper=q_upper, q_lower=q_lower)

    cd = cd_tether_fun(re_number, parameters)

    density = atmos.get_density(zz)
    drag = cd * 0.5 * density * ua_norm * diam * length_perp_to_wind * ua

    element_drag_fun = cas.Function('element_drag_fun', [info_sym, parameters], [drag])

    return element_drag_fun, info_to_add_to_applied_params_dict

def get_element_diameter(variables, upper_node, architecture):

    parent_map = architecture.parent_map
    lower_node = parent_map[upper_node]

    main_tether = (lower_node == 0)
    secondary_tether = (upper_node in architecture.kite_nodes)

    if main_tether:
        diam = variables['theta']['diam_t']
    elif secondary_tether:
        diam = variables['theta']['diam_s']
    else:
        # todo: add intermediate tether diameter
        diam = variables['theta']['diam_t']

    return diam


def get_upper_and_lower_pos_and_vel(variables, upper_node, architecture):
    parent_map = architecture.parent_map
    
    lower_node = parent_map[upper_node]
    q_upper = variables['x']['q' + str(upper_node) + str(lower_node)]
    dq_upper = variables['x']['dq' + str(upper_node) + str(lower_node)]

    if lower_node == 0:
        q_lower = cas.DM.zeros((3, 1))
        dq_lower = cas.DM.zeros((3, 1))
    else:
        grandparent = parent_map[lower_node]
        q_lower = variables['x']['q' + str(lower_node) + str(grandparent)]
        dq_lower = variables['x']['dq' + str(lower_node) + str(grandparent)]
        
    return q_upper, q_lower, dq_upper, dq_lower


def get_info_column_linear_division_of_tether_segment(segment_info_dict, element, n_elements):
    # divides a tether linearly into n_elements equal elements

    q_top = segment_info_dict['q_upper']
    q_bottom = segment_info_dict['q_lower']
    dq_top = segment_info_dict['dq_upper']
    dq_bottom = segment_info_dict['dq_lower']
    diam = segment_info_dict['diameter']

    lower_phi = float(element) / float(n_elements)
    upper_phi = float(element + 1) / float(n_elements)

    q_lower = q_bottom + (q_top - q_bottom) * lower_phi
    q_upper = q_bottom + (q_top - q_bottom) * upper_phi

    dq_lower = dq_bottom + (dq_top - dq_bottom) * lower_phi
    dq_upper = dq_bottom + (dq_top - dq_bottom) * upper_phi
    element_info = columnize_element_info(q_upper=q_upper, q_lower=q_lower, dq_upper=dq_upper, dq_lower=dq_lower, diameter=diam)

    return element_info

def get_element_upper_pos_and_vel(variables, upper_node, architecture, element, n_elements):
    # divides a tether linearly into n_elements equal elements
    q_top, q_bottom, dq_top, dq_bottom = get_upper_and_lower_pos_and_vel(variables, upper_node, architecture)
    segment_info_dict = {'q_upper': q_top,
                         'q_lower': q_bottom,
                         'dq_upper': dq_top,
                         'dq_lower': dq_bottom,
                         'diameter': None}
    element_info = get_info_column_linear_division_of_tether_segment(segment_info_dict, element, n_elements)
    return element_info['q_upper'], element_info['q_lower'], element_info['dq_upper'], element_info['dq_lower']