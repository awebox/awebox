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



def get_element_info_column(variables, upper_node, architecture, element, n_elements):

    q_upper, q_lower, dq_upper, dq_lower = get_element_upper_pos_and_vel(variables, upper_node, architecture, element, n_elements)
    diam = get_element_diameter(variables, upper_node, architecture)

    info_column = cas.vertcat(q_upper, q_lower, dq_upper, dq_lower, diam)

    return info_column


def get_uapp(q_upper, q_lower, dq_upper, dq_lower, wind):

    q_average = (q_upper + q_lower) / 2.
    zz = q_average[2]

    uw_average = wind.get_velocity(zz)

    dq_average = (dq_upper + dq_lower) / 2.
    ua = uw_average - dq_average

    return ua

def get_element_drag_fun(wind, atmos, cd_tether_fun):

    info_sym = cas.SX.sym('info_sym', (13, 1))

    q_upper = info_sym[:3]
    q_lower = info_sym[3:6]
    dq_upper = info_sym[6:9]
    dq_lower = info_sym[9:12]
    diam = info_sym[12]

    q_average = (q_upper + q_lower) / 2.
    zz = q_average[2]

    ua = get_uapp(q_upper, q_lower, dq_upper, dq_lower, wind)

    epsilon = 1.e-6

    ua_norm = vect_op.smooth_norm(ua, epsilon)
    ehat_ua = vect_op.smooth_normalize(ua, epsilon)

    tether = q_upper - q_lower

    length_sq = cas.mtimes(tether.T, tether)
    length_parallel_to_wind = cas.mtimes(tether.T, ehat_ua)
    length_perp_to_wind = vect_op.smooth_sqrt(length_sq - length_parallel_to_wind**2., epsilon)

    re_number = reynolds.get_reynolds_number(atmos, ua, diam, q_upper, q_lower)
    cd = cd_tether_fun(re_number)

    density = atmos.get_density(zz)
    drag = cd * 0.5 * density * ua_norm * diam * length_perp_to_wind * ua

    element_drag_fun = cas.Function('element_drag_fun', [info_sym], [drag])

    return element_drag_fun

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
    q_upper = variables['xd']['q' + str(upper_node) + str(lower_node)]
    dq_upper = variables['xd']['dq' + str(upper_node) + str(lower_node)]

    if lower_node == 0:
        q_lower = cas.DM.zeros((3, 1))
        dq_lower = cas.DM.zeros((3, 1))
    else:
        grandparent = parent_map[lower_node]
        q_lower = variables['xd']['q' + str(lower_node) + str(grandparent)]
        dq_lower = variables['xd']['dq' + str(lower_node) + str(grandparent)]
        
    return q_upper, q_lower, dq_upper, dq_lower


def get_element_upper_pos_and_vel(variables, upper_node, architecture, element, n_elements):
    # divides a tether linearly into n_elements equal elements

    q_top, q_bottom, dq_top, dq_bottom = get_upper_and_lower_pos_and_vel(variables, upper_node, architecture)

    lower_phi = float(element) / float(n_elements)
    upper_phi = float(element + 1) / float(n_elements)

    q_lower = q_bottom + (q_top - q_bottom) * lower_phi
    q_upper = q_bottom + (q_top - q_bottom) * upper_phi

    dq_lower = dq_bottom + (dq_top - dq_bottom) * lower_phi
    dq_upper = dq_bottom + (dq_top - dq_bottom) * upper_phi

    return q_upper, q_lower, dq_upper, dq_lower