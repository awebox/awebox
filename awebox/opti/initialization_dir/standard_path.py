#
#    This file is part of awebox.
#
#    awebox -- A modeling and optimization framework for multi-kite AWE systems.
#    Copyright (C) 2017-2020 Jochem De Schutter, Rachel Leuthold, Moritz Diehl,
#                            ALU Freiburg.
#    Copyright (C) 2018-2019 Thilo Bronnenmeyer, Kiteswarms Ltd.
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
initialization functions specific to the standard path scenario
_python _version 2.7 / casadi-3.4.5
- _author: rachel leuthold, jochem de schutter, thilo bronnenmeyer (alu-fr, 2017 - 20)
'''

import numpy as np
import casadi.tools as cas
import awebox.tools.vector_operations as vect_op
from awebox.logger.logger import Logger as awelogger
import awebox.tools.struct_operations as struct_op
import awebox.opti.initialization_dir.induction as induction
import awebox.opti.initialization_dir.tools as tools


def get_normalized_time_param_dict(ntp_dict, formulation):
    n_min = 0
    d_min = 0

    ntp_dict['n_min'] = n_min
    ntp_dict['d_min'] = d_min

    return ntp_dict

def set_normalized_time_params(formulation, V_init):
    xi_0_init = 0.0
    xi_f_init = 0.0

    V_init['xi', 'xi_0'] = xi_0_init
    V_init['xi', 'xi_f'] = xi_f_init

    return V_init

def guess_radius(init_options, model):
    tether_length, max_cone_angle = tools.get_hypotenuse_and_max_cone_angle(model, init_options)

    windings = init_options['windings']
    winding_period = init_options['winding_period']

    winding_period = tools.clip_winding_period(init_options, model.wind, winding_period)
    tf_guess = windings * winding_period

    init_options = tools.clip_speed_and_reset_options(init_options, model.wind)
    dq_kite_norm = init_options['dq_kite_norm']

    total_distance = dq_kite_norm * tf_guess
    circumference = total_distance / windings
    radius = circumference / 2. / np.pi

    radius = tools.clip_radius(init_options, max_cone_angle, tether_length, radius)

    return radius

def guess_final_time(init_options, model):

    windings = init_options['windings']
    radius = guess_radius(init_options, model)

    total_distance = 2. * np.pi * radius * windings

    dq_kite_norm = init_options['dq_kite_norm']
    tf_guess = total_distance / dq_kite_norm

    return tf_guess

def guess_values_at_time(t, init_options, model):
    ret = {}
    for name in struct_op.subkeys(model.variables, 'xd'):
        ret[name] = 0.0
    ret['e'] = 0.0

    ret['l_t'] = init_options['xd']['l_t']
    ret['dl_t'] = 0.0

    number_of_nodes = model.architecture.number_of_nodes
    parent_map = model.architecture.parent_map
    kite_nodes = model.architecture.kite_nodes

    kite_dof = model.kite_dof

    height_list, radius = tools.get_cone_height_and_radius(init_options, model, ret['l_t'])

    for node in range(1, number_of_nodes):

        parent = parent_map[node]
        if parent == 0:
            parent_position = np.zeros((3, 1))
        else:
            grandparent = parent_map[parent]
            parent_position = ret['q' + str(parent) + str(grandparent)]

        if not node in kite_nodes:
            ret['q' + str(node) + str(parent)] = get_tether_node_position(init_options, parent_position, node, ret['l_t'])
            ret['dq' + str(node) + str(parent)] = np.zeros((3, 1))

        else:
            if parent == 0:
                height = height_list[0]
            else:
                height = height_list[1]

            speed = init_options['dq_kite_norm']

            omega_norm = speed / radius

            n_rot_hat, y_rot_hat, z_rot_hat = tools.get_rotor_reference_frame(init_options)

            ehat_normal = n_rot_hat
            ehat_radial = tools.get_ehat_radial(t, init_options, model, node, ret)
            ehat_tangential = vect_op.normed_cross(ehat_normal, ehat_radial)

            omega_vector = ehat_normal * omega_norm

            tether_vector = ehat_radial * radius + ehat_normal * height
            position = parent_position + tether_vector

            velocity = speed * ehat_tangential

            ehat1 = -1. * ehat_tangential
            ehat3 = n_rot_hat
            ehat2 = vect_op.normed_cross(ehat3, ehat1)

            dcm = cas.horzcat(ehat1, ehat2, ehat3)
            if init_options['cross_tether']:
                if init_options['cross_tether_attachment'] in ['com','stick']:
                    dcm = get_cross_tether_dcm(init_options, dcm)
            dcm_column = cas.reshape(dcm, (9, 1))

            ret['q' + str(node) + str(parent)] = position
            ret['dq' + str(node) + str(parent)] = velocity

            if int(kite_dof) == 6:
                ret['omega' + str(node) + str(parent)] = omega_vector
                ret['r' + str(node) + str(parent)] = dcm_column

    return ret


def get_tether_node_position(init_options, parent_position, node, l_t):

    ehat_tether = tools.get_ehat_tether(init_options)

    seg_length = init_options['theta']['l_i']
    if node == 1:
        seg_length = l_t

    position = parent_position + seg_length * ehat_tether

    return position

def get_cross_tether_dcm(init_options, dcm):
    ang = -init_options['rotation_bounds'] * 1.05
    rotx = np.array([[1, 0, 0], [0, np.cos(ang), -np.sin(ang)], [0, np.sin(ang), np.cos(ang)]])
    dcm = cas.mtimes(dcm, rotx)
    return dcm


