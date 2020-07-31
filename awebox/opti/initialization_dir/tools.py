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
repeated tools to make initialization smoother
_python _version 2.7 / casadi-3.4.5
- _author: rachel leuthold, jochem de schutter, thilo bronnenmeyer (alu-fr, 2017 - 20)
'''


import numpy as np
import casadi.tools as cas
import awebox.tools.vector_operations as vect_op
from awebox.logger.logger import Logger as awelogger


def get_ehat_tether(init_options):
    inclination = init_options['inclination_deg'] * np.pi / 180.
    ehat_tether = np.cos(inclination) * vect_op.xhat() + np.sin(inclination) * vect_op.zhat()
    return ehat_tether

def get_rotor_reference_frame(init_options):
    n_rot_hat = get_ehat_tether(init_options)

    n_hat_is_x_hat = vect_op.abs(vect_op.norm(n_rot_hat - vect_op.xhat_np())) < 1.e-4
    if n_hat_is_x_hat:
        y_rot_hat = vect_op.yhat_np()
        z_rot_hat = vect_op.zhat_np()
    else:
        u_hat = vect_op.xhat_np()
        z_rot_hat = vect_op.normed_cross(u_hat, n_rot_hat)
        y_rot_hat = vect_op.normed_cross(z_rot_hat, n_rot_hat)

    return n_rot_hat, y_rot_hat, z_rot_hat

def get_ehat_radial(t, init_options, model, kite, ret={}):
    parent_map = model.architecture.parent_map
    level_siblings = model.architecture.get_all_level_siblings()

    parent = parent_map[kite]

    omega_norm = init_options['precompute']['angular_speed']
    psi = get_azimuthal_angle(t, level_siblings, kite, parent, omega_norm)

    ehat_radial = get_ehat_radial_from_azimuth(init_options, psi)

    return ehat_radial

def get_ehat_radial_from_azimuth(init_options, psi):
    _, y_rot_hat, z_rot_hat = get_rotor_reference_frame(init_options)

    cospsi_var = np.cos(psi)
    sinpsi_var = np.sin(psi)

    # for positive yaw(turns around +zhat, normal towards +yhat):
    #     rhat = zhat * cos(psi) - yhat * sin(psi)
    ehat_radial = z_rot_hat * cospsi_var - y_rot_hat * sinpsi_var

    return ehat_radial


def get_azimuthal_angle(t, level_siblings, node, parent, omega_norm):

    number_of_siblings = len(level_siblings[parent])
    if number_of_siblings == 1:
        psi0 = 0.
    else:
        idx = level_siblings[parent].index(node)
        psi0 = np.float(idx) / np.float(number_of_siblings) * 2. * np.pi

    psi = psi0 + omega_norm * t

    return psi








def insert_dict(dict, var_type, name, name_stripped, V_init):
    init_val = dict[name_stripped]

    for idx in range(init_val.shape[0]):
        V_init = insert_val(V_init, var_type, name, init_val[idx], idx)

    return V_init


def insert_val(V_init, var_type, name, init_val, idx = 0):

    # initialize on collocation nodes
    V_init['coll_var', :, :, var_type, name, idx] = init_val

    if var_type == 'xd':
        # initialize on interval nodes
        # V_init[var_type, :, :, name] = init_val

        V_init[var_type, :, name, idx] = init_val

    return V_init
