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
repeated tools to make initialization smoother
_python _version 2.7 / casadi-3.4.5
- _author: rachel leuthold, jochem de schutter, thilo bronnenmeyer (alu-fr, 2017 - 20)
'''


import numpy as np
import casadi.tools as cas
import awebox.tools.vector_operations as vect_op
from awebox.logger.logger import Logger as awelogger
import awebox.mdl.wind as wind


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

def get_rotating_reference_frame(t, init_options, model, node, ret):

    n_rot_hat = get_ehat_tether(init_options)

    ehat_normal = n_rot_hat
    ehat_radial = get_ehat_radial(t, init_options, model, node, ret)
    ehat_tangential = vect_op.normed_cross(ehat_normal, ehat_radial)

    return ehat_normal, ehat_radial, ehat_tangential

def get_ehat_radial(t, init_options, model, kite, ret={}):
    parent_map = model.architecture.parent_map
    level_siblings = model.architecture.get_all_level_siblings()

    parent = parent_map[kite]

    omega_norm = init_options['precompute']['angular_speed']
    psi = get_azimuthal_angle(t, init_options, level_siblings, kite, parent, omega_norm)

    ehat_radial = get_ehat_radial_from_azimuth(init_options, psi)

    return ehat_radial

def get_rotation_direction_sign(init_options):
    # rotation with right hand rule about + (positive) nhat
    clockwise_rotation_about_xhat = init_options['clockwise_rotation_about_xhat']
    if clockwise_rotation_about_xhat:
        sign = +1
    else:
        sign = -1.

    return sign

def get_ehat_radial_from_azimuth(init_options, psi):
    _, y_rot_hat, z_rot_hat = get_rotor_reference_frame(init_options)

    cospsi_var = np.cos(psi)
    sinpsi_var = np.sin(psi)

    sign = get_rotation_direction_sign(init_options)

    # for positive yaw(turns around +zhat, normal towards +yhat):
    #     rhat = zhat * cos(psi) - yhat * sin(psi)
    ehat_radial = z_rot_hat * cospsi_var - sign * y_rot_hat * sinpsi_var

    return ehat_radial

def get_dependent_rotation_direction_sign(t, init_options, model, node, ret):
    velocity = get_velocity_vector(t, init_options, model, node, ret)
    ehat_normal, ehat_radial, ehat_tangential = get_rotating_reference_frame(t, init_options, model, node, ret)
    forwards_speed = cas.mtimes(velocity.T, ehat_tangential)
    forwards_sign = forwards_speed / vect_op.norm(forwards_speed)

    return forwards_sign


def get_omega_vector(t, init_options, model, node, ret):

    forwards_sign = get_dependent_rotation_direction_sign(t, init_options, model, node, ret)
    omega_norm = init_options['precompute']['angular_speed']
    ehat_normal, ehat_radial, ehat_tangential = get_rotating_reference_frame(t, init_options, model, node, ret)

    omega_vector = forwards_sign * ehat_normal * omega_norm

    return omega_vector

def get_dpsi(init_options):
    omega_norm = init_options['precompute']['angular_speed']
    dpsi = omega_norm
    return dpsi

def get_azimuthal_angle(t, init_options, level_siblings, node, parent, omega_norm):
    number_of_siblings = len(level_siblings[parent])

    psi0_base = init_options['psi0_rad']

    if number_of_siblings == 1:
        psi0 = psi0_base + 0.
    else:
        idx = level_siblings[parent].index(node)
        psi0 = psi0_base + np.float(idx) / np.float(number_of_siblings) * 2. * np.pi

    psi = psi0 + omega_norm * t

    return psi

def get_velocity_vector(t, init_options, model, node, ret):

    groundspeed = init_options['precompute']['groundspeed']
    sign = get_rotation_direction_sign(init_options)

    ehat_normal, ehat_radial, ehat_tangential = get_rotating_reference_frame(t, init_options, model, node, ret)
    velocity = sign * groundspeed * ehat_tangential
    return velocity

def get_velocity_vector_from_psi(init_options, groundspeed, psi):

    n_rot_hat, _, _ = get_rotor_reference_frame(init_options)
    ehat_normal = n_rot_hat
    ehat_radial = get_ehat_radial_from_azimuth(init_options, psi)
    ehat_tangential = vect_op.normed_cross(ehat_normal, ehat_radial)
    sign = get_rotation_direction_sign(init_options)
    velocity = sign * groundspeed * ehat_tangential
    return velocity

def get_kite_dcm(t, init_options, model, node, ret):

    velocity = get_velocity_vector(t, init_options, model, node, ret)
    ehat_normal, ehat_radial, ehat_tangential = get_rotating_reference_frame(t, init_options, model, node, ret)

    forwards_speed = cas.mtimes(velocity.T, ehat_tangential)
    forwards_sign = forwards_speed / vect_op.norm(forwards_speed)
    ehat_forwards = forwards_sign * ehat_tangential

    ehat1 = -1. * ehat_forwards
    ehat3 = ehat_normal
    ehat2 = vect_op.normed_cross(ehat3, ehat1)

    kite_dcm = cas.horzcat(ehat1, ehat2, ehat3)

    return kite_dcm


def find_airspeed(init_options, groundspeed, psi):

    dq_kite = get_velocity_vector_from_psi(init_options, groundspeed, psi)

    l_t = init_options['xd']['l_t']
    ehat_tether = get_ehat_tether(init_options)
    zz = l_t * ehat_tether[2]

    wind_model = init_options['model']['wind_model']
    u_ref = init_options['model']['wind_u_ref']
    z_ref = init_options['model']['wind_z_ref']
    z0_air = init_options['model']['wind_z0_air']
    exp_ref = init_options['model']['wind_exp_ref']

    uu = wind.get_speed(wind_model, u_ref, z_ref, z0_air, exp_ref, zz) * vect_op.xhat_np()

    u_app = dq_kite - uu
    airspeed = float(vect_op.norm(u_app))

    return airspeed






def insert_dict(dict, var_type, name, name_stripped, V_init):
    init_val = dict[name_stripped]

    for idx in range(init_val.shape[0]):
        V_init = insert_val(V_init, var_type, name, init_val[idx], idx)

    return V_init


def insert_val(V_init, var_type, name, init_val, idx = 0):

    # initialize on collocation nodes
    V_init['coll_var', :, :, var_type, name, idx] = init_val

    if var_type in ['xd', 'xl','xa']:
        # initialize on interval nodes
        # V_init[var_type, :, :, name] = init_val

        V_init[var_type, :, name, idx] = init_val

    return V_init
