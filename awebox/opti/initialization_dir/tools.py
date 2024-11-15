#
#    This file is part of awebox.
#
#    awebox -- A modeling and optimization framework for multi-kite AWE systems.
#    Copyright (C) 2017-2021 Jochem De Schutter, Rachel Leuthold, Moritz Diehl,
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
- _author: rachel leuthold, jochem de schutter, thilo bronnenmeyer (alu-fr, 2017 - 21)
'''
import pdb

import numpy as np
import casadi.tools as cas
from sympy.physics.units import velocity, acceleration

import awebox.tools.vector_operations as vect_op
import awebox.tools.print_operations as print_op
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

    unit_vector_pointing_radially_outward = get_unit_vector_pointing_radially_outwards_by_azimuth(init_options, psi)
    ehat_radial = get_ehat_radial_from_unit_vector_pointing_outwards_radially(init_options, unit_vector_pointing_radially_outward)

    return ehat_radial

def get_unit_vector_pointing_outwards_radially_from_ehat_radial(init_options, ehat_radial):
    ehat_radial_points_inwards_during_counter_clockwise_motion = get_rotation_direction_sign(init_options)
    unit_vector_pointing_radially_outward = ehat_radial_points_inwards_during_counter_clockwise_motion * ehat_radial
    return unit_vector_pointing_radially_outward

def get_ehat_radial_from_unit_vector_pointing_outwards_radially(init_options, unit_vector_pointing_radially_outward):
    ehat_radial_points_inwards_during_counter_clockwise_motion = get_rotation_direction_sign(init_options)
    ehat_radial = ehat_radial_points_inwards_during_counter_clockwise_motion * unit_vector_pointing_radially_outward
    return ehat_radial

def get_rotation_direction_sign(init_options):
    # rotation with right hand rule about + (positive) nhat
    clockwise_rotation_about_xhat = init_options['clockwise_rotation_about_xhat']
    if clockwise_rotation_about_xhat:
        sign = +1
    else:
        sign = -1.

    return sign

def get_unit_vector_pointing_radially_outwards_by_azimuth(init_options, psi):
    _, y_rot_hat, z_rot_hat = get_rotor_reference_frame(init_options)

    cospsi_var = np.cos(psi)
    sinpsi_var = np.sin(psi)

    sign = get_rotation_direction_sign(init_options)

    # for positive yaw(turns around +zhat, normal towards +yhat):
    #     rhat = zhat * cos(psi) - yhat * sin(psi)
    ehat_radial = z_rot_hat * cospsi_var - sign * y_rot_hat * sinpsi_var

    return ehat_radial

def get_omega_vector(t, init_options, model, node, ret):
    rotation_direction_sign = get_rotation_direction_sign(init_options)
    omega_norm = init_options['precompute']['angular_speed']
    ehat2_in_body_frame = vect_op.zhat_dm()
    omega_vector = rotation_direction_sign * omega_norm * ehat2_in_body_frame
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
        psi0 = psi0_base + float(idx) / float(number_of_siblings) * 2. * np.pi

    psi = psi0 + omega_norm * t

    psi = np.mod(psi, 2. * np.pi)
    if psi < 0:
        psi += 2. * np.pi

    return psi

def get_velocity_vector(t, init_options, model, node, ret):
    groundspeed = init_options['precompute']['groundspeed']
    ehat_normal, ehat_radial, ehat_tangential = get_rotating_reference_frame(t, init_options, model, node, ret)
    velocity = groundspeed * ehat_tangential

    return velocity

def get_acceleration_vector(t, init_options, model, node, ret):

    radius = init_options['precompute']['radius']
    groundspeed = init_options['precompute']['groundspeed']

    ehat_normal, ehat_radial, ehat_tangential = get_rotating_reference_frame(t, init_options, model, node, ret)
    acceleration_magnitude = groundspeed**2 / radius

    unit_vector_pointing_radially_outward = get_unit_vector_pointing_outwards_radially_from_ehat_radial(init_options, ehat_radial)
    unit_vector_pointing_radially_inwards = -1. * unit_vector_pointing_radially_outward
    acceleration = acceleration_magnitude * unit_vector_pointing_radially_inwards
    return acceleration


def get_velocity_vector_from_psi(init_options, groundspeed, psi):

    n_rot_hat, _, _ = get_rotor_reference_frame(init_options)
    ehat_normal = n_rot_hat
    unit_vector_pointing_radially_outward = get_unit_vector_pointing_radially_outwards_by_azimuth(init_options, psi)
    ehat_radial = get_ehat_radial_from_unit_vector_pointing_outwards_radially(init_options, unit_vector_pointing_radially_outward)
    ehat_tangential = vect_op.normed_cross(ehat_normal, ehat_radial)
    velocity = groundspeed * ehat_tangential
    return velocity

def get_air_velocity(init_options, model, node, ret):

    position = ret['q' + str(node) + str(model.architecture.parent_map[node])]
    velocity = ret['dq' + str(node) + str(model.architecture.parent_map[node])]

    vec_u_infty = get_wind_velocity(init_options)
    vec_u_app = vec_u_infty - velocity

    return vec_u_app

def get_kite_dcm(init_options, model, node, ret):

    normal_vector_model = model.options['aero']['actuator']['normal_vector_model']
    if normal_vector_model == 'xhat':
        ehat_normal = vect_op.xhat_dm()
    elif normal_vector_model == 'tether_parallel':
        normal_vector = ret['q10']
        ehat_normal = vect_op.normalize(normal_vector)
    else:
        n_rot_hat, _, _ = get_rotor_reference_frame(init_options)
        ehat_normal = n_rot_hat

    kite_dcm_setting_method = init_options['kite_dcm']
    if kite_dcm_setting_method == 'aero_validity':

        vec_u_app = get_air_velocity(init_options, model, node, ret)

        ehat1 = vect_op.normalize(vec_u_app)
        ehat2 = vect_op.normed_cross(ehat_normal, ehat1)
        ehat3 = vect_op.normed_cross(ehat1, ehat2)

    elif kite_dcm_setting_method == 'circular':

        velocity = ret['dq' + str(node) + str(model.architecture.parent_map[node])]
        ehat_forwards = vect_op.normalize(velocity)

        ehat1 = -1. * ehat_forwards
        ehat3 = ehat_normal
        ehat2 = vect_op.normed_cross(ehat3, ehat1)

    else:
        message = 'unknown kite_dcm initialization option (' + kite_dcm_setting_method + ').'
        print_op.log_and_raise_error(message)

    kite_dcm = cas.horzcat(ehat1, ehat2, ehat3)

    return kite_dcm


def get_l_t_from_init_options(init_options):
    if 'l_t' in init_options.keys():
        l_t = init_options['l_t']
    elif 'x' in init_options.keys() and 'l_t' in init_options['x'].keys():
        l_t = init_options['x']['l_t']
    elif 'theta' in init_options.keys() and 'l_t' in init_options['theta'].keys():
        l_t = init_options['theta']['l_t']
    else:
        print_op.log_and_raise_error('missing l_t initialization information')
    return l_t


def find_airspeed(init_options, groundspeed, psi):

    dq_kite = get_velocity_vector_from_psi(init_options, groundspeed, psi)

    l_t = get_l_t_from_init_options(init_options)

    ehat_tether = get_ehat_tether(init_options)
    zz = l_t * ehat_tether[2]

    vec_u_infty = get_wind_velocity(init_options)
    vec_u_app = dq_kite - vec_u_infty
    airspeed = float(vect_op.norm(vec_u_app))

    return airspeed

def get_wind_velocity(init_options):

    l_t = get_l_t_from_init_options(init_options)

    ehat_tether = get_ehat_tether(init_options)
    zz = l_t * ehat_tether[2]

    wind_model = init_options['model']['wind_model']
    u_ref = init_options['model']['wind_u_ref']
    z_ref = init_options['model']['wind_z_ref']
    z0_air = init_options['model']['wind_z0_air']
    exp_ref = init_options['model']['wind_exp_ref']

    uu = wind.get_speed(wind_model, u_ref, z_ref, z0_air, exp_ref, zz) * vect_op.xhat_np()

    return uu

def insert_dict(dict, var_type, name, name_stripped, V_init):
    init_val = cas.DM(dict[name_stripped])
    for idx in range(init_val.shape[0]):
        V_init = insert_val(V_init, var_type, name, init_val[idx], idx)

    return V_init


def insert_val(V_init, var_type, name, init_val, idx=0):

    if not hasattr(init_val, 'shape'):
        init_val = cas.DM(init_val)

    # initialize on collocation nodes
    if '[coll_var,0,0,' + var_type + ',' + name + ',0]' in V_init.labels():
        if V_init['coll_var', 0, 0, var_type, name].shape == init_val.shape:
            V_init['coll_var', :, :, var_type, name, idx] = init_val[idx]
        elif init_val.shape == (1, 1):
            V_init['coll_var', :, :, var_type, name, idx] = init_val
        else:
            substructure = 'coll_var'
            message = 'something went wrong when trying to insert the ' + var_type + ' variable ' + name + "'s value (" + str(init_val) + ') into the ' + substructure + ' part of V_init'
            print_op.log_and_raise_error(message)

    if '[' + var_type + ',0,' + name + ',0]' in V_init.labels():
        if V_init[var_type, 0, name].shape == init_val.shape:
            V_init[var_type, :, name, idx] = init_val[idx]
        elif init_val.shape == (1, 1):
            V_init[var_type, :, name, idx] = init_val
        else:
            substructure = 'shooting node'
            message = 'something went wrong when trying to insert the ' + var_type + ' variable ' + name + "'s value (" + str(
                init_val) + ') into the ' + substructure + ' part of V_init'
            print_op.log_and_raise_error(message)

    return V_init
