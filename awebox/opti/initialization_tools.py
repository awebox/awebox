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

import pdb
import numpy as np
import casadi.tools as cas
import awebox.tools.vector_operations as vect_op
from awebox.logger.logger import Logger as awelogger



def guess_tf(initialization_options, model, formulation, n_min = None, d_min = None):

    if initialization_options['type'] in ['nominal_landing','compromised_landing']:
        tf_guess = guess_tf_nominal_or_compromised(initialization_options, formulation, n_min, d_min)

    elif initialization_options['type'] == 'transition':
        tf_guess = guess_tf_transition()

    else:
        _, tf_guess = guess_radius_and_tf_standard(initialization_options, model)

    return tf_guess


def guess_tf_transition():
    tf_guess = 30.
    return tf_guess


def guess_tf_nominal_or_compromised(initialization_options, formulation, n_min, d_min):
    l_0 = formulation.xi_dict['V_pickle_initial']['coll_var', n_min, d_min, 'xd', 'l_t'] * initialization_options['xd']['l_t']
    tf_guess = l_0 / initialization_options['landing_velocity']
    return tf_guess




def guess_radius_and_tf_standard(initialization_options, model):

    tether_length, max_cone_angle = get_hypotenuse_and_max_cone_angle(model, initialization_options)

    windings = initialization_options['windings']
    winding_period = initialization_options['winding_period']

    winding_period = clip_winding_period(initialization_options, model.wind, winding_period)
    tf_guess = windings * winding_period

    qdot_norm, initialization_options = approx_speed(initialization_options, model.wind)
    total_distance = qdot_norm * tf_guess
    circumference = total_distance / windings
    radius = circumference / 2. / np.pi

    radius = clip_radius(initialization_options, max_cone_angle, tether_length, radius)

    total_distance = 2. * np.pi * radius * windings
    tf_guess = total_distance / qdot_norm

    return radius, tf_guess

def airspeeds_at_four_quadrants_above_minimum(options, wind, dq_kite_norm):

    airspeed_limits = options['airspeed_limits']
    airspeed_min = airspeed_limits[0]

    above_at_quadrant = []
    for psi in [np.pi / 2., np.pi, 3. * np.pi / 2, 2. * np.pi]:
        airspeed = find_airspeed(options, wind, dq_kite_norm, psi)

        loc_bool = airspeed > airspeed_min
        above_at_quadrant += [loc_bool]

    return all(above_at_quadrant)


def airspeeds_at_four_quadrants_below_maximum(options, wind, dq_kite_norm):

    airspeed_limits = options['airspeed_limits']
    airspeed_max = airspeed_limits[1]

    below_at_quadrant = []
    for psi in [np.pi / 2., np.pi, 3. * np.pi / 2, 2. * np.pi]:
        airspeed = find_airspeed(options, wind, dq_kite_norm, psi)

        loc_bool = airspeed < airspeed_max
        below_at_quadrant += [loc_bool]

    return all(below_at_quadrant)


def find_airspeed(options, wind, dq_kite_norm, psi):

    n_hat, y_hat, z_hat = get_rotor_reference_frame(options)
    ehat_radial = get_ehat_radial_from_azimuth(options, psi)
    ehat_tangential = vect_op.normed_cross(n_hat, ehat_radial)
    vec_dq = dq_kite_norm * ehat_tangential

    vec_u_ref = options['sys_params_num']['wind']['u_ref'] * vect_op.xhat_np()
    vec_u_infty = vec_u_ref

    # l_t = options['xd']['l_t']
    # ehat_tether = get_ehat_tether(options)
    # q_end = l_t * ehat_tether
    # vec_u_infty = wind.get_velocity(q_end[2])

    vec_ua = vec_dq - vec_u_infty
    airspeed = float(vect_op.norm(vec_ua))

    return airspeed

def approx_speed(options, wind):
    dq_kite_norm = options['dq_kite_norm']
    airspeed_include = options['airspeed_include']

    if not airspeed_include:
        return dq_kite_norm, options
    else:

        adjust_count = 0
        max_adjustments = 10

        increment = 1

        while adjust_count < max_adjustments:

            above_min = airspeeds_at_four_quadrants_above_minimum(options, wind, dq_kite_norm)
            below_max = airspeeds_at_four_quadrants_below_maximum(options, wind, dq_kite_norm)

            if dq_kite_norm <= 0.:
                adjust_count = 10 + max_adjustments
                awelogger.logger.error(
                    'proposed initial kite speed is not positive. does not satisfy airspeed limits, and cannot be adjusted to do so.')

            elif (not above_min) and (not below_max):
                adjust_count = 10 + max_adjustments
                awelogger.logger.error(
                    'proposed initial kite speed does not satisfy airspeed limits, and cannot be adjusted to do so.')

            elif (not above_min):
                dq_kite_norm += increment
                awelogger.logger.warning(
                    'proposed initial kite speed does not satisfy the minimum airspeed limits. kite speed will be incremented to ' + str(
                        dq_kite_norm) + 'm/s.')

            elif (not below_max):
                dq_kite_norm -= increment
                awelogger.logger.warning(
                    'proposed initial kite speed does not satisfy the maximum airspeed limits. kite speed will be decremented to ' + str(
                        dq_kite_norm) + 'm/s.')

            else:
                options['dq_kite_norm'] = dq_kite_norm
                return dq_kite_norm, options

            adjust_count += 1

        awelogger.logger.error(
            'proposed initial kite speed does not satisfy airspeed limits, and could not be adjusted to do so within ' + str(max_adjustments) + ' adjustments.')

    dq_kite_norm_last_chance = 1.
    return dq_kite_norm_last_chance, options

def clip_winding_period(initialization_options, wind, winding_period):
    # acc = omega * ua = 2 pi ua / winding_period < hardware_limit
    acc_max = initialization_options['acc_max']
    qdot_norm, initialization_options = approx_speed(initialization_options, wind)

    omega = 2. * np.pi / winding_period
    acc_centripetal = qdot_norm * omega

    if acc_centripetal > acc_max:

        omega_clip = acc_max / qdot_norm
        winging_period = 2. * np.pi / omega_clip

        awelogger.logger.warning('proposed initial winding period implies centripetal acceleration above maximum acceleration. winding period will be clipped to ' + str(winging_period) + 's.')

    return winding_period

def clip_radius(initialization_options, max_cone_angle, tether_length, radius):
    b_ref = initialization_options['sys_params_num']['geometry']['b_ref']
    min_radius = initialization_options['min_rel_radius'] * b_ref
    current_cone = np.arcsin(radius / tether_length) * 180. / np.pi

    if radius < min_radius:
        radius = min_radius
        awelogger.logger.warning('proposed initial radius is below the minimum radius. radius will be clipped to ' + str(radius) + 's.')


    max_radius = np.sin(max_cone_angle * np.pi / 180.) * tether_length
    if radius > max_radius:
        radius = max_radius
        awelogger.logger.warning('proposed initial radius implies a cone angle (' + str(current_cone) + ' deg. ) above the maximum value. radius will be clipped to ' + str(radius) + 's.')

    return radius

def get_hypotenuse_and_max_cone_angle(model, initialization_options):
    max_cone_angle_multi = initialization_options['max_cone_angle_multi']
    max_cone_angle_single = initialization_options['max_cone_angle_single']

    number_kites = model.architecture.number_of_kites
    if number_kites == 1:
        tether_length = initialization_options['xd']['l_t']
        max_cone_angle = max_cone_angle_single
    else:
        tether_length = initialization_options['theta']['l_s']
        max_cone_angle = max_cone_angle_multi

    return tether_length, max_cone_angle


def get_cone_height_and_radius(options, model, l_t):

    # get radius and height of the cones in use
    # two cone types specified, based on main tether (single kite option) and secondary tether (multi-kite option)
    # radius is dependent on flight velocity
    # height is a dependent
    hypotenuse_list = cas.vertcat(l_t, options['theta']['l_s'])
    [radius, _] = guess_radius_and_tf_standard(options, model)

    height_list = []
    for hdx in range(hypotenuse_list.shape[0]):
        hypotenuse = hypotenuse_list[hdx]
        height = (hypotenuse**2. - radius**2.)**0.5
        height_list = cas.vertcat(height_list, height)

    return height_list, radius









def get_ehat_tether(options):
    inclination = options['incid_deg'] * np.pi / 180.
    ehat_tether = np.cos(inclination) * vect_op.xhat() + np.sin(inclination) * vect_op.zhat()
    return ehat_tether

def get_rotor_reference_frame(initialization_options):
    n_rot_hat = get_ehat_tether(initialization_options)

    n_hat_is_x_hat = vect_op.abs(vect_op.norm(n_rot_hat - vect_op.xhat_np())) < 1.e-4
    if n_hat_is_x_hat:
        y_rot_hat = vect_op.yhat_np()
        z_rot_hat = vect_op.zhat_np()
    else:
        u_hat = vect_op.xhat_np()
        z_rot_hat = vect_op.normed_cross(u_hat, n_rot_hat)
        y_rot_hat = vect_op.normed_cross(z_rot_hat, n_rot_hat)

    return n_rot_hat, y_rot_hat, z_rot_hat

def get_ehat_radial(t, options, model, kite, ret={}):
    parent_map = model.architecture.parent_map
    level_siblings = model.architecture.get_all_level_siblings()

    qdot_norm, options = approx_speed(options, model.wind)

    if ret == {}:
        l_t = options['xd']['l_t']
    else:
        l_t = ret['l_t']

    height_list, radius = get_cone_height_and_radius(options, model, l_t)

    parent = parent_map[kite]

    omega_norm = qdot_norm / radius
    psi = get_azimuthal_angle(t, level_siblings, kite, parent, omega_norm)

    ehat_radial = get_ehat_radial_from_azimuth(options, psi)

    return ehat_radial

def get_ehat_radial_from_azimuth(options, psi):
    _, y_rot_hat, z_rot_hat = get_rotor_reference_frame(options)

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
