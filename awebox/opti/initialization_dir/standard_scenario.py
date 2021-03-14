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
initialization functions specific to the standard path scenario
_python _version 2.7 / casadi-3.4.5
- _author: rachel leuthold, jochem de schutter, thilo bronnenmeyer (alu-fr, 2017 - 21)
'''

import numpy as np
import casadi.tools as cas
import awebox.tools.vector_operations as vect_op
from awebox.logger.logger import Logger as awelogger
import awebox.tools.struct_operations as struct_op
import awebox.opti.initialization_dir.induction as induction
import awebox.opti.initialization_dir.tools as tools

import awebox.tools.print_operations as print_op
import awebox.mdl.wind as wind

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
    radius = init_options['precompute']['radius']
    return radius

def guess_final_time(init_options, model):
    tf_guess = init_options['precompute']['time_final']
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
            height = init_options['precompute']['height']
            radius = init_options['precompute']['radius']

            ehat_normal, ehat_radial, ehat_tangential = tools.get_rotating_reference_frame(t, init_options, model, node, ret)

            tether_vector = ehat_radial * radius + ehat_normal * height

            position = parent_position + tether_vector
            velocity = tools.get_velocity_vector(t, init_options, model, node, ret)
            ret['q' + str(node) + str(parent)] = position
            ret['dq' + str(node) + str(parent)] = velocity

            rho = init_options['sys_params_num']['atmosphere']['rho_ref']
            diam = init_options['theta']['diam_s']
            cd_tether = init_options['sys_params_num']['tether']['cd']
            if 'CD' in init_options['sys_params_num']['aero'].keys():
                cd_aero = vect_op.norm(init_options['sys_params_num']['aero']['CD']['0'])
            elif 'CX' in init_options['sys_params_num']['aero'].keys():
                cd_aero = vect_op.norm(init_options['sys_params_num']['aero']['CX']['0'])
            else:
                cd_aero = 0.1
            planform_area = init_options['sys_params_num']['geometry']['s_ref']
            u_eff = init_options['sys_params_num']['wind']['u_ref'] * vect_op.xhat_np() - velocity
            approx_dyn_pressure = 0.5 * rho * vect_op.norm(u_eff) * u_eff
            ret['f_tether' + str(node) + str(parent)] = cd_tether * approx_dyn_pressure * vect_op.norm(tether_vector) * diam
            ret['f_aero' + str(node) + str(parent)] = cd_aero * approx_dyn_pressure * planform_area

            dcm = tools.get_kite_dcm(init_options, model, node, ret)
            if init_options['cross_tether']:
                if init_options['cross_tether_attachment'] in ['com','stick']:
                    dcm = get_cross_tether_dcm(init_options, dcm)
            dcm_column = cas.reshape(dcm, (9, 1))

            omega_vector = tools.get_omega_vector(t, init_options, model, node, ret)

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


############# precompute parameters

def precompute_path_parameters(init_options, model):
    adjustment_steps = 3

    init_options['precompute'] = {}

    init_options = set_fixed_hypotenuse(init_options, model)
    init_options = set_fixed_max_cone_angle(init_options, model)

    init_options = set_user_winding_period(init_options)
    init_options = set_user_groundspeed(init_options)

    # clipping and adjusting
    for step in range(adjustment_steps):
        init_options = clip_groundspeed(init_options)  # clipping depends on arguments of airspeed calculation
        init_options = set_precomputed_radius(init_options)  # depends on groundspeed and winding_period

        init_options = clip_radius(init_options)  # clipping depends on hypotenuse and max cone angle
        init_options = set_precomputed_winding_period(init_options)  # depends on radius and groundspeed

        init_options = clip_winding_period(init_options)  # clipping depends on groundspeed
        init_options = set_precomputed_groundspeed(init_options)  # depends on radius and winding_period

    init_options = set_dependent_time_final(init_options)  # depends on winding_period
    init_options = set_dependent_height(init_options)  # depends on radius and hypotenuse
    init_options = set_dependent_angular_speed(init_options)  # depends on radius and groundspeed
    return init_options


####### fixed values that cannot be changed

def set_fixed_hypotenuse(init_options, model):
    number_kites = model.architecture.number_of_kites
    if number_kites == 1:
        hypotenuse = init_options['xd']['l_t']
    else:
        hypotenuse = init_options['theta']['l_s']

    init_options['precompute']['hypotenuse'] = hypotenuse

    return init_options


def set_fixed_max_cone_angle(init_options, model):
    max_cone_angle_multi = init_options['max_cone_angle_multi']
    max_cone_angle_single = init_options['max_cone_angle_single']

    number_kites = model.architecture.number_of_kites
    if number_kites == 1:
        max_cone_angle = max_cone_angle_single
    else:
        max_cone_angle = max_cone_angle_multi

    init_options['precompute']['max_cone_angle'] = max_cone_angle

    return init_options


####### user given initial guesses - starting point for search for feasible point, but unreliable

def set_user_winding_period(init_options):
    init_options['precompute']['winding_period'] = init_options['winding_period']
    return init_options


def set_user_groundspeed(init_options):
    init_options['precompute']['groundspeed'] = init_options['groundspeed']
    return init_options


###### precompute various values, as dependents of givens and existing guesses


def set_precomputed_radius(init_options):
    winding_period = init_options['precompute']['winding_period']
    groundspeed = init_options['precompute']['groundspeed']

    circumference = groundspeed * winding_period
    radius = circumference / 2. / np.pi

    init_options['precompute']['radius'] = radius

    return init_options


def set_precomputed_winding_period(init_options):
    groundspeed = init_options['precompute']['groundspeed']
    radius = init_options['precompute']['radius']

    circumference = 2. * np.pi * radius
    winding_period = circumference / groundspeed

    init_options['precompute']['winding_period'] = winding_period

    return init_options


def set_precomputed_groundspeed(init_options):
    winding_period = init_options['precompute']['winding_period']
    radius = init_options['precompute']['radius']

    circumference = 2. * np.pi * radius
    groundspeed = circumference / winding_period

    init_options['precompute']['groundspeed'] = groundspeed

    return init_options


###### clipping functions to increase feasibility

def clip_winding_period(init_options):
    winding_period = init_options['precompute']['winding_period']
    groundspeed = init_options['precompute']['groundspeed']

    acc_max = init_options['acc_max']

    omega = 2. * np.pi / winding_period
    acc_centripetal = groundspeed * omega
    # acc = omega * ua = 2 pi ua / winding_period < hardware_limit

    if acc_centripetal > acc_max:
        omega_clip = acc_max / groundspeed
        winging_period = 2. * np.pi / omega_clip

        awelogger.logger.warning(
            'proposed initial winding period implies centripetal acceleration above maximum acceleration. winding period will be clipped to ' + str(
                winging_period) + 's.')

    init_options['precompute']['winding_period'] = winding_period

    return init_options


def clip_radius(init_options):
    radius = init_options['precompute']['radius']
    hypotenuse = init_options['precompute']['hypotenuse']
    max_cone_angle = init_options['precompute']['max_cone_angle']

    b_ref = init_options['sys_params_num']['geometry']['b_ref']
    min_radius = init_options['min_rel_radius'] * b_ref

    if radius < min_radius:
        radius = min_radius
        awelogger.logger.warning(
            'proposed initial radius is below the minimum radius. radius will be clipped to ' + str(radius) + 'm.')

    max_radius = np.sin(max_cone_angle * np.pi / 180.) * hypotenuse
    if radius > max_radius:
        radius = max_radius
        awelogger.logger.warning(
            'proposed initial radius implies a cone angle above the maximum value. radius will be clipped to ' + str(
                radius) + 'm.')

    init_options['precompute']['radius'] = radius

    return init_options


def clip_groundspeed(init_options):
    groundspeed = init_options['precompute']['groundspeed']
    airspeed_include = init_options['airspeed_include']

    if airspeed_include:

        adjust_count = 0
        max_adjustments = 60

        increment = 1

        while adjust_count < max_adjustments:

            above_min = airspeeds_at_four_quadrants_above_minimum(init_options, groundspeed)
            below_max = airspeeds_at_four_quadrants_below_maximum(init_options, groundspeed)

            if groundspeed <= 0.:
                adjust_count = 10 + max_adjustments
                awelogger.logger.error(
                    'proposed initial kite speed is not positive. does not satisfy airspeed limits, and cannot be adjusted to do so.')

            elif (not above_min) and (not below_max):
                adjust_count = 10 + max_adjustments
                awelogger.logger.error(
                    'proposed initial kite speed does not satisfy airspeed limits, and cannot be adjusted to do so.')

            elif (not above_min):
                groundspeed += increment
                awelogger.logger.warning(
                    'proposed initial kite speed does not satisfy the minimum airspeed limits. kite speed will be incremented to ' + str(
                        groundspeed) + 'm/s.')

            elif (not below_max):
                groundspeed -= increment
                awelogger.logger.warning(
                    'proposed initial kite speed does not satisfy the maximum airspeed limits. kite speed will be decremented to ' + str(
                        groundspeed) + 'm/s.')

            else:
                # we have finally found a working value!
                init_options['precompute']['groundspeed'] = groundspeed
                return init_options

            adjust_count += 1

        awelogger.logger.error(
            'proposed initial kite speed does not satisfy airspeed limits, and could not be adjusted to do so within ' + str(
                max_adjustments) + ' adjustments. kite speed remains as specified by user.')

    return init_options


####### helpful booleans

def airspeeds_at_four_quadrants_above_minimum(options, groundspeed):
    airspeed_limits = options['airspeed_limits']
    airspeed_min = airspeed_limits[0]

    above_at_quadrant = []
    for psi in [np.pi / 2., np.pi, 3. * np.pi / 2, 2. * np.pi]:
        airspeed = tools.find_airspeed(options, groundspeed, psi)

        loc_bool = airspeed > airspeed_min
        above_at_quadrant += [loc_bool]

    return all(above_at_quadrant)


def airspeeds_at_four_quadrants_below_maximum(options, groundspeed):
    airspeed_limits = options['airspeed_limits']
    airspeed_max = airspeed_limits[1]

    below_at_quadrant = []
    for psi in [np.pi / 2., np.pi, 3. * np.pi / 2, 2. * np.pi]:
        airspeed = tools.find_airspeed(options, groundspeed, psi)

        loc_bool = airspeed < airspeed_max
        below_at_quadrant += [loc_bool]

    return all(below_at_quadrant)


################ dependent values

def set_dependent_height(init_options):
    radius = init_options['precompute']['radius']
    hypotenuse = init_options['precompute']['hypotenuse']

    height = (hypotenuse ** 2. - radius ** 2.) ** 0.5
    init_options['precompute']['height'] = height

    return init_options


def set_dependent_angular_speed(init_options):
    groundspeed = init_options['precompute']['groundspeed']
    radius = init_options['precompute']['radius']
    angular_speed = groundspeed / radius
    init_options['precompute']['angular_speed'] = angular_speed
    return init_options


def set_dependent_time_final(init_options):
    windings = init_options['windings']
    winding_period = init_options['precompute']['winding_period']

    tf_guess = windings * winding_period

    init_options['precompute']['time_final'] = tf_guess

    return init_options

