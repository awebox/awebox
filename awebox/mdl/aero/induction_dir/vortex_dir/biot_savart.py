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
vortex model of awebox aerodynamics
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2021
'''
import pdb

import casadi.tools as cas
import numpy as np

import awebox.tools.vector_operations as vect_op
from awebox.logger.logger import Logger as awelogger
import awebox.mdl.aero.induction_dir.tools_dir.unit_normal as unit_normal
import awebox.tools.print_operations as print_op

def filament(seg_data):

    try:
        num = get_filament_numerator(seg_data)
        den = get_filament_denominator(seg_data)
        sol = num / den

    except:
        message = 'something went wrong while computing the filament biot-savart induction.'
        awelogger.logger.error(message)
        raise Exception(message)

    return sol

def get_filament_numerator(seg_data):

    point_obs = seg_data['x_obs']
    point_1 = seg_data['x_start']
    point_2 = seg_data['x_end']
    Gamma = seg_data['strength']

    vec_1 = point_obs - point_1
    vec_2 = point_obs - point_2

    r1 = vect_op.smooth_norm(vec_1)
    r2 = vect_op.smooth_norm(vec_2)

    factor = Gamma / (4. * np.pi)

    scale = (r1 + r2) * factor
    dir = vect_op.cross(vec_1, vec_2)
    num = dir * scale

    return num

def get_filament_denominator(seg_data):

    # for actual signs:
    # https: // openfast.readthedocs.io / en / master / source / user / aerodyn - olaf / OLAFTheory.html

    point_obs = seg_data['x_obs']
    point_1 = seg_data['x_start']
    point_2 = seg_data['x_end']
    r_core = seg_data['r_core']

    vec_1 = point_obs - point_1
    vec_2 = point_obs - point_2
    vec_0 = point_2 - point_1

    r1 = vect_op.smooth_norm(vec_1)
    r2 = vect_op.smooth_norm(vec_2)
    r0 = vect_op.smooth_norm(vec_0)

    den_ori = (r1 * r2) * (r1 * r2 + cas.mtimes(vec_1.T, vec_2))
    reg_den = r0**2. * r_core**2.

    den = den_ori + reg_den

    return den

def test_filament():

    point_obs = vect_op.yhat()
    point_1 = 1000. * vect_op.zhat()
    point_2 = -1. * point_1
    Gamma = 1.
    r_core = 0.

    seg_data = {'x_obs': point_obs,
                'x_start': point_1,
                'x_end': point_2,
                'r_core': r_core,
                'strength': Gamma
                }

    vec_found = filament(seg_data)
    val_normalize = 1. / (2. * np.pi)
    vec_norm = vec_found / val_normalize

    mag_test = (vect_op.norm(vec_norm) - 1.)**2.
    mag_thresh = 1.e-6
    if mag_test > mag_thresh:
        message = 'biot-savart filament induction magnitude test gives error of size: ' + str(mag_test)
        awelogger.logger.error(message)
        raise Exception(message)

    dir_test = vect_op.norm(vec_norm - vect_op.xhat() * cas.mtimes(vec_norm.T, vect_op.xhat()))
    dir_thresh = 1.e-6
    if dir_test > dir_thresh:
        message = 'biot-savart filament induction direction test gives error of size: ' + str(dir_test)
        awelogger.logger.error(message)
        raise Exception(message)

    return None

def k_squared_cylinder(r_obs, r_cyl, z_obs, epsilon):
    # see (36.76) from Branlard, 2017
    denom = (r_cyl + r_obs)**2. + z_obs**2. + epsilon**2.
    k_squared = 4. * r_cyl * r_obs / denom
    return k_squared

def get_cylinder_r_obs(cyl_data):
    x_obs = cyl_data['x_obs']
    x_center = cyl_data['x_center']
    l_hat = cyl_data['l_hat']

    x_axis_2 = x_center + l_hat
    r_obs = vect_op.get_altitude(x_center - x_obs, x_axis_2 - x_obs)
    return r_obs

def get_cylinder_r_cyl(cyl_data):
    x_center = cyl_data['x_center']
    x_kite = cyl_data['x_kite']
    l_hat = cyl_data['l_hat']

    x_axis_2 = x_center + l_hat
    r_cyl = vect_op.get_altitude(x_center - x_kite, x_axis_2 - x_kite)
    return r_cyl

def get_cylinder_z_obs(cyl_data):
    x_kite = cyl_data['x_kite']
    l_hat = cyl_data['l_hat']
    x_obs = cyl_data['x_obs']

    z_obs = cas.mtimes((x_obs - x_kite).T, l_hat)
    return z_obs

def get_cylinder_axes(cyl_data):
    x_center = cyl_data['x_center']
    l_hat = cyl_data['l_hat']

    if 'x_obs' in cyl_data.keys():
        x_point = cyl_data['x_obs']
    elif 'x_kite' in cyl_data.keys():
        x_point = cyl_data['x_kite']
    else:
        message = 'insufficient information given in cylinder data to determine a coordinate frame'
        awelogger.logger.error(message)
        raise Exception(message)

    vec_diff = x_point - x_center

    ehat_long = l_hat
    vec_theta = vect_op.cross(l_hat, vec_diff)
    ehat_theta = vect_op.normalize(vec_theta)
    ehat_r = vect_op.normed_cross(vec_theta, ehat_long)

    return ehat_r, ehat_theta, ehat_long


def longitudinal_cylinder(cyl_data):

    gamma_long = cyl_data['strength']
    epsilon = cyl_data['epsilon']

    r_obs = get_cylinder_r_obs(cyl_data)
    r_cyl = get_cylinder_r_cyl(cyl_data)
    z_obs = get_cylinder_z_obs(cyl_data)
    _, ehat_theta, _ = get_cylinder_axes(cyl_data)

    factor = gamma_long / 2. * r_cyl / r_obs
    part_1 = (r_obs - r_cyl) / 2. / vect_op.smooth_abs(r_cyl - r_obs) + 0.5

    k_z_squared = k_squared_cylinder(r_obs, r_cyl, z_obs, epsilon)
    k_z = vect_op.smooth_sqrt(k_z_squared)
    k_0_squared = k_squared_cylinder(r_obs, r_cyl, 0., epsilon)

    factor_a = z_obs * k_z / 2. / np.pi / vect_op.smooth_sqrt(r_cyl * r_obs)
    part_2_1 = vect_op.elliptic_k(m=k_z_squared)
    part_2_2 = -1. * (r_cyl - r_obs) / (r_cyl + r_obs) * vect_op.elliptic_pi(k_0_squared, k_z_squared)
    part_2 = factor_a * (part_2_1 + part_2_2)
    u_theta = factor * (part_1 + part_2)

    found = u_theta * ehat_theta

    return found

def test_longtitudinal_cylinder():

    r_cyl = 1.
    gamma_long = 1.

    l_hat = vect_op.xhat_dm()
    r_hat = vect_op.zhat_dm()

    x_center = 0. * l_hat
    x_kite = x_center + r_cyl * r_hat
    epsilon = 0.

    thresh = 1.e-6

    # check direction of induced velocity
    x_obs = 0. * l_hat + 0. * r_hat + 2. * r_cyl * vect_op.yhat_dm()
    expected_direction = vect_op.zhat_dm()
    cyl_data = {'x_kite': x_kite,
                'x_center': x_center,
                'l_hat': l_hat,
                'epsilon': epsilon,
                'strength': gamma_long,
                'x_obs': x_obs}
    found = longitudinal_cylinder(cyl_data)
    found_direction = vect_op.normalize(found)

    test = vect_op.norm(expected_direction - found_direction)
    if test > thresh:
        message = 'biot-savart longitudinal cylinder induction direction test gives error of size: ' + str(test)
        awelogger.logger.error(message)
        raise Exception(message)

    # check value on axis
    x_obs = 0. * l_hat + 0. * r_hat
    expected_value = 0.
    cyl_data = {'x_kite': x_kite,
                'x_center': x_center,
                'l_hat': l_hat,
                'epsilon': epsilon,
                'strength': gamma_long,
                'x_obs': x_obs}
    found = longitudinal_cylinder(cyl_data)
    found_value = vect_op.norm(found)

    test = (expected_value - found_value)
    if test > thresh:
        message = 'biot-savart longitudinal cylinder induction magnitude (axis) test gives error of size: ' + str(test)
        awelogger.logger.error(message)
        raise Exception(message)


    # check value inside cylinder
    r_obs = r_cyl / 2.
    x_obs = 0. * l_hat + r_obs * r_hat
    expected_value = 0.
    cyl_data = {'x_kite': x_kite,
                'x_center': x_center,
                'l_hat': l_hat,
                'epsilon': epsilon,
                'strength': gamma_long,
                'x_obs': x_obs}
    found = longitudinal_cylinder(cyl_data)
    found_value = vect_op.norm(found)

    test = (expected_value - found_value)
    if test > thresh:
        message = 'biot-savart longitudinal cylinder induction magnitude (inside) test gives error of size: ' + str(test)
        awelogger.logger.error(message)
        raise Exception(message)


    # check value near cylinder edge
    r_obs = r_cyl * 1.001
    x_obs = 0. * l_hat + r_obs * r_hat
    expected_value = gamma_long * r_cyl / 2. / r_obs
    cyl_data = {'x_kite': x_kite,
                'x_center': x_center,
                'l_hat': l_hat,
                'epsilon': epsilon,
                'strength': gamma_long,
                'x_obs': x_obs}
    found = longitudinal_cylinder(cyl_data)
    found_value = vect_op.norm(found)

    test = (expected_value - found_value)
    if test > thresh:
        message = 'biot-savart longitudinal cylinder induction magnitude (near-edge) test gives error of size: ' + str(test)
        awelogger.logger.error(message)
        raise Exception(message)

    # check value outside of cylinder
    r_obs = 3. * r_cyl
    x_obs = 0. * l_hat + r_obs * r_hat
    expected_value = gamma_long * r_cyl / 2. / r_obs
    cyl_data = {'x_kite': x_kite,
                'x_center': x_center,
                'l_hat': l_hat,
                'epsilon': epsilon,
                'strength': gamma_long,
                'x_obs': x_obs}
    found = longitudinal_cylinder(cyl_data)
    found_value = vect_op.norm(found)

    test = (expected_value - found_value)
    if test > thresh:
        message = 'biot-savart longitudinal cylinder induction magnitude (external) test gives error of size: ' + str(test)
        awelogger.logger.error(message)
        raise Exception(message)

    return None

def tangential_cylinder(cyl_data):

    gamma_tan = cyl_data['strength']
    epsilon = cyl_data['epsilon']

    r_obs = get_cylinder_r_obs(cyl_data)
    r_cyl = get_cylinder_r_cyl(cyl_data)
    z_obs = get_cylinder_z_obs(cyl_data)
    ehat_r, ehat_theta, ehat_long = get_cylinder_axes(cyl_data)

    k_z_squared = k_squared_cylinder(r_obs, r_cyl, z_obs, epsilon)
    k_z = vect_op.smooth_sqrt(k_z_squared)
    k_0_squared = k_squared_cylinder(r_obs, r_cyl, 0., epsilon)

    u_r_factor = - gamma_tan / 2. / np.pi * vect_op.smooth_sqrt(r_cyl / r_obs)
    u_r_part_1 = (2. - k_z_squared) / k_z * vect_op.elliptic_k(m = k_z_squared)
    u_r_part_2 = -2. / k_z * vect_op.elliptic_e(m = k_z_squared)
    u_r = u_r_factor * (u_r_part_1 + u_r_part_2)

    u_z_factor = gamma_tan / 2.
    u_z_part_1 = (r_obs - r_cyl) / 2. / vect_op.smooth_abs(r_cyl - r_obs) + 0.5
    u_z_part_2_factor = z_obs * k_z / (2. * np.pi * vect_op.smooth_sqrt(r_obs * r_cyl))
    u_z_part_2_a = vect_op.elliptic_k(m = k_z_squared)
    u_z_part_2_b = (r_cyl - r_obs) / (r_cyl + r_obs) * vect_op.elliptic_pi(n=k_0_squared, alpha = k_z_squared)
    u_z = u_z_factor * (u_z_part_1 + u_z_part_2_factor * (u_z_part_2_a + u_z_part_2_b))

    found = u_r * ehat_r + u_z * ehat_long
    # print_op.warn_about_temporary_funcationality_removal(location='biot_savart.long_cylinder')
    # found = cas.DM.zeros((3, 1))

    return found

def test_tangential_cylinder():

    r_cyl = 1.
    gamma_tan = 1.

    l_hat = vect_op.xhat_dm()
    r_hat = vect_op.zhat_dm()

    x_center = 0. * l_hat
    x_kite = x_center + r_cyl * r_hat

    epsilon = 0.
    thresh = 1.e-6

    # check axial induction on plane., within cylinder
    x_obs = 0. * l_hat + 0.2 * r_cyl * r_hat
    expected_axial = 0.
    cyl_data = {'x_kite': x_kite,
                'x_center': x_center,
                'l_hat': l_hat,
                'epsilon': epsilon,
                'strength': gamma_tan,
                'x_obs': x_obs}
    found = tangential_cylinder(cyl_data)
    found_axial = cas.mtimes(found.T, l_hat)

    test = vect_op.norm(expected_axial - found_axial)
    if test > thresh:
        message = 'biot-savart tangential cylinder axial-induction on-plane-within-cylinder test gives error of size: ' + str(test)
        awelogger.logger.error(message)
        raise Exception(message)

    # check axial induction on plane., outside cylinder
    x_obs = 0. * l_hat + 2. * r_cyl * r_hat
    expected_axial = gamma_tan / 2.
    cyl_data = {'x_kite': x_kite,
                'x_center': x_center,
                'l_hat': l_hat,
                'epsilon': epsilon,
                'strength': gamma_tan,
                'x_obs': x_obs}
    found = tangential_cylinder(cyl_data)
    found_axial = cas.mtimes(found.T, l_hat)

    test = vect_op.norm(expected_axial - found_axial)
    if test > thresh:
        message = 'biot-savart tangential cylinder axial-induction on-plane-outside-cylinder test gives error of size: ' + str(test)
        awelogger.logger.error(message)
        raise Exception(message)

    # check radial induction on axis
    x_obs = 0. * l_hat + 0. * r_hat
    expected_radial = 0.
    cyl_data = {'x_kite': x_kite,
                'x_center': x_center,
                'l_hat': l_hat,
                'epsilon': epsilon,
                'strength': gamma_tan,
                'x_obs': x_obs}
    found = tangential_cylinder(cyl_data)
    found_radial = cas.mtimes(found.T, r_hat)

    test = vect_op.norm(expected_radial - found_radial)
    if test > thresh:
        message = 'biot-savart tangential cylinder radial-induction on-axis test gives error of size: ' + str(test)
        awelogger.logger.error(message)
        raise Exception(message)

    return None
