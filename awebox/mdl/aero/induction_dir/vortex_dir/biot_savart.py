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

def list_filament_observer_and_normal_info(point_obs, filament_list, options, n_hat=None):
    # join the vortex_list to the observation data

    n_filaments = filament_list.shape[1]

    r_core = options['induction']['vortex_core_radius']

    point_obs_extended = []
    for jdx in range(3):
        point_obs_extended = cas.vertcat(point_obs_extended, vect_op.ones_sx((1, n_filaments)) * point_obs[jdx])
    eps_extended = vect_op.ones_sx((1, n_filaments)) * r_core

    seg_list = cas.vertcat(point_obs_extended, filament_list, eps_extended)

    if n_hat is not None:
        n_hat_ext = []
        for jdx in range(3):
            n_hat_ext = cas.vertcat(n_hat_ext, vect_op.ones_sx((1, n_filaments)) * n_hat[jdx])

        seg_list = cas.vertcat(seg_list, n_hat_ext)

    return seg_list


def list_filaments_kiteobs_and_normal_info(filament_list, options, variables, parameters, kite_obs, architecture, include_normal_info):

    n_filaments = filament_list.shape[1]

    parent_obs = architecture.parent_map[kite_obs]

    point_obs = variables['xd']['q' + str(kite_obs) + str(parent_obs)]

    seg_list = list_filament_observer_and_normal_info(point_obs, filament_list, options)

    if include_normal_info:

        n_vec_val = unit_normal.get_n_vec(options, parent_obs, variables, parameters, architecture)
        n_hat = vect_op.normalize(n_vec_val)

        n_hat_ext = []
        for jdx in range(3):
            n_hat_ext = cas.vertcat(n_hat_ext, vect_op.ones_sx((1, n_filaments)) * n_hat[jdx])

        seg_list = cas.vertcat(seg_list, n_hat_ext)

    return seg_list


def filament_normal(seg_data, r_core=1.e-2):
    n_hat = seg_data[-3:]
    return cas.mtimes(filament(seg_data, r_core).T, n_hat)

def filament(seg_data, r_core=1.e-2):

    try:
        num = get_filament_numerator(seg_data, r_core)
        den = get_filament_denominator(seg_data, r_core)
        sol = num / den
    except:
        message = 'something went wrong while computing the filament biot-savart induction.'
        awelogger.logger.error(message)
        raise Exception(message)

    return sol

def filament_resi(u_fil_var, seg_data, r_core=1.e-2):

    try:
        num = get_filament_numerator(seg_data, r_core)
        den = get_filament_denominator(seg_data, r_core)
        resi = (u_fil_var * den - num)
    except:
        message = 'something went wrong while computing the filament biot-savart residual.'
        awelogger.logger.error(message)
        raise Exception(message)

    return resi

def get_altitude(vec_1, vec_2):
    vec_a = vect_op.cross(vec_1, vec_2)
    altitude = vect_op.smooth_norm(vec_a) / vect_op.smooth_norm(vec_1 - vec_2)
    return altitude

def test_altitude():

    expected = 1.
    x_obs = expected * vect_op.zhat_dm()

    # right triangle
    x_1 = 0. * vect_op.xhat_dm()
    x_2 = 1. * vect_op.xhat_dm()
    difference = get_altitude(x_1 - x_obs, x_2 - x_obs) - expected
    thresh = 1.e-6
    if thresh < difference**2.:
        message = 'biot-savart right-triangle altitude test gives error of size: ' + str(difference)
        awelogger.logger.error(message)
        raise Exception(message)

    # obtuse triangle
    x_1 = 1. * vect_op.xhat_np()
    x_2 = 2. * vect_op.xhat_np()
    difference = get_altitude(x_1 - x_obs, x_2 - x_obs) - expected
    thresh = 1.e-6
    if thresh < difference**2.:
        message = 'biot-savart obtuse-triangle altitude test gives error of size: ' + str(difference)
        awelogger.logger.error(message)
        raise Exception(message)

    return None


def get_filament_numerator(seg_data, r_core):

    point_obs = seg_data[:3]
    point_1 = seg_data[3:6]
    point_2 = seg_data[6:9]
    Gamma = seg_data[9]

    vec_1 = point_obs - point_1
    vec_2 = point_obs - point_2
    vec_0 = point_2 - point_1

    r1 = vect_op.smooth_norm(vec_1)
    r2 = vect_op.smooth_norm(vec_2)
    r0 = vect_op.smooth_norm(vec_0)

    factor = Gamma / (4. * np.pi)

    scale = (r1 + r2) * factor
    dir = vect_op.cross(vec_1, vec_2)
    num = dir * scale

    return num

def get_filament_denominator(seg_data, r_core):

    # for actual signs:
    # https: // openfast.readthedocs.io / en / master / source / user / aerodyn - olaf / OLAFTheory.html

    point_obs = seg_data[:3]
    point_1 = seg_data[3:6]
    point_2 = seg_data[6:9]

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
    seg_data = cas.vertcat(point_obs, point_1, point_2, Gamma)

    r_core = 0.
    vec_found = filament(seg_data, r_core=r_core)
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

def k_squared_cylinder(r_obs, r_cyl, z_obs, epsilon=1.e-3):
    # see (36.76) from Branlard, 2017
    denom = (r_cyl + r_obs)**2. + z_obs**2. + epsilon**2.
    k_squared = 4. * r_cyl * r_obs / denom
    return k_squared

def get_cylinder_r_obs(cyl_data):
    x_obs = unpack_cylinder_x_obs(cyl_data)
    x_center = unpack_cylinder_x_center(cyl_data)
    l_hat = unpack_cylinder_l_hat(cyl_data)

    x_axis_2 = x_center + l_hat
    r_obs = get_altitude(x_center - x_obs, x_axis_2 - x_obs)
    return r_obs

def get_cylinder_r_cyl(cyl_data):
    x_center = unpack_cylinder_x_center(cyl_data)
    x_kite = unpack_cylinder_x_kite(cyl_data)
    l_hat = unpack_cylinder_l_hat(cyl_data)

    x_axis_2 = x_center + l_hat
    r_cyl = get_altitude(x_center - x_kite, x_axis_2 - x_kite)
    return r_cyl

def get_cylinder_z_obs(cyl_data):
    x_obs = unpack_cylinder_x_obs(cyl_data)
    x_kite = unpack_cylinder_x_kite(cyl_data)
    l_hat = unpack_cylinder_l_hat(cyl_data)

    z_obs = cas.mtimes((x_obs - x_kite).T, l_hat)
    return z_obs

def get_cylinder_axes(cyl_data):
    x_obs = unpack_cylinder_x_obs(cyl_data)
    x_center = unpack_cylinder_x_center(cyl_data)
    l_hat = unpack_cylinder_l_hat(cyl_data)

    vec_diff =  x_obs - x_center

    ehat_long = l_hat
    vec_theta = vect_op.cross(l_hat, vec_diff)
    ehat_theta = vect_op.normalize(vec_theta)
    ehat_r = vect_op.normed_cross(vec_theta, ehat_long)

    return ehat_r, ehat_theta, ehat_long

def assemble_cylinder_data(x_center=cas.DM.zeros((3, 1)), x_kite=None, l_hat=vect_op.xhat_dm(), gamma=1., x_obs=None):
    cyl_data = cas.vertcat(x_center, x_kite, l_hat, gamma, x_obs)
    return cyl_data

def unpack_cylinder_x_center(cyl_data):
    return cyl_data[0:3]

def unpack_cylinder_x_kite(cyl_data):
    return cyl_data[3:6]

def unpack_cylinder_l_hat(cyl_data):
    return cyl_data[6:9]

def unpack_cylinder_gamma(cyl_data):
    return cyl_data[9]

def unpack_cylinder_x_obs(cyl_data):
    return cyl_data[10:13]

def longitudinal_cylinder(cyl_data):

    gamma_long = unpack_cylinder_gamma(cyl_data)

    r_obs = get_cylinder_r_obs(cyl_data)
    r_cyl = get_cylinder_r_cyl(cyl_data)
    z_obs = get_cylinder_z_obs(cyl_data)
    _, ehat_theta, _ = get_cylinder_axes(cyl_data)

    factor = gamma_long / 2. * r_cyl / r_obs
    part_1 = (r_obs - r_cyl) / 2. / vect_op.smooth_abs(r_cyl - r_obs) + 0.5

    k_z_squared = k_squared_cylinder(r_obs, r_cyl, z_obs)
    k_z = vect_op.smooth_sqrt(k_z_squared)
    k_0_squared = k_squared_cylinder(r_obs, r_cyl, 0.)

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

    thresh = 1.e-6

    # check direction of induced velocity
    x_obs = 0. * l_hat + 0. * r_hat + 2. * r_cyl * vect_op.yhat_dm()
    expected_direction = vect_op.zhat_dm()
    cyl_data = assemble_cylinder_data(x_kite = x_kite,
                                      x_center = x_center,
                                      l_hat = l_hat,
                                      gamma = gamma_long,
                                      x_obs = x_obs)
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
    cyl_data = assemble_cylinder_data(x_kite = x_kite,
                                      x_center = x_center,
                                      l_hat = l_hat,
                                      gamma = gamma_long,
                                      x_obs = x_obs)
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
    cyl_data = assemble_cylinder_data(x_kite = x_kite,
                                      x_center = x_center,
                                      l_hat = l_hat,
                                      gamma = gamma_long,
                                      x_obs = x_obs)
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
    cyl_data = assemble_cylinder_data(x_kite = x_kite,
                                      x_center = x_center,
                                      l_hat = l_hat,
                                      gamma = gamma_long,
                                      x_obs = x_obs)
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
    cyl_data = assemble_cylinder_data(x_kite = x_kite,
                                      x_center = x_center,
                                      l_hat = l_hat,
                                      gamma = gamma_long,
                                      x_obs = x_obs)
    found = longitudinal_cylinder(cyl_data)
    found_value = vect_op.norm(found)

    test = (expected_value - found_value)
    if test > thresh:
        message = 'biot-savart longitudinal cylinder induction magnitude (external) test gives error of size: ' + str(test)
        awelogger.logger.error(message)
        raise Exception(message)

    return None

def tangential_cylinder(cyl_data):

    gamma_tan = unpack_cylinder_gamma(cyl_data)

    r_obs = get_cylinder_r_obs(cyl_data)
    r_cyl = get_cylinder_r_cyl(cyl_data)
    z_obs = get_cylinder_z_obs(cyl_data)
    ehat_r, ehat_theta, ehat_long = get_cylinder_axes(cyl_data)

    k_z_squared = k_squared_cylinder(r_obs, r_cyl, z_obs)
    k_z = vect_op.smooth_sqrt(k_z_squared)
    k_0_squared = k_squared_cylinder(r_obs, r_cyl, 0.)

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
    return found

def test_tangential_cylinder():

    r_cyl = 1.
    gamma_tan = 1.

    l_hat = vect_op.xhat_dm()
    r_hat = vect_op.zhat_dm()

    x_center = 0. * l_hat
    x_kite = x_center + r_cyl * r_hat

    thresh = 1.e-6

    # check axial induction on plane., within cylinder
    x_obs = 0. * l_hat + 0.2 * r_cyl * r_hat
    expected_axial = 0.
    cyl_data = assemble_cylinder_data(x_kite = x_kite,
                                      x_center = x_center,
                                      l_hat = l_hat,
                                      gamma = gamma_tan,
                                      x_obs = x_obs)
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
    cyl_data = assemble_cylinder_data(x_kite = x_kite,
                                      x_center = x_center,
                                      l_hat = l_hat,
                                      gamma = gamma_tan,
                                      x_obs = x_obs)
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
    cyl_data = assemble_cylinder_data(x_kite = x_kite,
                                      x_center = x_center,
                                      l_hat = l_hat,
                                      gamma = gamma_tan,
                                      x_obs = x_obs)
    found = tangential_cylinder(cyl_data)
    found_radial = cas.mtimes(found.T, r_hat)

    test = vect_op.norm(expected_radial - found_radial)
    if test > thresh:
        message = 'biot-savart tangential cylinder radial-induction on-axis test gives error of size: ' + str(test)
        awelogger.logger.error(message)
        raise Exception(message)

    return None
