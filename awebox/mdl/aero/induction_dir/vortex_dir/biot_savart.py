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

import casadi.tools as cas
import numpy as np

import awebox.tools.vector_operations as vect_op
from awebox.logger.logger import Logger as awelogger
import awebox.mdl.aero.induction_dir.tools_dir.unit_normal as unit_normal


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
        num = get_numerator(seg_data, r_core)
        den = get_denominator(seg_data, r_core)
        sol = num / den
    except:
        message = 'something went wrong while computing the filament biot-savart induction.'
        awelogger.logger.error(message)
        raise Exception(message)

    return sol

def filament_resi(u_fil_var, seg_data, r_core=1.e-2):

    try:
        num = get_numerator(seg_data, r_core)
        den = get_denominator(seg_data, r_core)
        resi = (u_fil_var * den - num)
    except:
        message = 'something went wrong while computing the filament biot-savart residual.'
        awelogger.logger.error(message)
        raise Exception(message)

    return resi

def get_altitude(vec_1, vec_2, r0):
    vec_a = vect_op.cross(vec_1, vec_2)
    altitude = vect_op.smooth_norm(vec_a) / r0
    return altitude

def get_numerator(seg_data, r_core):

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

def get_denominator(seg_data, r_core):

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

def test():

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
