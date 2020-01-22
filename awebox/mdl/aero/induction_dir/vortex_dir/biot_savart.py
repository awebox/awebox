#
#    This file is part of awebox.
#
#    awebox -- A modeling and optimization framework for multi-kite AWE systems.
#    Copyright (C) 2017-2019 Jochem De Schutter, Rachel Leuthold, Moritz Diehl,
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
vortex model of awebox aerodynamics
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2020
'''

import casadi.tools as cas
import numpy as np

import awebox.tools.vector_operations as vect_op

import awebox.mdl.aero.induction_dir.general_dir.geom as general_geom
import awebox.mdl.aero.induction_dir.general_dir.flow as general_flow
import awebox.mdl.aero.induction_dir.vortex_dir.tools as vortex_tools

import pdb

def get_induced_velocity_at_kite(filament_list, options, variables, kite, parent):

    processes = options['processing']['processes']

    include_normal_info = False
    segment_list = get_biot_savart_segment_list(filament_list, options, variables, kite, parent, include_normal_info)

    # define the symbolic function
    n_symbolics = segment_list.shape[0]
    seg_data_sym = cas.SX.sym('seg_data_sym', (n_symbolics, 1))
    filament_sym = filament(seg_data_sym)
    filament_fun = cas.Function('filament_fun', [seg_data_sym], [filament_sym])

    # evaluate the symbolic function
    total_u_vec_ind = evaluate_symbolic_on_segments_and_sum(filament_fun, segment_list, processes)

    return total_u_vec_ind


def get_induction_factor_at_kite(filament_list, options, wind, variables, kite, parent):

    processes = options['processing']['processes']

    include_normal_info = True
    segment_list = get_biot_savart_segment_list(filament_list, options, variables, kite, parent, include_normal_info)

    # define the symbolic function
    n_symbolics = segment_list.shape[0]
    seg_data_sym = cas.SX.sym('seg_data_sym', (n_symbolics, 1))
    filament_sym = normal_induction(seg_data_sym)
    filament_fun = cas.Function('filament_fun', [seg_data_sym], [filament_sym])

    # evaluate the symbolic function
    total_u_proj = evaluate_symbolic_on_segments_and_sum(filament_fun, segment_list, processes)

    u_app_kite = general_flow.get_kite_apparent_velocity(variables, wind, kite, parent)
    u_mag = vect_op.smooth_norm(u_app_kite)

    a_calc = -1. * total_u_proj / u_mag

    return a_calc



def evaluate_symbolic_on_segments_and_sum(filament_fun, segment_list, processes):

    n_filaments = segment_list.shape[1]
    filament_map = filament_fun.map(n_filaments, "thread", processes)
    all = filament_map(segment_list)

    n_dims = all.shape[0]

    total = []
    for jdx in range(n_dims):
        in_dim = vect_op.sum(all[jdx, :].T)
        total = cas.vertcat(total, in_dim)

    return total



def get_biot_savart_segment_list(filament_list, options, variables, kite, parent, include_normal_info):

    n_filaments = filament_list.shape[1]

    # join the vortex_list to the observation data
    point_obs = variables['xd']['q' + str(kite) + str(parent)]
    epsilon = options['aero']['vortex']['epsilon']

    point_obs_extended = []
    for jdx in range(3):
        point_obs_extended = cas.vertcat(point_obs_extended, vect_op.ones_sx((1, n_filaments)) * point_obs[jdx])
    eps_extended = vect_op.ones_sx((1, n_filaments)) * epsilon

    if include_normal_info:
        n_hat = general_geom.get_n_hat_var(variables, parent)

        n_hat_ext = []
        for jdx in range(3):
            n_hat_ext = cas.vertcat(n_hat_ext, vect_op.ones_sx((1, n_filaments)) * n_hat[jdx])

        segment_list = cas.vertcat(point_obs_extended, filament_list, eps_extended, n_hat_ext)
    else:
        segment_list = cas.vertcat(point_obs_extended, filament_list, eps_extended)

    return segment_list





def normal_induction(seg_data):

    n_hat = seg_data[10:13]
    scale, dir = filament_base_vals(seg_data)
    sol = scale * cas.mtimes(dir.T, n_hat)

    return sol


def filament(seg_data):

    scale, dir = filament_base_vals(seg_data)
    sol = scale * dir

    return sol


def filament_base_vals(seg_data):
    point_obs = seg_data[:3]
    point_1 = seg_data[3:6]
    point_2 = seg_data[6:9]
    Gamma = seg_data[9]
    epsilon = seg_data[10]

    vec_1 = point_obs - point_1
    vec_2 = point_obs - point_2
    vec_0 = point_2 - point_1

    r1 = vect_op.smooth_norm(vec_1)
    r2 = vect_op.smooth_norm(vec_2)
    r0 = vect_op.smooth_norm(vec_0)

    factor = Gamma / (4. * np.pi)
    num = (r1 + r2)

    den_ori = (r1 * r2) * (r1 * r2 + cas.mtimes(vec_1.T, vec_2))
    den_reg = (epsilon * r0) ** 2.
    den = den_ori + den_reg

    dir = vect_op.cross(vec_1, vec_2)
    scale = factor * num / den

    return scale, dir


def test_filament():

    point_obs = vect_op.yhat()
    point_1 = 1000. * vect_op.zhat()
    point_2 = -1. * point_1
    Gamma = 1.
    epsilon = 1.e-2
    seg_data = cas.vertcat(point_obs, point_1, point_2, Gamma, epsilon)

    vec_found = filament(seg_data)
    val_normalize = 1. / (2. * np.pi)
    vec_norm = vec_found / val_normalize

    difference = vec_norm - vect_op.xhat()
    resi = cas.mtimes(difference.T, difference)

    print("filament test residual is:" + str(resi))
    print()

    return None
