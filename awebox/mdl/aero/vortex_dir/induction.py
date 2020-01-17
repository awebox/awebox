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

import casadi as cas
import numpy as np
import pdb
from multiprocessing import Pool

from awebox.logger.logger import Logger as awelogger
import awebox.tools.vector_operations as vect_op
import awebox.mdl.aero.vortex_dir.tools as tools

def get_induced_velocity_at_kite(options, wind, variables, architecture, kite):

    # get a list of all the vortex segments
    n_k = options['aero']['vortex']['n_k']
    d = options['aero']['vortex']['d']
    periods_tracked = options['aero']['vortex']['periods_tracked']
    U_ref = wind.get_velocity_ref() * vect_op.xhat()

    enable_pool = options['processing']['enable_pool']
    processes = options['processing']['processes']

    vortex_list = tools.get_list_of_all_vortices(variables['xd'], variables['xl'], architecture, U_ref, periods_tracked, n_k, d, enable_pool=enable_pool, processes=processes)
    n_segments = vortex_list.shape[1]

    # join the vortex_list to the observation data
    parent = architecture.parent_map[kite]
    point_obs = variables['xd']['q' + str(kite) + str(parent)]
    epsilon = options['aero']['vortex']['epsilon']

    point_obs_ext = []
    for jdx in range(3):
        point_obs_ext = cas.vertcat(point_obs_ext, vect_op.ones_sx((1, n_segments)) * point_obs[jdx])
    eps_ext = vect_op.ones_sx((1, n_segments)) * epsilon

    segment_list = cas.vertcat(point_obs_ext, vortex_list, eps_ext)

    # define the symbolic function
    n_symbolics = segment_list.shape[0]
    seg_data_sym = cas.SX.sym('seg_data_sym', (n_symbolics, 1))
    filament_sym = filament(seg_data_sym)
    filament_fun = cas.Function('filament_fun', [seg_data_sym], [filament_sym])

    # evaluate the symbolic function
    total_U_ind = vect_op.zeros_sx((3, 1))
    for sdx in range(n_segments):
        seg_data = segment_list[:, sdx]
        local_U_ind = filament_fun(seg_data)
        total_U_ind = total_U_ind + local_U_ind

    return total_U_ind




def filament(seg_data):

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
    num = (r1 + r2) * vect_op.cross(vec_1, vec_2)

    den_ori = (r1 * r2) * (r1 * r2 + cas.mtimes(vec_1.T, vec_2))
    den_reg = (epsilon * r0)**2.
    den = den_ori + den_reg

    sol = factor * num / den

    return sol



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
