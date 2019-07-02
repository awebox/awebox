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
actuator_disk model of awebox aerodynamics
sets up the axial-induction actuator disk equation
currently for untilted rotor with no tcf.
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2017-19
- edit: jochem de schutter, alu-fr 2019
'''

import numpy as np
import casadi as cas
from awebox.logger import Logger as awelogger

import awebox.tools.vector_operations as vect_op


def get_nhat(model_options, parent, variables, parameters, architecture):

    model = model_options['aero']['actuator']['normal_vector_model']
    children = architecture.kites_map[parent]
    number_children = float(len(children))

    if model == 'least_squares' and number_children > 2:
        nhat = get_least_squares_nhat(parent, variables, parameters, architecture)

    elif model == 'binormal':
        nhat = get_binormal_nhat(parent, variables, parameters, architecture)

    elif model == 'tether_parallel' and number_children > 1:
        nhat = get_tether_parallel_multi_nhat(parent, variables, parameters, architecture)

    elif model == 'tether_parallel' and number_children == 1:
        nhat = get_tether_parallel_single_nhat(parent, variables, parameters, architecture)

    elif model == 'default' and number_children > 1:
        nhat = get_tether_parallel_multi_nhat(parent, variables, parameters, architecture)

    elif model == 'default' and number_children == 1:
        nhat = get_tether_parallel_single_nhat(parent, variables, parameters, architecture)

    else:
        awelogger.logger.warning('normal-vector model-type (for actuator disk) not supported. Consider checking the number of kites per layer.')
        nhat = vect_op.xhat_np()

    return nhat

#
# def get_exact_plane_nhat(parent, variables, parameters, architecture):
#
#     children = architecture.children_map[parent]
#     number_children = architecture.get_number_children(parent)
#
#     x = []
#     y = []
#     z = []
#     for idx in range(number_children):
#         kite = children[idx]
#         qk = variables['xd']['q' + str(kite) + str(parent)]
#
#         x = cas.vertcat(x, qk[[0]])
#         y = cas.vertcat(y, qk[[1]])
#         z = cas.vertcat(z, qk[[2]])
#
#     # phi = (yk, zk, 1)
#
#     phi_inv_11 = (-z[[1]] + z[[2]]) / (y[[2]] * (-z[[0]] + z[[1]]) + y[[1]] * (z[[0]] - z[[2]]) + y[[0]] * (-z[[1]] + z[[2]]))
#     phi_inv_12 = (z[[0]] - z[[2]]) / (y[[1]] * z[[0]] - y[[2]] * z[[0]] - y[[0]] * z[[1]] + y[[2]] * z[[1]] + y[[0]] * z[[2]] - y[[1]] * z[[2]])
#     phi_inv_13 = (z[[0]] - z[[1]]) / (y[[2]] * (z[[0]] - z[[1]]) + y[[0]] * (z[[1]] - z[[2]]) + y[[1]] * (-z[[0]] + z[[2]]))
#     phi_inv_21 = (y[[1]] - y[[2]]) / (y[[1]] * z[[0]] - y[[2]] * z[[0]] - y[[0]] * z[[1]] + y[[2]] * z[[1]] + y[[0]] * z[[2]] - y[[1]] * z[[2]])
#     phi_inv_22 = (y[[0]] - y[[2]]) / (y[[2]] * (z[[0]] - z[[1]]) + y[[0]] * (z[[1]] - z[[2]]) + y[[1]] * (-z[[0]] + z[[2]]))
#     phi_inv_23 = (y[[0]] - y[[1]]) / (y[[1]] * z[[0]] - y[[2]] * z[[0]] - y[[0]] * z[[1]] + y[[2]] * z[[1]] + y[[0]] * z[[2]] - y[[1]] * z[[2]])
#     phi_inv_31 = (y[[2]] * z[[1]] - y[[1]] * z[[2]]) / (y[[1]] * z[[0]] - y[[2]] * z[[0]] - y[[0]] * z[[1]] + y[[2]] * z[[1]] + y[[0]] * z[[2]] - y[[1]] * z[[2]])
#     phi_inv_32 = (y[[2]] * z[[0]] - y[[0]] * z[[2]]) / (y[[2]] * (z[[0]] - z[[1]]) + y[[0]] * (z[[1]] - z[[2]]) + y[[1]] * (-z[[0]] + z[[2]]))
#     phi_inv_33 = (y[[1]] * z[[0]] - y[[0]] * z[[1]]) / (y[[1]] * z[[0]] - y[[2]] * z[[0]] - y[[0]] * z[[1]] + y[[2]] * z[[1]] + y[[0]] * z[[2]] - y[[1]] * z[[2]])
#
#     phi_inv_1 = cas.horzcat(phi_inv_11, phi_inv_12, phi_inv_13)
#     phi_inv_2 = cas.horzcat(phi_inv_21, phi_inv_22, phi_inv_23)
#     phi_inv_3 = cas.horzcat(phi_inv_31, phi_inv_32, phi_inv_33)
#
#     phi_inv = cas.vertcat(phi_inv_1, phi_inv_2, phi_inv_3)
#
#     coords = cas.mtimes(phi_inv, x)
#
#     nvec = cas.vertcat(1., coords[[0]], coords[[1]])
#
#     nhat = vect_op.normalize(nvec)
#
#     return nhat


def get_factor_var(parent, variables, parameters):
    # b_ref = parameters['theta0', 'geometry', 'b_ref']
    factor = variables['xl']['fnorm' + str(parent)]
    return factor


def get_least_squares_nhat(parent, variables, parameters, architecture):

    children = architecture.kites_map[parent]

    matrix = []
    rhs = []
    for kite in children:
        qk = variables['xd']['q' + str(kite) + str(parent)]
        newline = cas.horzcat(qk[[1]], qk[[2]], 1)
        matrix = cas.vertcat(matrix, newline)
        rhs = cas.vertcat(rhs, qk[[0]])

    coords = cas.mtimes(cas.pinv(matrix), rhs)

    nvec = cas.vertcat(1., coords[[0]], coords[[1]])

    scale = 1.
    factor = scale * get_factor_var(parent, variables, parameters)
    nhat = nvec * factor

    return nhat


def get_tether_parallel_multi_nhat(parent, variables, parameters, architecture):

    grandparent = architecture.parent_map[parent]

    if grandparent == 0:
        nvec = variables['xd']['q' + str(parent) + str(grandparent)]
    else:
        great_grandparent = architecture.parent_map[grandparent]
        q_parent = variables['xd']['q' + str(parent) + str(grandparent)]
        q_grandparent = variables['xd']['q' + str(grandparent) + str(great_grandparent)]
        nvec = q_parent - q_grandparent

    scale = 1.
    factor = scale * get_factor_var(parent, variables, parameters)
    nhat = nvec * factor

    return nhat


def get_tether_parallel_single_nhat(parent, variables, parameters, architecture):

    kite = architecture.children_map[parent][0]
    nvec = variables['xd']['q' + str(kite) + str(parent)]

    scale = 0.01
    factor = scale * get_factor_var(parent, variables, parameters)
    nhat = nvec * factor

    return nhat

def get_binormal_nhat(parent, variables, parameters, architecture):

    children = architecture.kites_map[parent]

    nvec = np.zeros((3,1))
    for kite in children:
        dqk = variables['xddot']['dq' + str(kite) + str(parent)]
        ddqk = variables['xddot']['ddq' + str(kite) + str(parent)]

        binormal_dim = vect_op.cross(dqk, ddqk)

        nvec = nvec + binormal_dim

    scale = 1.e-3
    factor = scale * get_factor_var(parent, variables, parameters)
    nhat = nvec * factor

    return nhat
