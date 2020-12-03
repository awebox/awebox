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
actuator_disk model of awebox aerodynamics
sets up the axial-induction actuator disk equation
currently for untilted rotor with no tcf.
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2017-20
- edit: jochem de schutter, alu-fr 2019
'''

import numpy as np
import casadi.tools as cas
import awebox.tools.vector_operations as vect_op

def approx_center_point(model_options, siblings, variables, architecture):
    number_of_siblings = len(siblings)

    center = np.zeros((3, 1))
    for kite in siblings:

        approx_center = present_location_extrapolation_to_center(model_options, kite, variables, architecture)

        center = center + approx_center / float(number_of_siblings)

    return center

def approx_center_velocity(model_options, siblings, variables, architecture):

    parent_map = architecture.parent_map
    parent = parent_map[siblings[0]]
    velocity = get_parent_velocity(model_options, variables, parent, architecture)

    return velocity

def approx_kite_radius_vector(model_options, variables, kite, parent):
    radius = get_radius_of_curvature(variables, kite, parent)
    rhat = get_trajectory_normal(variables, kite, parent)

    radius_vec = radius * rhat

    return radius_vec


def get_radius_of_curvature(variables, kite, parent):

    dq = variables['xd']['dq' + str(kite) + str(parent)]
    ddq = variables['xddot']['ddq' + str(kite) + str(parent)]

    gamma_dot = dq
    gamma_ddot = ddq

    # from frenet vectors + curvature definition
    # r = || gamma' || / (e1' cdot e2)
    # e1 = gamma' / || gamma' ||
    # e1' = ( gamma" || gamma' ||^2  - gamma' (gamma' cdot gamma") ) / || gamma' ||^3
    # e2 = ebar2 / || ebar2 ||
    # ebar2 = gamma" - (gamma' cdot gamma") gamma' / || gamma' ||^2
    # ....
    # r = || gamma' ||^4 // || gamma" || gamma' ||^2 - gamma' (gamma' cdot gamma") ||

    num = cas.mtimes(gamma_dot.T, gamma_dot)**2. + 1.0e-8

    den_vec = gamma_ddot * cas.mtimes(gamma_dot.T, gamma_dot) - gamma_dot * cas.mtimes(gamma_dot.T, gamma_ddot)
    den = vect_op.smooth_norm(den_vec)

    radius = num / den
    return radius


def get_radius_inequality(model_options, variables, kite, parent, parameters):
    # no projection included...

    b_ref = parameters['theta0','geometry','b_ref']
    half_span = b_ref / 2.
    num_ref = model_options['model_bounds']['anticollision_radius']['num_ref']

    # half_span - radius < 0
    # half_span * den - num < 0

    dq = variables['xd']['dq' + str(kite) + str(parent)]
    ddq = variables['xddot']['ddq' + str(kite) + str(parent)]

    gamma_dot = cas.vertcat(0., dq[1], dq[2])
    gamma_ddot = cas.vertcat(0., ddq[1], ddq[2])

    num = cas.mtimes(gamma_dot.T, gamma_dot)**2.

    den_vec = gamma_ddot * cas.mtimes(gamma_dot.T, gamma_dot) - gamma_dot * cas.mtimes(gamma_dot.T, gamma_ddot)
    den = vect_op.norm(den_vec)

    inequality = (half_span * den - num) / num_ref

    return inequality


def get_trajectory_tangent(variables, kite, parent):
    dq = variables['xd']['dq' + str(kite) + str(parent)]
    tangent = vect_op.smooth_normalize(dq)
    return tangent

def get_trajectory_normal(variables, kite, parent):
    ddq = variables['xddot']['ddq' + str(kite) + str(parent)]
    normal = vect_op.smooth_normalize(ddq)
    return normal

def get_trajectory_binormal(variables, kite, parent):

    tangent = get_trajectory_tangent(variables, kite, parent)
    normal = get_trajectory_normal(variables, kite, parent)
    binormal = vect_op.smooth_normed_cross(tangent, normal)

    forwards_orientation = binormal[0] / vect_op.smooth_abs(binormal[0])

    forwards_binormal = forwards_orientation * binormal
    return forwards_binormal


def approx_normal_axis(model_options, siblings, variables, architecture):

    parent_map = architecture.parent_map

    normal = np.zeros((3, 1))
    for kite in siblings:
        parent = parent_map[kite]

        binormal = get_trajectory_binormal(variables, kite, parent)
        normal = normal + binormal

    nhat = vect_op.smooth_normalize(normal)
    return nhat

def radial_extrapolation_to_center(model_options, kite, variables, architecture):

    parent_map = architecture.parent_map
    parent = parent_map[kite]

    q = variables['xd']['q' + str(kite) + str(parent)]

    radius = get_radius_of_curvature(variables, kite, parent)
    radial = get_trajectory_normal(variables, kite, parent)

    approx_center = q - radius * radial

    return approx_center

def present_location_extrapolation_to_center(model_options, kite, variables, architecture):

    parent_map = architecture.parent_map
    parent = parent_map[kite]

    q = variables['xd']['q' + str(kite) + str(parent)]
    approx_center = q

    return approx_center


def get_tether_vector(model_options, variables, kite, parent, architecture):

    parent_map = architecture.parent_map
    q_kite = variables['xd']['q' + str(kite) + str(parent)]

    if parent > 0:
        grandparent = parent_map[parent]
        q_parent = variables['xd']['q' + str(parent) + str(grandparent)]
    else:
        q_parent = np.zeros((3, 1))

    tether_vec = q_kite - q_parent

    return tether_vec

def get_parent_velocity(model_options, variables, parent, architecture):

    parent_map = architecture.parent_map

    if parent > 0:
        grandparent = parent_map[parent]
        dq_parent = variables['xd']['dq' + str(parent) + str(grandparent)]
    else:
        dq_parent = np.zeros((3, 1))

    return dq_parent
