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
- author: rachel leuthold, alu-fr 2017-18
- edit: jochem de schutter, alu-fr 2019
'''

import numpy as np

import awebox.tools.vector_operations as vect_op
import awebox.mdl.aero.indicators as indicators

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
    radius = indicators.get_radius_of_curvature(variables, kite, parent)
    rhat = indicators.get_trajectory_normal(variables, kite, parent)

    radius_vec = radius * rhat

    return radius_vec

def approx_normal_axis(model_options, siblings, variables, architecture):

    parent_map = architecture.parent_map

    normal = np.zeros((3, 1))
    for kite in siblings:
        parent = parent_map[kite]

        binormal = indicators.get_trajectory_binormal(variables, kite, parent)
        normal = normal + binormal

    nhat = vect_op.smooth_normalize(normal)
    return nhat

def radial_extrapolation_to_center(model_options, kite, variables, architecture):

    parent_map = architecture.parent_map
    parent = parent_map[kite]

    q = variables['xd']['q' + str(kite) + str(parent)]

    radius = indicators.get_radius_of_curvature(variables, kite, parent)
    radial = indicators.get_trajectory_normal(variables, kite, parent)

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
