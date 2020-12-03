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

import awebox.tools.vector_operations as vect_op

def approx_center_point(parent, variables, architecture):

    children = architecture.kites_map[parent]
    number_children = float(len(children))

    center = np.zeros((3, 1))
    for kite in children:

        q_kite = variables['xd']['q' + str(kite) + str(parent)]
        center = center + q_kite / number_children

    return center

def approx_center_velocity(parent, variables, architecture):

    children = architecture.kites_map[parent]
    number_children = float(len(children))

    dcenter = np.zeros((3, 1))
    for kite in children:

        dq_kite = variables['xd']['dq' + str(kite) + str(parent)]
        dcenter = dcenter + dq_kite / number_children

    return dcenter

def approx_kite_radius_vector(variables, architecture, kite):

    parent = architecture.parent_map[kite]

    q_kite = variables['xd']['q' + str(kite) + str(parent)]
    center = approx_center_point(parent, variables, architecture)

    radius_vec = q_kite - center

    return radius_vec

def tether_vector(variables, architecture, parent):

    parent_map = architecture.parent_map
    grandparent = parent_map[parent]

    q_parent = variables['xd']['q' + str(parent) + str(grandparent)]

    if grandparent in parent_map.keys():
        great_grandparent = parent_map[grandparent]
        q_grandparent = variables['xd']['q' + str(grandparent) + str(great_grandparent)]
    else:
        q_grandparent = np.zeros((3, 1))

    tether = q_parent - q_grandparent

    return tether

def approx_kite_tang_vector(variables, architecture, kite):

    parent = architecture.parent_map[kite]

    z_vec = tether_vector(variables, architecture, parent)
    r_vec = approx_kite_radius_vector(variables, architecture, kite)

    t_vec_dimensioned = vect_op.cross(z_vec, r_vec)

    return t_vec_dimensioned

def approx_kite_normal_vector(variables, architecture, kite):

    # z cross r = t
    # r x t = n

    r_vec = approx_kite_radius_vector(variables, architecture, kite)
    t_vec = approx_kite_tang_vector(variables, architecture, kite)
    n_vec = vect_op.cross(r_vec, t_vec)

    return n_vec

def approx_normal_axis(parent, variables, architecture):

    children = architecture.kites_map[parent]

    normal_sum = np.zeros((3, 1))

    for kite in children:
        normal_sum = normal_sum + approx_kite_normal_vector(variables, architecture, kite)

    nhat = vect_op.smooth_normalize(normal_sum)
    return nhat