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
import awebox.tools.print_operations as print_op
import casadi.tools as cas
import awebox.tools.struct_operations as struct_op
from awebox.logger.logger import Logger as awelogger

def print_warning_if_relevant(architecture):
    kite_parent_is_at_groundstation = [architecture.parent_map[kite] == 0 for kite in architecture.kite_nodes]
    warning_is_relevant = any(kite_parent_is_at_groundstation)
    if warning_is_relevant:
        message = 'if the parent of a kite node is at the ground, the parent geometry will estimate the center of rotation and the velocity of that center as the position and velocity, respectively, of the groundstation. this may lead to distorted calculations.'
        awelogger.logger.warning(message)
    return None

def get_center_position(parent, variables, architecture):
    parent_map = architecture.parent_map

    if parent > 0:
        grandparent = parent_map[parent]
        q_parent = struct_op.get_variable_from_model_or_reconstruction(variables, 'x', 'q' + str(parent) + str(grandparent))
    else:
        # TODO: rocking mode : is this q_arm_tip or the origin ?
        q_parent = cas.DM.zeros((3, 1))

    center = q_parent

    return center

def get_center_velocity(parent, variables, architecture):
    parent_map = architecture.parent_map

    if parent > 0:
        grandparent = parent_map[parent]
        dq_parent = struct_op.get_variable_from_model_or_reconstruction(variables, 'x', 'dq' + str(parent) + str(grandparent))
    else:
        # TODO: rocking mode : cf. get_center_position
        dq_parent = cas.DM.zeros((3, 1))

    dcenter = dq_parent

    return dcenter
