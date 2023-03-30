#
#    This file is part of awebox.
#
#    awebox -- A modeling and optimization framework for multi-kite AWE systems.
#    Copyright (C) 2017-2022 Jochem De Schutter, Rachel Leuthold, Moritz Diehl,
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
geometry values needed for general induction modelling
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2017-22
- edit: jochem de schutter, alu-fr 2019
'''

import numpy as np

import awebox.tools.vector_operations as vect_op
import awebox.tools.struct_operations as struct_op
import awebox.tools.print_operations as print_op
from awebox.logger.logger import Logger as awelogger

def print_warning_if_relevant(architecture):
    warning_is_relevant = (architecture.number_of_kites == 1)
    if warning_is_relevant:
        message = 'in a single-kite situation, the averaged geometry will estimate the center of rotation and the velocity of that center as the position and velocity, respectively, of the single kite. this may lead to distorted calculations and divide-by-zero errors'
        awelogger.logger.warning(message)
    return None

def get_center_position(parent, variables, architecture):

    children = architecture.kites_map[parent]
    number_children = float(len(children))

    center = np.zeros((3, 1))
    for kite in children:
        q_kite = struct_op.get_variable_from_model_or_reconstruction(variables, 'x', 'q' + str(kite) + str(parent))
        center += q_kite / number_children

    return center

def get_center_velocity(parent, variables, architecture):

    children = architecture.kites_map[parent]
    number_children = float(len(children))

    dcenter = np.zeros((3, 1))
    for kite in children:
        dq_kite = struct_op.get_variable_from_model_or_reconstruction(variables, 'x', 'dq' + str(kite) + str(parent))
        dcenter += dq_kite / number_children

    return dcenter
