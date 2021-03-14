#
#    This file is part of awebox.
#
#    awebox -- A modeling and optimization framework for multi-kite AWE systems.
#    Copyright (C) 2017-2021 Jochem De Schutter, Rachel Leuthold, Moritz Diehl,
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
- author: rachel leuthold, alu-fr 2017-21
- edit: jochem de schutter, alu-fr 2019
'''

import casadi.tools as cas
import numpy as np
from awebox.logger.logger import Logger as awelogger

import awebox.mdl.aero.induction_dir.tools_dir.unit_normal as unit_normal
import awebox.mdl.aero.induction_dir.tools_dir.path_based_geom as path_based_geom
import awebox.mdl.aero.induction_dir.tools_dir.multi_kite_geom as multi_kite_geom

import awebox.tools.constraint_operations as cstr_op
import awebox.tools.vector_operations as vect_op

def get_center_point(model_options, parent, variables, architecture):

    children = architecture.kites_map[parent]
    number_children = float(len(children))

    if number_children > 1:
        center = multi_kite_geom.approx_center_point(parent, variables, architecture)
    else:
        center = path_based_geom.approx_center_point(model_options, children, variables, architecture)

    return center

def get_center_velocity(parent, variables, architecture):

    children = architecture.kites_map[parent]
    number_children = float(len(children))

    if (parent > 0) and (number_children > 1):
        dcenter = multi_kite_geom.approx_center_velocity(parent, variables, architecture)
    elif (number_children == 1):
        dq = variables['xd']['dq' + str(children[0]) + str(parent)]
        dcenter = dq
    else:
        message = 'actuator-center velocity not yet set-up for this architecture'
        awelogger.logger.error(message)
        raise Exception(message)

    return dcenter
