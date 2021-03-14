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
"""
general flow functions for the induction model
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2017-21
- edit: jochem de schutter, alu-fr 2019
"""

import awebox.mdl.aero.induction_dir.tools_dir.geom as general_geom

import awebox.tools.vector_operations as vect_op
import awebox.tools.print_operations as print_op

def get_kite_apparent_velocity(variables, wind, kite, parent):

    q_kite = variables['xd']['q' + str(kite) + str(parent)]
    u_infty = wind.get_velocity(q_kite[2])
    u_kite = variables['xd']['dq' + str(kite) + str(parent)]
    u_app_kite = u_infty - u_kite

    return u_app_kite

def get_uzero_vec(model_options, wind, parent, variables, architecture):

    u_infty = get_actuator_freestream_velocity(model_options, wind, parent, variables, architecture)
    u_actuator = general_geom.get_center_velocity(parent, variables, architecture)

    u_apparent = u_infty - u_actuator

    return u_apparent

def get_f_val(model_options, wind, parent, variables, architecture):
    dl_t = variables['xd']['dl_t']
    u_infty = get_actuator_freestream_velocity(model_options, wind, parent, variables, architecture)
    f_val = dl_t / vect_op.smooth_norm(u_infty)

    return f_val

def get_actuator_freestream_velocity(model_options, wind, parent, variables, architecture):

    center = general_geom.get_center_point(model_options, parent, variables, architecture)
    u_infty = wind.get_velocity(center[2])

    return u_infty