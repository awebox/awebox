#
#    This file is part of awebox.
#
#    awebox -- A modeling and optimization framework for multi-kite AWE systems.
#    Copyright (C) 2017-2020 Jochem De Schutter, Rachel Leuthold, Moritz Diehl,
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
"""
flow functions for the vortex based model
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2017-20
- edit: jochem de schutter, alu-fr 2019
"""

import casadi.tools as cas

import awebox.tools.vector_operations as vect_op
import awebox.mdl.aero.induction_dir.actuator_dir.geom as actuator_geom
import numpy as np
from awebox.logger.logger import Logger as awelogger
import awebox.mdl.aero.induction_dir.general_dir.flow as general_flow
import awebox.mdl.aero.induction_dir.vortex_dir.biot_savart as vortex_induction

def get_kite_effective_velocity(options, variables, wind, kite, architecture):


    parent = architecture.parent_map[kite]
    u_app_kite = general_flow.get_kite_apparent_velocity(variables, wind, kite, parent)

    u_ind_kite = vortex_induction.get_induced_velocity_at_kite(options, wind, variables, architecture, kite)

    u_eff_kite = u_app_kite + u_ind_kite

    return u_eff_kite

