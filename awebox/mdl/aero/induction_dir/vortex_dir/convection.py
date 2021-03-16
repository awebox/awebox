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
geometry functions for vortex model
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2019-2020
'''

import casadi as cas
import awebox.tools.vector_operations as vect_op
import awebox.mdl.aero.induction_dir.vortex_dir.tools as tools
import awebox.tools.print_operations as print_op

def get_state_repr_convection_residual(options, wind, variables_si, architecture):

    kite_nodes = architecture.kite_nodes
    wingtips = ['ext', 'int']
    wake_nodes = options['aero']['vortex']['wake_nodes']

    resi = []
    for kite in kite_nodes:
        for tip in wingtips:
            for wake_node in range(wake_nodes):

                wx_local = tools.get_wake_node_position_si(options, variables_si, kite, tip, wake_node)
                dwx_local = tools.get_wake_node_velocity_si(variables_si, kite, tip, wake_node)

                altitude = cas.mtimes(wx_local.T, vect_op.zhat())
                u_infty = wind.get_velocity(altitude)

                resi_local = (dwx_local - u_infty) / wind.get_velocity_ref()
                resi = cas.vertcat(resi, resi_local)

    return resi