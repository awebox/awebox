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
- author: rachel leuthold, alu-fr 2019-2021
'''

import casadi as cas
import awebox.tools.vector_operations as vect_op
import awebox.mdl.aero.induction_dir.vortex_dir.tools as tools
import awebox.tools.print_operations as print_op
import awebox.tools.constraint_operations as cstr_op

def get_state_repr_convection_cstr(options, wind, variables_si, architecture):

    cstr_list = cstr_op.ConstraintList()

    kite_nodes = architecture.kite_nodes
    wingtips = ['ext', 'int']
    wake_nodes = options['aero']['vortex']['wake_nodes']

    for kite in kite_nodes:
        for tip in wingtips:
            for wake_node in range(wake_nodes):

                var_name = 'wx_' + str(kite) + '_' + tip + '_' + str(wake_node)

                wx_local = variables_si['xd'][var_name]
                u_infty = wind.get_velocity(wx_local[2])

                dwx_local = variables_si['xddot']['d' + var_name]

                resi_local = (dwx_local - u_infty) / wind.get_velocity_ref()

                local_cstr = cstr_op.Constraint(expr = resi_local,
                                                name = 'convection_' + var_name,
                                                cstr_type = 'eq')
                cstr_list.append(local_cstr)

    return cstr_list
