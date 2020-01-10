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
constraints to create "intermediate condition" fixing constraints on the positions of the wake nodes,
to be referenced/used from ocp.constraints
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2020
'''

import casadi as cas
import numpy as np
from awebox.logger.logger import Logger as awelogger
import awebox.tools.vector_operations as vect_op
import pdb

def fixing_constraints_on_zeroth_period(options, g_list, g_bounds, V, Outputs, model):

    kite_nodes = model.architecture.kite_nodes
    parent_map = model.architecture.parent_map
    wingtips = ['ext', 'int']

    n_k = options['n_k']
    d = options['collocation']['d']

    period = 0

    for ndx in range(n_k):
        for ddx in range(d):
            for kite in kite_nodes:
                for tip in wingtips:

                    parent = parent_map[kite]

                    node_pos = []
                    for dim in ['x', 'y', 'z']:
                        var_name = 'w' + dim + '_' + tip + '_' + str(period) + '_' + str(kite) + str(parent)

                        var_column = V['coll_var', ndx, ddx, 'xd', var_name]
                        var_reshape = cas.reshape(var_column, (n_k, d))
                        var_local = var_reshape[ndx, ddx]
                        node_pos = cas.vertcat(node_pos, var_local)

                    wingtip_pos = Outputs['coll_outputs', ndx, ddx, 'aerodynamics', 'wingtip_' + tip + str(kite)]

                    fix = node_pos - wingtip_pos

                    g_list.append(fix)

                    g_bounds['ub'].append(np.zeros(fix.shape))
                    g_bounds['lb'].append(np.zeros(fix.shape))

    return [g_list, g_bounds]


def fixing_constraints_on_previous_period(options, g_list, g_bounds, V, Outputs, model, period):
    kite_nodes = model.architecture.kite_nodes
    parent_map = model.architecture.parent_map
    wingtips = ['ext', 'int']

    n_k = options['n_k']
    d = options['collocation']['d']

    for ndx_shed in range(n_k):
        for ddx_shed in range(d):
            for kite in kite_nodes:
                for tip in wingtips:

                    parent = parent_map[kite]

                    node_pos = []
                    prev_pos = []
                    for dim in ['x', 'y', 'z']:
                        var_name = 'w' + dim + '_' + tip + '_' + str(period) + '_' + str(kite) + str(parent)
                        prev_name = 'w' + dim + '_' + tip + '_' + str(period - 1) + '_' + str(kite) + str(parent)

                        var_column = V['coll_var', 0, 0, 'xd', var_name]
                        var_reshape = cas.reshape(var_column, (n_k, d))
                        var_local = var_reshape[ndx_shed, ddx_shed]
                        node_pos = cas.vertcat(node_pos, var_local)

                        prev_column = V['coll_var', n_k-1, d-1, 'xd', prev_name] # indexing starts at 0
                        prev_reshape = cas.reshape(prev_column, (n_k, d))
                        prev_local = prev_reshape[ndx_shed, ddx_shed]
                        prev_pos = cas.vertcat(prev_pos, prev_local)

                    fix = node_pos - prev_pos

                    g_list.append(fix)

                    g_bounds['ub'].append(np.zeros(fix.shape))
                    g_bounds['lb'].append(np.zeros(fix.shape))

    return [g_list, g_bounds]
