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

import numpy as np
import awebox.mdl.aero.induction_dir.vortex_dir.tools as tools

def fixing_constraints_on_zeroth_period(options, g_list, g_bounds, V, Outputs, model):

    kite_nodes = model.architecture.kite_nodes
    parent_map = model.architecture.parent_map
    wingtips = ['ext', 'int']
    dims = ['x', 'y', 'z']

    n_k = options['n_k']
    d = options['collocation']['d']

    period = 0

    for kite in kite_nodes:
        parent = parent_map[kite]
        for tip in wingtips:
            for jdx in range(len(dims)):
                dim = dims[jdx]
                var_name = 'w' + dim + '_' + tip + '_' + str(period) + '_' + str(kite) + str(parent)

                var_column = V['xd', 0, var_name]
                node_pos = tools.get_wake_var_at_ndx_ddx(n_k, d, var_column, start=True)

                # remember: periodicity! wingtip positions at end, must be equal to positions at start
                wingtip_pos = Outputs['coll_outputs', -1, -1, 'aerodynamics', 'wingtip_' + tip + str(kite)][jdx]

                fix = node_pos - wingtip_pos
                g_list.append(fix)
                g_bounds['ub'].append(np.zeros(fix.shape))
                g_bounds['lb'].append(np.zeros(fix.shape))

                for ndx in range(n_k):
                    for ddx in range(d):

                        var_column = V['coll_var', ndx, ddx, 'xd', var_name]
                        node_pos = tools.get_wake_var_at_ndx_ddx(n_k, d, var_column, ndx=ndx, ddx=ddx)
                        wingtip_pos = Outputs['coll_outputs', ndx, ddx, 'aerodynamics', 'wingtip_' + tip + str(kite)][jdx]

                        fix = node_pos - wingtip_pos
                        g_list.append(fix)
                        g_bounds['ub'].append(np.zeros(fix.shape))
                        g_bounds['lb'].append(np.zeros(fix.shape))

    return [g_list, g_bounds]


def fixing_constraints_on_previous_period(options, g_list, g_bounds, V, Outputs, model, period):
    kite_nodes = model.architecture.kite_nodes
    parent_map = model.architecture.parent_map
    wingtips = ['ext', 'int']
    dims = ['x', 'y', 'z']

    for kite in kite_nodes:
        parent = parent_map[kite]

        for tip in wingtips:
            for dim in dims:

                var_name = 'w' + dim + '_' + tip + '_' + str(period) + '_' + str(kite) + str(parent)
                prev_name = 'w' + dim + '_' + tip + '_' + str(period - 1) + '_' + str(kite) + str(parent)

                node_column = V['xd', 0, var_name]

                prev_column = V['coll_var', -1, -1, 'xd', prev_name]

                fix = node_column - prev_column
                g_list.append(fix)
                g_bounds['ub'].append(np.zeros(fix.shape))
                g_bounds['lb'].append(np.zeros(fix.shape))

    return [g_list, g_bounds]
