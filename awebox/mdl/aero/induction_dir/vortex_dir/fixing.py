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

import casadi.tools as cas
import numpy as np
import awebox.mdl.aero.induction_dir.vortex_dir.tools as tools
import awebox.tools.print_operations as print_op


def get_wake_fix_constraints(options, variables, model):
    # this function is just the placeholder. For the applied constraint, see constraints.append_wake_fix_constraints()
    #
    # eqs_dict = {}
    # ineqs_dict = {}
    # constraint_list = []
    #
    # comparison_labels = options['induction']['comparison_labels']
    # periods_tracked = options['induction']['vortex_periods_tracked']
    # kite_nodes = model.architecture.kite_nodes
    # wingtips = ['ext', 'int']
    #
    #
    # any_vor = any(label[:3] == 'vor' for label in comparison_labels)
    # if any_vor:
    #     n_k = options['n_k']
    #     d = options['collocation']['d']
    #
    #     for kite in kite_nodes:
    #         for tip in wingtips:
    #             for period in range(periods_tracked):
    #
    #                 parent_map = model.architecture.parent_map
    #                 parent = parent_map[kite]
    #
    #                 wake_pos_dir = {}
    #                 for dim in ['x', 'y', 'z']:
    #                     var_name = 'w' + dim + '_' + tip + '_' + str(period) + '_' + str(kite) + str(parent)
    #                     wake_pos_dir[dim] = variables['xd', var_name]
    #
    #                 n_nodes = n_k * d + 1
    #                 for ldx in range(n_nodes):
    #                     wake_pos = cas.vertcat(wake_pos_dir['x'][ldx], wake_pos_dir['y'][ldx], wake_pos_dir['z'][ldx])
    #
    #                     # reminder! this function is just the space-holder.
    #                     wing_tip_pos = 0. * vect_op.zhat()
    #                     resi = wake_pos - wing_tip_pos
    #
    #                     name = 'wake_fix_period' + str(period) + '_kite' + str(kite) + '_' + tip + str(ldx)
    #                     eqs_dict[name] = resi
    #                     constraint_list.append(resi)
    #
    # # generate initial constraints - empty struct containing both equalities and inequalitiess
    # wake_fix_constraints_struct = make_constraint_struct(eqs_dict, ineqs_dict)
    #
    # # fill in struct and create function
    # wake_fix_constraints = wake_fix_constraints_struct(cas.vertcat(*constraint_list))
    # wake_fix_constraints_fun = cas.Function('wake_fix_constraints_fun', [variables], [wake_fix_constraints.cat])

    cstr = []
    cstr_fun = cas.Function('wake_fix_constraints_fun', [variables], [cstr])

    print_op.warn_about_temporary_funcationality_removal(location='fixing')

    return cstr, cstr_fun


def append_wake_fix_constraints(options, g_list, g_bounds, V, Outputs, model):

    # comparison_labels = options['induction']['comparison_labels']
    # periods_tracked = options['induction']['vortex_periods_tracked']
    #
    # any_vor = any(label[:3] == 'vor' for label in comparison_labels)
    # if any_vor:
    #     g_list, g_bounds = fixing_constraints_on_zeroth_period(options, g_list, g_bounds, V, Outputs, model)
    #
    #     for period in range(1, periods_tracked):
    #         g_list, g_bounds = fixing_constraints_on_previous_period(options, g_list, g_bounds, V, Outputs, model, period)

    print_op.warn_about_temporary_funcationality_removal(location='fixing2')

    return g_list, g_bounds


def fixing_constraints_on_zeroth_period(options, g_list, g_bounds, V, Outputs, model):

    print_op.warn_about_temporary_funcationality_removal(location='vortex_fixing_0')

    #
    # kite_nodes = model.architecture.kite_nodes
    # parent_map = model.architecture.parent_map
    # wingtips = ['ext', 'int']
    # dims = ['x', 'y', 'z']
    #
    # n_k = options['n_k']
    # d = options['collocation']['d']
    #
    # period = 0
    #
    # for kite in kite_nodes:
    #     parent = parent_map[kite]
    #     for tip in wingtips:
    #         for jdx in range(len(dims)):
    #             dim = dims[jdx]
    #             var_name = 'w' + dim + '_' + tip + '_' + str(period) + '_' + str(kite) + str(parent)
    #
    #             var_column = V['xd', 0, var_name]
    #             node_pos = tools.get_wake_var_at_ndx_ddx(n_k, d, var_column, start=True)
    #
    #             # remember: periodicity! wingtip positions at end, must be equal to positions at start
    #             wingtip_pos = Outputs['coll_outputs', -1, -1, 'aerodynamics', 'wingtip_' + tip + str(kite)][jdx]
    #
    #             fix = node_pos - wingtip_pos
    #             g_list.append(fix)
    #             g_bounds['ub'].append(np.zeros(fix.shape))
    #             g_bounds['lb'].append(np.zeros(fix.shape))
    #
    #             for ndx in range(n_k):
    #                 for ddx in range(d):
    #
    #                     var_column = V['coll_var', ndx, ddx, 'xd', var_name]
    #                     node_pos = tools.get_wake_var_at_ndx_ddx(n_k, d, var_column, ndx=ndx, ddx=ddx)
    #                     wingtip_pos = Outputs['coll_outputs', ndx, ddx, 'aerodynamics', 'wingtip_' + tip + str(kite)][jdx]
    #
    #                     fix = node_pos - wingtip_pos
    #                     g_list.append(fix)
    #                     g_bounds['ub'].append(np.zeros(fix.shape))
    #                     g_bounds['lb'].append(np.zeros(fix.shape))

    return [g_list, g_bounds]


def fixing_constraints_on_previous_period(options, g_list, g_bounds, V, Outputs, model, period):

    # kite_nodes = model.architecture.kite_nodes
    # parent_map = model.architecture.parent_map
    # wingtips = ['ext', 'int']
    # dims = ['x', 'y', 'z']
    #
    # for kite in kite_nodes:
    #     parent = parent_map[kite]
    #
    #     for tip in wingtips:
    #         for dim in dims:
    #
    #             var_name = 'w' + dim + '_' + tip + '_' + str(period) + '_' + str(kite) + str(parent)
    #             prev_name = 'w' + dim + '_' + tip + '_' + str(period - 1) + '_' + str(kite) + str(parent)
    #
    #             node_column = V['xd', 0, var_name]
    #
    #             prev_column = V['coll_var', -1, -1, 'xd', prev_name]
    #
    #             fix = node_column - prev_column
    #             g_list.append(fix)
    #             g_bounds['ub'].append(np.zeros(fix.shape))
    #             g_bounds['lb'].append(np.zeros(fix.shape))

    print_op.warn_about_temporary_funcationality_removal(location='vortex_fixing_prev')


    return [g_list, g_bounds]
