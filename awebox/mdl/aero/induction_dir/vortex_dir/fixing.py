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
import awebox.tools.struct_operations as struct_op
import casadi.tools as cas
from awebox.logger.logger import Logger as awelogger
import awebox.tools.print_operations as print_op


######## the constraints : see opti.constraints

def get_fixing_constraints(options, g_list, g_bounds, V, Outputs, model):

    comparison_labels = options['induction']['comparison_labels']
    periods_tracked = options['induction']['vortex_periods_tracked']

    any_vor = any(label[:3] == 'vor' for label in comparison_labels)
    if any_vor:
        g_list, g_bounds = fixing_constraints_on_zeroth_period(options, g_list, g_bounds, V, Outputs, model)

        for period in range(1, periods_tracked):
            g_list, g_bounds = fixing_constraints_on_previous_period(options, g_list, g_bounds, V, model, period)

    return g_list, g_bounds


def fixing_constraints_on_zeroth_period(options, g_list, g_bounds, V, Outputs, model):

    kite_nodes = model.architecture.kite_nodes
    parent_map = model.architecture.parent_map
    wingtips = ['ext', 'int']
    dims = ['x', 'y', 'z']

    n_k = options['n_k']
    d = options['collocation']['d']

    Xdot = struct_op.construct_Xdot_struct(options, model)(0.)

    period = 0

    for kite in kite_nodes:
        parent = parent_map[kite]
        for tip in wingtips:
            for jdx in range(len(dims)):

                # this is the variable that we're fixing
                dim = dims[jdx]
                var_name = 'w' + dim + '_' + tip + '_' + str(period) + '_' + str(kite) + str(parent)

                # remember: periodicity! wingtip positions at end, must be equal to positions at start
                variables = struct_op.get_variables_at_time(options, V, Xdot, model, 0, ddx=None)
                wingtip_pos = Outputs['coll_outputs', -1, -1, 'aerodynamics', 'wingtip_' + tip + str(kite)][jdx]
                fix = get_zeroth_xd_fix(variables, var_name, options, wingtip_pos)

                g_list.append(fix)
                g_bounds = tools.append_bounds(g_bounds, fix)

                for ndx in range(n_k):
                    for ddx in range(d):

                        variables = struct_op.get_variables_at_time(options, V, Xdot, model, ndx, ddx=ddx)
                        wingtip_pos = Outputs['coll_outputs', ndx, ddx, 'aerodynamics', 'wingtip_' + tip + str(kite)][jdx]
                        fix = get_zeroth_xd_coll_fix(variables, var_name, options, wingtip_pos, ndx, ddx)

                        g_list.append(fix)
                        g_bounds = tools.append_bounds(g_bounds, fix)

    return g_list, g_bounds


def fixing_constraints_on_previous_period(options, g_list, g_bounds, V, model, period):
    kite_nodes = model.architecture.kite_nodes
    parent_map = model.architecture.parent_map
    wingtips = ['ext', 'int']
    dims = ['x', 'y', 'z']

    Xdot = struct_op.construct_Xdot_struct(options, model)(0.)

    for kite in kite_nodes:
        parent = parent_map[kite]

        for tip in wingtips:
            for dim in dims:

                var_name = 'w' + dim + '_' + tip + '_' + str(period) + '_' + str(kite) + str(parent)
                prev_name = 'w' + dim + '_' + tip + '_' + str(period - 1) + '_' + str(kite) + str(parent)

                variables = struct_op.get_variables_at_time(options, V, Xdot, model, 0, ddx=None)
                prev_variables = struct_op.get_variables_at_time(options, V, Xdot, model, -1, ddx=-1)

                fix = get_previous_fix(variables, var_name, prev_variables, prev_name)

                g_list.append(fix)
                g_bounds = tools.append_bounds(g_bounds, fix)

    return g_list, g_bounds


######## the placeholders : see ocp.operation


def get_placeholder_fixing_constraints(options, variables, model):
    eqs_dict = {}
    constraint_list = []

    comparison_labels = options['induction']['comparison_labels']
    periods_tracked = options['induction']['vortex_periods_tracked']

    any_vor = any(label[:3] == 'vor' for label in comparison_labels)
    if any_vor:
        eqs_dict, constraint_list = placeholder_fixing_constraints_on_zeroth_period(options, variables, model, eqs_dict, constraint_list)

        for period in range(1, periods_tracked):
            eqs_dict, constraint_list = placeholder_fixing_constraints_on_previous_period(variables, model, period, eqs_dict, constraint_list)

    return eqs_dict, constraint_list



def placeholder_fixing_constraints_on_zeroth_period(options, variables, model, eqs_dict, constraint_list):

    kite_nodes = model.architecture.kite_nodes
    parent_map = model.architecture.parent_map
    wingtips = ['ext', 'int']
    dims = ['x', 'y', 'z']

    n_k = options['n_k']
    d = options['collocation']['d']

    wingtip_pos = cas.DM(0.)

    period = 0

    for kite in kite_nodes:
        parent = parent_map[kite]
        for tip in wingtips:
            for jdx in range(len(dims)):

                # this is the variable that we're fixing
                dim = dims[jdx]
                var_name = 'w' + dim + '_' + tip + '_' + str(period) + '_' + str(kite) + str(parent)
                eq_name = 'wake_fix_dim_' + dim + '_tip_' + tip + '_period_' + str(period) + '_kite_' + str(kite)
                g_list = []

                fix = get_zeroth_xd_fix(variables, var_name, options, wingtip_pos)
                g_list = cas.vertcat(g_list, fix)

                for ndx in range(n_k):
                    for ddx in range(d):

                        fix = get_zeroth_xd_coll_fix(variables, var_name, options, wingtip_pos, ndx, ddx)
                        g_list = cas.vertcat(g_list, fix)

                eqs_dict[eq_name] = g_list
                constraint_list.append(g_list)

    return eqs_dict, constraint_list


def placeholder_fixing_constraints_on_previous_period(variables, model, period, eqs_dict, constraint_list):

    kite_nodes = model.architecture.kite_nodes
    parent_map = model.architecture.parent_map
    wingtips = ['ext', 'int']
    dims = ['x', 'y', 'z']

    for kite in kite_nodes:
        parent = parent_map[kite]

        for tip in wingtips:
            for dim in dims:

                var_name = 'w' + dim + '_' + tip + '_' + str(period) + '_' + str(kite) + str(parent)
                eq_name = 'wake_fix_dim_' + dim + '_tip_' + tip + '_period_' + str(period) + '_kite_' + str(kite)

                g_list = get_placeholder_previous_fix(variables, var_name)

                eqs_dict[eq_name] = g_list
                constraint_list.append(g_list)

    return eqs_dict, constraint_list



######### the helper functions

def get_zeroth_xd_fix(variables, var_name, options, wingtip_pos):
    n_k = options['n_k']
    d = options['collocation']['d']

    var_column = variables['xd', var_name]
    node_pos = tools.get_wake_var_at_ndx_ddx(n_k, d, var_column, start=True)

    print_op.warn_about_temporary_funcationality_removal(editor='rachel', location='vortex.fixing.get_zeroth_xd_fix')
    # fix = node_pos - wingtip_pos
    fix = []

    return fix


def get_zeroth_xd_coll_fix(variables, var_name, options, wingtip_pos, ndx, ddx):
    n_k = options['n_k']
    d = options['collocation']['d']

    var_column = variables['xd', var_name]
    node_pos = tools.get_wake_var_at_ndx_ddx(n_k, d, var_column, ndx=ndx, ddx=ddx)

    print_op.warn_about_temporary_funcationality_removal(editor='rachel', location='vortex.fixing.get_zeroth_xd_coll_fix')
    # fix = node_pos - wingtip_pos
    fix = []

    return fix


def get_previous_fix(variables, var_name, prev_variables, prev_name):
    var_column = variables['xd', var_name]
    prev_column = prev_variables['xd', prev_name]

    print_op.warn_about_temporary_funcationality_removal(editor='rachel', location='vortex.fixing.get_previous_fix')
    # fix = var_column - prev_column
    fix = []

    return fix

def get_placeholder_previous_fix(variables, var_name):
    var_column = variables['xd', var_name]

    print_op.warn_about_temporary_funcationality_removal(editor='rachel', location='vortex.fixing.get_placeholder_previous_fix')
    # fix = var_column
    fix = []

    return fix

