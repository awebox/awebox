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
import awebox.ocp.collocation as collocation
import awebox.ocp.var_struct as var_struct

import pdb

######## the constraints : see opti.constraints

def get_cstr_in_constraints_format(options, g_list, g_bounds, V, Outputs, model):

    resi = get_fixing_constraint_all(options, V, Outputs, model)

    g_list.append(resi)
    g_bounds = tools.append_bounds(g_bounds, resi)

    return g_list, g_bounds

######## the placeholders : see ocp.operation

def get_cstr_in_operation_format(options, variables, model):
    eqs_dict = {}
    constraint_list = []

    if 'collocation' not in options.keys():
        message = 'vortex model is not yet set up for any discretization ' \
                  'other than direct collocation'
        awelogger.logger.error(message)

    n_k = options['n_k']
    d = options['collocation']['d']
    scheme = options['collocation']['scheme']
    Collocation = collocation.Collocation(n_k, d, scheme)

    model_outputs = model.outputs
    V_mock = var_struct.setup_nlp_v(options, model, Collocation)

    entry_tuple = (cas.entry('coll_outputs', repeat = [n_k,d], struct = model_outputs))
    Outputs_mock = cas.struct_symMX([entry_tuple])

    resi_mock = get_fixing_constraint_all(options, V_mock, Outputs_mock, model)
    try:
        resi = cas.DM.ones(resi_mock.shape)
    except:
        resi = []

    eq_name = 'vortex_fixing'
    eqs_dict[eq_name] = resi
    constraint_list.append(resi)

    return eqs_dict, constraint_list

################# define the actual constraint

def get_fixing_constraint_all(options, V, Outputs, model):

    resi = []

    comparison_labels = options['induction']['comparison_labels']
    periods_tracked = options['induction']['vortex_periods_tracked']

    any_vor = any(label[:3] == 'vor' for label in comparison_labels)
    if any_vor:
        local_resi = fixing_constraints_on_zeroth_period(options, V, Outputs, model)
        resi = cas.vertcat(resi, local_resi)

        for period in range(1, periods_tracked):
            local_resi = fixing_constraints_on_previous_period(options, V, model, period)
            resi = cas.vertcat(resi, local_resi)

    return resi



def fixing_constraints_on_zeroth_period(options, V, Outputs, model):

    local_resi = []

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

                local_resi = cas.vertcat(local_resi, fix)

                for ndx in range(n_k):
                    for ddx in range(d):

                        variables = struct_op.get_variables_at_time(options, V, Xdot, model, ndx, ddx=ddx)
                        wingtip_pos = Outputs['coll_outputs', ndx, ddx, 'aerodynamics', 'wingtip_' + tip + str(kite)][jdx]
                        fix = get_zeroth_xd_coll_fix(variables, var_name, options, wingtip_pos, ndx, ddx)

                        local_resi = cas.vertcat(local_resi, fix)

    return local_resi


def fixing_constraints_on_previous_period(options, V, model, period):

    local_resi = []

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

                local_resi = cas.vertcat(local_resi, fix)

    return local_resi






######### the helper functions

def get_zeroth_xd_fix(variables, var_name, options, wingtip_pos):
    n_k = options['n_k']
    d = options['collocation']['d']

    var_column = variables['xd', var_name]
    node_pos = tools.get_wake_var_at_ndx_ddx(n_k, d, var_column, start=True)

    fix = node_pos - wingtip_pos

    return fix


def get_zeroth_xd_coll_fix(variables, var_name, options, wingtip_pos, ndx, ddx):
    n_k = options['n_k']
    d = options['collocation']['d']

    var_column = variables['xd', var_name]
    node_pos = tools.get_wake_var_at_ndx_ddx(n_k, d, var_column, ndx=ndx, ddx=ddx)

    fix = node_pos - wingtip_pos

    return fix


def get_previous_fix(variables, var_name, prev_variables, prev_name):
    var_column = variables['xd', var_name]
    prev_column = prev_variables['xd', prev_name]

    fix = var_column - prev_column

    return fix
