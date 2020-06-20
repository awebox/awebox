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
constraints to create the on-off switch on the vortex strength
to be referenced/used from ocp.constraints
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2020
'''

import casadi as cas
import numpy as np
import awebox.tools.struct_operations as struct_op
import awebox.mdl.aero.induction_dir.vortex_dir.tools as tools
from awebox.logger.logger import Logger as awelogger


######## the constraints : see opti.constraints

def get_vortex_strength_constraints(options, g_list, g_bounds, V, Outputs, model):

    comparison_labels = options['induction']['comparison_labels']
    periods_tracked = options['induction']['vortex_periods_tracked']

    if periods_tracked > 1:
        periods_tracked = 1

    any_vor = any(label[:3] == 'vor' for label in comparison_labels)
    if any_vor:
        for period in range(periods_tracked):
            g_list, g_bounds = strength_constraints(options, g_list, g_bounds, V, Outputs, model, period)

    return g_list, g_bounds


def strength_constraints(options, g_list, g_bounds, V, Outputs, model, period):
    n_k = options['n_k']
    d = options['collocation']['d']

    if period == 0:
        for ndx in range(n_k):
            for ddx in range(d):
                for ndx_shed in range(n_k):
                    for ddx_shed in range(d):
                        g_list, g_bounds = strength_constraints_on_zeroth_period(options, g_list, g_bounds, V, Outputs, model, ndx, ddx,
                                                              ndx_shed, ddx_shed)

    elif period == 1:
        for ndx in range(n_k):
            for ddx in range(d):
                for ndx_shed in range(n_k):
                    for ddx_shed in range(d):
                        g_list, g_bounds = strength_constraints_on_previous_period(options, g_list, g_bounds, V, Outputs, model, ndx, ddx, ndx_shed,
                                              ddx_shed)

    return g_list, g_bounds


def strength_constraints_on_zeroth_period(options, g_list, g_bounds, V, Outputs, model, ndx, ddx, ndx_shed, ddx_shed):

    period = 0
    architecture = model.architecture
    Xdot = struct_op.construct_Xdot_struct(options, model)(0.)

    for kite in architecture.kite_nodes:
        parent = architecture.parent_map[kite]

        gamma_name = 'wg' + '_' + str(period) + '_' + str(kite) + str(parent)
        variables = struct_op.get_variables_at_time(options, V, Xdot, model, ndx, ddx=ddx)

        if is_on_vortex(ndx, ddx, ndx_shed, ddx_shed):
            gamma_val = Outputs['coll_outputs', ndx_shed, ddx_shed, 'aerodynamics', 'gamma' + str(kite)]
        else:
            gamma_val = cas.DM(0.)

        resi = get_strength_resi(variables, gamma_name, ndx_shed, ddx_shed, options, gamma_val)

        g_list.append(resi)
        g_bounds = tools.append_bounds(g_bounds, resi)

    return g_list, g_bounds


def strength_constraints_on_previous_period(options, g_list, g_bounds, V, Outputs, model, ndx, ddx, ndx_shed,
                                              ddx_shed):

    period = 1
    architecture = model.architecture
    Xdot = struct_op.construct_Xdot_struct(options, model)(0.)

    for kite in architecture.kite_nodes:
        parent = architecture.parent_map[kite]

        gamma_name = 'wg' + '_' + str(period) + '_' + str(kite) + str(parent)
        variables = struct_op.get_variables_at_time(options, V, Xdot, model, ndx, ddx=ddx)

        gamma_val = Outputs['coll_outputs', ndx_shed, ddx_shed, 'aerodynamics', 'gamma' + str(kite)]

        resi = get_strength_resi(variables, gamma_name, ndx_shed, ddx_shed, options, gamma_val)

        g_list.append(resi)
        g_bounds = tools.append_bounds(g_bounds, resi)

    return g_list, g_bounds



######## the placeholders : see ocp.operation

def get_placeholder_vortex_strength_constraints(options, variables, model):

    eqs_dict = {}
    constraint_list = []

    comparison_labels = options['induction']['comparison_labels']
    periods_tracked = options['induction']['vortex_periods_tracked']

    if periods_tracked > 1:
        periods_tracked = 1

    any_vor = any(label[:3] == 'vor' for label in comparison_labels)
    if any_vor:
        for period in range(periods_tracked):
            eqs_dict, constraint_list = placeholder_strength_constraints(options, variables, model, period, eqs_dict, constraint_list)

    return eqs_dict, constraint_list

def placeholder_strength_constraints(options, variables, model, period, eqs_dict, constraint_list):
    n_k = options['n_k']
    d = options['collocation']['d']

    if (period == 0) or (period == 1):
        for ndx in range(n_k):
            for ddx in range(d):
                for ndx_shed in range(n_k):
                    for ddx_shed in range(d):
                        eqs_dict, constraint_list = placeholder_strength_constraints_on_any_period(options, variables, model, period, eqs_dict, \
                                                                       constraint_list, ndx_shed, ddx_shed)

    return eqs_dict, constraint_list


def placeholder_strength_constraints_on_any_period(options, variables, model, period, eqs_dict, constraint_list, ndx_shed, ddx_shed):
    architecture = model.architecture

    eq_name = 'wake_strength_period_' + str(period) + '_ndxs_' + str(ndx_shed) + '_ddxs_' + str(ddx_shed) + '_' + str(np.random.randint(0, 100000))
    g_list = []

    for kite in architecture.kite_nodes:
        parent = architecture.parent_map[kite]

        gamma_name = 'wg' + '_' + str(period) + '_' + str(kite) + str(parent)

        resi = get_placeholder_strength_resi(variables, gamma_name, ndx_shed, ddx_shed, options)
        g_list = cas.vertcat(g_list, resi)

    eqs_dict[eq_name] = g_list
    constraint_list.append(g_list)

    return eqs_dict, constraint_list


######### the helper functions

def is_on_vortex(ndx, ddx, ndx_shed, ddx_shed):
    if ndx > ndx_shed:
        return True
    elif ndx == ndx_shed and ddx > ddx_shed:
        return True
    elif ndx == ndx_shed and ddx == ddx_shed:
        return True
    else:
        return False

def get_wake_var_at_ndx_ddx(n_k, d, var, ndx, ddx):

    dimensions = (n_k, d)
    var_reshape = cas.reshape(var, dimensions)

    return var_reshape[ndx, ddx]

def get_strength_resi(variables, gamma_name, ndx_shed, ddx_shed, options, gamma_val):
    n_k = options['n_k']
    d = options['collocation']['d']

    var = variables['xl', gamma_name]
    gamma_var = get_wake_var_at_ndx_ddx(n_k, d, var, ndx_shed, ddx_shed)

    awelogger.logger.warning('rachel has temporarily removed this functionality, in order to improve the code flow. location: vortex.strength.get_strength_resi')

    # resi = gamma_var - gamma_val
    resi = []

    return resi


def get_placeholder_strength_resi(variables, gamma_name, ndx_shed, ddx_shed, options):
    n_k = options['n_k']
    d = options['collocation']['d']

    var = variables['xl', gamma_name]
    gamma_var = get_wake_var_at_ndx_ddx(n_k, d, var, ndx_shed, ddx_shed)

    awelogger.logger.warning(
        'rachel has temporarily removed this functionality, in order to improve the code flow. location: vortex.strength.get_placeholder_strength_resi')

    # resi = gamma_var
    resi = []

    return resi
