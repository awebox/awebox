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

import casadi.tools as cas
import numpy as np
import awebox.tools.struct_operations as struct_op
import awebox.mdl.aero.induction_dir.vortex_dir.tools as tools
from awebox.logger.logger import Logger as awelogger
import awebox.tools.print_operations as print_op
import awebox.ocp.collocation as collocation
import awebox.ocp.var_struct as var_struct


######## the constraints : see opti.constraints

def get_cstr_in_constraints_format(options, g_list, g_bounds, V, Outputs, model):

    resi = get_strength_constraint_all(options, V, Outputs, model)

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

    entry_tuple = (cas.entry('coll_outputs', repeat=[n_k, d], struct=model_outputs))
    Outputs_mock = cas.struct_symMX([entry_tuple])

    resi_mock = get_strength_constraint_all(options, V_mock, Outputs_mock, model)
    try:
        resi = cas.DM.ones(resi_mock.shape)
    except:
        resi = []

    eq_name = 'vortex_strength'
    eqs_dict[eq_name] = resi
    



    constraint_list.append(resi)

    return eqs_dict, constraint_list


################ actually define the constriants

def get_strength_constraint_all(options, V, Outputs, model):

    n_k = options['n_k']
    d = options['collocation']['d']
    control_intervals = n_k

    comparison_labels = options['induction']['comparison_labels']
    wake_nodes = options['induction']['vortex_wake_nodes']
    rings = wake_nodes - 1
    kite_nodes = model.architecture.kite_nodes
    wingtips = ['ext', 'int']

    Xdot = struct_op.construct_Xdot_struct(options, model.variables_dict)(0.)

    resi = []

    any_vor = any(label[:3] == 'vor' for label in comparison_labels)
    if any_vor:

        for kite in kite_nodes:
            for tip in wingtips:
                for ring in range(rings):
                    for ndx in range(n_k):
                        for ddx in range(d):

                            variables = struct_op.get_variables_at_time(options, V, Xdot, model.variables, ndx, ddx)
                            wg_local = tools.get_ring_strength(variables, kite, ring)

                            wg_ref = 3.
                            print_op.warn_about_temporary_funcationality_removal(location='strength')

                            resi_local = wg_local - wg_ref
                            resi = cas.vertcat(resi, resi_local)


    # comparison_labels = options['induction']['comparison_labels']
    # periods_tracked = options['induction']['vortex_periods_tracked']
    #
    # if periods_tracked > 2:
    #     periods_tracked = 2
    #
    # any_vor = any(label[:3] == 'vor' for label in comparison_labels)
    # if any_vor:
    #     for period in range(periods_tracked):
    #         local_resi = strength_constraints(options, V, Outputs, model, period)
    #         resi = cas.vertcat(resi, local_resi)

    return resi
#
# def strength_constraints(options, V, Outputs, model, period):
#     n_k = options['n_k']
#     d = options['collocation']['d']
#
#     resi = []
#
#     if period == 0:
#         for ndx in range(n_k):
#             for ddx in range(d):
#                 for ndx_shed in range(n_k):
#                     for ddx_shed in range(d):
#                         local_resi = strength_constraints_on_zeroth_period(options, V, Outputs, model, ndx, ddx,
#                                                               ndx_shed, ddx_shed)
#                         resi = cas.vertcat(resi, local_resi)
#
#     elif period == 1:
#         for ndx in range(n_k):
#             for ddx in range(d):
#                 for ndx_shed in range(n_k):
#                     for ddx_shed in range(d):
#                         local_resi = strength_constraints_on_previous_period(options, V, Outputs, model, ndx, ddx, ndx_shed,
#                                               ddx_shed)
#                         resi = cas.vertcat(resi, local_resi)
#
#     return resi
#
#
# def strength_constraints_on_zeroth_period(options, V, Outputs, model, ndx, ddx, ndx_shed, ddx_shed):
#
#     period = 0
#     architecture = model.architecture
#     Xdot = struct_op.construct_Xdot_struct(options, model.variables_dict)(0.)
#
#     resi = []
#
#     for kite in architecture.kite_nodes:
#         parent = architecture.parent_map[kite]
#
#         gamma_name = 'wg' + '_' + str(period) + '_' + str(kite) + str(parent)
#         variables = struct_op.get_variables_at_time(options, V, Xdot, model.variables, ndx, ddx=ddx)
#
#         if is_on_vortex(ndx, ddx, ndx_shed, ddx_shed):
#             gamma_val = Outputs['coll_outputs', ndx_shed, ddx_shed, 'aerodynamics', 'gamma' + str(kite)]
#         else:
#             gamma_val = cas.DM(0.)
#
#         local_resi = get_strength_resi(variables, gamma_name, ndx_shed, ddx_shed, options, gamma_val)
#         resi = cas.vertcat(resi, local_resi)
#
#     return resi
#
#
# def strength_constraints_on_previous_period(options, V, Outputs, model, ndx, ddx, ndx_shed,
#                                               ddx_shed):
#
#     period = 1
#     architecture = model.architecture
#     Xdot = struct_op.construct_Xdot_struct(options, model.variables_dict)(0.)
#
#     resi = []
#
#     for kite in architecture.kite_nodes:
#         parent = architecture.parent_map[kite]
#
#         gamma_name = 'wg' + '_' + str(period) + '_' + str(kite) + str(parent)
#         variables = struct_op.get_variables_at_time(options, V, Xdot, model.variables, ndx, ddx=ddx)
#
#         gamma_val = Outputs['coll_outputs', ndx_shed, ddx_shed, 'aerodynamics', 'gamma' + str(kite)]
#
#         local_resi = get_strength_resi(variables, gamma_name, ndx_shed, ddx_shed, options, gamma_val)
#         resi = cas.vertcat(resi, local_resi)
#
#     return resi
#
#
# ######### the helper functions
#
# def is_on_vortex(ndx, ddx, ndx_shed, ddx_shed):
#     if ndx > ndx_shed:
#         return True
#     elif ndx == ndx_shed and ddx > ddx_shed:
#         return True
#     elif ndx == ndx_shed and ddx == ddx_shed:
#         return True
#     else:
#         return False
#
# def get_wake_var_at_ndx_ddx(n_k, d, var, ndx, ddx):
#
#     dimensions = (n_k, d)
#     var_reshape = cas.reshape(var, dimensions)
#
#     return var_reshape[ndx, ddx]
#
# def get_strength_resi(variables, gamma_name, ndx_shed, ddx_shed, options, gamma_val):
#     n_k = options['n_k']
#     d = options['collocation']['d']
#
#     var = tools.get_strength_var_column(variables, gamma_name, options)
#
#     gamma_var = get_wake_var_at_ndx_ddx(n_k, d, var, ndx_shed, ddx_shed)
#
#     resi_unscaled = gamma_var - gamma_val
#     scale = tools.get_strength_scale(options)
#     resi = resi_unscaled / scale
#
#     return resi
#
