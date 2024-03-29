#
#    This file is part of awebox.
#
#    awebox -- A modeling and optimization framework for multi-kite AWE systems.
#    Copyright (C) 2017-2021 Jochem De Schutter, Rachel Leuthold, Moritz Diehl,
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
- author: rachel leuthold, alu-fr 2020-21
'''

import casadi.tools as cas
import numpy as np
import awebox.tools.struct_operations as struct_op
import awebox.mdl.aero.induction_dir.vortex_dir.tools as tools
from awebox.logger.logger import Logger as awelogger
import awebox.tools.print_operations as print_op
import awebox.ocp.collocation as collocation
import awebox.ocp.var_struct as var_struct
import awebox.tools.constraint_operations as cstr_op

################ actually define the constriants


def get_strength_constraint(options, V, Outputs, model):

    vortex_representation = options['induction']['vortex_representation']

    if vortex_representation == 'state':
        return get_state_repr_strength_constraint(options, V, Outputs, model)
    elif vortex_representation == 'alg':
        return get_alg_repr_strength_constraint(options, V, Outputs, model)
    else:
        message = 'specified vortex representation ' + vortex_representation + ' is not allowed'
        awelogger.logger.error(message)
        raise Exception(message)

def get_state_repr_strength_constraint(options, V, Outputs, model):

    cstr_list = cstr_op.ConstraintList()

    n_k = options['n_k']
    d = options['collocation']['d']

    comparison_labels = options['induction']['comparison_labels']
    wake_nodes = options['induction']['vortex_wake_nodes']
    rings = wake_nodes - 1
    kite_nodes = model.architecture.kite_nodes

    any_vor = any(label[:3] == 'vor' for label in comparison_labels)
    if any_vor:

        for kite in kite_nodes:
            for ring in range(rings):

                for ndx in range(n_k):
                    if ndx > 0:
                        cont_cstr = get_continuity_strength_constraint(V, kite, ring, ndx)
                        cstr_list.append(cont_cstr)
                    else:
                        starting_cstr = get_state_starting_strength_constraint(options, V, Outputs, model, kite, ring)
                        cstr_list.append(starting_cstr)

                    for ddx in range(d):
                        local_cstr = get_local_state_repr_strength_constraint(options, V, Outputs, model, kite, ring, ndx, ddx)
                        cstr_list.append(local_cstr)

    return cstr_list

def get_local_state_repr_strength_constraint(options, V, Outputs, model, kite, ring, ndx, ddx):

    wake_node = ring
    n_k = options['n_k']
    d = options['collocation']['d']

    local_name = 'vortex_strength_' + str(kite) + '_' + str(ring) + '_' + str(ndx) + '_' + str(ddx)

    Xdot = struct_op.construct_Xdot_struct(options, model.variables_dict)(0.)
    variables_scaled = struct_op.get_variables_at_time(options, V, Xdot, model.variables, ndx, ddx)
    wg_local = tools.get_ring_strength_si(variables_scaled, kite, ring, model.scaling)

    ndx_shed = n_k - 1 - wake_node
    ddx_shed = d - 1

    # working out:
    # n_k = 3
    # if ndx = 0 and ddx = 0 -> shed: wn >= n_k
    #     wn: 0 sheds at ndx = 2, ddx = -1 : unshed,    period = 0
    #     wn: 1 sheds at ndx = 1, ddx = -1 : unshed,    period = 0
    #     wn: 2 sheds at ndx = 0, ddx = -1 : unshed,    period = 0
    #     wn: 3 sheds at ndx = -1, ddx = -1 : SHED      period = 1
    #     wn: 4 sheds at ndx = -2,                      period = 1
    #     wn: 5 sheds at ndx = -3                       period = 1
    #     wn: 6 sheds at ndx = -4                       period = 2
    # if ndx = 1 and ddx = 0 -> shed: wn >= n_k - ndx
    #     wn: 0 sheds at ndx = 2, ddx = -1 : unshed,
    #     wn: 1 sheds at ndx = 1, ddx = -1 : unshed,
    #     wn: 2 sheds at ndx = 0, ddx = -1 : SHED,
    #     wn: 3 sheds at ndx = -1, ddx = -1 : SHED
    # if ndx = 0 and ddx = -1 -> shed:
    #     wn: 0 sheds at ndx = 2, ddx = -1 : unshed,
    #     wn: 1 sheds at ndx = 1, ddx = -1 : unshed,
    #     wn: 2 sheds at ndx = 0, ddx = -1 : SHED,
    #     wn: 3 sheds at ndx = -1, ddx = -1 : SHED

    already_shed = False
    if (ndx > ndx_shed):
        already_shed = True
    elif ((ndx == ndx_shed) and (ddx == ddx_shed)):
        already_shed = True

    if already_shed:

        # working out:
        # n_k = 3
        # period_0 -> wn 0, wn 1, wn 2 -> floor(ndx_shed / n_k)
        # period_1 -> wn 3, wn 4, wn 5

        period_number = int(np.floor(float(ndx_shed) / float(n_k)))
        ndx_shed_w_periodicity = ndx_shed - period_number * n_k

        gamma_val = Outputs['coll_outputs', ndx_shed_w_periodicity, ddx_shed, 'aerodynamics', 'circulation' + str(kite)]
        wg_ref = 1. * gamma_val
    else:
        wg_ref = 0.

    local_resi = (wg_local - wg_ref) / tools.get_strength_scale(model.variables_dict, model.scaling)

    local_cstr = cstr_op.Constraint(expr=local_resi,
                                    name=local_name,
                                    cstr_type='eq')

    return  local_cstr


def get_state_starting_strength_constraint(options, V, Outputs, model, kite, ring):

    ndx = 0

    wake_node = ring
    n_k = options['n_k']
    d = options['collocation']['d']

    ndx_equiv = -1
    ddx_equiv = d - 1

    local_name = 'vortex_strength_' + str(kite) + '_' + str(ring) + '_' + str(ndx)

    Xdot = struct_op.construct_Xdot_struct(options, model.variables_dict)(0.)
    variables_scaled = struct_op.get_variables_at_time(options, V, Xdot, model.variables, ndx)
    wg_local = tools.get_ring_strength_si(variables_scaled, kite, ring, model.scaling)

    ndx_shed = n_k - 1 - wake_node
    ddx_shed = d - 1

    # working out:
    # n_k = 3
    # if ndx = -1 and ddx = -1 -> shed: wn >= n_k
    #     wn: 0 sheds at ndx = 2, ddx = -1 : unshed,    period = 0
    #     wn: 1 sheds at ndx = 1, ddx = -1 : unshed,    period = 0
    #     wn: 2 sheds at ndx = 0, ddx = -1 : unshed,    period = 0
    #     wn: 3 sheds at ndx = -1, ddx = -1 : SHED      period = 1
    #     wn: 4 sheds at ndx = -2,                      period = 1
    #     wn: 5 sheds at ndx = -3                       period = 1
    #     wn: 6 sheds at ndx = -4                       period = 2

    already_shed = False
    if (ndx_equiv > ndx_shed):
        already_shed = True
    elif ((ndx_equiv == ndx_shed) and (ddx_equiv == ddx_shed)):
        already_shed = True

    if already_shed:

        # working out:
        # n_k = 3
        # period_0 -> wn 0, wn 1, wn 2 -> floor(ndx_shed / n_k)
        # period_1 -> wn 3, wn 4, wn 5

        period_number = int(np.floor(float(ndx_shed) / float(n_k)))
        ndx_shed_w_periodicity = ndx_shed - period_number * n_k

        gamma_val = Outputs['coll_outputs', ndx_shed_w_periodicity, ddx_shed, 'aerodynamics', 'circulation' + str(kite)]
        wg_ref = 1. * gamma_val
    else:
        wg_ref = 0.

    local_resi = (wg_local - wg_ref) / tools.get_strength_scale(model.variables_dict, model.scaling)

    local_cstr = cstr_op.Constraint(expr=local_resi,
                                    name=local_name,
                                    cstr_type='eq')

    return local_cstr


def get_alg_repr_strength_constraint(options, V, Outputs, model):

    n_k = options['n_k']
    d = options['collocation']['d']

    comparison_labels = options['induction']['comparison_labels']
    wake_nodes = options['induction']['vortex_wake_nodes']
    kite_nodes = model.architecture.kite_nodes
    rings = wake_nodes - 1

    cstr_list = cstr_op.ConstraintList()

    any_vor = any(label[:3] == 'vor' for label in comparison_labels)
    if any_vor:
        for kite in kite_nodes:
            for ring in range(rings):

                for ndx in range(n_k):

                    if ndx > 0:
                        cont_cstr = get_continuity_strength_constraint(V, kite, ring, ndx)
                        cstr_list.append(cont_cstr)
                    else:
                        period_cstr = get_alg_periodic_strength_constraint(V, kite, ring)
                        cstr_list.append(period_cstr)

                    for ddx in range(d):
                        local_cstr = get_local_alg_repr_strength_constraint(options, V, Outputs, model, kite, ring, ndx, ddx)
                        cstr_list.append(local_cstr)

    return cstr_list

def get_local_alg_repr_strength_constraint(options, V, Outputs, model, kite, ring, ndx, ddx):

    n_k = options['n_k']

    local_name = 'wake_strength_' + str(kite) + '_' + str(ring) + '_' + str(ndx) + ',' + str(ddx)

    var_name = 'wg_' + str(kite) + '_' + str(ring)
    wg_local_scaled = V['coll_var', ndx, ddx, 'z', var_name]
    wg_local = struct_op.var_scaled_to_si('z', var_name, wg_local_scaled, model.scaling)

    subtracted_ndx = ndx - ring
    shedding_ndx = np.mod(subtracted_ndx, n_k)
    if ring == 0:
        shedding_ddx = ddx
    else:
        shedding_ddx = -1

    gamma_val = Outputs['coll_outputs', shedding_ndx, shedding_ddx, 'aerodynamics', 'circulation' + str(kite)]

    local_resi_si = (wg_local - gamma_val)
    local_resi = struct_op.var_si_to_scaled('z', var_name, local_resi_si, model.scaling)

    local_cstr = cstr_op.Constraint(expr=local_resi,
                                    name=local_name,
                                    cstr_type='eq')

    return local_cstr


def get_continuity_strength_constraint(V, kite, ring, ndx):

    local_name = 'continuity_wake_strength_' + str(kite) + '_' + str(ring) + '_' + str(ndx)

    var_name = 'wg_' + str(kite) + '_' + str(ring)
    wx_coll = V['coll_var', ndx-1, -1, 'z', var_name]
    wx_upper = V['z', ndx, var_name]

    local_resi = wx_coll - wx_upper

    local_cstr = cstr_op.Constraint(expr=local_resi,
                                    name=local_name,
                                    cstr_type='eq')

    return local_cstr

def get_alg_periodic_strength_constraint(V, kite, ring):

    local_name = 'periodic_wake_fixing_' + str(kite) + '_' + str(ring)

    var_name = 'wg_' + str(kite) + '_' + str(ring)
    wx_coll = V['coll_var', -1, -1, 'z', var_name]
    wx_upper = V['z', 0, var_name]

    local_resi = wx_coll - wx_upper

    local_cstr = cstr_op.Constraint(expr=local_resi,
                                    name=local_name,
                                    cstr_type='eq')

    return local_cstr
