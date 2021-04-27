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
constraints to create "intermediate condition" fixing constraints on the positions of the wake nodes,
to be referenced/used from ocp.constraints
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2020-21
'''
import pdb

import numpy as np
import awebox.mdl.aero.induction_dir.vortex_dir.tools as tools
import awebox.tools.struct_operations as struct_op
import casadi.tools as cas
from awebox.logger.logger import Logger as awelogger
import awebox.tools.print_operations as print_op
import awebox.tools.vector_operations as vect_op
import awebox.ocp.var_struct as var_struct

import awebox.tools.constraint_operations as cstr_op

################# define the actual constraint

def get_fixing_constraint(options, V, Outputs, model, time_grids):

    comparison_labels = options['induction']['comparison_labels']
    any_vor = any(label[:3] == 'vor' for label in comparison_labels)
    if any_vor:

        cstr_list = cstr_op.ConstraintList()

        vortex_representation = options['induction']['vortex_representation']

        if vortex_representation == 'state':
            cstr_list.append(get_state_repr_fixing_constraint(options, V, Outputs, model))
        elif vortex_representation == 'alg':
            cstr_list.append(get_alg_repr_fixing_constraint(options, V, Outputs, model, time_grids))
        else:
            message = 'specified vortex representation ' + vortex_representation + ' is not allowed'
            awelogger.logger.error(message)
            raise Exception(message)

    cstr_list.append(get_farwake_convection_velocity_constraint(options, V, model))

    return cstr_list

############# state representation

def get_state_repr_fixing_constraint(options, V, Outputs, model):

    n_k = options['n_k']

    wake_nodes = options['induction']['vortex_wake_nodes']
    kite_nodes = model.architecture.kite_nodes
    wingtips = ['ext', 'int']

    cstr_list = cstr_op.ConstraintList()

    for kite in kite_nodes:
        for tip in wingtips:
            for wake_node in range(wake_nodes):
                local_name = 'wake_fixing_' + str(kite) + '_' + str(tip) + '_' + str(wake_node)

                if wake_node < n_k:

                    # working out:
                    # n_k = 3
                    # wn:0 fixed at shooting node 3, corresponds to ndx=2, ddx=-1
                    # wn:1 fixed at shooting node 2, corresponds to ndx=1, ddx=-1
                    # wn:2 fixed at shooting node 1, corresponds to ndx=0, ddx=-1
                    # wn   fixed at shooting node n_k - wn, corresponds to ndx=n_k - wn - 1, ddx=-1
                    # ... then, switch to periodic fixing

                    shooting_ndx = n_k - wake_node
                    collocation_ndx = shooting_ndx - 1

                    var_name = 'wx_' + str(kite) + '_' + str(tip) + '_' + str(wake_node)
                    wx_scaled = V['xd', shooting_ndx, var_name]
                    wx_si = struct_op.var_scaled_to_si('xd', var_name, wx_scaled, model.scaling)

                    wingtip_pos_si = Outputs['coll_outputs', collocation_ndx, -1, 'aerodynamics', 'wingtip_' + tip + str(kite)]

                    local_resi_si = wx_si - wingtip_pos_si
                    local_resi = struct_op.var_si_to_scaled('xd', var_name, local_resi_si, model.scaling)

                else:

                    # working out for n_k = 3
                    # wn:0, n_k-1=2
                    # wn:1, n_k-2=1
                    # wn:2=n_k-1, n_k-3=0
                    # ... switch to periodic fixing
                    # wn:3 at t_0 must be equal to -> wn:0 at t_final
                    # wn:4 at t_0 must be equal to -> wn:1 at t_final
                    # wn:5 at t_0 must be equal to -> wn:2 at t_final
                    # wn:6 at t_0 must be equal to -> wn:3 at t_final
                    # wn:7 at t_0 must be equal to -> wn:4 at t_final

                    var_name_local = 'wx_' + str(kite) + '_' + str(tip) + '_' + str(wake_node)
                    wx_local = V['xd', 0, var_name_local]

                    wake_node_upstream = wake_node - n_k
                    var_name_upsteam = 'wx_' + str(kite) + '_' + str(tip) + '_' + str(wake_node_upstream)
                    wx_upstream = V['xd', -1, var_name_upsteam]

                    local_resi = wx_local - wx_upstream

                local_cstr = cstr_op.Constraint(expr = local_resi,
                                                name = local_name,
                                                cstr_type='eq')
                cstr_list.append(local_cstr)

    return cstr_list


################## algebraic representation



def get_alg_repr_fixing_constraint(options, V, Outputs, model, time_grids):

    n_k = options['n_k']
    d = options['collocation']['d']

    comparison_labels = options['induction']['comparison_labels']
    wake_nodes = options['induction']['vortex_wake_nodes']
    kite_nodes = model.architecture.kite_nodes
    wingtips = ['ext', 'int']

    cstr_list = cstr_op.ConstraintList()

    for kite in kite_nodes:
        for tip in wingtips:
            for wake_node in range(wake_nodes):

                for ndx in range(n_k):

                    shooting_cstr = get_local_algebraic_repr_shooting_position_constraint(V, model, kite, tip, wake_node, ndx)
                    cstr_list.append(shooting_cstr)

                    for ddx in range(d):
                        local_cstr = get_local_algebraic_repr_collocation_position_constraint(options, V, Outputs, model, time_grids,
                                                                                              kite, tip, wake_node, ndx, ddx)
                        cstr_list.append(local_cstr)


    return cstr_list




def get_local_algebraic_repr_collocation_position_constraint(options, V, Outputs, model, time_grids, kite, tip, wake_node, ndx, ddx):

    local_name = 'wake_fixing_' + str(kite) + '_' + str(tip) + '_' + str(wake_node) + '_' + str(ndx) + ',' + str(ddx)

    var_name = 'wx_' + str(kite) + '_' + tip + '_' + str(wake_node)

    wx_local_scaled = V['coll_var', ndx, ddx, 'xl', var_name]
    wx_local = struct_op.var_scaled_to_si('xl', var_name, wx_local_scaled, model.scaling)

    wx_val = get_local_algebraic_repr_collocation_position_value(options, V, Outputs, model, time_grids, kite, tip, wake_node, ndx, ddx)

    local_resi_si = wx_local - wx_val
    local_resi = struct_op.var_si_to_scaled('xl', var_name, local_resi_si, model.scaling)

    local_cstr = cstr_op.Constraint(expr=local_resi,
                                    name=local_name,
                                    cstr_type='eq')

    return local_cstr


def get_local_algebraic_repr_shooting_position_constraint(V, model, kite, tip, wake_node, ndx):

    local_name = 'wake_fixing_' + str(kite) + '_' + str(tip) + '_' + str(wake_node) + '_' + str(ndx)

    var_name = 'wx_' + str(kite) + '_' + tip + '_' + str(wake_node)

    wx_local_scaled = V['xl', ndx, var_name]
    wx_local = struct_op.var_scaled_to_si('xl', var_name, wx_local_scaled, model.scaling)

    wx_val = get_local_algebraic_repr_shooting_position_value(V, model, kite, tip, wake_node, ndx)

    local_resi_si = wx_local - wx_val
    local_resi = struct_op.var_si_to_scaled('xl', var_name, local_resi_si, model.scaling)

    local_cstr = cstr_op.Constraint(expr=local_resi,
                                    name=local_name,
                                    cstr_type='eq')

    return local_cstr


def get_local_algebraic_repr_shooting_position_value(V, model, kite, tip, wake_node, ndx):
    var_name = 'wx_' + str(kite) + '_' + tip + '_' + str(wake_node)
    wx_scaled = V['coll_var', ndx-1, -1, 'xl', var_name]
    wx_si = struct_op.var_scaled_to_si('xl', var_name, wx_scaled, model.scaling)
    return wx_si


def get_local_algebraic_repr_collocation_position_value(options, V, Outputs, model, time_grids, kite, tip, wake_node, ndx, ddx):

    t_f = V['theta', 't_f']
    tgrid = time_grids['coll'](t_f)
    current_time = tgrid[ndx, ddx]

    n_k = options['n_k']

    # # if wake_node = 0, then shed at ndx
    # # if wake_node = 1, then shed at (ndx - 1) ---- > corresponds to (ndx - 2), ddx = -1
    # # .... if shedding_ndx is 1, then shedding_ndx -> 1
    # # ....  if shedding_ndx is 0, then shedding_ndx -> n_k
    # # ....  if shedding_ndx is -1, then shedding_ndx -> n_k - 1
    # # .... so, shedding_ndx -> np.mod(ndx - wake_node, n_k) -----> np.mod(ndx - wake_node - 1, n_k), ddx=-1
    subtracted_ndx = ndx - wake_node
    shedding_ndx = np.mod(subtracted_ndx, n_k)
    periods_passed = np.floor(subtracted_ndx / n_k)

    if wake_node == 0:
        shedding_ddx = ddx
    else:
        shedding_ddx = -1

    wingtip_pos = Outputs['coll_outputs', shedding_ndx, shedding_ddx, 'aerodynamics', 'wingtip_' + tip + str(kite)]

    u_local = model.wind.get_velocity(wingtip_pos[2])
    t_period = tgrid[-1, -1]
    shedding_time = t_period * periods_passed + tgrid[shedding_ndx, shedding_ddx]
    delta_t = current_time - shedding_time

    wx_found = wingtip_pos + delta_t * u_local

    return wx_found

################ farwake

def get_farwake_convection_velocity_constraint(options, V, model):

    n_k = options['n_k']

    kite_nodes = model.architecture.kite_nodes
    wingtips = ['ext', 'int']

    cstr_list = cstr_op.ConstraintList()

    for kite in kite_nodes:
        for tip in wingtips:
            var_name = 'wu_farwake_' + str(kite) + '_' + tip

            for ndx in range(n_k):

                local_name = 'far_wake_convection_velocity_' + str(kite) + '_' + str(tip) + '_' + str(ndx)

                wu_scaled = V['xl', ndx, var_name]
                wu_si = struct_op.var_scaled_to_si('xd', var_name, wu_scaled, model.scaling)

                velocity = get_far_wake_velocity_val(options, V, model, kite, ndx)

                local_resi_si = wu_si - velocity
                local_resi = struct_op.var_si_to_scaled('xl', var_name, local_resi_si, model.scaling)

                local_cstr = cstr_op.Constraint(expr = local_resi,
                                                name = local_name,
                                                cstr_type='eq')
                cstr_list.append(local_cstr)

                for ddx in range(options['collocation']['d']):
                    local_name = 'far_wake_convection_velocity_' + str(kite) + '_' + str(tip) + '_' + str(ndx) + ',' + str(ddx)

                    wu_scaled = V['coll_var', ndx, ddx, 'xl', var_name]
                    wu_si = struct_op.var_scaled_to_si('xl', var_name, wu_scaled, model.scaling)

                    velocity = get_far_wake_velocity_val(options, V, model, kite, ndx, ddx)

                    local_resi_si = wu_si - velocity
                    local_resi = struct_op.var_si_to_scaled('xl', var_name, local_resi_si, model.scaling)

                    local_cstr = cstr_op.Constraint(expr=local_resi,
                                                    name=local_name,
                                                    cstr_type='eq')
                    cstr_list.append(local_cstr)

    return cstr_list

def get_far_wake_velocity_val(options, V, model, kite, ndx, ddx=None):

    parent = model.architecture.parent_map[kite]

    vortex_far_wake_model = options['induction']['vortex_far_wake_model']
    vortex_representation = options['induction']['vortex_representation']

    n_k = options['n_k']

    wake_nodes = options['induction']['vortex_wake_nodes']
    wake_node = wake_nodes - 1

    if vortex_far_wake_model == 'freestream_filament':
        velocity = model.wind.get_speed_ref(from_parameters=False) * vect_op.xhat()

    elif (vortex_far_wake_model == 'pathwise_filament') and (vortex_representation == 'state'):
        shooting_ndx = n_k - wake_node
        collocation_ndx = shooting_ndx - 1
        modular_ndx = np.mod(collocation_ndx, n_k)

        q_kite_scaled = V['coll_var', modular_ndx, -1, 'xd', 'q' + str(kite) + str(parent)]
        q_kite = struct_op.var_scaled_to_si('xd', 'q' + str(kite) + str(parent), q_kite_scaled,
                                            model.scaling)
        u_infty = model.wind.get_velocity(q_kite[2])

        dq_kite_scaled = V['coll_var', modular_ndx, -1, 'xd', 'dq' + str(kite) + str(parent)]
        dq_kite = struct_op.var_scaled_to_si('xd', 'dq' + str(kite) + str(parent), dq_kite_scaled,
                                             model.scaling)

        velocity = u_infty - dq_kite

    elif (vortex_far_wake_model == 'pathwise_filament') and (vortex_representation == 'alg'):

        if ddx is None:
            ndx_collocation = ndx - 1
            ddx_collocation = -1
        else:
            ndx_collocation = ndx
            ddx_collocation = ddx

        subtracted_ndx = ndx_collocation - wake_node
        shedding_ndx = np.mod(subtracted_ndx, n_k)

        if wake_node == 0:
            shedding_ddx = ddx_collocation
        else:
            shedding_ddx = -1

        q_kite_scaled = V['coll_var', shedding_ndx, shedding_ddx, 'xd', 'q' + str(kite) + str(parent)]
        q_kite = struct_op.var_scaled_to_si('xd', 'q' + str(kite) + str(parent), q_kite_scaled,
                                            model.scaling)
        u_infty = model.wind.get_velocity(q_kite[2])

        dq_kite_scaled = V['coll_var', shedding_ndx, shedding_ddx, 'xd', 'dq' + str(kite) + str(parent)]
        dq_kite = struct_op.var_scaled_to_si('xd', 'dq' + str(kite) + str(parent), dq_kite_scaled,
                                             model.scaling)

        velocity = u_infty - dq_kite

    else:
        message = 'unknown vortex far wake model specified: ' + vortex_far_wake_model
        awelogger.logger.error(message)
        raise Exception(message)

    return velocity