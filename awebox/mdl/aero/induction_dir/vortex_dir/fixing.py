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
import awebox.tools.vector_operations as vect_op
import awebox.ocp.var_struct as var_struct

import awebox.tools.constraint_operations as cstr_op

################# define the actual constraint

def get_fixing_constraint(options, V, Outputs, model, time_grids):

    vortex_representation = options['induction']['vortex_representation']

    if vortex_representation == 'state':
        return get_state_repr_fixing_constraint(options, V, Outputs, model)
    elif vortex_representation == 'alg':
        return get_alg_repr_fixing_constraint(options, V, Outputs, model, time_grids)
    else:
        message = 'specified vortex representation ' + vortex_representation + ' is not allowed'
        awelogger.logger.error(message)
        raise Exception(message)

def get_state_repr_fixing_constraint(options, V, Outputs, model):

    n_k = options['n_k']

    comparison_labels = options['induction']['comparison_labels']
    wake_nodes = options['induction']['vortex_wake_nodes']
    kite_nodes = model.architecture.kite_nodes
    wingtips = ['ext', 'int']

    Xdot = struct_op.construct_Xdot_struct(options, model.variables_dict)(0.)

    cstr_list = cstr_op.ConstraintList()

    any_vor = any(label[:3] == 'vor' for label in comparison_labels)
    if any_vor:

        for kite in kite_nodes:
            for tip in wingtips:
                for wake_node in range(wake_nodes):
                    local_name = 'wake_fixing_' + str(kite) + '_' + str(tip) + '_' + str(wake_node)

                    if wake_node < n_k:

                        # working out:
                        # n_k = 3
                        # wn:0, n_k-1=2
                        # wn:1, n_k-2=1
                        # wn:2=n_k-1, n_k-3=0
                        # ... switch to periodic fixing

                        reverse_index = n_k - 1 - wake_node
                        variables_at_shed = struct_op.get_variables_at_time(options, V, Xdot, model.variables,
                                                                            reverse_index, -1)

                        wx_local = tools.get_wake_node_position_si(options, variables_at_shed, kite, tip, wake_node, model.scaling)
                        wingtip_pos = Outputs[
                            'coll_outputs', reverse_index, -1, 'aerodynamics', 'wingtip_' + tip + str(kite)]

                        local_resi = wx_local - wingtip_pos

                        local_cstr = cstr_op.Constraint(expr = local_resi,
                                                        name = local_name,
                                                        cstr_type='eq')
                        cstr_list.append(local_cstr)

                    else:

                        # working out:
                        # n_k = 3
                        # wn:0, n_k-1=2
                        # wn:1, n_k-2=1
                        # wn:2=n_k-1, n_k-3=0
                        # ... switch to periodic fixing
                        # wn:3 at ndx = 0 must be equal to -> wn:0 at ndx = -1, ddx = -1
                        # wn:4 at ndx = 0 must be equal to -> wn:1 at ndx = -1, ddx = -1

                        variables_at_initial = struct_op.get_variables_at_time(options, V, Xdot, model.variables, 0)
                        variables_at_final = struct_op.get_variables_at_time(options, V, Xdot, model.variables, -1, -1)

                        upstream_node = wake_node - n_k
                        wx_local = tools.get_wake_node_position_si(options, variables_at_initial, kite, tip, wake_node, model.scaling)
                        wx_upstream = tools.get_wake_node_position_si(options, variables_at_final, kite, tip, upstream_node, model.scaling)

                        local_resi = wx_local - wx_upstream
                        local_cstr = cstr_op.Constraint(expr = local_resi,
                                                        name = local_name,
                                                        cstr_type='eq')
                        cstr_list.append(local_cstr)

    return cstr_list


def get_alg_repr_fixing_constraint(options, V, Outputs, model, time_grids):

    n_k = options['n_k']
    d = options['collocation']['d']

    t_f = V['theta', 't_f']

    comparison_labels = options['induction']['comparison_labels']
    wake_nodes = options['induction']['vortex_wake_nodes']
    kite_nodes = model.architecture.kite_nodes
    wingtips = ['ext', 'int']

    Xdot = struct_op.construct_Xdot_struct(options, model.variables_dict)(0.)

    cstr_list = cstr_op.ConstraintList()

    any_vor = any(label[:3] == 'vor' for label in comparison_labels)
    if any_vor:

        tgrid = time_grids['coll'](t_f)

        for ndx in range(n_k):
            for ddx in range(d):
                current_time = tgrid[ndx, ddx]

                for kite in kite_nodes:
                    for tip in wingtips:
                        for wake_node in range(wake_nodes):

                            local_name = 'wake_fixing_' + str(kite) + '_' + str(tip) + '_' + str(wake_node) + '_' + str(ndx) + ',' + str(ddx)

                            local_variables = struct_op.get_variables_at_time(options, V, Xdot, model.variables, ndx, ddx)
                            wx_local = tools.get_wake_node_position_si(options, local_variables, kite, tip, wake_node, model.scaling)

                            # if wake_node = 0, then shed at ndx
                            # if wake_node = 1, then shed at (ndx - 1) ---- > corresponds to (ndx - 2), ddx = -1
                            # .... if shedding_ndx is 1, then shedding_ndx -> 1
                            # ....  if shedding_ndx is 0, then shedding_ndx -> n_k
                            # ....  if shedding_ndx is -1, then shedding_ndx -> n_k - 1
                            # .... so, shedding_ndx -> np.mod(ndx - wake_node, n_k) -----> np.mod(ndx - wake_node - 1, n_k), ddx=-1
                            subtracted_ndx = ndx - wake_node
                            shedding_ndx = np.mod(subtracted_ndx, n_k)
                            periods_passed = np.floor(subtracted_ndx / n_k)


                            if wake_node == 0:
                                shedding_ddx = ddx
                            else:
                                shedding_ddx = -1

                            wingtip_pos = Outputs['coll_outputs', shedding_ndx, shedding_ddx, 'aerodynamics', 'wingtip_' + tip + str(kite)]

                            u_local = model.wind.get_velocity(wingtip_pos[2])
                            shedding_time = t_f[1] * periods_passed + tgrid[shedding_ndx, shedding_ddx]
                            delta_t = current_time - shedding_time

                            wx_found = wingtip_pos + delta_t * u_local

                            local_resi = wx_local - wx_found

                            local_cstr = cstr_op.Constraint(expr = local_resi,
                                                            name = local_name,
                                                            cstr_type='eq')
                            cstr_list.append(local_cstr)


    return cstr_list