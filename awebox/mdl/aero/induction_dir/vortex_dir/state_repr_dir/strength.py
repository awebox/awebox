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

# ################ actually define the constriants
#
#
# def get_strength_constraint(options, V, Outputs, model):
#
#     vortex_representation = options['induction']['vortex_representation']
#
#     if vortex_representation == 'state':
#         return get_state_repr_strength_constraint(options, V, Outputs, model)
#     elif vortex_representation == 'alg':
#         return get_algebraic_repr_strength_constraint(options, V, Outputs, model)
#     else:
#         message = 'specified vortex representation ' + vortex_representation + ' is not allowed'
#         awelogger.logger.error(message)
#         raise Exception(message)
#
#
#
#
#
# ################# state representation
#
# def get_state_repr_strength_constraint(options, V, Outputs, model):
#
#     cstr_list = cstr_op.ConstraintList()
#
#     n_k = options['n_k']
#     d = options['collocation']['d']
#
#     comparison_labels = options['induction']['comparison_labels']
#     wake_nodes = options['induction']['vortex_wake_nodes']
#     rings = wake_nodes
#     kite_nodes = model.architecture.kite_nodes
#
#     any_vor = any(label[:3] == 'vor' for label in comparison_labels)
#     if any_vor:
#
#         for kite in kite_nodes:
#             for ring in range(rings):
#
#                 for ndx in range(n_k):
#
#                     if ndx == 0:
#                         starting_cstr = get_local_state_repr_starting_strength_constraint(options, V, Outputs, model, kite, ring)
#                         cstr_list.append(starting_cstr)
#                     else:
#                         cont_cstr = get_local_state_repr_shooting_strength_constraint(V, model, kite, ring, ndx)
#                         cstr_list.append(cont_cstr)
#
#                     for ddx in range(d):
#                         local_cstr = get_local_state_repr_collocation_strength_constraint(options, V, Outputs, model, kite, ring, ndx, ddx)
#                         cstr_list.append(local_cstr)
#
#     return cstr_list
#
# def get_local_state_repr_shooting_strength_constraint(V, model, kite, ring, ndx):
#
#     local_name = 'wake_strength_' + str(kite) + '_' + str(ring) + '_' + str(ndx)
#
#     var_name = 'wg_' + str(kite) + '_' + str(ring)
#
#     wg_local_scaled = V['xl', ndx, var_name]
#     wg_local = struct_op.var_scaled_to_si('xl', var_name, wg_local_scaled, model.scaling)
#
#     var_name = 'wg_' + str(kite) + '_' + str(ring)
#     gamma_val_scaled = V['coll_var', ndx-1, -1, 'xl', var_name]
#     gamma_val = struct_op.var_si_to_scaled('xl', var_name, gamma_val_scaled, model.scaling)
#
#     local_resi_si = (wg_local - gamma_val)
#     local_resi = struct_op.var_si_to_scaled('xl', var_name, local_resi_si, model.scaling)
#
#     local_cstr = cstr_op.Constraint(expr=local_resi,
#                                     name=local_name,
#                                     cstr_type='eq')
#
#     return local_cstr
#
#
# def get_local_state_repr_collocation_strength_constraint(options, V, Outputs, model, kite, ring, ndx, ddx):
#
#     local_name = 'vortex_strength_' + str(kite) + '_' + str(ring) + '_' + str(ndx) + '_' + str(ddx)
#     var_name = 'wg_' + str(kite) + '_' + str(ring)
#
#     wg_local_scaled = V['coll_var', ndx, ddx, 'xl', var_name]
#     wg_local = struct_op.var_scaled_to_si('xl', var_name, wg_local_scaled, model.scaling)
#
#
#     gamma_val = get_local_state_repr_collocation_strength_val(options, Outputs, kite, ring, ndx, ddx)
#
#     local_resi_si = (wg_local - gamma_val)
#     local_resi = struct_op.var_si_to_scaled('xl', var_name, local_resi_si, model.scaling)
#
#     local_cstr = cstr_op.Constraint(expr=local_resi,
#                                     name=local_name,
#                                     cstr_type='eq')
#
#     return local_cstr
#
#
# def get_local_state_repr_starting_strength_constraint(options, V, Outputs, model, kite, ring):
#
#     ndx = 0
#
#     local_name = 'vortex_strength_' + str(kite) + '_' + str(ring) + '_' + str(ndx)
#     var_name = 'wg_' + str(kite) + '_' + str(ring)
#
#     wg_local_scaled = V['xl', ndx, var_name]
#     wg_local = struct_op.var_scaled_to_si('xl', var_name, wg_local_scaled, model.scaling)
#
#     gamma_val = get_local_state_repr_collocation_strength_val(options, Outputs, kite, ring)
#
#     local_resi_si = (wg_local - gamma_val)
#     local_resi = struct_op.var_si_to_scaled('xl', var_name, local_resi_si, model.scaling)
#
#     local_cstr = cstr_op.Constraint(expr=local_resi,
#                                     name=local_name,
#                                     cstr_type='eq')
#
#     return local_cstr
#
#
#
# def get_local_state_repr_collocation_strength_val(options, Outputs, kite, ring, ndx = None, ddx = None):
#
#     wake_node = ring
#     n_k = options['n_k']
#     d = options['collocation']['d']
#
#     if (ndx is None) and (ddx is None):
#         ndx = -1
#         ddx = d - 1
#
#     ndx_shed = n_k - 1 - wake_node
#     ddx_shed = d - 1
#
#     # working out:
#     # n_k = 3
#     # if ndx = 0 and ddx = 0 -> shed: wn >= n_k
#     #     wn: 0 sheds at ndx = 2, ddx = -1 : unshed,    period = 0
#     #     wn: 1 sheds at ndx = 1, ddx = -1 : unshed,    period = 0
#     #     wn: 2 sheds at ndx = 0, ddx = -1 : unshed,    period = 0
#     #     wn: 3 sheds at ndx = -1, ddx = -1 : SHED      period = 1
#     #     wn: 4 sheds at ndx = -2,                      period = 1
#     #     wn: 5 sheds at ndx = -3                       period = 1
#     #     wn: 6 sheds at ndx = -4                       period = 2
#     # if ndx = 1 and ddx = 0 -> shed: wn >= n_k - ndx
#     #     wn: 0 sheds at ndx = 2, ddx = -1 : unshed,
#     #     wn: 1 sheds at ndx = 1, ddx = -1 : unshed,
#     #     wn: 2 sheds at ndx = 0, ddx = -1 : SHED,
#     #     wn: 3 sheds at ndx = -1, ddx = -1 : SHED
#     # if ndx = 0 and ddx = -1 -> shed:
#     #     wn: 0 sheds at ndx = 2, ddx = -1 : unshed,
#     #     wn: 1 sheds at ndx = 1, ddx = -1 : unshed,
#     #     wn: 2 sheds at ndx = 0, ddx = -1 : SHED,
#     #     wn: 3 sheds at ndx = -1, ddx = -1 : SHED
#
#     already_shed = False
#     if (ndx > ndx_shed):
#         already_shed = True
#     elif ((ndx == ndx_shed) and (ddx == ddx_shed)):
#         already_shed = True
#
#     if already_shed:
#
#         # working out:
#         # n_k = 3
#         # period_0 -> wn 0, wn 1, wn 2 -> floor(ndx_shed / n_k)
#         # period_1 -> wn 3, wn 4, wn 5
#
#         period_number = int(np.floor(float(ndx_shed) / float(n_k)))
#         ndx_shed_w_periodicity = ndx_shed - period_number * n_k
#
#         gamma_val = Outputs['coll_outputs', ndx_shed_w_periodicity, ddx_shed, 'aerodynamics', 'circulation' + str(kite)]
#         gamma_on_off = 1. * gamma_val
#     else:
#         gamma_on_off = 0.
#
#     return gamma_on_off
#
#
#
#
#
#
#
#
# ############## algebraic representation
#
#
# def get_algebraic_repr_strength_constraint(options, V, Outputs, model):
#
#     n_k = options['n_k']
#     d = options['collocation']['d']
#
#     comparison_labels = options['induction']['comparison_labels']
#     wake_nodes = options['induction']['vortex_wake_nodes']
#     kite_nodes = model.architecture.kite_nodes
#     rings = wake_nodes
#
#     cstr_list = cstr_op.ConstraintList()
#
#     any_vor = any(label[:3] == 'vor' for label in comparison_labels)
#     if any_vor:
#         for kite in kite_nodes:
#             for ring in range(rings):
#
#                 for ndx in range(n_k):
#
#                     shooting_cstr = get_local_algebraic_repr_strength_constraint(options, V, Outputs, model, kite, ring, ndx)
#                     cstr_list.append(shooting_cstr)
#
#                     for ddx in range(d):
#                         coll_cstr = get_local_algebraic_repr_strength_constraint(options, V, Outputs, model, kite, ring, ndx, ddx)
#                         cstr_list.append(coll_cstr)
#
#     return cstr_list
#
#
# def get_local_algebraic_repr_strength_constraint(options, V, Outputs, model, kite, ring, ndx, ddx=None):
#
#     local_name = 'wake_strength_' + str(kite) + '_' + str(ring) + '_' + str(ndx)
#     var_name = 'wg_' + str(kite) + '_' + str(ring)
#
#     if ddx is None:
#         wg_local_scaled = V['xl', ndx, var_name]
#         gamma_val = get_local_algebraic_repr_shooting_strength_val(V, model, kite, ring, ndx)
#
#     else:
#         local_name += ',' + str(ddx)
#         wg_local_scaled = V['coll_var', ndx, ddx, 'xl', var_name]
#         gamma_val = get_local_algebraic_repr_collocation_strength_val(options, Outputs, kite, ring, ndx, ddx)
#
#     wg_local = struct_op.var_scaled_to_si('xl', var_name, wg_local_scaled, model.scaling)
#
#     local_resi_si = (wg_local - gamma_val)
#     local_resi = struct_op.var_si_to_scaled('xl', var_name, local_resi_si, model.scaling)
#
#     local_cstr = cstr_op.Constraint(expr=local_resi,
#                                     name=local_name,
#                                     cstr_type='eq')
#
#     return local_cstr
#
#
# def get_local_algebraic_repr_shooting_strength_val(V, model, kite, ring, ndx):
#
#     var_name = 'wg_' + str(kite) + '_' + str(ring)
#     gamma_val_scaled = V['coll_var', ndx-1, -1, 'xl', var_name]
#     gamma_val = struct_op.var_scaled_to_si('xl', var_name, gamma_val_scaled, model.scaling)
#
#     return gamma_val
#
#
# def get_local_algebraic_repr_collocation_strength_val(options, Outputs, kite, ring, ndx, ddx):
#
#     n_k = options['n_k']
#
#     subtracted_ndx = ndx - ring
#     shedding_ndx = np.mod(subtracted_ndx, n_k)
#     if ring == 0:
#         shedding_ddx = ddx
#     else:
#         shedding_ddx = -1
#
#     gamma_val = Outputs['coll_outputs', shedding_ndx, shedding_ddx, 'aerodynamics', 'circulation' + str(kite)]
#
#     return gamma_val
