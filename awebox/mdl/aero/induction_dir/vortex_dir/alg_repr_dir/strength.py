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


def get_constraint(options, V, Outputs, model):

    n_k = options['n_k']
    d = options['collocation']['d']

    comparison_labels = options['induction']['comparison_labels']
    wake_nodes = options['induction']['vortex_wake_nodes']
    kite_nodes = model.architecture.kite_nodes
    rings = wake_nodes

    cstr_list = cstr_op.ConstraintList()

    any_vor = any(label[:3] == 'vor' for label in comparison_labels)
    if any_vor:
        for kite in kite_nodes:
            for ring in range(rings):

                for ndx in range(n_k):

                    shooting_cstr = get_local_algebraic_repr_strength_constraint(options, V, Outputs, model, kite, ring, ndx)
                    cstr_list.append(shooting_cstr)

                    for ddx in range(d):
                        coll_cstr = get_local_algebraic_repr_strength_constraint(options, V, Outputs, model, kite, ring, ndx, ddx)
                        cstr_list.append(coll_cstr)

    return cstr_list


def get_local_algebraic_repr_strength_constraint(options, V, Outputs, model, kite, ring, ndx, ddx=None):

    local_name = 'wake_strength_' + str(kite) + '_' + str(ring) + '_' + str(ndx)
    var_name = 'wg_' + str(kite) + '_' + str(ring)

    if ddx is None:
        wg_local_scaled = V['xl', ndx, var_name]
        gamma_val = get_local_algebraic_repr_shooting_strength_val(V, model, kite, ring, ndx)

    else:
        local_name += ',' + str(ddx)
        wg_local_scaled = V['coll_var', ndx, ddx, 'xl', var_name]
        gamma_val = get_local_algebraic_repr_collocation_strength_val(options, Outputs, kite, ring, ndx, ddx)

    wg_local = struct_op.var_scaled_to_si('xl', var_name, wg_local_scaled, model.scaling)

    local_resi_si = (wg_local - gamma_val)
    local_resi = struct_op.var_si_to_scaled('xl', var_name, local_resi_si, model.scaling)

    local_cstr = cstr_op.Constraint(expr=local_resi,
                                    name=local_name,
                                    cstr_type='eq')

    return local_cstr


def get_local_algebraic_repr_shooting_strength_val(V, model, kite, ring, ndx):

    var_name = 'wg_' + str(kite) + '_' + str(ring)
    gamma_val_scaled = V['coll_var', ndx-1, -1, 'xl', var_name]
    gamma_val = struct_op.var_scaled_to_si('xl', var_name, gamma_val_scaled, model.scaling)

    return gamma_val


def get_local_algebraic_repr_collocation_strength_val(options, Outputs, kite, ring, ndx, ddx):

    n_k = options['n_k']

    subtracted_ndx = ndx - ring
    shedding_ndx = np.mod(subtracted_ndx, n_k)
    if ring == 0:
        shedding_ddx = ddx
    else:
        shedding_ddx = -1

    gamma_val = Outputs['coll_outputs', shedding_ndx, shedding_ddx, 'aerodynamics', 'circulation' + str(kite)]

    return gamma_val
