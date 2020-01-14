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
from awebox.logger.logger import Logger as awelogger
import awebox.tools.vector_operations as vect_op
import pdb
import awebox.mdl.aero.vortex_dir.geom as geom

def fix_strengths_of_on_vortices(options, g_list, g_bounds, V, Outputs, model, period):

    n_k = options['n_k']
    d = options['collocation']['d']
    n_nodes = n_k * d

    architecture = model.architecture

    for kite in architecture.kite_nodes:
        parent = architecture.parent_map[kite]

        gamma_name = 'wg' + '_' + str(period) + '_' + str(kite) + str(parent)

        for ndx in range(n_k):
            for ddx in range(d):
                var = V['coll_var', ndx, ddx, 'xl', gamma_name]

                for ndx_shed in range(n_k):
                    for ddx_shed in range(d):

                        gamma_var = geom.get_wake_var_at_ndx_ddx(n_k, d, var, False, ndx_shed, ddx_shed)

                        gamma_cross_val = Outputs['coll_outputs', ndx_shed, ddx_shed, 'aerodynamics', 'gamma_cross' + str(kite)]
                        gamma_cl_val = Outputs['coll_outputs', ndx, ddx, 'aerodynamics', 'gamma_cl' + str(kite)]

                        pdb.set_trace()

                        fix = gamma_var - 2. * np.ones((n_nodes, 1))
                        g_list.append(fix)
                        g_bounds['ub'].append(np.zeros(fix.shape))
                        g_bounds['lb'].append(np.zeros(fix.shape))

    return g_list, g_bounds

def vortex_is_on(ndx, ddx, ndx_shed, ddx_shed):
    return False

def get_residual_for_bound_vortex(V, Outputs, model, n_k, d, kite, ndx, ddx):
    period = 0

    architecture = model.architecture
    parent = architecture.parent_map[kite]

    gamma_name = 'wg' + '_' + str(period) + '_' + str(kite) + str(parent)
    var = V['coll_var', ndx, ddx, 'xl', gamma_name]
    gamma_var = geom.get_wake_var_at_ndx_ddx(n_k, d, var, False, ndx, ddx)

    gamma_cross_val = Outputs['coll_outputs', ndx, ddx, 'aerodynamics', 'gamma_cross' + str(kite)]
    gamma_cl_val = Outputs['coll_outputs', ndx, ddx, 'aerodynamics', 'gamma_cl' + str(kite)]

    resi = gamma_var - gamma_cl_val
    return resi