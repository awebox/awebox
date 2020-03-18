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

def get_wake_var_at_ndx_ddx(n_k, d, var, ndx, ddx):

    dimensions = (n_k, d)
    var_reshape = cas.reshape(var, dimensions)

    return var_reshape[ndx, ddx]


def fix_vortex_strengths(options, g_list, g_bounds, V, Outputs, model, period):
    n_k = options['n_k']
    d = options['collocation']['d']

    architecture = model.architecture

    if period == 0:
        for kite in architecture.kite_nodes:
            for ndx in range(n_k):
                for ddx in range(d):
                    for ndx_shed in range(n_k):
                        for ddx_shed in range(d):

                            if is_bound_vortex(ndx, ddx, ndx_shed, ddx_shed):
                                resi = get_residual_for_bound_vortex(V, Outputs, model, n_k, d, kite, ndx, ddx)

                            elif is_on_vortex(ndx, ddx, ndx_shed, ddx_shed):
                                resi = get_residual_for_on_vortex(V, Outputs, model, n_k, d, kite, ndx, ddx, ndx_shed, ddx_shed)

                            else:
                                resi = get_residual_for_off_vortex(V, Outputs, model, n_k, d, kite, ndx, ddx, ndx_shed, ddx_shed)

                            g_list.append(resi)
                            g_bounds['ub'].append(np.zeros(resi.shape))
                            g_bounds['lb'].append(np.zeros(resi.shape))

    else:
        g_list, g_bounds = fix_strengths_of_periodic_vortices(options, g_list, g_bounds, V, Outputs, model, period)

    return g_list, g_bounds


def is_bound_vortex(ndx, ddx, ndx_shed, ddx_shed):
    if ndx == ndx_shed and ddx == ddx_shed:
        return True
    else:
        return False

def is_on_vortex(ndx, ddx, ndx_shed, ddx_shed):
    if ndx > ndx_shed:
        return True
    elif ndx == ndx_shed and ddx > ddx_shed:
        return True
    else:
        return False



def get_residual_for_bound_vortex(V, Outputs, model, n_k, d, kite, ndx, ddx):
    period = 0

    architecture = model.architecture
    parent = architecture.parent_map[kite]

    gamma_name = 'wg' + '_' + str(period) + '_' + str(kite) + str(parent)
    var = V['coll_var', ndx, ddx, 'xl', gamma_name]
    gamma_var = get_wake_var_at_ndx_ddx(n_k, d, var, ndx, ddx)

    gamma_val = Outputs['coll_outputs', ndx, ddx, 'aerodynamics', 'gamma' + str(kite)]

    resi = gamma_var - gamma_val
    return resi

def get_residual_for_on_vortex(V, Outputs, model, n_k, d, kite, ndx, ddx, ndx_shed, ddx_shed):
    period = 0

    architecture = model.architecture
    parent = architecture.parent_map[kite]

    gamma_name = 'wg' + '_' + str(period) + '_' + str(kite) + str(parent)
    var = V['coll_var', ndx, ddx, 'xl', gamma_name]
    gamma_var = get_wake_var_at_ndx_ddx(n_k, d, var, ndx_shed, ddx_shed)

    gamma_val = Outputs['coll_outputs', ndx_shed, ddx_shed, 'aerodynamics', 'gamma' + str(kite)]

    resi = gamma_var - gamma_val
    return resi

def get_residual_for_off_vortex(V, Outputs, model, n_k, d, kite, ndx, ddx, ndx_shed, ddx_shed):
    period = 0

    architecture = model.architecture
    parent = architecture.parent_map[kite]

    gamma_name = 'wg' + '_' + str(period) + '_' + str(kite) + str(parent)
    var = V['coll_var', ndx, ddx, 'xl', gamma_name]
    gamma_var = get_wake_var_at_ndx_ddx(n_k, d, var, ndx_shed, ddx_shed)

    gamma_val = 0.

    resi = gamma_var - gamma_val
    return resi





def fix_strengths_of_periodic_vortices(options, g_list, g_bounds, V, Outputs, model, period):
    n_k = options['n_k']
    d = options['collocation']['d']

    architecture = model.architecture

    for kite in architecture.kite_nodes:
        for ndx in range(n_k):
            for ddx in range(d):
                for ndx_shed in range(n_k):
                    for ddx_shed in range(d):
                        resi = get_residual_for_periodic_vortex(V, Outputs, model, n_k, d, kite, ndx, ddx, ndx_shed, ddx_shed, period)

                        g_list.append(resi)
                        g_bounds['ub'].append(np.zeros(resi.shape))
                        g_bounds['lb'].append(np.zeros(resi.shape))

    return g_list, g_bounds


def get_residual_for_periodic_vortex(V, Outputs, model, n_k, d, kite, ndx, ddx, ndx_shed, ddx_shed, period):
    architecture = model.architecture
    parent = architecture.parent_map[kite]

    gamma_name = 'wg' + '_' + str(period) + '_' + str(kite) + str(parent)
    var = V['coll_var', ndx, ddx, 'xl', gamma_name]
    gamma_var = get_wake_var_at_ndx_ddx(n_k, d, var, ndx_shed, ddx_shed)

    gamma_val = Outputs['coll_outputs', ndx_shed, ddx_shed, 'aerodynamics', 'gamma' + str(kite)]

    resi = gamma_var - gamma_val
    return resi