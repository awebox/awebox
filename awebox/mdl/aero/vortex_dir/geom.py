#
#    This file is part of awebox.
#
#    awebox -- A modeling and optimization framework for multi-kite AWE systems.
#    Copyright (C) 2017-2019 Jochem De Schutter, Rachel Leuthold, Moritz Diehl,
#                            ALU Freiburg.
#    Copyright (C) 2018-2019 Thilo Bronnenmeyer, Kiteswarms Ltd.
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
geometry functions for vortex model
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2019
'''

import casadi as cas
import numpy as np
from awebox.logger.logger import Logger as awelogger
import awebox.tools.vector_operations as vect_op
import pdb

def reshape_wake_var(options, var):
    n_k = options['aero']['vortex']['n_k']
    d = options['aero']['vortex']['d']
    dimensions = (n_k, d)
    var_reshape = cas.reshape(var, dimensions)
    return var_reshape

def get_vector_var(options, variables, xd_xddot, ext_int, kite, ndx, ddx, architecture):
    parent = architecture.parent_map[kite]

    sym = 'w'
    if xd_xddot == 'xddot':
        sym = 'd' + sym

    vect = []
    dims = ['x', 'y', 'z']
    for dim in dims:
        comp_all = variables[xd_xddot][sym + dim + '_' + ext_int + str(kite) + str(parent)]
        comp_reshape = reshape_wake_var(options, comp_all)
        comp = comp_reshape[ndx, ddx]
        vect = cas.vertcat(vect, comp)

    return vect

def get_pos_wake_var(options, variables, ext_int, kite, ndx, ddx, architecture):
    pos = get_vector_var(options, variables, 'xd', ext_int, kite, ndx, ddx, architecture)
    return pos

def get_vel_wake_var(options, variables, ext_int, kite, ndx, ddx, architecture):
    vel = get_vector_var(options, variables, 'xddot', ext_int, kite, ndx, ddx, architecture)
    return vel

def get_convection_residual(options, variables, architecture):
    n_k = options['aero']['vortex']['n_k']
    d = options['aero']['vortex']['d']
    kite_nodes = architecture.kite_nodes
    ext_int_combi = ['ext']

    resi = []
    for kite in kite_nodes:
        for ndx in range(n_k):
            for ddx in range(d):
                for ext_int in ext_int_combi:
                    dx = get_vel_wake_var(options, variables, ext_int, kite, ndx, ddx, architecture)
                    local = dx - vect_op.xhat()
                    resi = cas.vertcat(resi, local)

    return resi
