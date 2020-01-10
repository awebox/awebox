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
geometry functions for vortex model
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2019-2020
'''

import casadi as cas
import numpy as np
from awebox.logger.logger import Logger as awelogger
import awebox.tools.vector_operations as vect_op
import awebox.mdl.wind as wind
import pdb

# name = 'w' + dim + '_' + tip + '_' + str(period) + '_' + str(kite) + str(parent)

def get_wake_var_at_ndx_ddx(n_k, d, var, start=bool(False), ndx=0, ddx=0):
    if start:
        return var[0]
    else:
        var_regular = var[1:]

        dimensions = (n_k, d)
        var_reshape = cas.reshape(var_regular, dimensions)

        return var_reshape[ndx, ddx]

def get_vector_var(options, variables, pos_vel, tip, period, kite, architecture, start=bool(False), ndx=0, ddx=0):
    n_k = options['aero']['vortex']['n_k']
    d = options['aero']['vortex']['d']
    parent = architecture.parent_map[kite]

    loc = 'xd'
    dims = ['x', 'y', 'z']

    if pos_vel == 'pos':
        sym = 'w'
    elif pos_vel == 'vel':
        sym = 'dw'
    else:
        pdb.set_trace()

    vect = []
    for dim in dims:
        name = sym + dim + '_' + tip + '_' + str(period) + '_' + str(kite) + str(parent)

        try:
            comp_all = variables[loc][name]
        except:
            pdb.set_trace()

        comp = get_wake_var_at_ndx_ddx(n_k, d, comp_all, start=start, ndx=ndx, ddx=ddx)
        vect = cas.vertcat(vect, comp)

    return vect

def get_pos_wake_var(options, variables, tip, period, kite, architecture, start=bool(False), ndx=0, ddx=0):
    pos = get_vector_var(options, variables, 'pos', tip, period, kite, architecture, start=start, ndx=ndx, ddx=ddx)
    return pos

def get_vel_wake_var(options, variables, tip, period, kite, architecture, start=bool(False), ndx=0, ddx=0):
    vel = get_vector_var(options, variables, 'vel', tip, period, kite, architecture, start=start, ndx=ndx, ddx=ddx)
    return vel

def get_convection_residual(options, wind, variables, architecture):
    n_k = options['aero']['vortex']['n_k']
    d = options['aero']['vortex']['d']
    kite_nodes = architecture.kite_nodes
    wingtips = ['ext', 'int']
    periods_tracked = options['aero']['vortex']['periods_tracked']

    resi = []
    for kite in kite_nodes:
        for tip in wingtips:
            for period in range(periods_tracked):

                vel_resi = get_convection_residual_local(options, variables, tip, period, kite, architecture, wind, start=True)
                resi = cas.vertcat(resi, vel_resi)

                for ndx in range(n_k):
                    for ddx in range(d):
                        vel_resi = get_convection_residual_local(options, variables, tip, period, kite, architecture, wind, ndx=ndx, ddx=ddx)
                        resi = cas.vertcat(resi, vel_resi)

    return resi

def get_convection_residual_local(options, variables, tip, period, kite, architecture, wind, start=bool(False), ndx=0, ddx=0):

    vel_var = get_vel_wake_var(options, variables, tip, period, kite, architecture, start=start, ndx=ndx, ddx=ddx)
    pos_var = get_pos_wake_var(options, variables, tip, period, kite, architecture, start=start, ndx=ndx, ddx=ddx)

    z_var = cas.mtimes(pos_var.T, vect_op.zhat())
    vel_comp = wind.get_velocity(z_var)

    vel_resi = vel_var - vel_comp

    return vel_resi