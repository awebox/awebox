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
import awebox.tools.vector_operations as vect_op
import awebox.mdl.aero.induction_dir.vortex_dir.tools as tools


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

    vel_var = tools.get_vel_wake_var(options, variables, tip, period, kite, architecture, start=start, ndx=ndx, ddx=ddx)
    pos_var = tools.get_pos_wake_var(options, variables, tip, period, kite, architecture, start=start, ndx=ndx, ddx=ddx)

    z_var = cas.mtimes(pos_var.T, vect_op.zhat())
    vel_comp = wind.get_velocity(z_var)

    vel_resi = vel_var - vel_comp

    return vel_resi