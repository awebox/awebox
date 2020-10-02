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
various structural tools for the vortex model
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2019-2020
'''

import casadi.tools as cas
from awebox.logger.logger import Logger as awelogger
import awebox.tools.vector_operations as vect_op
from awebox.logger.logger import Logger as awelogger
import awebox.tools.print_operations as print_op
import awebox.tools.struct_operations as struct_op

def get_wake_node_position_si(options, variables, kite, tip, wake_node):

    wx_local = get_wake_node_position(variables, kite, tip, wake_node)
    wx_scale = get_position_scale(options)
    wx_rescaled = wx_local * wx_scale

    return wx_rescaled

def get_wake_node_position(variables, kite, tip, wake_node):
    coord_name = 'wx_' + str(kite) + '_' + tip + '_' + str(wake_node)
    wx_local = struct_op.get_variable_from_model_or_reconstruction(variables, 'xd', coord_name)
    return wx_local

def get_wake_node_velocity_si(options, variables, kite, tip, wake_node):
    dwx_local = get_wake_node_velocity(variables, kite, tip, wake_node)
    dwx_scale = get_position_scale(options)
    dwx_rescaled = dwx_local * dwx_scale

    return dwx_rescaled

def get_wake_node_velocity(variables, kite, tip, wake_node):
    coord_name = 'dwx_' + str(kite) + '_' + tip + '_' + str(wake_node)
    dwx_local = struct_op.get_variable_from_model_or_reconstruction(variables, 'xd', coord_name)
    return dwx_local

def get_ring_strength_si(options, variables, kite, ring):

    wg_local = get_ring_strength(variables, kite, ring)

    wg_scale = get_strength_scale(options)
    wg_rescaled = wg_local * wg_scale

    return wg_rescaled

def get_ring_strength(variables, kite, ring):
    coord_name = 'wg_' + str(kite) + '_' + str(ring)
    wg_local = struct_op.get_variable_from_model_or_reconstruction(variables, 'xl', coord_name)

    return wg_local

def evaluate_symbolic_on_segments_and_sum(filament_fun, segment_list):

    n_filaments = segment_list.shape[1]
    filament_map = filament_fun.map(n_filaments, 'openmp')
    all = filament_map(segment_list)

    total = cas.sum2(all)

    return total

def get_position_scale(options):

    wx_scale = options['induction']['vortex_position_scale']
    return wx_scale

def get_velocity_scale(options):
    dwx_scale = options['induction']['vortex_u_ref']
    return dwx_scale

def get_strength_scale(options):
    wg_scale = options['induction']['vortex_gamma_scale']
    return wg_scale

def append_bounds(g_bounds, fix):

    if (type(fix) == type([])) and fix == []:
        return g_bounds

    else:
        try:
            fix_shape = fix.shape
        except:
            message = 'An attempt to append bounds was passed a vortex-related constraint with an unaccepted structure.'
            awelogger.logger.error(message)
            raise Exception(message)

        g_bounds['ub'].append(cas.DM.zeros(fix_shape))
        g_bounds['lb'].append(cas.DM.zeros(fix_shape))

        return g_bounds

