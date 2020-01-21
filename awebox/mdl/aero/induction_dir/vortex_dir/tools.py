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

import casadi as cas
import numpy as np
from awebox.logger.logger import Logger as awelogger
import awebox.tools.vector_operations as vect_op
import awebox.mdl.wind as wind
import pdb
from multiprocessing import Pool

def get_wake_var_at_ndx_ddx(n_k, d, var, start=bool(False), ndx=0, ddx=0):
    if start:
        return var[0]
    else:
        var_regular = var[1:]

        dimensions = (n_k, d)
        var_reshape = cas.reshape(var_regular, dimensions)

        return var_reshape[ndx, ddx]

def get_strength_at_ndx_ddx(variables_xl, architecture, n_k, d, period, kite, ndx, ddx):
    parent_map = architecture.parent_map
    parent = parent_map[kite]

    gamma_name = 'wg' + '_' + str(period) + '_' + str(kite) + str(parent)

    var_regular = variables_xl[gamma_name]

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

def get_list_of_all_vortices(variables_xd, variables_xl, architecture, U_infty, periods_tracked, n_k, d, enable_pool=False, processes=3):

    if enable_pool:
        vortex_list = get_list_of_all_vortices_parallel(variables_xd, variables_xl, architecture, U_infty, periods_tracked, n_k, d, processes)
    else:
        vortex_list = get_list_of_all_vortices_singular(variables_xd, variables_xl, architecture, U_infty, periods_tracked, n_k, d)

    return vortex_list

def get_list_of_all_vortices_parallel(variables_xd, variables_xl, architecture, U_infty, periods_tracked, n_k, d, processes=3):

    args = {}

    args['variables_xd'] = variables_xd
    args['variables_xl'] = variables_xl
    args['architecture'] = architecture
    args['U_infty'] = U_infty
    args['periods_tracked'] = periods_tracked
    args['n_k'] = n_k
    args['d'] = d

    kite_nodes = architecture.kite_nodes
    args_list = []
    for kite in kite_nodes:
        new_arg = args
        new_arg['kite'] = kite
        args_list += [new_arg]

    with Pool(processes=processes) as pool:
        list_set = pool.map(get_parallel_sublist_of_all_vortices, args_list)

    vortex_list = []
    for sublist in list_set:
        vortex_list = cas.horzcat(vortex_list, sublist)

    return vortex_list

def get_parallel_sublist_of_all_vortices(args):

    variables_xd = args['variables_xd']
    variables_xl = args['variables_xl']
    architecture = args['architecture']
    U_infty = args['U_infty']
    periods_tracked = args['periods_tracked']
    n_k = args['n_k']
    d = args['d']
    kite = args['kite']

    vortex_list = []

    # add all "typical/ladder" vortex rings
    for period in range(periods_tracked):
        for ndx_shed in range(n_k):
            for ddx_shed in range(d):

                points = get_points_for_vortex_ring(variables_xd, architecture, n_k, d, period, kite, ndx_shed, ddx_shed)
                ring_list = get_list_entries_for_vortex_ring(points, variables_xl, architecture, n_k, d, period, kite, ndx_shed, ddx_shed)
                vortex_list = cas.horzcat(vortex_list, ring_list)

    # add infinite trailing vortices
    infinite_list = get_semi_infinite_trailing_vortices(variables_xd, variables_xl, architecture, U_infty, periods_tracked, n_k, d, kite)
    vortex_list = cas.horzcat(vortex_list, infinite_list)

    return vortex_list



def get_list_of_all_vortices_singular(variables_xd, variables_xl, architecture, U_infty, periods_tracked, n_k, d):

    vortex_list = []

    # add all "typical/ladder" vortex rings
    kite_nodes = architecture.kite_nodes
    for kite in kite_nodes:
        for period in range(periods_tracked):
            for ndx_shed in range(n_k):
                for ddx_shed in range(d):

                    points = get_points_for_vortex_ring(variables_xd, architecture, n_k, d, period, kite, ndx_shed, ddx_shed)
                    ring_list = get_list_entries_for_vortex_ring(points, variables_xl, architecture, n_k, d, period, kite, ndx_shed, ddx_shed)
                    vortex_list = cas.horzcat(vortex_list, ring_list)

    # add infinite trailing vortices
    for kite in kite_nodes:
        infinite_list = get_semi_infinite_trailing_vortices(variables_xd, variables_xl, architecture, U_infty, periods_tracked, n_k, d, kite)
        vortex_list = cas.horzcat(vortex_list, infinite_list)

    return vortex_list


def get_semi_infinite_trailing_vortices(variables_xd, variables_xl, architecture, U_infty, periods_tracked, n_k, d, kite):

    infinite_list = []

    period = periods_tracked - 1
    ndx_shed = 0
    ddx_shed = 0
    points = get_points_for_vortex_ring(variables_xd, architecture, n_k, d, period, kite, ndx_shed, ddx_shed)
    strength = get_strength_at_ndx_ddx(variables_xl, architecture, n_k, d, period, kite, ndx_shed,
                                       ddx_shed)

    ## semi-infinite vortex: interior wingtip
    start_point = points['int_trailing'] + U_infty * 1000.
    end_point = points['int_trailing']
    new_vortex = cas.vertcat(start_point, end_point, strength)
    infinite_list = cas.horzcat(infinite_list, new_vortex)

    ## semi-infinite vortex: interior wingtip
    start_point = points['ext_trailing']
    end_point = points['ext_trailing'] + U_infty * 1000.
    new_vortex = cas.vertcat(start_point, end_point, strength)
    infinite_list = cas.horzcat(infinite_list, new_vortex)

    return infinite_list


def get_list_entries_for_vortex_ring(points, variables_xl, architecture, n_k, d, period, kite, ndx_shed, ddx_shed):

    ring_list = []
    # identify every given vortex segment by starting point of segment
    # do not add trailing edge vortex segment

    ## vortex segment
    # from: interior, trailing point
    # to: interior, leading point
    start_point = points['int_trailing']
    end_point = points['int_leading']
    strength = get_strength_at_ndx_ddx(variables_xl, architecture, n_k, d, period, kite, ndx_shed, ddx_shed)
    new_vortex = cas.vertcat(start_point, end_point, strength)
    ring_list = cas.horzcat(ring_list, new_vortex)

    ## vortex segment
    # from: interior, leading point
    # to: exterior, leading point
    start_point = points['int_leading']
    end_point = points['ext_leading']
    strength_leading_edge = get_strength_at_ndx_ddx(variables_xl, architecture, n_k, d, period, kite, ndx_shed, ddx_shed)

    if (period == 0) and (ndx_shed == n_k - 1) and (ddx_shed == d - 1):
        strength_trailing_edge = 0.
    elif (ndx_shed == n_k - 1) and (ddx_shed == d - 1):
        strength_trailing_edge = get_strength_at_ndx_ddx(variables_xl, architecture, n_k, d, period - 1, kite, 0, 0)
    elif (ddx_shed == d - 1):
        strength_trailing_edge = get_strength_at_ndx_ddx(variables_xl, architecture, n_k, d, period, kite, ndx_shed + 1, 0)
    else:
        strength_trailing_edge = get_strength_at_ndx_ddx(variables_xl, architecture, n_k, d, period, kite, ndx_shed, ddx_shed + 1)

    strength = strength_leading_edge - strength_trailing_edge
    new_vortex = cas.vertcat(start_point, end_point, strength)
    ring_list = cas.horzcat(ring_list, new_vortex)

    ## vortex segment
    # from: exterior, leading point
    # to: exterior, trailing point
    start_point = points['ext_leading']
    end_point = points['ext_trailing']
    strength = get_strength_at_ndx_ddx(variables_xl, architecture, n_k, d, period, kite, ndx_shed, ddx_shed)
    new_vortex = cas.vertcat(start_point, end_point, strength)
    ring_list = cas.horzcat(ring_list, new_vortex)

    return ring_list



def get_points_for_vortex_ring(variables_xd, architecture, n_k, d, period, kite, ndx_shed, ddx_shed):
    parent_map = architecture.parent_map
    parent = parent_map[kite]

    wingtips = ['ext', 'int']
    longitudes = ['leading', 'trailing']
    dims = ['x', 'y', 'z']

    # reserve space for ring position information
    points = {}
    for tip in wingtips:
        for long in longitudes:
            entry_name = tip + '_' + long
            points[entry_name] = vect_op.zeros_sx((3, 1))

    # fill in information
    for tip in wingtips:
        for long in longitudes:
            entry_name = tip + '_' + long

            for jdx in range(len(dims)):
                point_name = 'w' + dims[jdx] + '_' + tip + '_' + str(period) + '_' + str(kite) + str(
                    parent)

                var = variables_xd[point_name]

                if long == 'trailing' and ndx_shed == 0 and ddx_shed == 0:
                    points[entry_name][jdx] += get_wake_var_at_ndx_ddx(n_k, d, var, start=True)

                elif long == 'trailing' and ddx_shed == 0:
                    points[entry_name][jdx] += get_wake_var_at_ndx_ddx(n_k, d, var, start=False, ndx=ndx_shed - 1,
                                                                      ddx=d - 1)

                elif long == 'trailing':
                    points[entry_name][jdx] += get_wake_var_at_ndx_ddx(n_k, d, var, start=False,
                                                                      ndx=ndx_shed, ddx=ddx_shed - 1)

                elif long == 'leading':
                    points[entry_name][jdx] += get_wake_var_at_ndx_ddx(n_k, d, var, start=False,
                                                                      ndx=ndx_shed, ddx=ddx_shed)

                else:
                    awelogger.logger.error(
                        'requested vortex position does not belong to recognized interior/exterior or leading/trailing position')

    return points
