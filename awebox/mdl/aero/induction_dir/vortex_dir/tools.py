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

import numpy as np

import pdb


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
        awelogger.logger.error('Unknown vector type. Please choose either pos (position) or vel (velocity).')

    vect = []
    for dim in dims:
        name = sym + dim + '_' + tip + '_' + str(period) + '_' + str(kite) + str(parent)

        try:
            comp_all = variables[loc][name]
        except:
            awelogger.logger.error('No such variable known.')

        comp = get_wake_var_at_ndx_ddx(n_k, d, comp_all, start=start, ndx=ndx, ddx=ddx)
        vect = cas.vertcat(vect, comp)

    return vect

def get_pos_wake_var(options, variables, tip, period, kite, architecture, start=bool(False), ndx=0, ddx=0):
    pos = get_vector_var(options, variables, 'pos', tip, period, kite, architecture, start=start, ndx=ndx, ddx=ddx)
    return pos

def get_vel_wake_var(options, variables, tip, period, kite, architecture, start=bool(False), ndx=0, ddx=0):
    vel = get_vector_var(options, variables, 'vel', tip, period, kite, architecture, start=start, ndx=ndx, ddx=ddx)
    return vel













def get_time_ordered_wake_var_without_start(n_k, d, var):
    dimensions = (n_k, d)
    n_regular = n_k * d

    var_regular = var[1:]
    var_reshape = cas.reshape(var_regular, dimensions)
    time_ordered = cas.reshape(var_reshape.T, (n_regular, 1))

    oldest_behind = time_ordered[::-1]

    return oldest_behind


def get_time_ordered_wake_var_with_start(n_k, d, var):

    var_start = var[0]
    without_start = get_time_ordered_wake_var_without_start(n_k, d, var)
    oldest_behind = cas.vertcat(without_start, var_start)

    return oldest_behind


def get_time_ordered_strength(n_k, d, var):
    dimensions = (n_k, d)
    n_regular = n_k * d

    var_regular = var
    var_reshape = cas.reshape(var_regular, dimensions)
    time_ordered = cas.reshape(var_reshape.T, (n_regular, 1))

    oldest_behind = time_ordered[::-1]

    return oldest_behind

def get_all_time_ordered_points_by_kite_and_tip(variables_xd, n_k, d, periods_tracked, kite, parent, tip):

    dims = ['x', 'y', 'z']

    all_dims = []
    for dim in dims:

        all_ordered = []
        for period in range(periods_tracked - 1):
            var_name = 'w' + dim + '_' + tip + '_' + str(period) + '_' + str(kite) + str(parent)
            var = variables_xd[var_name]
            var_ordered = get_time_ordered_wake_var_without_start(n_k, d, var)
            all_ordered = cas.vertcat(all_ordered, var_ordered)

        period = periods_tracked - 1
        var_name = 'w' + dim + '_' + tip + '_' + str(period) + '_' + str(kite) + str(parent)
        var = variables_xd[var_name]
        var_ordered = get_time_ordered_wake_var_with_start(n_k, d, var)
        all_ordered = cas.vertcat(all_ordered, var_ordered)

        all_dims = cas.horzcat(all_dims, all_ordered)

    return all_dims

def get_padded_points_by_kite_and_tip(variables_xd, n_k, d, periods_tracked, kite, parent, tip, u_vec_ref, infinite_time):

    regular = get_all_time_ordered_points_by_kite_and_tip(variables_xd, n_k, d, periods_tracked, kite, parent, tip)

    leading = regular[0, :]
    trailing = regular[-1, :]

    infinite_leading = leading + infinite_time * (-1.) * u_vec_ref.T
    infinite_trailing = trailing + infinite_time * u_vec_ref.T

    padded = cas.vertcat(infinite_leading, regular, infinite_trailing)

    return padded


def get_strength_var_column(variables_xl_or_variables, gamma_name, options):
    try:
        var_unscaled = variables_xl_or_variables[gamma_name]
    except:
        try:
            var_unscaled = variables_xl_or_variables['xl', gamma_name]
        except:
            message = 'gamma variable is incorrect format to pull from given variables (xl)'
            awelogger.logger.error(message)

            pdb.set_trace()

    var = get_strength_scale(options) * var_unscaled

    return var

def get_strength_scale(options):
    gamma_scale = options['aero']['vortex']['gamma_scale']
    return gamma_scale


def get_all_time_ordered_strengths_by_kite(variables_xl, n_k, d, periods_tracked, kite, parent, options):

    all_ordered = []
    for period in range(periods_tracked):

        period_chopped = period
        if period > 0:
            period_chopped = 1

        gamma_name = 'wg' + '_' + str(period_chopped) + '_' + str(kite) + str(parent)
        var = get_strength_var_column(variables_xl, gamma_name, options)

        var_ordered = get_time_ordered_strength(n_k, d, var)
        all_ordered = cas.vertcat(all_ordered, var_ordered)

    return all_ordered

def get_padded_strengths_by_kite(variables, n_k, d, periods_tracked, kite, parent, options):
    regular = get_all_time_ordered_strengths_by_kite(variables, n_k, d, periods_tracked, kite, parent, options)

    trailing = regular[-1]

    infinite_leading = cas.DM(0)
    infinite_trailing = trailing

    padded = cas.vertcat(infinite_leading, regular, infinite_trailing)

    return padded


def get_vortex_ring_filaments(padded_strengths, padded_points_ext, padded_points_int, rdx):

    points = {}
    points['int_trailing'] = padded_points_int[rdx + 1, :]
    points['int_leading'] = padded_points_int[rdx, :]
    points['ext_leading'] = padded_points_ext[rdx, :]
    points['ext_trailing'] = padded_points_ext[rdx + 1, :]

    strength_leading = padded_strengths[rdx]
    strength_trailing = padded_strengths[rdx - 1]

    ring_list = []

    ## vortex segment
    # from: interior, trailing point
    # to: interior, leading point
    start_point = points['int_trailing']
    end_point = points['int_leading']
    strength = strength_leading
    new_vortex = cas.horzcat(start_point, end_point, strength)
    ring_list = cas.vertcat(ring_list, new_vortex)

    ## vortex segment
    # from: interior, leading point
    # to: exterior, leading point
    start_point = points['int_leading']
    end_point = points['ext_leading']
    strength = strength_leading - strength_trailing
    new_vortex = cas.horzcat(start_point, end_point, strength)
    ring_list = cas.vertcat(ring_list, new_vortex)

    ## vortex segment
    # from: exterior, leading point
    # to: exterior, trailing point
    start_point = points['ext_leading']
    end_point = points['ext_trailing']
    strength = strength_leading
    new_vortex = cas.horzcat(start_point, end_point, strength)
    ring_list = cas.vertcat(ring_list, new_vortex)

    return ring_list


def get_number_of_rings_per_kite(n_k, d, periods_tracked):
    # attention! count includes leading- and trailing-infinite rings!
    n_rings = (periods_tracked * n_k * d) + 2
    return n_rings


def get_list_of_filaments_by_kite(args, options):

    variables = args['variables']
    u_vec_ref = args['u_vec_ref']
    infinite_time = args['infinite_time']
    periods_tracked = args['periods_tracked']
    n_k = args['n_k']
    d = args['d']
    kite = args['kite']
    parent = args['parent']

    filaments = []

    padded_strengths = get_padded_strengths_by_kite(variables, n_k, d, periods_tracked, kite, parent, options)
    padded_points_int = get_padded_points_by_kite_and_tip(variables, n_k, d, periods_tracked, kite, parent, 'int',
                                                          u_vec_ref, infinite_time)
    padded_points_ext = get_padded_points_by_kite_and_tip(variables, n_k, d, periods_tracked, kite, parent, 'ext',
                                                          u_vec_ref, infinite_time)

    n_rings = padded_strengths.shape[0]
    for rdx in range(1, n_rings):
        ring_filaments = get_vortex_ring_filaments(padded_strengths, padded_points_ext, padded_points_int, rdx)
        filaments = cas.vertcat(filaments, ring_filaments)

    return filaments


def get_list_of_all_filaments(variables, architecture, u_vec_ref, periods_tracked, n_k, d, options):

    kite_nodes = architecture.kite_nodes
    parent_map = architecture.parent_map

    infinite_time = 1000.

    args = {}
    args['variables'] = variables
    args['u_vec_ref'] = u_vec_ref
    args['infinite_time'] = infinite_time
    args['periods_tracked'] = periods_tracked
    args['n_k'] = n_k
    args['d'] = d

    filaments = []

    for kite in kite_nodes:
        parent = parent_map[kite]

        args['kite'] = kite
        args['parent'] = parent

        new_filaments = get_list_of_filaments_by_kite(args, options)
        filaments = cas.vertcat(filaments, new_filaments)

    horz_filaments = filaments.T

    return horz_filaments


def get_filament_list(options, wind, variables, architecture):
    n_k = options['aero']['vortex']['n_k']
    d = options['aero']['vortex']['d']
    periods_tracked = options['aero']['vortex']['periods_tracked']
    u_vec_ref = wind.get_velocity_ref() * vect_op.xhat()

    filament_list = get_list_of_all_filaments(variables, architecture, u_vec_ref,
                                                           periods_tracked, n_k, d, options)

    return filament_list


def get_list_of_filaments_by_kite_and_ring(options, variables, wind, kite, parent, rdx):

    n_k = options['aero']['vortex']['n_k']
    d = options['aero']['vortex']['d']
    periods_tracked = options['aero']['vortex']['periods_tracked']
    u_vec_ref = wind.get_velocity_ref() * vect_op.xhat()

    infinite_time = 1000.

    padded_strengths = get_padded_strengths_by_kite(variables, n_k, d, periods_tracked, kite, parent, options)
    padded_points_int = get_padded_points_by_kite_and_tip(variables, n_k, d, periods_tracked, kite, parent,
                                                                'int',
                                                                u_vec_ref, infinite_time)
    padded_points_ext = get_padded_points_by_kite_and_tip(variables, n_k, d, periods_tracked, kite, parent,
                                                                'ext',
                                                                u_vec_ref, infinite_time)

    ring_filaments = get_vortex_ring_filaments(padded_strengths, padded_points_ext, padded_points_int, rdx)

    return ring_filaments





def evaluate_symbolic_on_segments_and_sum(filament_fun, segment_list):

    n_filaments = segment_list.shape[1]
    filament_map = filament_fun.map(n_filaments, 'openmp')
    all = filament_map(segment_list)

    total = cas.sum2(all)

    return total


def append_bounds(g_bounds, fix):

    if (type(fix) == type([])) and fix == []:
        return g_bounds

    else:

        fix_shape = (1,1)
        try:
            fix_shape = fix.shape
        except:
            message = 'An attempt to append bounds was passed a vortex-related constraint with an unaccepted structure.'
            awelogger.logger.error(message)

        g_bounds['ub'].append(cas.DM.zeros(fix_shape))
        g_bounds['lb'].append(cas.DM.zeros(fix_shape))

        return g_bounds

def get_vortex_verification_mu_vals():

    radius = 155.77
    b_ref = 68.

    varrho = radius / b_ref

    mu_center_by_exterior = varrho / (varrho + 0.5)
    mu_min_by_exterior = (varrho - 0.5) / (varrho + 0.5)
    mu_max_by_exterior = 1.

    mu_min_by_path = (varrho - 0.5) / varrho
    mu_max_by_path = (varrho + 0.5) / varrho
    mu_center_by_path = 1.

    mu_vals = {}
    mu_vals['mu_center_by_exterior'] = mu_center_by_exterior
    mu_vals['mu_min_by_exterior'] = mu_min_by_exterior
    mu_vals['mu_max_by_exterior'] = mu_max_by_exterior
    mu_vals['mu_min_by_path'] = mu_min_by_path
    mu_vals['mu_max_by_path'] = mu_max_by_path
    mu_vals['mu_center_by_path'] = mu_center_by_path

    return mu_vals
