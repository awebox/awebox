#
#    This file is part of awebox.
#
#    awebox -- A modeling and optimization framework for multi-kite AWE systems.
#    Copyright (C) 2017-2020 Jochem De Schutter, Rachel Leuthold, Moritz Diehl,
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
import casadi.tools as cas
import matplotlib.pyplot as plt
import awebox.tools.vector_operations as vect_op
import numpy as np
import pdb
import awebox.viz.tools as tools


def draw_wake_nodes(ax, side, plot_dict, index):

    vals, gammas = prepare_wake_node_information(plot_dict, index)

    plot_trailing_vortices(ax, side, plot_dict, vals, gammas)
    plot_infinite_trailing_vortex(ax, side, plot_dict, vals, gammas)

    plot_starting_shed_vortex(ax, side, plot_dict, vals, gammas)

    return None


def get_gamma_extrema(plot_dict):
    n_k = plot_dict['n_k']
    d = plot_dict['d']
    kite_nodes = plot_dict['architecture'].kite_nodes
    parent_map = plot_dict['architecture'].parent_map
    periods_tracked = plot_dict['options']['model']['aero']['vortex']['periods_tracked']

    gamma_max = -1.e5
    gamma_min = 1.e5

    for kite in kite_nodes:
        parent = parent_map[kite]
        for period in range(periods_tracked):

            for ndx in range(n_k):
                for ddx in range(d):
                    gamma_name = 'wg' + '_' + str(period) + '_' + str(kite) + str(parent)
                    var = plot_dict['V_plot']['coll_var', ndx, ddx, 'xl', gamma_name]

                    gamma_max = np.max(np.array(cas.vertcat(gamma_max, var)))
                    gamma_min = np.min(np.array(cas.vertcat(gamma_min, var)))

    gamma_max = np.max(np.array([gamma_max, -1. * gamma_min]))
    gamma_min = np.min(np.array([gamma_min, -1. * gamma_max]))

    return gamma_min, gamma_max


def convert_gamma_to_color(gamma_val, plot_dict):

    gamma_min, gamma_max = get_gamma_extrema(plot_dict)
    cmap = plt.get_cmap('seismic')
    gamma_scaled = float( (gamma_val - gamma_min) / (gamma_max - gamma_min) )
    color = cmap(gamma_scaled)
    return color


def plot_starting_shed_vortex(ax, side, plot_dict, vals, gammas):

    kite_nodes = plot_dict['architecture'].kite_nodes
    parent_map = plot_dict['architecture'].parent_map
    dims = ['x', 'y', 'z']
    wingtips = ['ext', 'int']

    period = 0

    for kite in kite_nodes:
        parent = parent_map[kite]

        points = []
        for tip in wingtips:
            name = 'w' + '_' + tip + '_' + str(period) + '_' + str(kite) + str(parent)

            tip_point = []
            for dim in dims:
                tip_point = cas.horzcat(tip_point, vals[name][dim]['start'])

            points = cas.vertcat(points, tip_point)

        wake_color = convert_gamma_to_color(gammas[name][0, 0], plot_dict)
        tools.make_side_plot(ax, points, side, wake_color)

    return None



def plot_trailing_vortices(ax, side, plot_dict, vals, gammas):

    n_k = plot_dict['n_k']
    d = plot_dict['d']
    dims = ['x', 'y', 'z']

    for name in vals.keys():

        for ndx_shed in range(n_k):
            for ddx_shed in range(d):

                start_point = []
                end_point = []
                for dim in dims:
                    end_point = cas.horzcat(end_point, vals[name][dim]['reg'][ndx_shed, ddx_shed])
                    if ndx_shed == 0 and ddx_shed == 0:
                        start_point = cas.horzcat(start_point, vals[name][dim]['start'])
                    elif ddx_shed == 0:
                        start_point = cas.horzcat(start_point, vals[name][dim]['reg'][ndx_shed - 1, 0])
                    else:
                        start_point = cas.horzcat(start_point, vals[name][dim]['reg'][ndx_shed, ddx_shed - 1])

                wake_color = convert_gamma_to_color(gammas[name][ndx_shed, ddx_shed], plot_dict)

                points = cas.vertcat(start_point, end_point)
                tools.make_side_plot(ax, points, side, wake_color)
    return None

def plot_infinite_trailing_vortex(ax, side, plot_dict, vals, gammas):

    kite_nodes = plot_dict['architecture'].kite_nodes
    parent_map = plot_dict['architecture'].parent_map
    dims = ['x', 'y', 'z']
    wingtips = ['ext', 'int']

    n_k = plot_dict['n_k']
    d = plot_dict['d']
    periods_tracked = plot_dict['options']['model']['aero']['vortex']['periods_tracked']

    period = periods_tracked - 1
    ndx_shed = n_k - 1
    ddx_shed = d - 1

    U_ref = plot_dict['options']['model']['params']['wind']['u_ref'] * vect_op.xhat_np()


    for kite in kite_nodes:
        parent = parent_map[kite]

        for tip in wingtips:
            name = 'w' + '_' + tip + '_' + str(period) + '_' + str(kite) + str(parent)

            start_point = []
            for dim in dims:
                start_point = cas.horzcat(start_point, vals[name][dim]['reg'][ndx_shed, ddx_shed])
            end_point = start_point + 100. * U_ref.T

            wake_color = convert_gamma_to_color(gammas[name][ndx_shed, ddx_shed], plot_dict)

            points = cas.vertcat(start_point, end_point)
            tools.make_side_plot(ax, points, side, wake_color)
    return None


def prepare_wake_node_information(plot_dict, index):

    n_k = plot_dict['n_k']
    d = plot_dict['d']
    n_nodes = n_k * d + 1
    kite_nodes = plot_dict['architecture'].kite_nodes
    parent_map = plot_dict['architecture'].parent_map
    dims = ['x', 'y', 'z']
    wingtips = ['ext', 'int']
    periods_tracked = plot_dict['options']['model']['aero']['vortex']['periods_tracked']

    vals = {}
    for kite in kite_nodes:
        parent = parent_map[kite]
        for period in range(periods_tracked):
            for tip in wingtips:
                short_name = 'w' + '_' + tip + '_' + str(period) + '_' + str(kite) + str(parent)
                vals[short_name] = {}

                for dim in dims:
                    var_name = 'w' + dim + '_' + tip + '_' + str(period) + '_' + str(kite) + str(parent)
                    vals[short_name][dim] = {}

                    vals[short_name][dim]['all'] = []

                    for node in range(n_nodes):
                        val_col = plot_dict['xd'][var_name][node][index]
                        vals[short_name][dim]['all'] = cas.vertcat(vals[short_name][dim]['all'], val_col)

    gammas = {}
    for kite in kite_nodes:
        parent = parent_map[kite]
        for period in range(periods_tracked):
            for tip in wingtips:
                short_name = 'w' + '_' + tip + '_' + str(period) + '_' + str(kite) + str(parent)
                gamma_name = 'wg_' + str(period) + '_' + str(kite) + str(parent)

                gammas[short_name] = []
                for node in range(n_k * d):
                    gamma_col = plot_dict['xl'][gamma_name][node][index]
                    gammas[short_name] = cas.vertcat(gammas[short_name], gamma_col)

    # reshape into (ndx, ddx) shape for easy interpretation
    for name in vals.keys():
        gammas[name] = cas.reshape(gammas[name], (n_k, d))
        for dim in dims:
            start = vals[name][dim]['all'][0]
            vals[name][dim]['start'] = start

            regular = vals[name][dim]['all'][1:]
            vals[name][dim]['reg'] = cas.reshape(regular, (n_k, d))

    return vals, gammas
