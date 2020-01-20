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
import numpy as np
import pdb
import awebox.viz.tools as tools
import awebox.mdl.aero.induction_dir.vortex_dir.tools as vortex_tools
import awebox.tools.vector_operations as vect_op


def draw_wake_nodes(ax, side, plot_dict, index):
    variables_xd = tools.assemble_variable_slice_from_interpolated_data(plot_dict, index, 'xd')
    variables_xl = tools.assemble_variable_slice_from_interpolated_data(plot_dict, index, 'xl')

    n_k = plot_dict['n_k']
    d = plot_dict['d']
    architecture = plot_dict['architecture']
    periods_tracked = plot_dict['options']['model']['aero']['vortex']['periods_tracked']

    U_ref = plot_dict['options']['model']['params']['wind']['u_ref'] * vect_op.xhat_np()

    enable_pool = plot_dict['cosmetics']['processing']['enable_pool']
    processes = plot_dict['cosmetics']['processing']['processes']

    vortex_list = vortex_tools.get_list_of_all_vortices(variables_xd, variables_xl, architecture, U_ref,
                                                        periods_tracked, n_k, d, enable_pool=enable_pool,
                                                        processes=processes)

    n_segments = vortex_list.shape[1]
    for sdx in range(n_segments):
        seg_data = vortex_list[:, sdx]
        start_point = seg_data[:3].T
        end_point = seg_data[3:6].T
        gamma = seg_data[6]

        points = cas.vertcat(start_point, end_point)
        wake_color = convert_gamma_to_color(gamma, plot_dict)
        try:
            tools.make_side_plot(ax, points, side, wake_color)
        except:
            pdb.set_trace()

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

