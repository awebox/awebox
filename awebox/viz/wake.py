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
from awebox.logger.logger import Logger as awelogger
import awebox.viz.tools as tools
import awebox.tools.vector_operations as vect_op
import awebox.tools.struct_operations as struct_op
import awebox.mdl.aero.induction_dir.vortex_dir.flow as vortex_flow
import awebox.mdl.aero.induction_dir.vortex_dir.tools as vortex_tools

import pdb

def draw_wake_nodes(ax, side, plot_dict, index):

    vortex_info_exists = determine_if_vortex_info_exists(plot_dict)
    if vortex_info_exists:

        filament_list = reconstruct_filament_list(plot_dict, index)

        n_filaments = filament_list.shape[1]

        for fdx in range(n_filaments):
            seg_data = filament_list[:, fdx]
            start_point = seg_data[:3].T
            end_point = seg_data[3:6].T
            gamma = seg_data[6]

            points = cas.vertcat(start_point, end_point)
            wake_color = convert_gamma_to_color(gamma, plot_dict)
            try:
                tools.make_side_plot(ax, points, side, wake_color)
            except:
                awelogger.logger.error('error in side plot production')

    return None

def determine_if_vortex_info_exists(plot_dict):
    vortex_exists = 'vortex' in plot_dict['outputs'].keys()
    filament_list_exists = vortex_exists and ('filament_list' in plot_dict['outputs']['vortex'].keys())

    return filament_list_exists

def reconstruct_filament_list(plot_dict, index):

    all_time = plot_dict['outputs']['vortex']['filament_list']
    n_entries = len(all_time)
    n_filaments = int(n_entries / 7)

    filament_list = []
    for edx in range(n_entries):
        new_entry = all_time[edx][index]
        filament_list = cas.vertcat(filament_list, new_entry)

    filament_list = cas.reshape(filament_list, (7, n_filaments))

    return filament_list

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

            if period > 1:
                period = 1

            for ndx in range(n_k):
                for ddx in range(d):
                    gamma_name = 'wg' + '_' + str(period) + '_' + str(kite) + str(parent)
                    var = plot_dict['V_plot']['coll_var', ndx, ddx, 'xl', gamma_name]

                    gamma_max = np.max(np.array(cas.vertcat(gamma_max, var)))
                    gamma_min = np.min(np.array(cas.vertcat(gamma_min, var)))

    # so that gamma = 0 vortex filaments will be drawn in white...
    gamma_max = np.max(np.array([gamma_max, -1. * gamma_min]))
    gamma_min = -1. * gamma_max

    return gamma_min, gamma_max


def convert_gamma_to_color(gamma_val, plot_dict):

    gamma_min, gamma_max = get_gamma_extrema(plot_dict)
    cmap = plt.get_cmap('seismic')
    gamma_scaled = float( (gamma_val - gamma_min) / (gamma_max - gamma_min) )

    color = cmap(gamma_scaled)
    return color

def compute_vortex_verification_points(plot_dict, cosmetics, idx_at_eval):

    architecture = plot_dict['architecture']
    architecture = plot_dict['architecture']
    wind = plot_dict['wind']

    V_plot = plot_dict['V_plot']

    Xdot = struct_op.construct_Xdot_struct(plot_dict['options']['nlp'], plot_dict['variables_dict'])(0.)
    variables = plot_dict['variables'](struct_op.get_variables_at_time(plot_dict['options']['nlp'], V_plot, Xdot, plot_dict['variables'], idx_at_eval))
    parameters = plot_dict['parameters'](struct_op.get_parameters_at_time(plot_dict['options']['nlp'], plot_dict['p_fix_num'], V_plot, Xdot, plot_dict['variables'], plot_dict['parameters']))

    parent = 1
    radius = 155.77

    verification_points = plot_dict['options']['model']['aero']['vortex']['verification_points']
    half_points = int(verification_points / 2.) + 1

    mu_grid_min = 0.4
    mu_grid_max = 1.6
    psi_grid_min = 1. * np.pi / float(architecture.number_of_kites)
    psi_grid_max = 3. * np.pi / float(architecture.number_of_kites)

    # define mu with respect to kite mid-span radius
    mu_grid_points = np.linspace(mu_grid_min, mu_grid_max, verification_points, endpoint=True)
    # psi_grid_points = np.linspace(psi_grid_min, psi_grid_max, n_points, endpoint=True)

    beta = np.linspace(0., np.pi / 2, half_points)
    cos_front = 0.5 * (1. - np.cos(beta))
    cos_back = -1. * cos_front[::-1]
    psi_grid_unscaled = cas.vertcat(cos_back[:-1], cos_front) + 0.5
    psi_grid_points_cas = psi_grid_unscaled * (psi_grid_max - psi_grid_min) + psi_grid_min

    psi_grid_points = []
    for idx in range(psi_grid_points_cas.shape[0]):
        psi_grid_points += [float(psi_grid_points_cas[idx])]

    haas_grid = {}
    center = plot_dict['outputs']['performance']['actuator_center1']
    counter = 0
    for mu_val in mu_grid_points:
        for psi_val in psi_grid_points:
            r_val = mu_val * radius

            ehat_radial = vect_op.zhat_np() * cas.cos(psi_val) - vect_op.yhat_np() * cas.sin(psi_val)
            added = r_val * ehat_radial
            point_obs = center + added

            unscaled = mu_val * ehat_radial

            # a_ind = vortex_flow.get_induction_factor_at_observer(point_obs, plot_dict['options']['model'], wind,
            #                                                      variables, parameters, parent, architecture)

            pdb.set_trace()

            a_ind = vortex_flow.get_induction_factor_at_observer(point_obs,
                                                                 plot_dict['options']['model'],
                                                                 wind,
                                                                 plot_dict['variables'],
                                                                 plot_dict['parameters'],
                                                                 parent, architecture)

            pdb.set_trace()


            a_ind = 0.

            local_info = cas.vertcat(unscaled[1], unscaled[2], a_ind)
            haas_grid['p' + str(counter)] = local_info

            counter += 1


def plot_vortex_verification(plot_dict, cosmetics, fig_name, fig_num=None):

    idx_at_eval = 0

    verification_points = plot_dict['options']['model']['aero']['vortex']['verification_points']
    mu_vals = vortex_tools.get_vortex_verification_mu_vals()

    vortex_structure_modelled = 'vortex' in plot_dict['outputs'].keys()
    if vortex_structure_modelled:

        haas_grid = compute_vortex_verification_points(plot_dict, cosmetics, idx_at_eval)

        mu_min_by_path = mu_vals['mu_min_by_path']
        mu_max_by_path = mu_vals['mu_max_by_path']

        number_entries = len(haas_grid.keys())

        slice_index = -1

        y_matr = []
        z_matr = []
        a_matr = []
        idx = 0

        y_row = []
        z_row = []
        a_row = []

        for ndx in range(number_entries):

            idx += 1

            local_y = haas_grid['p' + str(ndx)][0][slice_index]
            local_z = haas_grid['p' + str(ndx)][1][slice_index]
            local_a = haas_grid['p' + str(ndx)][2][slice_index]

            y_row = cas.horzcat(y_row, local_y)
            z_row = cas.horzcat(z_row, local_z)
            a_row = cas.horzcat(a_row, local_a)

            if float(idx) == (verification_points):
                y_matr = cas.vertcat(y_matr, y_row)
                z_matr = cas.vertcat(z_matr, z_row)
                a_matr = cas.vertcat(a_matr, a_row)

                y_row = []
                z_row = []
                a_row = []
                idx = 0

        y_matr = np.array(y_matr)
        z_matr = np.array(z_matr)
        a_matr = np.array(a_matr)

        y_matr_list = np.array(cas.vertcat(y_matr))
        z_matr_list = np.array(cas.vertcat(z_matr))

        max_y = np.max(y_matr_list)
        min_y = np.min(y_matr_list)
        max_z = np.max(z_matr_list)
        min_z = np.min(z_matr_list)
        max_axes = np.max(np.array([max_y, -1. * min_y, max_z, -1. * min_z]))

        ### points plot

        fig_points, ax_points = plt.subplots(1, 1)
        add_annulus_background(ax_points, mu_min_by_path, mu_max_by_path)
        ax_points.scatter(y_matr_list, z_matr_list)
        plt.grid(True)
        plt.title('induction factors over the kite plane')
        plt.xlabel("y/r [-]")
        plt.ylabel("z/r [-]")
        ax_points.set_xlim([-1. * max_axes, max_axes])
        ax_points.set_ylim([-1. * max_axes, max_axes])

        fig_points.savefig('points.pdf')

        #### contour plot

        fig_contour, ax_contour = plt.subplots(1, 1)
        add_annulus_background(ax_contour, mu_min_by_path, mu_max_by_path)

        levels = [-0.05, 0., 0.2]
        colors = ['red', 'green', 'blue']

        cs = ax_contour.contour(y_matr, z_matr, a_matr, levels, colors=colors)
        plt.clabel(cs, inline=1, fontsize=10)
        for i in range(len(levels)):
            cs.collections[i].set_label(levels[i])
        plt.legend(loc='lower right')

        plt.grid(True)
        plt.title('induction factors over the kite plane')
        plt.xlabel("y/r [-]")
        plt.ylabel("z/r [-]")
        ax_contour.set_xlim([-1. * max_axes, max_axes])
        ax_contour.set_ylim([-1. * max_axes, max_axes])

        fig_contour.savefig('contour.pdf')


def add_annulus_background(ax, mu_min_by_path, mu_max_by_path):
    n, radii = 50, [mu_min_by_path, mu_max_by_path]
    theta = np.linspace(0, 2 * np.pi, n, endpoint=True)
    xs = np.outer(radii, np.cos(theta))
    ys = np.outer(radii, np.sin(theta))

    # in order to have a closed area, the circles
    # should be traversed in opposite directions
    xs[1, :] = xs[1, ::-1]
    ys[1, :] = ys[1, ::-1]

    color = (0.83,0.83,0.83,0.5)

    ax.fill(np.ravel(xs), np.ravel(ys), color=color)
