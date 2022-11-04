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
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import pdb

import casadi.tools as cas
import numpy as np
from awebox.logger.logger import Logger as awelogger

import awebox.mdl.wind as wind

import awebox.viz.tools as tools

import awebox.tools.vector_operations as vect_op
import awebox.tools.struct_operations as struct_op
import awebox.tools.print_operations as print_op



def plot_wake(plot_dict, cosmetics, fig_name, side):

    fig = plt.figure()

    if side == 'xy':
        ax = plt.subplot(1, 1, 1)
        plt.axis('equal')
        ax.set_xlabel('x [m]', **cosmetics['trajectory']['axisfont'])
        ax.set_ylabel('y [m]', **cosmetics['trajectory']['axisfont'])

    elif side == 'xz':
        ax = plt.subplot(1, 1, 1)
        plt.axis('equal')
        ax.set_xlabel('x [m]', **cosmetics['trajectory']['axisfont'])
        ax.set_ylabel('z [m]', **cosmetics['trajectory']['axisfont'])

    elif side == 'yz':
        ax = plt.subplot(1, 1, 1)
        plt.axis('equal')
        ax.set_xlabel('y [m]', **cosmetics['trajectory']['axisfont'])
        ax.set_ylabel('z [m]', **cosmetics['trajectory']['axisfont'])

    elif side == 'isometric':
        ax = plt.subplot(111, projection='3d')
        ax.set_xlabel('\n x [m]', **cosmetics['trajectory']['axisfont'])
        ax.set_ylabel('\n y [m]', **cosmetics['trajectory']['axisfont'])
        ax.set_zlabel('z [m]', **cosmetics['trajectory']['axisfont'])
        ax.xaxis._axinfo['label']['space_factor'] = 2.8
        ax.yaxis._axinfo['label']['space_factor'] = 2.8
        ax.zaxis._axinfo['label']['space_factor'] = 2.8

    index = -1
    draw_wake_nodes(ax, side, plot_dict, cosmetics, index)

    if cosmetics['trajectory']['kite_bodies']:
        init_colors = False
        tools.draw_all_kites(ax, plot_dict, index, cosmetics, side, init_colors)

    ax.tick_params(labelsize=cosmetics['trajectory']['ylabelsize'])
    plt.suptitle(fig_name)

    return None

def draw_wake_nodes(ax, side, plot_dict, cosmetics, index):

    if 'wake' in plot_dict.keys():

        model_variables = plot_dict['variables']
        model_parameters = plot_dict['parameters']
        model_scaling = plot_dict['scaling']

        variables_si = tools.assemble_variable_slice_from_interpolated_data(plot_dict, index)
        variables_scaled = struct_op.variables_si_to_scaled(model_variables, variables_si, model_scaling)

        parameters = plot_dict['parameters_plot']

        wake = plot_dict['wake']
        wake.draw(ax, side, variables_scaled=variables_scaled, parameters=parameters, cosmetics=cosmetics)

    return None

def compute_observer_coordinates_for_radial_distribution_in_yz_plane(plot_dict, cosmetics, idx_at_eval, kdx):

    kite_plane_induction_params = get_kite_plane_induction_params(plot_dict, idx_at_eval)

    architecture = plot_dict['architecture']
    number_of_kites = architecture.number_of_kites

    radius = kite_plane_induction_params['average_radius']
    center = kite_plane_induction_params['center']
    u_zero = kite_plane_induction_params['u_zero']

    verification_points = plot_dict['options']['model']['aero']['vortex']['verification_points']
    half_points = int(verification_points / 2.) + 1

    psi0_base = plot_dict['options']['solver']['initialization']['psi0_rad']

    mu_grid_min = kite_plane_induction_params['mu_start_by_path']
    mu_grid_max = kite_plane_induction_params['mu_end_by_path']
    psi_grid_min = psi0_base - np.pi / float(number_of_kites) + float(kdx) * 2. * np.pi / float(number_of_kites)
    psi_grid_max = psi0_base + np.pi / float(number_of_kites) + float(kdx) * 2. * np.pi / float(number_of_kites)

    # define mu with respect to kite mid-span radius
    mu_grid_points = np.linspace(mu_grid_min, mu_grid_max, verification_points, endpoint=True)
    length_mu = mu_grid_points.shape[0]

    verification_uniform_distribution = plot_dict['options']['model']['aero']['vortex']['verification_uniform_distribution']
    if verification_uniform_distribution:
        psi_grid_unscaled = np.linspace(0., 1., 2 * half_points)
    else:
        beta = np.linspace(0., np.pi / 2, half_points)
        cos_front = 0.5 * (1. - np.cos(beta))
        cos_back = -1. * cos_front[::-1]
        psi_grid_unscaled = cas.vertcat(cos_back, cos_front) + 0.5
    psi_grid_points_cas = psi_grid_unscaled * (psi_grid_max - psi_grid_min) + psi_grid_min

    psi_grid_points_np = np.array(psi_grid_points_cas)
    psi_grid_points_recenter = np.deg2rad(np.rad2deg(psi_grid_points_np))
    psi_grid_points = np.unique(psi_grid_points_recenter)

    length_psi = psi_grid_points.shape[0]

    # reserve mesh space
    y_matr = np.ones((length_psi, length_mu))
    z_matr = np.ones((length_psi, length_mu))

    for mdx in range(length_mu):
        mu_val = mu_grid_points[mdx]

        for pdx in range(length_psi):
            psi_val = psi_grid_points[pdx]

            r_val = mu_val * radius

            ehat_radial = vect_op.zhat_np() * cas.cos(psi_val) - vect_op.yhat_np() * cas.sin(psi_val)
            added = r_val * ehat_radial

            unscaled = mu_val * ehat_radial
            unscaled_x_val = float(unscaled[0])
            unscaled_y_val = float(unscaled[1])
            unscaled_z_val = float(unscaled[2])

            y_matr[pdx, mdx] = unscaled_y_val
            z_matr[pdx, mdx] = unscaled_z_val

    return y_matr, z_matr

def compute_induction_factor_at_specified_observer_coordinates(plot_dict, cosmetics, idx_at_eval, kdx, specified='radial_yz'):

    print_op.warn_about_temporary_functionality_alteration()
    #
    # local_variables = get_local_variables()
    # local_parameters = get_local_parameters()
    # stacked_x_hat_and_n_hat_sym = cas.SX.sym('x_hat_and_n_hat_sym', (6, 1))
    # local_x_hat_sym = stacked_x_hat_and_n_hat_sym[:3]
    # local_n_hat_sym = stacked_x_hat_and_n_hat_sym[-3:]
    # print(local_x_hat_sym.shape)
    # print(local_n_hat_sym.shape)
    # pdb.set_trace()
    #
    # local_projected_induction_sym = cas.DM.zeros((1,1))
    # for name, object in plot_dict['vortex_objects_dir'].items():
    #
    #     if object.model_projected_induction_fun is not None:
    #         object.make_awebox_model_induction_functions(plot_dict['variables'], plot_dict['parameters'])
    #
    #     local_projected_induction_sym += object.model_projected_induction_fun(local_variables, local_parameters, local_x_hat_sym, local_n_hat_sym)
    #
    # induction_factor_normalizing_speed = cosmetics['vortex']['induction_factor_normalizing_speed']
    # if induction_factor_normalizing_speed == 'u_zero':
    #     normalizing_speed =
    #
    # induction_factor_sym = vortex_flow.compute_induction_factor(local_projected_induction_sym, u_zero)
    #
    #
    # compute_induction_factor(u_projected, u_zero)
    #
    # # Get the name of the distribution of points
    # if specified == 'radial_yz':
    #     y_matr, z_matr = compute_observer_coordinates_for_radial_distribution_in_yz_plane(plot_dict, cosmetics,
    #                                                                                       idx_at_eval, kdx)
    # else:
    #     message = 'desired observer-coordinate-distribution (' + specified + ') is not yet available'
    #     print_op.log_and_raise_error(message)
    #
    #
    # # Calculated the points according to that distribution
    # # Reformat entries as inputs to map
    # # Compute map at points
    # # Reformat points into mesh
    #
    # # get the observer position
    #
    # x_matr, y_matr, z_matr = compute_observer_coordinates_for_radial_distribution_in_yz_plane(plot_dict, cosmetics, idx_at_eval, kdx)
    # a_matr = np.ones(x_matr.shape)
    #
    # for pdx in range(x_matr.shape[0]):
    #     for mdx in range(x_matr.shape[1]):
    #         x_val = x_matr[pdx, mdx]
    #         y_val = y_matr[pdx, mdx]
    #         z_val = z_matr[pdx, mdx]
    #
    #         x_obs = cas.vertcat(x_val, y_val, z_val)
    #
    #
    # pdb.set_trace()
    #
    # kite_plane_induction_params = get_kite_plane_induction_params(plot_dict, idx_at_eval)
    #
    # architecture = plot_dict['architecture']
    # filament_list = reconstruct_filament_list(plot_dict, idx_at_eval)
    # number_of_kites = architecture.number_of_kites
    #
    # radius = kite_plane_induction_params['average_radius']
    # center = kite_plane_induction_params['center']
    # u_zero = kite_plane_induction_params['u_zero']
    #
    # verification_points = plot_dict['options']['model']['aero']['vortex']['verification_points']
    # half_points = int(verification_points / 2.) + 1
    #
    # psi0_base = plot_dict['options']['solver']['initialization']['psi0_rad']
    #
    # mu_grid_min = kite_plane_induction_params['mu_start_by_path']
    # mu_grid_max = kite_plane_induction_params['mu_end_by_path']
    # psi_grid_min = psi0_base - np.pi / float(number_of_kites) + float(kdx) * 2. * np.pi / float(number_of_kites)
    # psi_grid_max = psi0_base + np.pi / float(number_of_kites) + float(kdx) * 2. * np.pi / float(number_of_kites)
    #
    # # define mu with respect to kite mid-span radius
    # mu_grid_points = np.linspace(mu_grid_min, mu_grid_max, verification_points, endpoint=True)
    # length_mu = mu_grid_points.shape[0]
    #
    # verification_uniform_distribution = plot_dict['options']['model']['aero']['vortex']['verification_uniform_distribution']
    # if verification_uniform_distribution:
    #     psi_grid_unscaled = np.linspace(0., 1., 2 * half_points)
    # else:
    #     beta = np.linspace(0., np.pi / 2, half_points)
    #     cos_front = 0.5 * (1. - np.cos(beta))
    #     cos_back = -1. * cos_front[::-1]
    #     psi_grid_unscaled = cas.vertcat(cos_back, cos_front) + 0.5
    # psi_grid_points_cas = psi_grid_unscaled * (psi_grid_max - psi_grid_min) + psi_grid_min
    #
    # psi_grid_points_np = np.array(psi_grid_points_cas)
    # psi_grid_points_recenter = np.deg2rad(np.rad2deg(psi_grid_points_np))
    # psi_grid_points = np.unique(psi_grid_points_recenter)
    #
    # length_psi = psi_grid_points.shape[0]
    #
    # # reserve mesh space
    # y_matr = np.ones((length_psi, length_mu))
    # z_matr = np.ones((length_psi, length_mu))
    # a_matr = np.ones((length_psi, length_mu))
    #
    # for mdx in range(length_mu):
    #     mu_val = mu_grid_points[mdx]
    #
    #     for pdx in range(length_psi):
    #         psi_val = psi_grid_points[pdx]
    #
    #         r_val = mu_val * radius
    #
    #         ehat_radial = vect_op.zhat_np() * cas.cos(psi_val) - vect_op.yhat_np() * cas.sin(psi_val)
    #         added = r_val * ehat_radial
    #         x_obs = center + added
    #
    #         unscaled = mu_val * ehat_radial
    #
    #         a_ind = float(vortex_flow.get_induction_factor_at_observer(plot_dict['options']['model'], filament_list, x_obs, u_zero, n_hat=vect_op.xhat()))
    #
    #         unscaled_y_val = float(cas.mtimes(unscaled.T, vect_op.yhat_np()))
    #         unscaled_z_val = float(cas.mtimes(unscaled.T, vect_op.zhat_np()))
    #
    #         y_matr[pdx, mdx] = unscaled_y_val
    #         z_matr[pdx, mdx] = unscaled_z_val
    #         a_matr[pdx, mdx] = a_ind
    #
    # y_list = np.array(cas.reshape(y_matr, (length_psi * length_mu, 1)))
    # z_list = np.array(cas.reshape(z_matr, (length_psi * length_mu, 1)))
    #
    # return y_matr, z_matr, a_matr, y_list, z_list
    return None

def get_kite_plane_induction_params(plot_dict, idx_at_eval):

    kite_plane_induction_params = {}
    
    layer_nodes = plot_dict['architecture'].layer_nodes
    layer = int( np.min(np.array(layer_nodes)) )
    kite_plane_induction_params['layer'] = layer
    
    b_ref = plot_dict['options']['model']['params']['geometry']['b_ref']
    average_radius = plot_dict['outputs']['performance']['average_radius' + str(layer)][0][idx_at_eval]
    kite_plane_induction_params['average_radius'] = average_radius

    center_x = plot_dict['outputs']['performance']['trajectory_center' + str(layer)][0][idx_at_eval]
    center_y = plot_dict['outputs']['performance']['trajectory_center' + str(layer)][1][idx_at_eval]
    center_z = plot_dict['outputs']['performance']['trajectory_center' + str(layer)][2][idx_at_eval]

    center = cas.vertcat(center_x, center_y, center_z)
    kite_plane_induction_params['center'] = center

    wind_model = plot_dict['options']['model']['wind']['model']
    u_ref = plot_dict['options']['user_options']['wind']['u_ref']
    z_ref = plot_dict['options']['model']['params']['wind']['z_ref']
    z0_air = plot_dict['options']['model']['params']['wind']['log_wind']['z0_air']
    exp_ref = plot_dict['options']['model']['params']['wind']['power_wind']['exp_ref']
    u_infty = wind.get_speed(wind_model, u_ref, z_ref, z0_air, exp_ref, center[2])
    kite_plane_induction_params['u_infty'] = u_infty

    u_zero_x = plot_dict['outputs']['performance']['u_zero' + str(layer)][0][idx_at_eval]
    u_zero_y = plot_dict['outputs']['performance']['u_zero' + str(layer)][1][idx_at_eval]
    u_zero_z = plot_dict['outputs']['performance']['u_zero' + str(layer)][2][idx_at_eval]
    u_zero = cas.vertcat(u_zero_x, u_zero_y, u_zero_z)
    kite_plane_induction_params['u_zero'] = u_zero

    varrho = average_radius / b_ref
    kite_plane_induction_params['varrho'] = varrho

    mu_center_by_exterior = varrho / (varrho + 0.5)
    mu_min_by_exterior = (varrho - 0.5) / (varrho + 0.5)
    mu_max_by_exterior = 1.

    mu_min_by_path = (varrho - 0.5) / varrho
    mu_max_by_path = (varrho + 0.5) / varrho
    mu_center_by_path = 1.

    mu_start_by_path = 0.1
    mu_end_by_path = 1.6

    kite_plane_induction_params['mu_center_by_exterior'] = mu_center_by_exterior
    kite_plane_induction_params['mu_min_by_exterior'] = mu_min_by_exterior
    kite_plane_induction_params['mu_max_by_exterior'] = mu_max_by_exterior
    kite_plane_induction_params['mu_min_by_path'] = mu_min_by_path
    kite_plane_induction_params['mu_max_by_path'] = mu_max_by_path
    kite_plane_induction_params['mu_center_by_path'] = mu_center_by_path
    kite_plane_induction_params['mu_start_by_path'] = mu_start_by_path
    kite_plane_induction_params['mu_end_by_path'] = mu_end_by_path

    return kite_plane_induction_params


def plot_vortex_verification(plot_dict, cosmetics, fig_name, fig_num=None):

    vortex_info_exists = determine_if_vortex_info_exists(plot_dict)
    if vortex_info_exists:

        idx_at_eval = plot_dict['options']['visualization']['cosmetics']['animation']['snapshot_index']
        number_of_kites = plot_dict['architecture'].number_of_kites
        kite_plane_induction_params = get_kite_plane_induction_params(plot_dict, idx_at_eval)
        max_axes = -100.

        mu_min_by_path = kite_plane_induction_params['mu_min_by_path']
        mu_max_by_path = kite_plane_induction_params['mu_max_by_path']

        ### points plot
        fig_points, ax_points = plt.subplots(1, 1)
        add_annulus_background(ax_points, mu_min_by_path, mu_max_by_path)
        plt.grid(True)
        plt.title('induction factors over the kite plane')
        plt.xlabel("y/r [-]")
        plt.ylabel("z/r [-]")
        ax_points.set_aspect(1.)

        #### contour plot
        fig_contour, ax_contour = plt.subplots(1, 1)
        add_annulus_background(ax_contour, mu_min_by_path, mu_max_by_path)

        levels = [-0.05, 0., 0.2]
        linestyles = ['dotted', 'solid', 'dashed']
        colors = ['k', 'k', 'k']

        emergency_levels = 5
        emergency_colors = 'k'
        emergency_linestyles = 'dashdot'

        plt.grid(True)
        plt.title('induction factors over the kite plane')
        plt.xlabel("y/r [-]")
        plt.ylabel("z/r [-]")
        ax_contour.set_aspect(1.)

        for kdx in range(number_of_kites):
            y_matr, z_matr, a_matr, y_list, z_list = compute_induction_factor_at_specified_observer_coordinates(plot_dict, cosmetics, idx_at_eval, kdx)

            max_y = np.min(y_list)
            min_y = np.min(y_list)
            max_z = np.max(z_list)
            min_z = np.min(z_list)
            max_axes = np.max(np.array([max_axes, max_y, -1. * min_y, max_z, -1. * min_z]))

            ### points plot
            ax_points.scatter(y_list, z_list, c='k')

            #### contour plot
            if (np.any(a_matr < levels[0])) and (np.any(a_matr > levels[-1])):
                cs = ax_contour.contour(y_matr, z_matr, a_matr, levels, colors=colors, linestyles=linestyles)
                # plt.clabel(cs, inline=1, fontsize=10)
                # for ldx in range(len(cs.collections)):
                #     cs.collections[ldx].set_label(levels[ldx])

            else:
                cs = ax_contour.contour(y_matr, z_matr, a_matr, emergency_levels, colors=emergency_colors, linestyles=emergency_linestyles)
                ax_contour.clabel(cs, inline=True, fontsize=10)

            # plt.legend(loc='lower right')

        ax_points.set_xlim([-1. * max_axes, max_axes])
        ax_points.set_ylim([-1. * max_axes, max_axes])
        fig_points.savefig('points.pdf')

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
