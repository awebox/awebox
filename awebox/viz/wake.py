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
import copy

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import pdb

import casadi.tools as cas
import numpy as np
from awebox.logger.logger import Logger as awelogger

import awebox.mdl.wind as wind
import awebox.mdl.aero.induction_dir.actuator_dir.actuator as actuator
import awebox.mdl.aero.induction_dir.general_dir.flow as general_flow

import awebox.viz.tools as tools

import awebox.tools.vector_operations as vect_op
import awebox.tools.struct_operations as struct_op
import awebox.tools.print_operations as print_op

def plot_actuator(plot_dict, cosmetics, fig_name, side):

    fig, ax = tools.setup_axes_for_side(cosmetics, side)

    index = -1
    draw_actuator(ax, side, plot_dict, cosmetics, index)

    if cosmetics['trajectory']['kite_bodies']:
        init_colors = False
        tools.draw_all_kites(ax, plot_dict, index, cosmetics, side, init_colors)

    if cosmetics['trajectory']['kite_aero_dcm']:
        tools.draw_kite_aero_dcm(ax, side, plot_dict, cosmetics, index)

    ax.tick_params(labelsize=cosmetics['trajectory']['ylabelsize'])
    plt.suptitle(fig_name)

    return None

def draw_actuator(ax, side, plot_dict, cosmetics, index):
    actuator.draw_actuator(ax, side, plot_dict, cosmetics, index)
    return None

def plot_wake(plot_dict, cosmetics, fig_name, side):
    fig, ax = tools.setup_axes_for_side(cosmetics, side)

    index = -1
    draw_wake_nodes(ax, side, plot_dict, cosmetics, index)

    strength_max = cosmetics['trajectory']['circulation_max_estimate']
    strength_min = -1. * strength_max

    norm = plt.Normalize(strength_min, strength_max)
    cmap = plt.get_cmap('seismic')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = ax.figure.colorbar(sm)
    cbar.set_label('vortex filament strength [m$^2$/s]', rotation=270)

    if cosmetics['trajectory']['kite_bodies']:
        init_colors = False
        tools.draw_all_kites(ax, plot_dict, index, cosmetics, side, init_colors)

    ax.tick_params(labelsize=cosmetics['trajectory']['ylabelsize'])
    plt.suptitle(fig_name)

    return None


def get_variables_scaled(plot_dict, cosmetics, index):
    model_variables = plot_dict['model_variables']
    model_scaling = model_variables(plot_dict['model_scaling'])

    if cosmetics['variables']['si_or_scaled'] == 'scaled':
        variables_scaled = tools.assemble_variable_slice_from_interpolated_data(plot_dict, index)
    else:
        variables_si = tools.assemble_variable_slice_from_interpolated_data(plot_dict, index)
        variables_scaled = struct_op.variables_si_to_scaled(model_variables, variables_si, model_scaling)

    return variables_scaled


def draw_wake_nodes(ax, side, plot_dict, cosmetics, index):

    if ('wake' in plot_dict.keys()) and (plot_dict['wake'] is not None):

        variables_scaled = get_variables_scaled(plot_dict, cosmetics, index)
        parameters = plot_dict['parameters_plot']
        wake = plot_dict['wake']

        wake.draw(ax, side, variables_scaled=variables_scaled, parameters=parameters, cosmetics=cosmetics)

    return None


def compute_observer_coordinates_for_radial_distribution_in_yz_plane(plot_dict, idx_at_eval, kdx):

    kite_plane_induction_params = get_kite_plane_induction_params(plot_dict, idx_at_eval)

    architecture = plot_dict['architecture']
    number_of_kites = architecture.number_of_kites

    radius = kite_plane_induction_params['average_radius']
    center = kite_plane_induction_params['center']

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
    dimensionless_coordinates = {}
    dimensioned_coordinates = {}
    index_to_dimensions = get_index_to_dimensions_dict()
    for idx, dim in index_to_dimensions.items():
        dimensionless_coordinates[dim] = np.ones((length_psi, length_mu))
        dimensioned_coordinates[dim] = np.ones((length_psi, length_mu))

    n_hat, a_hat, b_hat = get_coordinate_axes_for_haas_verification()

    for mdx in range(length_mu):
        mu_val = mu_grid_points[mdx]

        for pdx in range(length_psi):
            psi_val = psi_grid_points[pdx]

            ehat_radial = a_hat * cas.cos(psi_val) + b_hat * cas.sin(psi_val)

            dimensionless = mu_val * ehat_radial
            dimensioned = center + radius * dimensionless

            for idx, dim in index_to_dimensions.items():
                dimensionless_coordinates[dim][pdx, mdx] = float(dimensionless[idx])
                dimensioned_coordinates[dim][pdx, mdx] = float(dimensioned[idx])

    return dimensionless_coordinates, dimensioned_coordinates


def get_index_to_dimensions_dict():
    index_to_dimensions = {0: 'x', 1: 'y', 2: 'z'}
    return index_to_dimensions


def get_distributed_coordinates_about_actuator_center(plot_dict, kdx, orientation='radial', plane='yz', idx_at_eval=0):
    if (orientation == 'radial') and (plane == 'yz'):
        dimensionless_coordinates, dimensioned_coordinates = compute_observer_coordinates_for_radial_distribution_in_yz_plane(
            plot_dict, idx_at_eval, kdx)

    else:
        message = 'only radial and yz options are currently available via get_distributed_coordinates_about_actuator_center function'
        print_op.log_and_raise_error(message)

    return dimensionless_coordinates, dimensioned_coordinates


def get_coordinate_axes_for_haas_verification():
    n_hat = vect_op.xhat_dm()
    a_hat = vect_op.zhat_dm()
    print_op.warn_about_temporary_functionality_alteration()
    # b_hat = -1. * vect_op.yhat_dm()
    b_hat = vect_op.yhat_dm()
    return n_hat, a_hat, b_hat


def compute_induction_factor_at_specified_observer_coordinates(plot_dict, cosmetics, kdx, orientation='radial', plane='yz', idx_at_eval=0):

    variables_scaled = get_variables_scaled(plot_dict, cosmetics, idx_at_eval)
    parameters = plot_dict['parameters_plot']
    wake = plot_dict['wake']

    kite_plane_induction_params = get_kite_plane_induction_params(plot_dict, idx_at_eval)

    dimensionless_coordinates, dimensioned_coordinates = get_distributed_coordinates_about_actuator_center(plot_dict, kdx, orientation, plane, idx_at_eval)

    x_obs_sym = cas.SX.sym('x_obs_sym', (3, 1))
    u_ind_sym = wake.calculate_total_biot_savart_residual_at_x_obs(variables_scaled, parameters, x_obs=x_obs_sym)

    n_hat, _, _ = get_coordinate_axes_for_haas_verification()

    model_options = plot_dict['options']['model']
    induction_factor_normalizing_speed = model_options['aero']['vortex']['induction_factor_normalizing_speed']
    if induction_factor_normalizing_speed == 'u_zero':
        u_normalizing = kite_plane_induction_params['u_zero']
    else:
        message = 'computing induction factor at specific points is not yet defined for normalizing speed ' + induction_factor_normalizing_speed
        print_op.log_and_raise_error(message)

    a_sym = general_flow.compute_induction_factor(u_ind_sym, n_hat, u_normalizing)
    a_fun = cas.Function('a_fun', [x_obs_sym], [a_sym])

    index_to_dimensions = get_index_to_dimensions_dict()
    x_obs_dimensioned_stacked = []
    for pdx in range(dimensioned_coordinates['y'].shape[0]):
        for mdx in range(dimensioned_coordinates['y'].shape[1]):
            local_x_obs_dimensioned = cas.DM.ones((3, 1))
            for idx, dim in index_to_dimensions.items():
                local_x_obs_dimensioned[idx] = dimensioned_coordinates[dim][pdx, mdx]
            x_obs_dimensioned_stacked = cas.horzcat(x_obs_dimensioned_stacked, local_x_obs_dimensioned)

    a_map = a_fun.map(x_obs_dimensioned_stacked.shape[1], 'openmp')
    all_a = a_map(x_obs_dimensioned_stacked)
    a_matr = cas.reshape(all_a, dimensioned_coordinates['y'].shape)

    y_matr = dimensionless_coordinates['y']
    z_matr = dimensionless_coordinates['z']

    return y_matr, z_matr, a_matr


def get_kite_plane_induction_params(plot_dict, idx_at_eval):

    kite_plane_induction_params = {}
    
    layer_nodes = plot_dict['architecture'].layer_nodes
    layer = int( np.min(np.array(layer_nodes)) )
    kite_plane_induction_params['layer'] = layer
    
    b_ref = plot_dict['options']['model']['params']['geometry']['b_ref']
    average_radius = plot_dict['outputs']['geometry']['average_radius' + str(layer)][0][idx_at_eval]
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

    vec_u_zero = []
    for dim in range(3):
        local_u_zero = plot_dict['outputs']['geometry']['vec_u_zero' + str(layer)][0][idx_at_eval]
        vec_u_zero = cas.vertcat(vec_u_zero, local_u_zero)
    kite_plane_induction_params['vec_u_zero'] = vec_u_zero
    kite_plane_induction_params['u_zero'] = vect_op.norm(vec_u_zero)

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


def plot_haas_verification_test(plot_dict, cosmetics, fig_name, fig_num=None):

    vortex_info_exists = (plot_dict['wake'] is not None)
    if vortex_info_exists:

        orientation = 'radial'
        plane = 'yz'

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
        plt.title('induction factors over the kite plane:\n sample points')
        plt.xlabel("y/r [-]")
        plt.ylabel("z/r [-]")
        ax_points.set_aspect(1.)

        ### draw in vortices
        if ('wake' in plot_dict.keys()) and (plot_dict['wake'] is not None):
            variables_scaled = get_variables_scaled(plot_dict, cosmetics, idx_at_eval)
            parameters = plot_dict['parameters_plot']
            wake = plot_dict['wake']
            bound_wake = wake.get_substructure('bound')
            finite_filament_list = bound_wake.get_list('finite_filament')

            center = kite_plane_induction_params['center']
            average_radius = kite_plane_induction_params['average_radius']
            for elem in finite_filament_list.list:
                unpacked = elem.unpack_info(elem.evaluate_info(variables_scaled, parameters))
                x_start = unpacked['x_start']
                x_end = unpacked['x_end']

                x_start_shifted = (x_start - center) / average_radius
                x_end_shifted = (x_end - center) / average_radius

                y_over_r_vals = [float(x_start_shifted[1]), float(x_end_shifted[1])]
                z_over_r_vals = [float(x_start_shifted[2]), float(x_end_shifted[2])]
                plt.plot(y_over_r_vals, z_over_r_vals)

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
            y_matr, z_matr, a_matr = compute_induction_factor_at_specified_observer_coordinates(plot_dict, cosmetics, kdx, orientation, plane, idx_at_eval)

            y_list = np.array(vect_op.columnize(y_matr))
            z_list = np.array(vect_op.columnize(z_matr))

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
