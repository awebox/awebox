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
import pdb
from platform import architecture

import matplotlib
from pandas.core.ops import unpack_zerodim_and_defer

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import pdb

import casadi.tools as cas
import numpy as np
from awebox.logger.logger import Logger as awelogger

import awebox.mdl.wind as wind
import awebox.mdl.aero.induction_dir.actuator_dir.actuator as actuator
import awebox.mdl.aero.induction_dir.general_dir.flow as general_flow
import awebox.mdl.aero.indicators as aero_indicators

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
    if cosmetics['trajectory']['trajectory_rotation_dcm']:
        tools.draw_trajectory_rotation_dcm(ax, side, plot_dict, cosmetics, index)

    ax.tick_params(labelsize=cosmetics['trajectory']['ylabelsize'])
    plt.suptitle(fig_name)

    return None

def draw_actuator(ax, side, plot_dict, cosmetics, index):
    actuator.draw_actuator(ax, side, plot_dict, cosmetics, index)
    return None

def plot_wake_legend(plot_dict, cosmetics, fig_name):
    fig, ax = plt.subplots(figsize=(6, 1), layout='constrained')

    strength_max = cosmetics['trajectory']['circulation_max_estimate']
    strength_min = -1. * strength_max

    norm = plt.Normalize(strength_min, strength_max)
    cmap = plt.get_cmap('seismic')
    label = 'vortex filament strength [m$^2$/s]'

    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                 cax=ax, orientation='horizontal', label=label)

    return None


def plot_wake(plot_dict, cosmetics, fig_name, side, ref=False):

    fig, ax = tools.setup_axes_for_side(cosmetics, side)

    all_points = {0: [], 1: [], 2: []}
    list_of_x_position_variable_names = []
    list_of_z_position_variable_names = []
    for kite in plot_dict['architecture'].kite_nodes:
        parent = plot_dict['architecture'].parent_map[kite]
        list_of_x_position_variable_names += ['q' + str(kite) + str(parent)]
    for var_name in plot_dict['variables_dict']['z'].keys():
        if var_name[0:2] == 'wx':
            list_of_z_position_variable_names += [var_name]

    for position_var_name in list_of_x_position_variable_names:
        for dim in range(3):
            all_points[dim] = cas.vertcat(all_points[dim], plot_dict['interpolation_si']['x'][position_var_name][dim])
    for position_var_name in list_of_z_position_variable_names:
        for dim in range(3):
            all_points[dim] = cas.vertcat(all_points[dim], plot_dict['interpolation_si']['z'][position_var_name][dim])

    for dim in range(3):
        all_points[dim] = vect_op.columnize(all_points[dim])

    b_ref = plot_dict['options']['model']['params']['geometry']['b_ref']
    range_extension = 1. * b_ref
    max_vals = {}
    min_vals = {}
    range_vals = {}
    center_vals = {}
    for dim in range(3):
        max_vals[dim] = np.max(np.array(all_points[dim]))
        min_vals[dim] = np.min(np.array(all_points[dim]))
        range_vals[dim] = max_vals[dim] - min_vals[dim]
        center_vals[dim] = 0.5 * (max_vals[dim] + min_vals[dim])

    if len(side) == 2:
        if side == 'xy':
            adx = 0
            bdx = 1
        elif side == 'xz':
            adx = 0
            bdx = 2
        elif side == 'yz':
            adx = 1
            bdx = 2
        range_extension = 2. * b_ref
        larger_range = np.max(np.array([range_vals[adx], range_vals[bdx]])) + range_extension
        ax.set_xlim(center_vals[adx] - larger_range / 2., center_vals[adx] + larger_range / 2.)
        ax.set_ylim(center_vals[bdx] - larger_range / 2., center_vals[bdx] + larger_range / 2.)
        ax.set_aspect('equal', adjustable='box')

        strength_max = cosmetics['trajectory']['circulation_max_estimate']
        strength_min = -1. * strength_max
        norm = plt.Normalize(strength_min, strength_max)
        cmap = plt.get_cmap('seismic')
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = ax.figure.colorbar(sm, ax=plt.gca())
        cbar.set_label('vortex filament strength [m$^2$/s]', rotation=270)

    elif side == 'isometric':
        larger_range = np.max(np.array([val for val in range_vals.values()])) + range_extension
        ax.set_xlim(center_vals[0] - larger_range / 2., center_vals[0] + larger_range / 2.)
        ax.set_ylim(center_vals[1] - larger_range / 2., center_vals[1] + larger_range / 2.)
        ax.set_zlim(center_vals[2] - larger_range / 2., center_vals[2] + larger_range / 2.)

    index = -1
    draw_wake_nodes(ax, side, plot_dict, cosmetics, index)

    if cosmetics['trajectory']['kite_bodies']:
        init_colors = False
        tools.draw_all_kites(ax, plot_dict, index, cosmetics, side, init_colors)

    ax.tick_params(labelsize=cosmetics['trajectory']['ylabelsize'])
    plt.suptitle(fig_name)
    ax.margins(0.)
    plt.tight_layout()

    return None


def get_variables_scaled(plot_dict, cosmetics, index):
    variables_scaled = tools.assemble_variable_slice_from_interpolated_data(plot_dict, index, si_or_scaled='scaled')
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

    n_hat, a_hat, b_hat = get_coordinate_axes_for_haas_verification(plot_dict, idx_at_eval)

    for mdx in range(length_mu):
        mu_val = mu_grid_points[mdx]

        for pdx in range(length_psi):
            psi_val = psi_grid_points[pdx]

            ehat_radial = b_hat * cas.cos(psi_val) + a_hat * cas.sin(psi_val)

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


def get_coordinate_axes_for_haas_verification(plot_dict, idx_at_eval):

    architecture = plot_dict['architecture']
    top_parent = architecture.parent_map[architecture.number_of_nodes - 1]

    n_hat = []
    for dim in range(3):
        n_hat = cas.vertcat(n_hat,
                                 plot_dict['interpolation_si']['outputs']['rotation']['ehat_normal' + str(top_parent)][dim][idx_at_eval])

    b_hat_temp = vect_op.zhat_dm()
    a_hat = vect_op.normed_cross(b_hat_temp, n_hat)
    b_hat = vect_op.normed_cross(n_hat, a_hat)

    # n_hat = vect_op.xhat_dm()
    # a_hat = vect_op.yhat_dm()
    # b_hat = vect_op.zhat_dm()

    return n_hat, a_hat, b_hat

def plot_velocity_distribution_comparison_only(plot_dict, cosmetics, fig_name):

    comparison_data_for_velocity_distribution = plot_dict['options']['visualization']['cosmetics']['induction']['comparison_data_for_velocity_distribution']

    if comparison_data_for_velocity_distribution == 'not_in_use':
        message = 'request to produce a velocity-distribution-comparison plot is being ignored because no comparison velocity distribution was input'
        print_op.base_print(message, level='info')
        return None

    add_rancourt_comparison_to_velocity_distribution = (comparison_data_for_velocity_distribution == 'rancourt')
    add_trevisi_comparison_to_velocity_distribution = (comparison_data_for_velocity_distribution == 'trevisi')

    all_xi = np.linspace(-0.5, 0.5, 500)
    xi_cas = cas.DM(all_xi).reshape((1, len(all_xi)))

    if add_rancourt_comparison_to_velocity_distribution:
        rancourt_dict = get_rancourt_velocity_distribution(all_xi)

    if add_trevisi_comparison_to_velocity_distribution:
        trevisi_a_normal_dict, trevisi_a_radial_dict, other_trevisi_dict = get_trevisi_induction_factor_distributions(all_xi)

    adx = {}

    if add_rancourt_comparison_to_velocity_distribution:
        adx['radius'] = 0
        adx['cl'] = adx['radius'] + 1
        adx['lift_per_unit_span'] = adx['cl'] + 1
        adx['app'] = adx['lift_per_unit_span'] + 1
        adx['eff'] = adx['app']
        adx['norm_minus'] = adx['eff'] + 1

    if add_trevisi_comparison_to_velocity_distribution:
        adx['a_n'] = 0
        adx['a_r'] = adx['a_n'] + 1

    rows = 1 + int(np.max(np.array([val for val in adx.values()])))
    cols = 1
    plot_height = 3
    plot_width = 2 * plot_height
    fig, ax = plt.subplots(rows, cols, squeeze=False, sharex=True, figsize=(plot_width, rows * plot_height))
    fig.suptitle('spanwise velocity distribution')

    if add_trevisi_comparison_to_velocity_distribution:
        # for label, values in other_trevisi_dict.items():
        #     ax[adx[label]][0].plot(values['xi'], values['vals'], linestyle='--', label=label + ' Trevisi 2023')

        for label, values in trevisi_a_normal_dict.items():
            ax[adx['a_n']][0].plot(values['short_xi'], values['short_vals'], label=label)
        for label, values in trevisi_a_radial_dict.items():
            ax[adx['a_r']][0].plot(values['short_xi'], values['short_vals'], label=label)

        ax[adx['a_r']][0].set_ylabel('radial\n induction factor [-]')
        ax[adx['a_n']][0].set_ylabel('normal\n induction factor [-]')


    if add_rancourt_comparison_to_velocity_distribution:
        ax[adx['cl']][0].set_ylabel('2D lift\n coefficient \n [-]')
        ax[adx['lift_per_unit_span']][0].set_ylabel('lift per unit\nspan [N/m]')
        ax[adx['radius']][0].set_ylabel('radial distance\n from center of\n rotation [m]')
        ax[adx['radius']][0].set_ylabel('radial distance\n from center of\n rotation [m]')
        ax[adx['eff']][0].set_ylabel('inflow [m/s]')
        ax[adx['norm_minus']][0].set_ylabel('difference in inflow [m/s]')

        ax[adx['norm_minus']][0].set_ylim(-1, 1)

        for type in ['app', 'eff', 'norm_minus', 'radius', 'cl', 'lift_per_unit_span']:
            extra_label = ''
            if type in ['app', 'eff']:
                extra_label = '||u_' + type + '||'
            elif type == 'norm_minus':
                extra_label = '||u_eff|| - ||u_app||'
            if extra_label == '':
                ax[adx[type]][0].plot(rancourt_dict[type]['short_xi'], rancourt_dict[type]['short_vals'])
            else:
                ax[adx[type]][0].plot(rancourt_dict[type]['short_xi'], rancourt_dict[type]['short_vals'], label= extra_label)

    ax[-1][0].set_xlabel('non-dimensional spanwise position [-]')
    ax[-1][0].set_xticks(np.linspace(-0.5, 0.5, 5))

    shrink_factor = 0.3
    for adx_val in range(rows):
        ax[adx_val][0].grid(True)  # Add a grid for better readability
        # Put a legend to the right of the current axis
        ax[adx_val][0].legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    plt.tight_layout()
    # Adjust the subplots with the new bottom values, leaving space for the legend
    plt.subplots_adjust(right=1.-shrink_factor)

    return None


def plot_velocity_distribution(plot_dict, cosmetics, fig_name):
    idx_at_eval = plot_dict['options']['visualization']['cosmetics']['animation']['snapshot_index']
    parallelization_type = plot_dict['options']['model']['construction']['parallelization']['type']
    architecture = plot_dict['architecture']

    comparison_data_for_velocity_distribution = plot_dict['options']['visualization']['cosmetics']['induction']['comparison_data_for_velocity_distribution']
    add_rancourt_comparison_to_velocity_distribution = (comparison_data_for_velocity_distribution == 'rancourt')
    add_trevisi_comparison_to_velocity_distribution = (comparison_data_for_velocity_distribution == 'trevisi')

    all_xi = np.linspace(-0.5, 0.5, 500)
    xi_cas = cas.DM(all_xi).reshape((1, len(all_xi)))

    if add_rancourt_comparison_to_velocity_distribution:
        rancourt_dict = get_rancourt_velocity_distribution(all_xi)

    if add_trevisi_comparison_to_velocity_distribution:
        trevisi_a_normal_dict, trevisi_a_radial_dict, other_trevisi_dict = get_trevisi_induction_factor_distributions(all_xi)

    adx = {}
    adx['radius'] = 0
    adx['app'] = adx['radius'] + 1
    adx['eff'] = adx['app']
    if add_rancourt_comparison_to_velocity_distribution:
        adx['norm_minus'] = adx['eff'] + 1
        adx['a_n'] = adx['norm_minus'] + 1
    else:
        adx['a_n'] = adx['eff'] + 1
    adx['a_r'] = adx['a_n'] + 1
    adx['a_t'] = adx['a_r'] + 1
    adx['alpha_app'] = adx['a_t'] + 1
    adx['alpha_eff'] = adx['alpha_app'] + 1

    rows = 1 + int(np.max(np.array([val for val in adx.values()])))
    cols = 1
    plot_height = 2
    plot_width = 4 * plot_height
    fig, ax = plt.subplots(rows, cols, squeeze=False, sharex=True, figsize=(plot_width, rows * plot_height))
    fig.suptitle('spanwise velocity distribution')

    for kdx in range(len(architecture.kite_nodes)):
        kite = architecture.kite_nodes[kdx]

        fun_dict = get_velocity_distribution_at_spanwise_position_functions(plot_dict, cosmetics, kite, idx_at_eval=idx_at_eval)

        parallelization_type = plot_dict['options']['model']['construction']['parallelization']['type']
        all_dict = {}
        for name, local_fun in fun_dict.items():
            if parallelization_type in ['serial', 'openmp', 'thread']:
                local_map = local_fun.map(all_xi.shape[0], parallelization_type)
                local_all = local_map(all_xi)
            elif parallelization_type == 'concurrent_futures':
                local_all = np.array(struct_op.concurrent_future_map(local_fun, xi_cas)).reshape(all_xi.shape)
            else:
                message = 'sorry, but the awebox has not yet set up ' + parallelization_type + ' parallelization'
                print_op.log_and_raise_error(message)
            all_dict[name] = np.array(local_all).reshape(all_xi.shape)

        if kite > 1:
            kite_label = ', on kite ' + str(kite)
        else:
            kite_label = ''

        for name in adx.keys():
            extra_label = name
            if name in ['app', 'eff']:
                extra_label = '||u_' + name + '||'
            elif name == 'norm_minus':
                extra_label = '||u_eff|| - ||u_app||'

            ax[adx[name]][0].plot(all_xi, all_dict[name], label=extra_label + kite_label)

    if add_trevisi_comparison_to_velocity_distribution:

        for label, values in other_trevisi_dict.items():
            if label in ['app', 'eff']:
                extra_label = '||u_' + label + '||'
            else:
                extra_label = label
            ax[adx[label]][0].plot(values['xi'], values['vals'], linestyle='--', label=extra_label + ' Trevisi 2023')

        for label, values in trevisi_a_normal_dict.items():
            ax[adx['a_n']][0].plot(values['short_xi'], values['short_vals'], linestyle='--', label=label)
        for label, values in trevisi_a_radial_dict.items():
            ax[adx['a_r']][0].plot(values['short_xi'], values['short_vals'], linestyle='--', label=label)

    ax[adx['radius']][0].set_ylabel('radial distance\n from center of\n rotation [m]')
    ax[adx['eff']][0].set_ylabel('inflow [m/s]')
    ax[adx['a_r']][0].set_ylabel('radial\n induction factor [-]')
    ax[adx['a_t']][0].set_ylabel('tangential\n induction factor [-]')
    ax[adx['a_n']][0].set_ylabel('normal\n induction factor [-]')

    if add_rancourt_comparison_to_velocity_distribution:
        ax[adx['norm_minus']][0].set_ylabel('change in inflow [m/s]')
        ax[adx['norm_minus']][0].set_ylim(-1, 1)

    if add_rancourt_comparison_to_velocity_distribution:
        for type in ['app', 'eff', 'norm_minus', 'radius']:
            extra_label = ''
            if type in ['app', 'eff']:
                extra_label = '||u_' + type + '|| '
            elif type == 'norm_minus':
                extra_label = '||u_eff|| - ||u_app|| '
            ax[adx[type]][0].plot(rancourt_dict[type]['short_xi'], rancourt_dict[type]['short_vals'], linestyle='--', label= extra_label + 'Rancourt 2018')

    ax[-1][0].set_xlabel('non-dimensional spanwise position [-]')
    ax[-1][0].set_xticks(np.linspace(-0.5, 0.5, 5))

    shrink_factor = 0.3
    for adx_val in range(rows):
        ax[adx_val][0].grid(True)  # Add a grid for better readability
        # Put a legend to the right of the current axis
        ax[adx_val][0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    # Adjust the subplots with the new bottom values, leaving space for the legend
    plt.subplots_adjust(right=1.-shrink_factor)

    return None


def get_rancourt_velocity_distribution(all_xi):

    # Efficient Aerodynamic Method for Interacting Lifting Surfaces over Long Distances
    # David Rancourt and Dimitri N. Mavris
    # Journal of Aircraft 2018 55:6, 2466-2475

    rancourt_b_ref = 8. #m
    rancourt_c_ref = 1. #m
    rancourt_rho_air = 1.225 #kg/m^3
    rancourt_radius = 24. #m
    rancourt_groundspeed = 30. #m/s
    rancourt_uinfty = 5. #m/s
    rancourt_omega = 2. * np.pi / (2. * np.pi * rancourt_radius / rancourt_groundspeed)

    xi_data_lift = [-0.49595875, -0.49450375, -0.49304875, -0.491595, -0.49014, -0.488685, -0.48723, -0.485545,
                    -0.483735, -0.481925, -0.4801275, -0.47834375, -0.47656125, -0.47461125, -0.4720675, -0.46952375,
                    -0.46695375, -0.46378125, -0.46060875, -0.45743625, -0.45406375, -0.45063, -0.447195, -0.44323625,
                    -0.438885, -0.434535, -0.4300075, -0.4252025, -0.4203975, -0.41542875, -0.4094875, -0.40354625,
                    -0.39760625, -0.3907675, -0.3838675, -0.37696875, -0.36946, -0.361755, -0.35405, -0.34582375,
                    -0.337145, -0.32846625, -0.319645, -0.31045125, -0.3012575, -0.29202, -0.28249375, -0.2729675,
                    -0.2634425, -0.254355, -0.2453375, -0.23632125, -0.226415, -0.21605, -0.20568625, -0.19492,
                    -0.18372875, -0.17253625, -0.16129375, -0.14988375, -0.138475, -0.12706625, -0.1155325, -0.10399875,
                    -0.09246375, -0.08091875, -0.06937125, -0.05782375, -0.04629125, -0.034765, -0.02324, -0.0117325,
                    -0.00025375, 0.011225, 0.0227, 0.03415625, 0.04561125, 0.05706625, 0.06836875, 0.07967125,
                    0.0909725, 0.10225, 0.1135225, 0.124795, 0.1360125, 0.14719375, 0.158375, 0.16951875, 0.18060875,
                    0.19169875, 0.20277625, 0.21380125, 0.22482625, 0.23585125, 0.24702875, 0.25821, 0.26939,
                    0.27958375, 0.28948625, 0.2993875, 0.30879, 0.31780625, 0.32682375, 0.3354975, 0.34357, 0.35164375,
                    0.35951125, 0.3666375, 0.37376375, 0.38089, 0.386895, 0.39287125, 0.39884875, 0.404825, 0.4106575,
                    0.416035, 0.42141375, 0.42676125, 0.43073625, 0.43471125, 0.43868625, 0.44244, 0.4461475, 0.449855,
                    0.45333125, 0.456715, 0.46009875, 0.4632075, 0.46600875, 0.46881]
    lift_data_lift = [73.522, 78.039, 82.556, 87.073, 91.589, 96.106, 100.623, 105.127, 109.624, 114.12, 118.618,
                      123.117, 127.615, 132.101, 136.543, 140.985, 145.424, 149.802, 154.181, 158.56, 162.915, 167.263,
                      171.611, 175.885, 180.103, 184.322, 188.511, 192.653, 196.795, 200.902, 204.809, 208.715, 212.621,
                      216.291, 219.945, 223.598, 227.056, 230.45, 233.844, 237.031, 240.038, 243.045, 245.983, 248.742,
                      251.501, 254.235, 256.812, 259.39, 261.967, 264.778, 267.626, 270.474, 272.77, 274.782, 276.793,
                      278.376, 279.501, 280.627, 281.655, 282.363, 283.071, 283.778, 284.002, 284.224, 284.446, 284.485,
                      284.49, 284.496, 284.321, 284.047, 283.773, 283.415, 282.925, 282.436, 281.933, 281.364, 280.796,
                      280.228, 279.302, 278.373, 277.444, 276.471, 275.488, 274.506, 273.434, 272.301, 271.169, 269.984,
                      268.72, 267.456, 266.177, 264.829, 263.48, 262.131, 260.993, 259.86, 258.728, 256.665, 254.327,
                      251.99, 249.368, 246.529, 243.689, 240.7, 237.448, 234.197, 230.875, 227.296, 223.717, 220.139,
                      216.254, 212.361, 208.469, 204.577, 200.652, 196.626, 192.6, 188.568, 184.296, 180.024, 175.752,
                      171.448, 167.139, 162.829, 158.49, 154.139, 149.788, 145.407, 140.992, 136.576]

    lift_per_unit_span_interpolated = vect_op.spline_interpolation(xi_data_lift, lift_data_lift, all_xi)

    xi_data_CL = [-0.49411625, -0.49272625, -0.49133625, -0.48969375, -0.4877825, -0.48583125, -0.48312375, -0.48041625,
                  -0.47718375, -0.47338, -0.46954125, -0.4649275, -0.46031375, -0.4554975, -0.45050625, -0.44551625,
                  -0.4394475, -0.4333625, -0.42684, -0.41990625, -0.412965, -0.40552375, -0.39808125, -0.390255,
                  -0.3820875, -0.37392, -0.365295, -0.35666375, -0.3476825, -0.33844625, -0.32920875, -0.3199725,
                  -0.310735, -0.30137, -0.2919125, -0.28243375, -0.2728275, -0.26322125, -0.25408875, -0.24525625,
                  -0.2362975, -0.22648125, -0.21666625, -0.20651625, -0.1961925, -0.18585125, -0.17543875, -0.1650275,
                  -0.15450375, -0.14391875, -0.1333125, -0.122565, -0.11181875, -0.10097, -0.09007, -0.07916,
                  -0.0682125, -0.057265, -0.04622625, -0.03514875, -0.02408, -0.01304125, -0.00200125, 0.0090675,
                  0.020145, 0.0312575, 0.04245125, 0.053645, 0.0648375, 0.07603125, 0.08725375, 0.0985275, 0.1098025,
                  0.1210975, 0.1323925, 0.14373375, 0.15511375, 0.1664925, 0.1778725, 0.1892525, 0.2006325, 0.21201125,
                  0.22339125, 0.23477125, 0.24615125, 0.25753125, 0.26891, 0.28033375, 0.29182625, 0.30331875, 0.31436,
                  0.32539, 0.33595125, 0.34616, 0.35617125, 0.36538875, 0.37460625, 0.382855, 0.39069125, 0.39852875,
                  0.406365, 0.41255375, 0.4185175, 0.4244825, 0.43041375, 0.434835, 0.43925625, 0.4436775, 0.44809875,
                  0.45252, 0.45564875, 0.458495, 0.4613425, 0.46418875, 0.46703625, 0.4698825, 0.47232125, 0.47446,
                  0.4765975, 0.478735, 0.48087375, 0.48301125, 0.4848925, 0.486415, 0.4879375, 0.48946125, 0.49098375,
                  0.49237625, 0.49314125, 0.493905]
    CL_data_CL = [0.08078, 0.09192, 0.10306, 0.11416, 0.12522, 0.13628, 0.14719, 0.1581, 0.16885, 0.17945, 0.19003,
                  0.20031, 0.2106, 0.22079, 0.23091, 0.24102, 0.25057, 0.2601, 0.26934, 0.27831, 0.28727, 0.29585,
                  0.30442, 0.31265, 0.32057, 0.3285, 0.33595, 0.34339, 0.35042, 0.35714, 0.36387, 0.37059, 0.37731,
                  0.38386, 0.39028, 0.39668, 0.40289, 0.4091, 0.41593, 0.42315, 0.4302, 0.43609, 0.44199, 0.4473,
                  0.4523, 0.45727, 0.4621, 0.46693, 0.47152, 0.47598, 0.48039, 0.48447, 0.48854, 0.49235, 0.49602,
                  0.49967, 0.5032, 0.50674, 0.50999, 0.51312, 0.51627, 0.51952, 0.52278, 0.52593, 0.52906, 0.53206,
                  0.53477, 0.53748, 0.54019, 0.5429, 0.54549, 0.54786, 0.55023, 0.55251, 0.55478, 0.55681, 0.55865,
                  0.56048, 0.56231, 0.56414, 0.56597, 0.5678, 0.56964, 0.57147, 0.5733, 0.57513, 0.57696, 0.57773,
                  0.57678, 0.57584, 0.57262, 0.56934, 0.56495, 0.55973, 0.55421, 0.54747, 0.54072, 0.53294, 0.52471,
                  0.51648, 0.50825, 0.49881, 0.48921, 0.4796, 0.46999, 0.45962, 0.44926, 0.4389, 0.42854, 0.41817,
                  0.40739, 0.39652, 0.38565, 0.37478, 0.3639, 0.35303, 0.34207, 0.33105, 0.32002, 0.309, 0.29797,
                  0.28695, 0.27588, 0.26476, 0.25364, 0.24252, 0.23139, 0.22026, 0.20906, 0.19787]
    cl_interpolated = vect_op.spline_interpolation(xi_data_CL, CL_data_CL, all_xi)

    # lift_per_unit_span = cl (1/2 rho norm(u_eff)^2) chord
    # 2 lpus / (chord cl rho) = norm(u_eff)^2
    # norm(u_eff) = ( 2 lpus / chord cl rho)**0.5

    radius_list = []
    norm_u_app_list = []
    norm_u_eff_list = []
    norm_dq_kite_list = []
    for idx in range(len(all_xi)):

        xi  = all_xi[idx]
        local_radius = rancourt_radius - xi * rancourt_b_ref
        radius_list = cas.vertcat(radius_list, local_radius)

        norm_dq_kite = local_radius * rancourt_omega
        norm_dq_kite_list = cas.vertcat(norm_dq_kite_list, norm_dq_kite)

        u_eff_squared = (2. * lift_per_unit_span_interpolated[idx]) / (rancourt_c_ref * cl_interpolated[idx] * rancourt_rho_air)
        norm_u_eff = u_eff_squared**0.5
        norm_u_eff_list = cas.vertcat(norm_u_eff_list, norm_u_eff)

        norm_u_app = (norm_dq_kite**2. + rancourt_uinfty**2.)**0.5
        norm_u_app_list = cas.vertcat(norm_u_app_list, norm_u_app)

    rancourt_dict = {}
    rancourt_dict['lift_per_unit_span'] = {'vals': np.array(lift_per_unit_span_interpolated)}
    rancourt_dict['cl'] = {'vals': np.array(cl_interpolated)}
    rancourt_dict['radius'] = {'vals': np.array(radius_list)}
    rancourt_dict['app'] = {'vals': np.array(norm_u_app_list)}
    rancourt_dict['eff'] = {'vals': np.array(norm_u_eff_list)}
    rancourt_dict['dq'] = {'vals': np.array(norm_dq_kite_list)}

    rancourt_dict = add_shortened_distributions(rancourt_dict, all_xi)

    norm_minus = []
    sdx = 0
    for ldx in range(all_xi.shape[0]):
        xi_val = all_xi[ldx]
        if xi_val in rancourt_dict['eff']['short_xi']:
            local_eff = rancourt_dict['eff']['short_vals'][sdx]
            local_app = rancourt_dict['app']['vals'][ldx]

            local_norm_minus = local_eff - local_app
            norm_minus = cas.vertcat(norm_minus, local_norm_minus)
            sdx += 1

    rancourt_dict['norm_minus'] = {'short_xi': rancourt_dict['eff']['short_xi'],
                                   'short_vals': np.array(norm_minus)
                                   }
    return rancourt_dict

def add_shortened_distributions(rancourt_dict, all_xi):
    for label, values in rancourt_dict.items():
        short_xi = []
        short_ax = []
        for idx in range(all_xi.shape[0]):
            if np.isfinite(values['vals'][idx]) and (values['vals'][idx] != 0.):
                short_xi += [all_xi[idx]]
                short_ax += [values['vals'][idx]]
        rancourt_dict[label]['short_xi'] = np.array(short_xi)
        rancourt_dict[label]['short_vals'] = np.array(short_ax)
    return rancourt_dict


def get_trevisi_induction_factor_distributions(all_xi):

    #  Vortex model of the aerodynamic wake of airborne wind energy systems
    #     Filippo Trevisi Carlo E. D. Riboldi Alessandro Croce
    #     Wind Energy Science vol. 8 issue 6 (2023) pp: 999-1016

    ax_interpolations = {}

    trevisi_blank_name = 'T0'
    trevisi_trevisi_name = 'TT'
    trevisi_gaunaa_name = 'TG'
    gaunaa_gaunaa_name = 'GG'
    trevisi_kheiri_name = 'TK'
    trevisi_kheiri_minus_trevisi_blank_name = 'KT'
    kheiri_full_name = 'KK'
    qblade_name = 'QB'


    wingspan = 5.5
    u_infty = 5.
    omega = 3.88221
    trevisi_radius = []
    trevisi_app = []
    for xi in all_xi:
        local_radius = (wingspan / 0.3) + xi * wingspan
        trevisi_radius += [local_radius]
        local_u_tan = omega * local_radius
        local_u_app = (local_u_tan**2. + u_infty**2.)**0.5
        trevisi_app += [local_u_app]
    trevisi_radius = np.array(trevisi_radius).reshape(all_xi.shape)
    trevisi_app = np.array(trevisi_app).reshape(all_xi.shape)
    other_trevisi_dict = {'radius': {'xi': all_xi, 'vals': trevisi_radius},
                          'app': {'xi': all_xi, 'vals': trevisi_app}}

    trevisi_blank_xi = [-0.459632497636364, -0.450145848218182, -0.422512565127273, -0.395952613418181, -0.333559393527273, -0.283076151981818, -0.199572970472728, -0.0756831957272723, 0.0518732390181821, 0.154653052145455, 0.283129485218182, 0.321656081836364, 0.362956006745454, 0.394159283345454, 0.425419226509091, 0.447339186654546, 0.465142487618182]
    trevisi_blank_ax = [0, 0.20339, 0.27702, 0.24587, 0.25218, 0.26846, 0.27855, 0.3024, 0.32501, 0.34385, 0.36771, 0.36402, 0.37405, 0.38033, 0.42402, 0.36043, 0.0037]
    trevisi_blank_ax_interpolated = np.interp(all_xi, trevisi_blank_xi, trevisi_blank_ax, left=np.nan, right=np.nan)
    ax_interpolations[trevisi_blank_name] = {'vals': trevisi_blank_ax_interpolated}

    trevisi_trevisi_xi = [-0.460552495963637, -0.448135851872727, -0.447179186945455, -0.421395900490909, -0.395755947109091, -0.333359393890909, -0.283792817345455, -0.171839687563636, -0.0975064893818182, 0.0227099587090907, 0.270479508218182, 0.29892945649091, 0.331032731454546, 0.375082651363637, 0.396195946309091, 0.424699227818182, 0.447535852963636, 0.466085819236364]
    trevisi_trevisi_ax = [0, 0.31815, 0.34435, 0.40675, 0.3756, 0.38191, 0.39819, 0.41704, 0.42961, 0.45221, 0.49493, 0.49996, 0.49625, 0.50379, 0.51255, 0.55375, 0.48892, 0.01992]
    trevisi_trevisi_ax_interpolated = np.interp(all_xi, trevisi_trevisi_xi, trevisi_trevisi_ax, left=np.nan, right=np.nan)
    ax_interpolations[trevisi_trevisi_name] = {'vals': trevisi_trevisi_ax_interpolated}

    trevisi_gaunaa_xi = [-0.462372492654545, -0.449989181836364, -0.424202562054546, -0.392139287018182, -0.312302765509091, -0.256316200636364, -0.1718930208, 0.0171499688181817, 0.21719960509091, 0.273179503309091, 0.2924528016, 0.32272607989091, 0.355759353163636, 0.382375971436364, 0.405332596363636, 0.426489224563636, 0.4520625114, 0.464282489181819]
    trevisi_gaunaa_ax = [0.00878, 0.30568, 0.37182, 0.33943, 0.3545, 0.36829, 0.38211, 0.41603, 0.44871, 0.46001, 0.46377, 0.46381, 0.46635, 0.47387, 0.48762, 0.52506, 0.449, 0.04112]
    trevisi_gaunaa_ax_interpolated = np.interp(all_xi, trevisi_gaunaa_xi, trevisi_gaunaa_ax, left=np.nan, right=np.nan)
    ax_interpolations[trevisi_gaunaa_name] = {'vals': trevisi_gaunaa_ax_interpolated}

    gaunaa_gaunaa_ax = np.array(cas.DM.ones(all_xi.shape[0], 1) * 0.900289)
    ax_interpolations[gaunaa_gaunaa_name] = {'vals': gaunaa_gaunaa_ax}

    trevisi_kheiri_xi = [-0.469709145981818, -0.4498625154, -0.422225898981818, -0.391999287272727, -0.333276060709091, -0.271776172527273, -0.157073047745455, 0.0925364984181814, 0.285249481363636, 0.323779411309091, 0.390769289509091, 0.423862562672727, 0.448549184454545, 0.469762479218182]
    trevisi_kheiri_ax = [0.00877, 0.38925, 0.46288, 0.43298, 0.43805, 0.45808, 0.47319, 0.51841, 0.55358, 0.55238, 0.56369, 0.60614, 0.55129, 0.02491]
    trevisi_kheiri_ax_interpolated = np.interp(all_xi, trevisi_kheiri_xi, trevisi_kheiri_ax, left=np.nan, right=np.nan)
    ax_interpolations[trevisi_kheiri_name] = {'vals': trevisi_kheiri_ax_interpolated}

    trevisi_kheiri_minus_trevisi_blank_ax = trevisi_kheiri_ax_interpolated - trevisi_blank_ax_interpolated
    ax_interpolations[trevisi_kheiri_minus_trevisi_blank_name] = {'vals': trevisi_kheiri_minus_trevisi_blank_ax}

    kheiri_full_ax = np.array(cas.DM.ones(all_xi.shape[0], 1) * 0.136719)
    ax_interpolations[kheiri_full_name] = {'vals': kheiri_full_ax}

    qblade_xi_ax = [-0.4976390952, -0.496015764818182, -0.495519099054546, -0.495172433018182, -0.494805767018182, -0.494415767727273,
         -0.494025768436364, -0.493632435818182, -0.493242436527273, -0.492849103909091, -0.492459104618182, \
         -0.492069105327273, -0.491675772709091, -0.491285773418182, -0.490895774127273, -0.490502441509091, \
         -0.490112442218182, -0.4897191096, -0.489019110872727, -0.488305778836364, -0.486892448072727, -0.484649118818182, \
         -0.483319121236364, -0.482209123254545, -0.481082458636364, -0.479925794072728, -0.478769129509091, \
         -0.477339132109091, -0.475872468109091, -0.474259137709091, -0.472369141145455, -0.470479144581819, \
         -0.468349148454545, -0.466192485709091, -0.464115822818182, -0.462229159581819, -0.460339163018182, \
         -0.454292507345454, -0.447609186163636, -0.440245866218182, -0.431825881527273, -0.423405896836363, \
         -0.414762579218182, -0.406092594981819, -0.397252611054545, -0.388229294127273, -0.3792793104, -0.370325993345454, \
         -0.3612760098, -0.352226026254545, -0.343139376109091, -0.334029392672727, -0.324919409236364, -0.315866092363636, \
         -0.306812775490909, -0.297802791872727, -0.288816141545455, -0.279822824563636, -0.270769507690909, \
         -0.261716190818182, -0.252662873945454, -0.243609557072727, -0.234559573527273, -0.225506256654546, \
         -0.216452939781818, -0.207356289654546, -0.198249639545454, -0.189166322727272, -0.180113005854546, \
         -0.171059688981818, -0.162009705436364, -0.152956388563637, -0.143873071745455, -0.134766421636363, \
         -0.125666438181818, -0.116606454654546, -0.1075431378, -0.0984864876000002, -0.089433170727273, \
         -0.0803731872000003, -0.071266537090909, -0.0621598869818185, -0.0530565701999996, -0.0439499200909091, \
         -0.0348632699454547, -0.0258099530727275, -0.0167566362000002, -0.00765665274545442, 0.00144999736363616, \
         0.0105599808000004, 0.0196699642363634, 0.0287799476727276, 0.0378865977818182, 0.0469932478909088, \
         0.0560998979999994, 0.0652032147818182, 0.0743098648909088, 0.0834231816545454, 0.092533165090909, 0.1016398152, \
         0.110746465309091, 0.119853115418182, 0.128959765527273, 0.138063082309091, 0.147203065690908, 0.156349715727273, \
         0.165483032454545, 0.174589682563636, 0.183696332672728, 0.192802982781818, 0.201906299563636, 0.211016283, \
         0.220126266436364, 0.2292395832, 0.238346233309091, 0.247449550090909, 0.2565562002, 0.265662850309091, \
         0.274769500418181, 0.283916150454546, 0.293059467163636, 0.302179450581818, 0.311282767363637, 0.320376084163636, \
         0.329429401036364, 0.338482717909091, 0.3475260348, 0.356569351690909, 0.365582668636363, 0.374569318963636, \
         0.383545969309091, 0.39217262029091, 0.4008026046, 0.409409255618182, 0.418005906654546, 0.427052556872727, \
         0.436102540418182, 0.444965857636364, 0.453829174854546, 0.456929169218182, 0.458639166109091, 0.463822490018182, \
         0.466689151472727, 0.468895814127273, 0.471102476781819, 0.473309139436364, 0.475515802090909, 0.477079132581818, \
         0.478509129981818, 0.479835794236363, 0.480909125618182, 0.481982457, 0.483055788381818, 0.484129119763636, \
         0.485215784454545, 0.486335782418182, 0.487455780381818, 0.488535778418181, 0.489609109800001, 0.490682441181818, \
         0.491759105890909, 0.492832437272727, 0.493905768654545, 0.494979100036364]
    qblade_ax = [0.34304, 0.33077, 0.31833, 0.30586, 0.2934, 0.28094, 0.26848, 0.25602, 0.24356, 0.2311, 0.21863, 0.20617, 0.19371, \
         0.18125, 0.16879, 0.15633, 0.14387, 0.13141, 0.11897, 0.10654, 0.10536, 0.11746, 0.12979, 0.14218, 0.15457, \
         0.16694, 0.17932, 0.19165, 0.20397, 0.21625, 0.22847, 0.24068, 0.25282, 0.26496, 0.27711, 0.28933, 0.30154, 0.3106, \
         0.31917, 0.32634, 0.33133, 0.33631, 0.34052, 0.34463, 0.34599, 0.34506, 0.34784, 0.35061, 0.35269, 0.35475, \
         0.35651, 0.35801, 0.35952, 0.36158, 0.36364, 0.366, 0.36855, 0.37104, 0.3731, 0.37516, 0.37722, 0.37929, 0.38135, \
         0.38341, 0.38547, 0.38711, 0.38867, 0.39044, 0.3925, 0.39457, 0.39663, 0.39869, 0.40047, 0.40203, 0.40365, 0.40563, \
         0.40761, 0.40964, 0.4117, 0.41368, 0.41524, 0.4168, 0.41835, 0.41991, 0.42164, 0.4237, 0.42576, 0.42737, 0.42893, \
         0.43046, 0.43196, 0.43346, 0.43501, 0.43657, 0.43813, 0.43969, 0.44124, 0.44274, 0.44423, 0.44577, 0.44733, \
         0.44889, 0.45044, 0.452, 0.45313, 0.45414, 0.45529, 0.45685, 0.45841, 0.45997, 0.46153, 0.46306, 0.46456, 0.46605, \
         0.46761, 0.46917, 0.47072, 0.47228, 0.47381, 0.47485, 0.4759, 0.47729, 0.47884, 0.48053, 0.48259, 0.48465, 0.48678, \
         0.48893, 0.49127, 0.49383, 0.49643, 0.5007, 0.50498, 0.50933, 0.51372, 0.51194, 0.51028, 0.51353, 0.51678, 0.51607, \
         0.50649, 0.49621, 0.48452, 0.47242, 0.46032, 0.44823, 0.43613, 0.42385, 0.41153, 0.3992, 0.38681, 0.37443, 0.36205, \
         0.34966, 0.33728, 0.3249, 0.31253, 0.30014, 0.28776, 0.27538, 0.26299, 0.25061, 0.23823, 0.22584]
    qblade_ax_interpolated = np.interp(all_xi, qblade_xi_ax, qblade_ax, left=np.nan, right=np.nan)
    ax_interpolations[qblade_name] = {'vals': qblade_ax_interpolated}

    ar_interpolations = {}

    trevisi_trevisi_ar = np.array(cas.DM.ones(all_xi.shape[0], 1) * 0.03383)
    ar_interpolations[trevisi_trevisi_name] = {'vals': trevisi_trevisi_ar}

    gaunaa_gaunaa_ar = np.array(cas.DM.ones(all_xi.shape[0], 1) * 1e-8)
    ar_interpolations[gaunaa_gaunaa_name] = {'vals': gaunaa_gaunaa_ar}

    kheiri_full_ar = np.array(cas.DM.ones(all_xi.shape[0], 1) * 1e-8)
    ar_interpolations[kheiri_full_name] = {'vals': kheiri_full_ar}

    qblade_xi_ar = [-0.497175762709091, -0.488699111454545, -0.481165791818182, -0.4745758038, -0.467042484163637,
                    -0.460452496145455, -0.451975844890909, -0.442559195345454, -0.429379219309091, -0.411485918509091,
                    -0.389829291218182, -0.367229332309091, -0.344632706727273, -0.322032747818182, -0.298492790618182,
                    -0.274952833418182, -0.251412876218182, -0.228812917309091, -0.205272960109091, -0.1826730012,
                    -0.159133044, -0.136533085090909, -0.113936459509091, -0.0913365006, -0.068736541690909,
                    0.00564998972727244, 0.0282499486363635, 0.0508465742181821, 0.0734465331272732, 0.0960464920363635,
                    0.119586449236363, 0.142183074818182, 0.164783033727273, 0.188322990927273, 0.210922949836363,
                    0.234462907036363, 0.257062865945454, 0.280602823145455, 0.304142780345454, 0.327682737545455,
                    0.350282696454546, 0.371939323745455, 0.393595951036364, 0.413369248418182, 0.426552557781818,
                    0.440675865436364, 0.450092514981818, 0.458569166236364, 0.466102485872728, 0.472692473890909,
                    0.478342463618182, 0.485875783254546]
    qblade_ar = [0.116809, 0.107021, 0.096809, 0.086596, 0.076383, 0.06617, 0.056383, 0.046596, 0.038085, 0.031277,
                 0.027447, 0.024468, 0.022766, 0.021915, 0.021489, 0.021489, 0.021489, 0.021915, 0.022766, 0.023617,
                 0.024894, 0.02617, 0.027447, 0.029149, 0.030426, 0.035957, 0.03766, 0.039362, 0.041064, 0.042766,
                 0.044043, 0.045319, 0.047021, 0.047872, 0.048723, 0.049149, 0.05, 0.049574, 0.049574, 0.048723,
                 0.047447, 0.045319, 0.041915, 0.03766, 0.031702, 0.023617, 0.01383, 0.004043, -0.00617, -0.016383,
                 -0.026596, -0.036809]
    qblade_ar_interpolated = np.interp(all_xi, qblade_xi_ar, qblade_ar, left=np.nan, right=np.nan)
    ar_interpolations[qblade_name] = {'vals': qblade_ar_interpolated}

    ax_interpolations = add_shortened_distributions(ax_interpolations, all_xi)
    ar_interpolations = add_shortened_distributions(ar_interpolations, all_xi)

    return ax_interpolations, ar_interpolations, other_trevisi_dict



def get_x_obs_for_spanwise_distribution(xi_sym, plot_dict, kite, idx_at_eval):

    search_name = 'interpolation_si'
    parent = plot_dict['architecture'].parent_map[kite]
    x_vals = plot_dict[search_name]['x']

    # kite position information
    q_kite = []
    ehat_2 = []
    for j in range(3):
        q_kite = cas.vertcat(q_kite, x_vals['q' + str(kite) + str(parent)][j][idx_at_eval])
        ehat_2 = cas.vertcat(ehat_2, plot_dict[search_name]['outputs']['aerodynamics']['ehat_span' + str(kite)][j][idx_at_eval])

    b_ref = plot_dict['options']['model']['params']['geometry']['b_ref']
    vec_offset = xi_sym * b_ref * ehat_2
    x_obs = q_kite + vec_offset
    return x_obs, vec_offset

def get_velocity_distribution_at_spanwise_position_functions(plot_dict, cosmetics, kite, idx_at_eval):

    xi_sym = cas.SX.sym('xi_sym', (1, 1))

    search_name = 'interpolation_si'
    x_vals = plot_dict[search_name]['x']

    parent_map = plot_dict['architecture'].parent_map
    parent = parent_map[kite]

    # kite position information
    q_kite = []
    dq_kite = []
    ehat_1 = []
    ehat_2 = []
    ehat_3 = []
    x_center = []
    ehat_radial = []
    ehat_tangential = []
    ehat_normal = []
    for j in range(3):
        q_kite = cas.vertcat(q_kite, x_vals['q' + str(kite) + str(parent)][j][idx_at_eval])
        dq_kite = cas.vertcat(dq_kite, x_vals['dq' + str(kite) + str(parent)][j][idx_at_eval])
        ehat_1 = cas.vertcat(ehat_1,
                             plot_dict[search_name]['outputs']['aerodynamics']['ehat_chord' + str(kite)][j][idx_at_eval])
        ehat_2 = cas.vertcat(ehat_2,
                             plot_dict[search_name]['outputs']['aerodynamics']['ehat_span' + str(kite)][j][idx_at_eval])
        ehat_3 = cas.vertcat(ehat_3,
                             plot_dict[search_name]['outputs']['aerodynamics']['ehat_up' + str(kite)][j][idx_at_eval])
        x_center = cas.vertcat(x_center, plot_dict[search_name]['outputs']['geometry']['x_center' + str(parent)][j][idx_at_eval])
        ehat_radial = cas.vertcat(ehat_radial,
                               plot_dict[search_name]['outputs']['rotation']['ehat_radial' + str(kite)][j][idx_at_eval])
        ehat_tangential = cas.vertcat(ehat_tangential,
                               plot_dict[search_name]['outputs']['rotation']['ehat_tangential' + str(kite)][j][idx_at_eval])
        ehat_normal = cas.vertcat(ehat_normal,
                               plot_dict[search_name]['outputs']['rotation']['ehat_normal' + str(parent)][j][idx_at_eval])

    x_obs, vec_offset = get_x_obs_for_spanwise_distribution(xi_sym, plot_dict, kite, idx_at_eval)
    vec_to_center = x_obs - x_center
    radius_to_center = vect_op.abs(cas.mtimes(vec_to_center.T, ehat_radial))

    wind_model = plot_dict['options']['model']['wind']['model']
    u_ref = plot_dict['options']['user_options']['wind']['u_ref']
    z_ref = plot_dict['options']['model']['params']['wind']['z_ref']
    z0_air = plot_dict['options']['model']['params']['wind']['log_wind']['z0_air']
    exp_ref = plot_dict['options']['model']['params']['wind']['power_wind']['exp_ref']
    u_infty = wind.get_speed(wind_model, u_ref, z_ref, z0_air, exp_ref, x_obs[2])
    vec_u_infty = u_infty * vect_op.xhat_dm()

    omega_name = 'omega' + str(kite) + str(parent)
    if omega_name in x_vals.keys():
        omega_body_axes = []
        for j in range(3):
            omega_body_axes = cas.vertcat(omega_body_axes, x_vals[omega_name][j][idx_at_eval])
        omega_earth_axes = omega_body_axes[0] * ehat_1 + omega_body_axes[1] * ehat_2 + omega_body_axes[2] * ehat_3
        dq_rotational = vect_op.cross(omega_earth_axes, vec_offset)
        # because, remember, omega is defined wrt the *body-fixed* axes.
    else:
        dq_rotational = cas.DM.zeros((3, 1))

    dq_local = dq_kite + dq_rotational
    vec_u_app = vec_u_infty - dq_local

    kite_dcm = cas.horzcat(ehat_1, ehat_2, ehat_3)
    alpha_app = aero_indicators.get_alpha(vec_u_app, kite_dcm) * 180. / np.pi
    beta_app = aero_indicators.get_beta(vec_u_app, kite_dcm) * 180. / np.pi

    u_normalizing = get_induction_factor_normalizing_speed(plot_dict, idx_at_eval)
    if ('wake' in plot_dict.keys()) and ('interpolation_scaled' in plot_dict.keys()) and ('parameters_plot' in plot_dict.keys()):
        variables_scaled = get_variables_scaled(plot_dict, cosmetics, idx_at_eval)
        parameters = plot_dict['parameters_plot']
        wake = plot_dict['wake']
        vec_u_ind = wake.calculate_total_biot_savart_at_x_obs(variables_scaled, parameters, x_obs=x_obs)

        a_normal = -1. * cas.mtimes(vec_u_ind.T, ehat_normal) / u_normalizing
        a_tangential = -1. * cas.mtimes(vec_u_ind.T, ehat_tangential) / u_normalizing
        a_radial = cas.mtimes(vec_u_ind.T, ehat_radial) / u_normalizing

        kite_dcm = cas.horzcat(ehat_1, ehat_2, ehat_3)

        vec_u_eff = vec_u_app + vec_u_ind

        alpha_eff = aero_indicators.get_alpha(vec_u_eff, kite_dcm) * 180. / np.pi
        beta_eff = aero_indicators.get_beta(vec_u_eff, kite_dcm) * 180. / np.pi
        delta_alpha = (alpha_eff - alpha_app)
        delta_beta = (beta_eff - beta_app)

    else:
        vec_u_eff = vec_u_app
        a_normal = cas.DM.zeros((1, 1))
        a_tangential = cas.DM.zeros((1, 1))
        a_radial = cas.DM.zeros((1, 1))
        alpha_eff = alpha_app
        beta_eff = beta_app
        delta_alpha = cas.DM.zeros((1, 1))
        delta_beta = cas.DM.zeros((1, 1))

    outputs = {
        'radius': radius_to_center,
        'app': vect_op.norm(vec_u_app),
        'eff': vect_op.norm(vec_u_eff),
        'dq': vect_op.norm(dq_local),
        'a_n': a_normal,
        'a_r': a_radial,
        'a_t': a_tangential,
        'alpha_app': alpha_app,
        'alpha_eff': alpha_eff,
        'd_alpha': delta_alpha,
        'd_beta': delta_beta
        }
    outputs['norm_minus'] = outputs['eff'] - outputs['app']

    fun_dict = {}
    for name, val in outputs.items():
        local_fun = cas.Function(name + '_fun', [xi_sym], [val])
        fun_dict[name] = local_fun

    return fun_dict

def get_total_biot_savart_at_observer_function(plot_dict, cosmetics, idx_at_eval=0):

    variables_scaled = get_variables_scaled(plot_dict, cosmetics, idx_at_eval)
    parameters = plot_dict['parameters_plot']
    wake = plot_dict['wake']

    x_obs_sym = cas.SX.sym('x_obs_sym', (3, 1))
    u_ind_sym = wake.calculate_total_biot_savart_at_x_obs(variables_scaled, parameters, x_obs=x_obs_sym)

    u_ind_fun = cas.Function('a_fun', [x_obs_sym], [u_ind_sym])

    return u_ind_fun

def get_the_induction_factor_at_observer_function(plot_dict, cosmetics, idx_at_eval=0):

    variables_scaled = get_variables_scaled(plot_dict, cosmetics, idx_at_eval)
    parameters = plot_dict['parameters_plot']
    wake = plot_dict['wake']
    x_obs_sym = cas.SX.sym('x_obs_sym', (3, 1))
    u_ind_sym = wake.calculate_total_biot_savart_at_x_obs(variables_scaled, parameters, x_obs=x_obs_sym)

    n_hat, _, _ = get_coordinate_axes_for_haas_verification(plot_dict, idx_at_eval)

    u_normalizing = get_induction_factor_normalizing_speed(plot_dict, idx_at_eval)

    a_sym = general_flow.compute_induction_factor(u_ind_sym, n_hat, u_normalizing)
    a_fun = cas.Function('a_fun', [x_obs_sym], [a_sym])

    return a_fun

def get_induction_factor_normalizing_speed(plot_dict, idx_at_eval):
    induction_factor_normalizing_speed = plot_dict['options']['model']['aero']['vortex']['induction_factor_normalizing_speed']
    if induction_factor_normalizing_speed == 'u_zero':
        kite_plane_induction_params = get_kite_plane_induction_params(plot_dict, idx_at_eval)
        u_normalizing = kite_plane_induction_params['u_zero']

        if plot_dict['architecture'].number_of_kites == 1:
            message = 'please be advised that the computation of the rotor apparent velocity vec_u_zero does not yet work well for single-kite systems'
            print_op.base_print(message, level='warning')

    elif induction_factor_normalizing_speed == 'u_ref':
        u_ref = plot_dict['options']['user_options']['wind']['u_ref']
        u_normalizing = u_ref

    else:
        message = 'computing induction factor at specific points is not yet defined for normalizing speed ' + induction_factor_normalizing_speed
        print_op.log_and_raise_error(message)

    return u_normalizing


def compute_induction_factor_at_specified_observer_coordinates(plot_dict, cosmetics, kdx, orientation='radial', plane='yz', idx_at_eval=0):

    dimensionless_coordinates, dimensioned_coordinates = get_distributed_coordinates_about_actuator_center(plot_dict, kdx, orientation, plane, idx_at_eval)

    a_fun = get_the_induction_factor_at_observer_function(plot_dict, cosmetics, idx_at_eval)

    index_to_dimensions = get_index_to_dimensions_dict()
    x_obs_dimensioned_stacked = []
    for pdx in range(dimensioned_coordinates['y'].shape[0]):
        for mdx in range(dimensioned_coordinates['y'].shape[1]):
            local_x_obs_dimensioned = cas.DM.ones((3, 1))
            for idx, dim in index_to_dimensions.items():
                local_x_obs_dimensioned[idx] = dimensioned_coordinates[dim][pdx, mdx]
            x_obs_dimensioned_stacked = cas.horzcat(x_obs_dimensioned_stacked, local_x_obs_dimensioned)

    parallelization_type = plot_dict['options']['model']['construction']['parallelization']['type']
    if parallelization_type in ['serial', 'openmp', 'thread']:
        a_map = a_fun.map(x_obs_dimensioned_stacked.shape[1], parallelization_type)
        all_a = a_map(x_obs_dimensioned_stacked)
    elif parallelization_type == 'concurrent_futures':
        all_a = struct_op.concurrent_future_map(a_fun, x_obs_dimensioned_stacked)
    else:
        message = 'sorry, but the awebox has not yet set up ' + parallelization_type + ' parallelization'
        print_op.log_and_raise_error(message)

    a_matr = cas.reshape(all_a, dimensioned_coordinates['y'].shape)

    y_matr = dimensionless_coordinates['y']
    z_matr = dimensionless_coordinates['z']

    return y_matr, z_matr, a_matr


def compute_the_scaled_haas_error(plot_dict, cosmetics):

    # reference points digitized from haas 2017
    data02 = [(0, -1.13084, -0.28134), (0, -0.89466, -0.55261), (0, -0.79354, -0.41195), (0, -0.79127, 0.02826), (0, -0.04449, 1.01208), (0, 0.22119, 0.83066), (0, 0.71947, -0.96407), (0, 0.52187, -0.66125), (0, 0.45227, 1.09891), (0, 0.92121, -0.46219), (0, 0.95625, -0.62566), (0, 0.19546, 1.12782)]
    data00 = [(0, -0.69372, 1.24284), (0, -0.5894, -0.17062), (0, -0.83795, -1.18138), (0, 0.27077, 1.35428), (0, -1.20982, -0.71437), (0, -0.10682, 0.54558), (0, -0.3789, -0.3959), (0, 0.5081, -0.34622), (0, 0.23767, 0.61635), (0, 0.57386, -0.09835), (0, 1.10956, -0.867), (0, 1.4683, -0.06319)]
    datan005 = [(0, -1.27365, 0.33177), (0, -1.21638, 0.65872), (0, -0.64295, 0.15234), (0, -0.50651, 0.31282), (0, -0.01271, -1.39519), (0, 0.03669, -0.60921), (0, 0.20264, -0.63812), (0, 0.38143, -1.27208), (0, 0.49584, 0.49075), (0, 0.52938, 0.32778), (0, 0.96982, 0.94579), (0, 1.3061, 0.48278)]

    data = {-0.05: datan005, 0.0: data00, 0.2: data02}

    idx_at_eval = plot_dict['options']['visualization']['cosmetics']['animation']['snapshot_index']

    kite_plane_induction_params = get_kite_plane_induction_params(plot_dict, idx_at_eval)
    radius = kite_plane_induction_params['average_radius']
    x_center = kite_plane_induction_params['center']

    a_fun = get_the_induction_factor_at_observer_function(plot_dict, cosmetics, idx_at_eval)

    total_squared_error = 0.
    baseline_squared_error = 0.
    for aa_haas in data.keys():
        coord_list = data[aa_haas]
        for scaled_x_obs_tuple in coord_list:

            baseline_squared_error += aa_haas ** 2.

            scaled_x_obs = []
            for dim in range(3):
                scaled_x_obs = cas.vertcat(scaled_x_obs, scaled_x_obs_tuple[dim])
            x_obs = scaled_x_obs * radius + x_center

            aa_computed = a_fun(x_obs)
            diff = aa_haas - aa_computed
            total_squared_error += diff ** 2.

    scaled_squared_error = total_squared_error / baseline_squared_error
    return scaled_squared_error

def get_kite_plane_induction_params(plot_dict, idx_at_eval):

    kite_plane_induction_params = {}

    interpolated_outputs_si = plot_dict['interpolation_si']['outputs']

    layer_nodes = plot_dict['architecture'].layer_nodes
    layer = int( np.min(np.array(layer_nodes)) )
    kite_plane_induction_params['layer'] = layer
    
    b_ref = plot_dict['options']['model']['params']['geometry']['b_ref']
    average_radius = interpolated_outputs_si['geometry']['average_radius' + str(layer)][0][idx_at_eval]
    kite_plane_induction_params['average_radius'] = average_radius

    center = []
    for dim in range(3):
        local_center = interpolated_outputs_si['performance']['trajectory_center' + str(layer)][dim][idx_at_eval]
        center = cas.vertcat(center, local_center)
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
        local_u_zero = interpolated_outputs_si['geometry']['vec_u_zero' + str(layer)][dim][idx_at_eval]
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


def plot_induction_contour_on_kmp(plot_dict, cosmetics, fig_name, fig_num=None):

    vortex_info_exists = ('wake' in plot_dict.keys()) and (plot_dict['wake'] is not None)
    if vortex_info_exists:

        tau_style_dict = tools.get_temporal_orientation_epigraphs_taus_and_linestyles(plot_dict)
        for tau_at_eval in tau_style_dict.keys():

            n_points_contour = 300

            n_points_interpolation = plot_dict['cosmetics']['interpolation']['n_points']
            idx_at_eval = int(np.floor((float(n_points_interpolation) -1.)  * tau_at_eval))

            kite_plane_induction_params = get_kite_plane_induction_params(plot_dict, idx_at_eval)
            radius = kite_plane_induction_params['average_radius']
            x_center = kite_plane_induction_params['center']

            variables_scaled = get_variables_scaled(plot_dict, cosmetics, idx_at_eval)
            parameters = plot_dict['parameters_plot']
            wake = plot_dict['wake']
            a_fun = get_the_induction_factor_at_observer_function(plot_dict, cosmetics, idx_at_eval)

            ### compute the induction factors
            plot_radius_scaled = 1.6
            plot_radius = plot_radius_scaled * radius
            sym_start_plot = -1. * plot_radius
            sym_end_plot = plot_radius
            delta_plot = plot_radius / float(n_points_contour)
            yy, zz = np.meshgrid(np.arange(sym_start_plot, sym_end_plot, delta_plot),
                                 np.arange(sym_start_plot, sym_end_plot, delta_plot))

            aa = np.zeros(yy.shape)

            print_op.base_print('making induction contour plot...')
            total_progress = yy.shape[0] * yy.shape[1]
            progress_index = 0
            for idx in range(yy.shape[0]):
                for jdx in range(yy.shape[1]):
                    print_op.print_progress(progress_index, total_progress)

                    x_obs_centered = cas.vertcat(0., yy[idx, jdx], zz[idx, jdx])
                    x_obs = x_obs_centered + x_center
                    aa_computed = a_fun(x_obs)
                    aa[idx, jdx] = float(aa_computed)

                    progress_index += 1

            print_op.close_progress()

            ### initialize the figure
            fig, ax = plt.subplots()

            ### draw the swept annulus
            draw_swept_background(ax, plot_dict)

            ### draw the contour
            levels = [-0.05, 0., 0.2]
            linestyles = ['dashdot', 'solid', 'dashed']
            colors = ['k', 'k', 'k']
            emergency_levels = 5
            emergency_colors = 'k'
            emergency_linestyles = 'solid'
            yy_scaled = yy / radius
            zz_scaled = zz / radius
            if (np.any(aa < levels[0])) and (np.any(aa > levels[-1])):
                cs = ax.contour(yy_scaled, zz_scaled, aa, levels, colors=colors, linestyles=linestyles)
            else:
                cs = ax.contour(yy_scaled, zz_scaled, aa, emergency_levels, colors=emergency_colors,
                            linestyles=emergency_linestyles)
            ax.clabel(cs, cs.levels, inline=True)

            ### draw the vortex positions
            if vortex_info_exists:
                for elem in wake.get_substructure('bound').get_list('finite_filament').list:
                    unpacked = elem.unpack_info(elem.evaluate_info(variables_scaled, parameters))
                    x_start = unpacked['x_start']
                    x_end = unpacked['x_end']

                    x_start_shifted = (x_start - x_center) / radius
                    x_end_shifted = (x_end - x_center) / radius

                    y_over_r_vals = [float(x_start_shifted[1]), float(x_end_shifted[1])]
                    z_over_r_vals = [float(x_start_shifted[2]), float(x_end_shifted[2])]

                    ax.plot(y_over_r_vals, z_over_r_vals)

            scaled_haas_error = compute_the_scaled_haas_error(plot_dict, cosmetics)

            tau_rounded = np.round(tau_at_eval, 2)
            ax.grid(True)
            title = 'Induction factor over the kite plane \n t=' + str(tau_rounded)
            ax.set_title(title)
            ax.set_xlabel("y/r [-]")
            ax.set_ylabel("z/r [-]")
            ax.set_aspect(1.)

            ticks_points = [-1.6, -1.5, -1., -0.8, -0.5, 0., 0.5, 0.8, 1.0, 1.5, 1.6]
            ax.set_xlim([-1. * plot_radius_scaled, plot_radius_scaled])
            ax.set_ylim([-1. * plot_radius_scaled, plot_radius_scaled])
            ax.set_xticks(ticks_points)
            ax.set_yticks(ticks_points)
            plt.xticks(rotation=60)
            plt.tight_layout()

            fig.savefig(plot_dict['name'] + '_induction_contour_on_kmp_tau' + str(tau_rounded) + '.pdf')

        return None





def draw_swept_background(ax, plot_dict):
    idx_at_eval = plot_dict['options']['visualization']['cosmetics']['animation']['snapshot_index']

    kite_plane_induction_params = get_kite_plane_induction_params(plot_dict, idx_at_eval)

    radius = kite_plane_induction_params['average_radius']
    x_center = kite_plane_induction_params['center']
    n_hat, a_hat, b_hat = get_coordinate_axes_for_haas_verification(plot_dict, idx_at_eval)

    side = (x_center, a_hat, b_hat, radius)

    for kite in plot_dict['architecture'].kite_nodes:
        for zeta in np.arange(-0.5, 0.5, 1./100.):
            tools.plot_path_of_wingtip(ax, side, plot_dict, kite, zeta, color='gray', alpha=0.2)

    return None

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
