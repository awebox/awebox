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
from matplotlib import ticker
from awebox.logger.logger import Logger as awelogger

import awebox.mdl.wind as module_wind
import awebox.mdl.aero.induction_dir.actuator_dir.actuator as actuator
import awebox.mdl.aero.induction_dir.general_dir.flow as general_flow
import awebox.mdl.aero.indicators as aero_indicators

import awebox.viz.trajectory as traj
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


def get_index_to_dimensions_dict():
    index_to_dimensions = {0: 'x', 1: 'y', 2: 'z'}
    return index_to_dimensions


def get_coordinate_axes(plot_dict, idx_at_eval, direction='normal'):

    # direction should be 'normal' or 'wind'

    architecture = plot_dict['architecture']
    top_parent = architecture.parent_map[architecture.number_of_nodes - 1]

    n_hat = []
    for dim in range(3):
        n_hat = cas.vertcat(n_hat, plot_dict['interpolation_si']['outputs']['rotation']['ehat_' + direction + str(top_parent)][dim][idx_at_eval])
    # roughly: nhat -> xhat

    b_hat_temp = vect_op.zhat_dm()
    a_hat = vect_op.normed_cross(b_hat_temp, n_hat) # zhat x xhat = yhat
    b_hat = vect_op.normed_cross(n_hat, a_hat) # xhat x yhat = zhat

    # therefore, mental-approximates read as:
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
                ax[adx[type]][0].plot(rancourt_dict[type]['short_xi'], rancourt_dict[type]['short_vals'],
                                      label=extra_label)

    ax[-1][0].set_xlabel('non-dimensional spanwise position [-]')
    ax[-1][0].set_xticks(np.linspace(-0.5, 0.5, 5))

    shrink_factor = 0.3
    for adx_val in range(rows):
        ax[adx_val][0].grid(True)  # Add a grid for better readability
        # Put a legend to the right of the current axis
        ax[adx_val][0].legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    plt.tight_layout()
    # Adjust the subplots with the new bottom values, leaving space for the legend
    plt.subplots_adjust(right=1. - shrink_factor)

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
    plt.subplots_adjust(right=1. - shrink_factor)

    return None


def get_rancourt_velocity_distribution(all_xi):

    # Efficient Aerodynamic Method for Interacting Lifting Surfaces over Long Distances
    # David Rancourt and Dimitri N. Mavris
    # Journal of Aircraft 2018 55:6, 2466-2475

    rancourt_b_ref = 8.  # m
    rancourt_c_ref = 1.  # m
    rancourt_rho_air = 1.225  # kg/m^3
    rancourt_radius = 24.  # m
    rancourt_groundspeed = 30.  # m/s
    rancourt_uinfty = 5.  # m/s
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
        xi = all_xi[idx]
        local_radius = rancourt_radius - xi * rancourt_b_ref
        radius_list = cas.vertcat(radius_list, local_radius)

        norm_dq_kite = local_radius * rancourt_omega
        norm_dq_kite_list = cas.vertcat(norm_dq_kite_list, norm_dq_kite)

        u_eff_squared = (2. * lift_per_unit_span_interpolated[idx]) / (rancourt_c_ref * cl_interpolated[idx] * rancourt_rho_air)
        norm_u_eff = u_eff_squared**0.5
        norm_u_eff_list = cas.vertcat(norm_u_eff_list, norm_u_eff)

        norm_u_app = (norm_dq_kite ** 2. + rancourt_uinfty ** 2.) ** 0.5
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
        local_u_app = (local_u_tan ** 2. + u_infty ** 2.) ** 0.5
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

    wind = plot_dict['wind']
    model_parameters = plot_dict['model_parameters']
    parameters = model_parameters(plot_dict['parameters_plot'])
    vec_u_infty = wind.get_velocity(x_obs[2], external_parameters=parameters)

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

    model_parameters = plot_dict['model_parameters']
    parameters = plot_dict['parameters_plot']
    fun_dict = {}
    for name, val in outputs.items():

        local_fun = cas.Function(name + '_fun', [xi_sym], [val])
        fun_dict[name] = local_fun

    return fun_dict


def plot_velocity_deficits(plot_dict, cosmetics, fig_num=None):

    x_over_d_vals = [0., 2.0, 4.0, 6.0]

    b_ref = plot_dict['options']['model']['params']['geometry']['b_ref']

    kite_plane_induction_params = get_kite_plane_induction_params(plot_dict, 0)

    vortex_info_exists = ('wake' in plot_dict.keys()) and (plot_dict['wake'] is not None)
    if vortex_info_exists:
        variables_scaled = get_variables_scaled(plot_dict, cosmetics, 0)
        scaling = plot_dict['model_variables'](plot_dict['model_scaling'])
        variables_si = struct_op.variables_scaled_to_si(plot_dict['model_variables'], variables_scaled, scaling)
        this_is_haas2019_test = is_this_a_haas2019_test(plot_dict, kite_plane_induction_params, variables_si)
    else:
        this_is_haas2019_test = False

    if this_is_haas2019_test:
        diam = 280
        z0_planned = 260.
        z_planned_under = z0_planned - diam / 2.
        z_planned_over = z0_planned + diam / 2.
        z0_current = kite_plane_induction_params['center'][2]
        # z0_current + z_offset = z_planned
        z_offset_under = z_planned_under - z0_current
        z_offset_over = z_planned_over - z0_current
        z_offset_dict = {0: 0., 1: z_offset_under, 2: z_offset_over}
        y_labels_dict = {0: 'z [m] (y=0)', 1: 'y [m] (z=' + str(z_planned_under) + ')', 2: 'y [m] (z=' + str(z_planned_over) + ')'}
    else:
        diam = kite_plane_induction_params['average_radius'] * 2.
        z_offset_dict = {0: 0., 1: -diam/2., 2: diam/2.}
        y_labels_dict = {0: 'z [m] (y=0)', 1: 'y [m] (z=1r below z_center)', 2: 'y [m] (z=1r above z_center)'}

    plot_table_r = 3
    plot_table_c = len(x_over_d_vals)

    fig = plt.figure(num=fig_num)
    axes = fig.axes
    if len(axes) == 0:  # if figure does not exist yet
        fig, axes = plt.subplots(num=fig_num, nrows=plot_table_r, ncols=plot_table_c)

    slice_axes_dict = {0: vect_op.zhat_dm(), 1: vect_op.yhat_dm(), 2: vect_op.yhat_dm()}

    total_subplots = len(list(slice_axes_dict.keys())) * len(x_over_d_vals)
    pdx = 0
    print_op.base_print('plotting temporal average velocity deficits...', level='info')

    for idx in range(len(x_over_d_vals)):
        x_over_d_local = x_over_d_vals[idx]
        x_offset = x_over_d_local * diam

        for rdx in [0, 1, 2]:

            z_offset_local = z_offset_dict[rdx]

            add_label_legends = (idx == len(x_over_d_vals) - 1)
            suppress_wind_options_import_warning = (pdx > 0)

            try:
                ax = axes[rdx, idx]
            except:
                pdb.set_trace()
            make_individual_time_averaged_velocity_deficit_subplot(ax, plot_dict, cosmetics, x_offset=x_offset, z_offset=z_offset_local,
                                                                   slice_axes=slice_axes_dict[rdx],
                                                                   add_legend_labels=add_label_legends,
                                                                   suppress_wind_options_import_warning=suppress_wind_options_import_warning)
            if this_is_haas2019_test:
                add_haas2019_velocity_deficit_curves(ax=ax, z_offset_local=z_offset_local, x_over_d_local=x_over_d_local)

            if rdx == 0:
                ax.set_title('x/d = ' + str(x_over_d_local))
            # todo.

            if idx == 0:
                ax.set_ylabel(y_labels_dict[rdx])

            if idx > 0:
                ax.sharex(axes[rdx, 0])
                ax.sharey(axes[rdx, 0])

            if add_label_legends:
                ax.legend(loc='center right', bbox_to_anchor=(1.25, 0.5))

            print_op.print_progress(pdx, total_subplots)
            pdx += 1
            fig.canvas.draw()

    print_op.close_progress()

    return None


def add_haas2019_velocity_deficit_curves(ax, z_offset_local, x_over_d_local):
    curve_dict = {'horizontal_light': {}, 'horizontal_dark': {}, 'vertical': {}}
    curve_dict['vertical'][0.] = {'x': [3.78, 4.47, 5.16, 5.85, 6.57, 7.56, 8.55, 9.53, 10.52, 11.55, 12.59, 13.63, 15.74, 18.55, 21.36, 28.11, 35.22, 42.48, 49.79, 57.9, 68.13, 78.35, 88.76, 99.56, 110.35, 120.71, 129.68, 137.59, 143.58, 148.18, 153.38, 163.92, 174.4, 184.69, 194.98, 205.26, 215.55, 226.04, 236.65, 247.25, 257.85, 268.45, 279.01, 289.55, 300.08, 310.61, 321.3, 332, 342.74, 351.36, 357.89, 360.61, 363.34, 365.65, 367.88, 370.1, 371.89, 373.63, 375.45, 377.85, 380.24, 383.67, 389.95, 393.37, 396.21, 400.22, 404.23, 407.71, 410.82, 413.92, 416.89, 419.7, 422.5, 425.34, 428.22, 432.23, 439.95, 450.73, 461.46, 472.21, 483, 493.79, 504.58, 515.37, 526.17, 536.96, 547.75, 558.53, 569.32, 580.12, 590.91, 601.7, 612.5, 623.29, 634.06, 644.84, 655.61, 666.39, 677.17, 687.96, 698.76, 709.55, 720.34, 731.12, 741.89, 752.67, 763.45, 774.23, 785, 795.78],
                                 'u': [0.5, 0.51769, 0.53539, 0.55308, 0.57077, 0.58843, 0.60609, 0.62374, 0.6414, 0.65905, 0.6767, 0.69434, 0.71167, 0.72879, 0.74591, 0.75957, 0.77291, 0.78603, 0.79909, 0.81009, 0.81579, 0.82149, 0.82535, 0.82535, 0.82535, 0.83016, 0.83922, 0.85128, 0.86574, 0.88169, 0.89412, 0.89792, 0.90214, 0.90752, 0.91289, 0.91827, 0.92364, 0.9277, 0.93104, 0.93438, 0.93773, 0.94107, 0.94475, 0.94865, 0.95254, 0.95644, 0.95893, 0.96126, 0.96303, 0.95537, 0.94169, 0.92454, 0.90738, 0.89007, 0.87272, 0.85536, 0.83788, 0.82038, 0.80291, 0.78562, 0.76833, 0.75296, 0.76219, 0.77893, 0.79603, 0.8125, 0.82896, 0.84572, 0.8627, 0.87969, 0.89673, 0.91385, 0.93097, 0.94808, 0.96516, 0.98162, 0.99184, 0.99104, 0.98914, 0.9877, 0.98827, 0.98878, 0.98926, 0.98974, 0.99022, 0.99085, 0.99149, 0.99213, 0.99277, 0.99321, 0.9936, 0.99398, 0.99437, 0.99488, 0.99599, 0.99709, 0.9982, 0.99931, 1.00016, 1.0006, 1.00104, 1.00148, 1.0021, 1.00312, 1.00414, 1.00515, 1.00617, 1.00724, 1.00831, 1.00939]
                                 }
    curve_dict['vertical'][2.] = {'x': [3.2, 3.67, 4.14, 4.61, 5.02, 5.36, 5.69, 6.02, 6.75, 7.5, 8.25, 8.28, 8.3, 9.27, 10.9, 12.53, 14.78, 17.09, 20.82, 25.66, 31.87, 40.48, 50.35, 61.12, 67.03, 72.35, 77.32, 81.07, 84.91, 89.11, 93.72, 99.62, 109.04, 117.3, 127.73, 136.1, 141.49, 146.15, 151.31, 156.48, 161.65, 166.23, 170.61, 175, 179.15, 183.31, 187.75, 193.32, 200.09, 208.61, 217.97, 227.99, 238.66, 249.36, 260.13, 270.85, 281.4, 291.97, 302.67, 313.38, 324.18, 333.77, 341.72, 348.08, 354.49, 363.42, 373.16, 382.28, 393.07, 403.05, 412.37, 420.78, 427.4, 433.06, 439.3, 445.92, 453.57, 464.08, 474.81, 485.45, 496.17, 506.94, 517.71, 528.47, 539.23, 550, 560.76, 571.54, 582.33, 593.12, 603.91, 614.71, 625.5, 636.3, 647.05, 657.8, 668.55, 679.36, 690.16, 700.96, 711.74, 722.52, 733.3, 744.08, 754.88, 765.67, 776.46, 787.24, 797.98],
                                 'u': [0.46293, 0.481, 0.49906, 0.51713, 0.5352, 0.55328, 0.57135, 0.58943, 0.60747, 0.62551, 0.64356, 0.66164, 0.67972, 0.69769, 0.71557, 0.73345, 0.75114, 0.76881, 0.78565, 0.80182, 0.81643, 0.8271, 0.83437, 0.83366, 0.81877, 0.80306, 0.78703, 0.77008, 0.7532, 0.73655, 0.72023, 0.70562, 0.69689, 0.68524, 0.68354, 0.69437, 0.70974, 0.72607, 0.74196, 0.75786, 0.77376, 0.79014, 0.80668, 0.82321, 0.83992, 0.85662, 0.87313, 0.88846, 0.90258, 0.91361, 0.92247, 0.92904, 0.93194, 0.93438, 0.9357, 0.93763, 0.9416, 0.9453, 0.9478, 0.95027, 0.95027, 0.94449, 0.93244, 0.91784, 0.90352, 0.89356, 0.88581, 0.87616, 0.87531, 0.88181, 0.89094, 0.90222, 0.9163, 0.93171, 0.94648, 0.96075, 0.97354, 0.97574, 0.97761, 0.98076, 0.98288, 0.98441, 0.98593, 0.98749, 0.98907, 0.99065, 0.99222, 0.99342, 0.99443, 0.99543, 0.99641, 0.997, 0.99758, 0.99816, 0.99996, 1.00182, 1.00362, 1.00362, 1.00362, 1.00366, 1.00489, 1.00613, 1.00736, 1.00846, 1.00933, 1.0102, 1.01107, 1.01219, 1.01419]
                                 }
    curve_dict['vertical'][4.] = {'x': [3.78, 4.06, 4.35, 4.63, 4.92, 5.23, 5.55, 5.86, 6.32, 6.84, 7.35, 8.27, 9.46, 10.65, 11.84, 13.82, 16.74, 21.28, 29.78, 39.56, 49.21, 55.91, 64.23, 74.17, 84.61, 95.27, 105.97, 116.47, 126.74, 136.96, 146.27, 155.52, 164.16, 172.88, 181.75, 190.63, 200.07, 209.52, 218.54, 227.56, 236.13, 244.59, 252.97, 261.31, 271.42, 281.93, 292.53, 303.18, 313.59, 322.97, 330.61, 338.86, 349.14, 359.91, 370.61, 381.33, 392.08, 402.81, 413.35, 423.89, 433.87, 443.44, 452.53, 460.3, 469.09, 479.12, 489.73, 500.33, 511.12, 521.9, 532.69, 543.47, 554.24, 565.02, 575.79, 586.57, 597.35, 608.13, 618.91, 629.69, 640.47, 651.25, 662.03, 672.81, 683.58, 694.36, 705.14, 715.92, 726.7, 737.48, 748.26, 759.05, 769.83, 780.61, 791.39],
                                 'u': [0.49644, 0.51423, 0.53202, 0.5498, 0.56759, 0.58538, 0.60316, 0.62095, 0.63873, 0.6565, 0.67427, 0.69199, 0.70968, 0.72736, 0.74505, 0.76249, 0.77959, 0.79568, 0.80665, 0.81091, 0.80414, 0.79046, 0.77998, 0.77314, 0.772, 0.77484, 0.7772, 0.78101, 0.78653, 0.79219, 0.80121, 0.81035, 0.82103, 0.83153, 0.84165, 0.85178, 0.86039, 0.86899, 0.87877, 0.88855, 0.89935, 0.91042, 0.92163, 0.93292, 0.93832, 0.94235, 0.94572, 0.94864, 0.94729, 0.93923, 0.92665, 0.9152, 0.91237, 0.91236, 0.91468, 0.91682, 0.91847, 0.92036, 0.92417, 0.92803, 0.93483, 0.94289, 0.95251, 0.96472, 0.9744, 0.98086, 0.98419, 0.98752, 0.98831, 0.98908, 0.98986, 0.99083, 0.99197, 0.9931, 0.99424, 0.99524, 0.99615, 0.99706, 0.99797, 0.99895, 0.99995, 1.00095, 1.00197, 1.00306, 1.00416, 1.00526, 1.00633, 1.00724, 1.00815, 1.00906, 1.00997, 1.01087, 1.01178, 1.01269, 1.0136]
                                 }
    curve_dict['vertical'][6.] = {'x': [3.78, 4.14, 4.51, 4.87, 5.02, 5.17, 5.33, 6.18, 7.61, 9.24, 11.41, 14.38, 19.35, 26.98, 36.56, 46.85, 56.46, 64.92, 73.48, 83.24, 94.03, 104.8, 115.4, 125.78, 136.57, 147.15, 157.14, 167.13, 177.14, 187.15, 197.31, 207.58, 217.64, 227.29, 236.95, 246.64, 256.44, 266.23, 275.89, 285.58, 295.86, 306.07, 314.81, 323.61, 333.68, 343.82, 354.15, 364.75, 375.31, 385.81, 396.27, 406.65, 417.4, 428.19, 438.99, 449.63, 460.06, 469.83, 479.69, 489.72, 500.42, 511.12, 521.82, 532.57, 543.35, 554.13, 564.91, 575.7, 586.49, 597.28, 608.06, 618.83, 629.61, 640.39, 651.17, 661.96, 672.76, 683.56, 694.35, 705.13, 715.9, 726.68, 737.46, 748.23, 759.01, 769.79, 780.57, 791.32],
                                 'u': [0.58654, 0.60401, 0.62148, 0.63896, 0.65644, 0.67392, 0.6914, 0.7088, 0.72613, 0.7434, 0.76052, 0.77719, 0.79271, 0.8047, 0.81031, 0.80597, 0.81031, 0.82102, 0.83147, 0.8382, 0.83763, 0.83656, 0.83378, 0.82973, 0.83062, 0.83294, 0.83958, 0.8462, 0.85275, 0.85931, 0.86517, 0.87058, 0.87682, 0.88464, 0.89246, 0.90017, 0.9075, 0.91486, 0.92268, 0.93036, 0.93569, 0.93275, 0.92271, 0.9126, 0.917, 0.92302, 0.92791, 0.93116, 0.93483, 0.93888, 0.9432, 0.94801, 0.94902, 0.9493, 0.9493, 0.95222, 0.9563, 0.96374, 0.97085, 0.9773, 0.97962, 0.98195, 0.98428, 0.98573, 0.98668, 0.98762, 0.98857, 0.9892, 0.98982, 0.99046, 0.99144, 0.99243, 0.99342, 0.99441, 0.9954, 0.99563, 0.99563, 0.99563, 0.99585, 0.99688, 0.99792, 0.99895, 0.99998, 1.00101, 1.00205, 1.00308, 1.00411, 1.00563]
                                 }

    curve_dict['horizontal_light'][0.] = {'x': [0, 25.44, 103.55, 181.07, 217.16, 252.07, 282.84, 349.7, 401.78, 454.44, 489.94, 529.59, 568.64, 612.43, 682.25, 798.22],
                                         'u': [1.01365, 1.01075, 1.00987, 0.99827, 0.98857, 0.97692, 0.96331, 0.94779, 0.94688, 0.94695, 0.95089, 0.96556, 0.97925, 0.99392, 1.00181, 1.00487]
                                         }
    curve_dict['horizontal_dark'][0.] = {'x': [0.59, 12.42, 24.25, 36.08, 47.91, 59.74, 71.56, 83.4, 95.23, 107.06, 118.89, 130.71, 142.53, 154.34, 166.16, 177.97, 189.78, 201.58, 213.32, 225.06, 236.75, 248.37, 260.01, 271.76, 280.8, 286.35, 291.5, 297.55, 303.61, 310.48, 317.68, 325.99, 334.45, 342.48, 351.12, 361.25, 372.73, 384.56, 396.39, 408.13, 419.71, 430.69, 440.76, 450.17, 459.29, 466.73, 474, 481.04, 488.96, 497.17, 503.64, 510.57, 520.48, 530.97, 542.61, 554.44, 566.26, 578.03, 589.8, 601.57, 613.33, 625.1, 636.87, 648.69, 660.5, 672.33, 684.16, 695.99, 707.83, 719.66, 731.5, 743.33, 755.17, 767, 778.83, 790.67],
                                        'u': [1.01267, 1.01209, 1.0115, 1.01092, 1.01123, 1.01184, 1.01246, 1.0129, 1.01317, 1.01344, 1.01371, 1.0146, 1.01576, 1.01692, 1.01811, 1.01935, 1.0206, 1.02202, 1.02449, 1.02697, 1.02991, 1.03365, 1.03707, 1.03932, 1.03039, 1.01325, 0.99573, 0.97898, 0.96224, 0.94641, 0.93095, 0.91713, 0.90352, 0.8892, 0.87604, 0.86649, 0.86199, 0.86133, 0.86159, 0.86348, 0.86751, 0.87403, 0.88426, 0.89609, 0.90844, 0.92362, 0.939, 0.95468, 0.96913, 0.98318, 0.99941, 1.01477, 1.02543, 1.03335, 1.0299, 1.02992, 1.02951, 1.02745, 1.02539, 1.02333, 1.02129, 1.01925, 1.01727, 1.01614, 1.01501, 1.01442, 1.01435, 1.01427, 1.0142, 1.01413, 1.01405, 1.01398, 1.01391, 1.01383, 1.01376, 1.01369]
                                        }

    curve_dict['horizontal_light'][2.] = {'x': [1.18, 52.07, 109.47, 175.15, 185.21, 192.9, 201.18, 214.79, 227.22, 233.14, 242.01, 256.21, 277.51, 295.27, 303.55, 311.24, 323.08, 334.91, 341.42, 350.3, 355.03, 367.46, 376.33, 382.84, 391.12, 412.43, 421.89, 428.4, 440.24, 457.99, 466.27, 471.6, 484.02, 494.08, 500, 506.51, 514.2, 527.22, 541.42, 561.54, 571.6, 585.8, 599.41, 613.02, 636.09, 656.21, 691.72, 799.41],
                                         'u': [1.02079, 1.01683, 1.02277, 1.02376, 1.02376, 1.01782, 0.99604, 0.97228, 0.94752, 0.94356, 0.92277, 0.89109, 0.8505, 0.8198, 0.81188, 0.81188, 0.8198, 0.83168, 0.83663, 0.83267, 0.82079, 0.79703, 0.78218, 0.78119, 0.78317, 0.78416, 0.78713, 0.79703, 0.82277, 0.84851, 0.8495, 0.8495, 0.83168, 0.80693, 0.79307, 0.79109, 0.79307, 0.80891, 0.83069, 0.84158, 0.85248, 0.87327, 0.91584, 0.95644, 1.00495, 1.01386, 1.01485, 1.02079]
                                         }
    curve_dict['horizontal_dark'][2.] = {'x': [1.18, 13.02, 24.85, 36.68, 48.51, 60.34, 72.18, 84.01, 95.84, 107.67, 119.51, 131.34, 143.17, 155, 166.83, 178.66, 190.49, 202.31, 214.14, 225.96, 237.38, 248.27, 258.92, 269.64, 279.96, 289.43, 299.16, 309.17, 319.14, 330.08, 341.91, 353.71, 365.51, 377.33, 389.16, 400.99, 412.81, 424.61, 436.41, 448.2, 459.57, 469.46, 479.57, 490.15, 500.5, 510.62, 520.09, 529.74, 541.05, 552.48, 564.3, 576.11, 587.93, 599.75, 611.58, 623.41, 635.24, 647.08, 658.91, 670.74, 682.58, 694.41, 706.25, 718.08, 729.91, 741.75, 753.58, 765.42, 777.25, 789.08],
                                        'u': [1.01881, 1.01844, 1.01807, 1.0177, 1.01733, 1.01696, 1.01708, 1.01746, 1.01784, 1.01822, 1.0186, 1.01863, 1.01821, 1.01779, 1.01737, 1.01696, 1.01622, 1.01534, 1.01485, 1.01462, 1.00947, 1.00189, 0.99325, 0.98487, 0.97539, 0.96351, 0.95225, 0.94169, 0.93108, 0.92716, 0.92752, 0.92608, 0.92465, 0.9239, 0.92424, 0.92459, 0.92556, 0.92708, 0.9286, 0.9271, 0.92813, 0.93899, 0.94924, 0.9581, 0.96768, 0.97794, 0.98971, 1.00061, 1.00644, 1.01114, 1.01221, 1.01328, 1.01435, 1.01517, 1.01578, 1.01639, 1.01687, 1.01701, 1.01715, 1.0173, 1.01744, 1.01758, 1.01772, 1.01786, 1.018, 1.01814, 1.01829, 1.01843, 1.01857, 1.01871]
                                        }

    curve_dict['horizontal_light'][4.] = {'x': [1.19, 95.38, 135.92, 157.97, 172.88, 190.76, 210.43, 228.32, 244.41, 256.93, 265.87, 276.01, 287.33, 309.39, 318.33, 328.46, 340.39, 348.73, 356.48, 363.64, 374.37, 382.71, 389.87, 397.62, 408.94, 424.44, 438.75, 450.07, 461.4, 471.54, 481.67, 494.19, 512.67, 525.78, 536.51, 552.01, 572.88, 582.41, 604.47, 623.55, 652.16, 674.22, 708.79, 798.81],
                                         'u': [1.0167, 1.01948, 1.01299, 1.00928, 0.99536, 0.97959, 0.96197, 0.95455, 0.95269, 0.9564, 0.95362, 0.94527, 0.93043, 0.90631, 0.89889, 0.89981, 0.90631, 0.90631, 0.90445, 0.89332, 0.88126, 0.87291, 0.8757, 0.88683, 0.90816, 0.92022, 0.93135, 0.93135, 0.92301, 0.90538, 0.89054, 0.87848, 0.88776, 0.89981, 0.91095, 0.92022, 0.92022, 0.92579, 0.93414, 0.95269, 0.97774, 1.00464, 1.01763, 1.01763]
                                         }
    curve_dict['horizontal_dark'][4.] = {'x': [0, 11.92, 23.84, 35.77, 47.69, 59.61, 71.53, 83.45, 95.38, 107.3, 119.22, 131.14, 143.07, 154.99, 166.91, 178.8, 190.66, 202.5, 214.34, 225.96, 237.42, 248.88, 260.27, 271.65, 283.11, 294.59, 306.37, 318.22, 330.08, 342, 353.92, 365.84, 377.77, 389.69, 401.59, 413.44, 425.29, 437.14, 449.04, 460.93, 472.83, 484.75, 496.59, 508.02, 519.16, 530.19, 541.19, 552.12, 563.06, 574.71, 586.47, 598.34, 610.26, 622.17, 634.09, 646.01, 657.94, 669.86, 681.78, 693.7, 705.62, 717.54, 729.47, 741.39, 753.31, 765.23, 777.15, 789.08],
                                        'u': [1.0167, 1.01688, 1.01705, 1.01723, 1.01741, 1.01759, 1.01763, 1.01763, 1.01763, 1.01763, 1.01763, 1.01763, 1.01763, 1.01763, 1.01763, 1.01649, 1.0146, 1.01248, 1.01023, 1.00635, 1.00125, 0.99614, 0.99063, 0.98512, 0.97998, 0.97501, 0.97238, 0.97033, 0.96846, 0.96846, 0.96846, 0.96846, 0.96846, 0.96846, 0.96797, 0.96589, 0.96381, 0.96211, 0.96334, 0.96458, 0.96573, 0.9662, 0.96744, 0.97271, 0.9793, 0.98632, 0.99348, 1.00088, 1.00828, 1.01192, 1.01497, 1.01624, 1.01688, 1.01752, 1.01778, 1.01798, 1.01817, 1.01836, 1.01855, 1.01874, 1.01893, 1.01912, 1.01931, 1.01951, 1.0197, 1.01989, 1.02008, 1.02027]
                                        }

    curve_dict['horizontal_light'][6.] = {'x': [-0.59, 82.84, 133.73, 204.14, 239.05, 250.89, 276.33, 305.33, 321.89, 336.09, 345.56, 355.62, 367.46, 388.76, 410.06, 428.99, 447.93, 463.31, 472.19, 485.21, 498.82, 514.79, 533.73, 552.07, 577.51, 598.22, 623.67, 694.67, 796.45],
                                         'u': [1.01724, 1.01437, 1.01341, 0.98467, 0.97031, 0.96264, 0.96169, 0.95594, 0.9636, 0.97222, 0.97222, 0.96743, 0.95402, 0.94636, 0.94923, 0.95785, 0.9636, 0.95594, 0.94061, 0.92816, 0.92816, 0.93678, 0.94636, 0.95115, 0.95402, 0.96264, 0.97414, 1.00958, 1.01341]
                                         }
    curve_dict['horizontal_dark'][6.] = {'x': [1.18, 13.02, 24.85, 36.68, 48.52, 60.35, 72.18, 84, 95.82, 107.65, 119.47, 131.31, 143.14, 154.98, 166.81, 178.64, 190.48, 202.31, 214.13, 224.54, 235.7, 247.34, 259.01, 270.78, 282.55, 294.34, 306.17, 318.01, 329.84, 341.67, 353.51, 365.34, 377.17, 388.99, 400.82, 412.65, 424.44, 436.23, 448.06, 459.89, 471.72, 483.56, 495.36, 507.16, 518.92, 530.67, 542.37, 554.04, 565.54, 576.98, 588.6, 600.3, 612.1, 623.92, 635.76, 647.59, 659.42, 671.26, 683.09, 694.93, 706.76, 718.6, 730.43, 742.26, 754.1, 765.93, 777.77, 789.6],
                                        'u': [1.01628, 1.01647, 1.01666, 1.01684, 1.01703, 1.01721, 1.01653, 1.0157, 1.01487, 1.01404, 1.01341, 1.01341, 1.01341, 1.01341, 1.01341, 1.01341, 1.01341, 1.01341, 1.01261, 1.00413, 0.99861, 0.99515, 0.99197, 0.99004, 0.9881, 0.98659, 0.98659, 0.98659, 0.98659, 0.98689, 0.98738, 0.98786, 0.98835, 0.98896, 0.98963, 0.9903, 0.98891, 0.98766, 0.9881, 0.98853, 0.98894, 0.98935, 0.98839, 0.98688, 0.98841, 0.99068, 0.99355, 0.9967, 1.00109, 1.00593, 1.00958, 1.01218, 1.01361, 1.01437, 1.01437, 1.01437, 1.01437, 1.01437, 1.01437, 1.01437, 1.01437, 1.01437, 1.01437, 1.01437, 1.01437, 1.01437, 1.01437, 1.01437]
                                        }

    if z_offset_local == 0:
        local_series = 'vertical'
    elif z_offset_local < 0:
        local_series = 'horizontal_light'
    elif z_offset_local > 0:
        local_series = 'horizontal_dark'
    else:
        message = 'something went wrong with assigning Haas2019 reference velocity deficits'
        print_op.log_and_raise_error(message)

    local_curve = curve_dict[local_series][x_over_d_local]

    if 'horizontal' in local_series:
        local_curve['x'] = list(np.array(local_curve['x']) - np.array([400.] * len(local_curve['x'])))

    ax.plot(local_curve['u'], local_curve['x'], linestyle='dotted', c='b', label='Haas 2019')
    return None


def make_individual_time_averaged_velocity_deficit_subplot(ax, plot_dict, cosmetics, x_offset=0., z_offset=0., slice_axes=vect_op.zhat_dm(), add_legend_labels=False, suppress_wind_options_import_warning=True):
    s_sym = cas.SX.sym('s_sym')

    # if not suppress_wind_options_import_warning:
    #     wind.warn_about_importing_from_options()

    kite_plane_induction_params = get_kite_plane_induction_params(plot_dict, 0)
    x_center = kite_plane_induction_params['center']

    z_min = 0 #plot_dict['options']['model.system_bounds.x.q'][0][2]
    z_max = 800 #x_center[2] + kite_plane_induction_params['mu_end_by_path'] * radius
    y_offset = (z_max - z_min) / 2.
    y_max = x_center[1] + y_offset #kite_plane_induction_params['mu_end_by_path'] * radius
    y_min = x_center[1] - y_offset

    u_infty_proj_list = []
    u_wind_proj_list = []

    n_interpolation = plot_dict['interpolation_si']['x']['q10'][0].shape[0]
    for idx_at_eval in range(n_interpolation):

        x_obs = x_center + x_offset * vect_op.xhat_dm() + z_offset * vect_op.zhat_dm() + s_sym * slice_axes
        u_wind_proj, u_infty_proj = get_total_wind_at_observer_function(plot_dict, cosmetics, idx_at_eval=idx_at_eval, direction_induction='wind', x_obs=x_obs)

        u_infty_proj_list = cas.vertcat(u_infty_proj_list, u_infty_proj)
        u_wind_proj_list = cas.vertcat(u_wind_proj_list, u_wind_proj)

    temp_average_u_infty = vect_op.average(u_infty_proj_list)
    temp_average_u_wind = vect_op.average(u_wind_proj_list)

    u_infty_avg_proj_fun = cas.Function('u_infty_avg_proj_fun', [s_sym], [temp_average_u_infty])
    u_wind_avg_proj_fun = cas.Function('u_wind_avg_proj_fun', [s_sym], [temp_average_u_wind])

    n_points = int(z_max - z_min) # evaluate every meter
    param_range = np.arange(0., 1. + 1./float(n_points), 1./float(n_points))
    z_vals = cas.DM([z_min + (z_max - z_min) * p_val for p_val in param_range]).T
    y_vals = cas.DM([y_min + (y_max - y_min) * p_val for p_val in param_range]).T

    if slice_axes[1] == 1:
        s_vals = y_vals
        y_axis_vals = y_vals
    elif slice_axes[2] == 1:
        # notice above !!
        # z_obs = z_center + z_offset + s
        # s = z_target - z_center - z_offset
        s_vals = []
        for idx in range(z_vals.shape[1]):
            local_s = z_vals[0, idx] - x_center[2] - z_offset
            s_vals = cas.horzcat(s_vals, local_s)
        y_axis_vals = z_vals

    else:
        message = 'awebox is not yet set up to make velocity deficit plots across axis ' + repr(slice_axes)
        message += '. skipping this plotting request.'
        print_op.base_print(message, level='warning')
        return None

    u_normalizing = u_infty_avg_proj_fun(np.max(np.array(s_vals)))

    u_infty_nn_fun = cas.Function('u_infty_nn_fun', [s_sym], [u_infty_avg_proj_fun(s_sym) / u_normalizing])
    u_wind_nn_fun = cas.Function('u_wind_nn_fun', [s_sym], [u_wind_avg_proj_fun(s_sym) / u_normalizing])

    entries = s_vals.shape[1]
    parallelization_type = plot_dict['options']['model']['construction']['parallelization']['type']
    u_infty_nn_map = u_infty_nn_fun.map(entries, parallelization_type)
    u_wind_nn_map = u_wind_nn_fun.map(entries, parallelization_type)

    def prep_to_plot(cas_dm_array):
        return list(np.array(vect_op.columnize(cas_dm_array)))

    u_infty_nn_vals = prep_to_plot(u_infty_nn_map(s_vals))
    u_wind_nn_vals = prep_to_plot(u_wind_nn_map(s_vals))
    y_axis_vals = prep_to_plot(y_axis_vals)

    if add_legend_labels:
        ax.plot(u_infty_nn_vals, y_axis_vals, linestyle='--', label='free_stream', c='k')
        ax.plot(u_wind_nn_vals, y_axis_vals, linestyle='-', label='modeled', c='b')
    else:
        ax.plot(u_infty_nn_vals, y_axis_vals, linestyle='--', c='k')
        ax.plot(u_wind_nn_vals, y_axis_vals, linestyle='-', c='b')

    ax.set_xlabel("u/u_limit [-]")
    ax.set_xlim([0.5, 1.25])

    return None


def get_total_wind_at_observer_function(plot_dict, cosmetics, idx_at_eval=0, direction_induction=None, x_obs=None):

    if x_obs is not None:
        x_obs_sym = x_obs
    else:
        x_obs_sym = cas.SX.sym('x_obs_sym', (3, 1))

    u_ind_fun = get_total_biot_savart_at_observer_function(plot_dict, cosmetics, idx_at_eval=idx_at_eval)
    u_ind = u_ind_fun(x_obs_sym)

    wind = plot_dict['wind']
    model_parameters = plot_dict['model_parameters']
    parameters = model_parameters(plot_dict['parameters_plot'])
    u_infty = wind.get_velocity(x_obs_sym[2], external_parameters=parameters)
    ehat_wind = wind.get_wind_direction()

    u_wind = u_infty + u_ind

    if direction_induction is not None:
        if direction_induction == 'wind':
            u_wind = cas.mtimes(u_wind.T, ehat_wind)
            u_infty = cas.mtimes(u_infty.T, ehat_wind)
        else:
            message = 'function is not yet set up for projections in the ' + direction_induction + ' direction'
            print_op.log_and_raise_error(message)

    if x_obs is None:
        u_wind_fun = cas.Function('u_wind_fun', [x_obs_sym], [u_wind])
        u_infty_fun = cas.Function('u_infty_fun', [x_obs_sym], [u_infty])
        return u_wind_fun, u_infty_fun
    else:
        return u_wind, u_infty


def get_total_biot_savart_at_observer_function(plot_dict, cosmetics, idx_at_eval=0):

    wake = plot_dict['wake']
    x_obs_sym = cas.SX.sym('x_obs_sym', (3, 1))

    if wake is not None:
        parameters = plot_dict['parameters_plot']
        variables_scaled = get_variables_scaled(plot_dict, cosmetics, idx_at_eval)
        u_ind_sym = wake.calculate_total_biot_savart_at_x_obs(variables_scaled, parameters, x_obs=x_obs_sym)
        u_ind_fun = cas.Function('u_ind_fun', [x_obs_sym], [u_ind_sym])

    else:
        u_ind_fun = cas.Function('u_ind_fun', [x_obs_sym], [cas.DM.zeros((3, 1))])

    return u_ind_fun

def get_the_induction_factor_at_observer_function(plot_dict, cosmetics, idx_at_eval=0, direction='normal'):

    x_obs_sym = cas.SX.sym('x_obs_sym', (3, 1))
    if 'wake' in plot_dict.keys():
        variables_scaled = get_variables_scaled(plot_dict, cosmetics, idx_at_eval)
        parameters = plot_dict['parameters_plot']
        wake = plot_dict['wake']
        u_ind_sym = wake.calculate_total_biot_savart_at_x_obs(variables_scaled, parameters, x_obs=x_obs_sym)

        n_hat, _, _ = get_coordinate_axes(plot_dict, idx_at_eval, direction)

        u_normalizing = get_induction_factor_normalizing_speed(plot_dict, idx_at_eval)

        a_sym = general_flow.compute_induction_factor(u_ind_sym, n_hat, u_normalizing)
    else:
        a_sym = cas.DM.zero((1, 1))
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


def compute_the_scaled_haas_error(plot_dict, cosmetics, direction_induction='normal'):

    # reference points digitized from haas 2017
    data02 = [(0, -1.13084, -0.28134), (0, -0.89466, -0.55261), (0, -0.79354, -0.41195), (0, -0.79127, 0.02826), (0, -0.04449, 1.01208), (0, 0.22119, 0.83066), (0, 0.71947, -0.96407), (0, 0.52187, -0.66125), (0, 0.45227, 1.09891), (0, 0.92121, -0.46219), (0, 0.95625, -0.62566), (0, 0.19546, 1.12782)]
    data00 = [(0, -0.69372, 1.24284), (0, -0.5894, -0.17062), (0, -0.83795, -1.18138), (0, 0.27077, 1.35428), (0, -1.20982, -0.71437), (0, -0.10682, 0.54558), (0, -0.3789, -0.3959), (0, 0.5081, -0.34622), (0, 0.23767, 0.61635), (0, 0.57386, -0.09835), (0, 1.10956, -0.867), (0, 1.4683, -0.06319)]
    datan005 = [(0, -1.27365, 0.33177), (0, -1.21638, 0.65872), (0, -0.64295, 0.15234), (0, -0.50651, 0.31282), (0, -0.01271, -1.39519), (0, 0.03669, -0.60921), (0, 0.20264, -0.63812), (0, 0.38143, -1.27208), (0, 0.49584, 0.49075), (0, 0.52938, 0.32778), (0, 0.96982, 0.94579), (0, 1.3061, 0.48278)]

    data = {-0.05: datan005, 0.0: data00, 0.2: data02}

    idx_at_eval = plot_dict['options']['visualization']['cosmetics']['animation']['snapshot_index']

    kite_plane_induction_params = get_kite_plane_induction_params(plot_dict, idx_at_eval)
    radius = kite_plane_induction_params['average_radius']
    x_center = kite_plane_induction_params['center']

    a_fun = get_the_induction_factor_at_observer_function(plot_dict, cosmetics, idx_at_eval, direction=direction_induction)

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

def get_kite_plane_induction_params(plot_dict, idx_at_eval, suppress_wind_options_import_warning=True):

    kite_plane_induction_params = {}

    interpolated_outputs_si = plot_dict['interpolation_si']['outputs']

    layer_nodes = plot_dict['architecture'].layer_nodes
    layer = int( np.min(np.array(layer_nodes)) )
    kite_plane_induction_params['layer'] = layer

    b_ref = plot_dict['options']['model']['params']['geometry']['b_ref']
    average_radius = np.average(np.array(interpolated_outputs_si['geometry']['average_radius' + str(layer)][0]))
    kite_plane_induction_params['average_radius'] = average_radius

    center = []
    for dim in range(3):
        local_center = np.average(np.array(interpolated_outputs_si['performance']['trajectory_center' + str(layer)][dim]))
        center = cas.vertcat(center, local_center)
    kite_plane_induction_params['center'] = center

    wind_model = plot_dict['options']['model']['wind']['model']
    u_ref = plot_dict['options']['user_options']['wind']['u_ref']
    z_ref = plot_dict['options']['model']['params']['wind']['z_ref']
    z0_air = plot_dict['options']['model']['params']['wind']['log_wind']['z0_air']
    exp_ref = plot_dict['options']['model']['params']['wind']['power_wind']['exp_ref']
    wind = plot_dict['wind']
    # u_infty = wind.get_speed(wind_model, u_ref, z_ref, z0_air, exp_ref, center[2])
    # if not suppress_wind_options_import_warning:
    #     wind.warn_about_importing_from_options()
    kite_plane_induction_params['u_infty'] = wind.get_velocity(center[2])[0]

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


def is_this_a_haas2017_test(plot_dict, kite_plane_induction_params, variables_si, thresh=0.01):
    test_name = 'Haas 2017'
    verification_dict = {'radius': {'found': kite_plane_induction_params['average_radius'],
                                    'expected': 155.77},
                         'period': {'found': plot_dict['time_grids']['ip'][-1],
                                    'expected': 2. * np.pi / (7. * 10. / 155.77)},
                         'diam_t': {'found': variables_si['theta', 'diam_t'],
                                    'expected': 5e-3},
                         'x_center_altitude': {'found': kite_plane_induction_params['center'][2],
                                               'expected': 0.},
                         'kite_span': {'found': plot_dict['options']['model']['params']['geometry']['b_ref'],
                                       'expected': 68},
                         'kite_dof': {'found': plot_dict['options']['user_options']['system_model']['kite_dof'],
                                      'expected': 6},
                         'u_ref': {'found': plot_dict['options']['user_options']['wind']['u_ref'],
                                   'expected': 10.}
                       }
    try:
        verification_dict['l_s'] = {'found': variables_si['theta', 'l_s'],
                'expected': 400.}
        verification_dict['diam_s'] = {'found': variables_si['theta', 'diam_s'],
                   'expected': 5e-3}
    except:
        pass
    return is_this_a_verification_test(test_name, verification_dict, thresh=thresh)


def is_this_a_haas2019_test(plot_dict, kite_plane_induction_params, variables_si, thresh=0.01):
    test_name = 'Haas 2019'
    verification_dict = {'period': {'found': plot_dict['time_grids']['ip'][-1],
                                    'expected': 43.8178},
                         'diam_t': {'found': variables_si['theta', 'diam_t'],
                                    'expected': 9.55e-2},
                         'kite_span': {'found': plot_dict['options']['model']['params']['geometry']['b_ref'],
                                       'expected': 60},
                         'kite_dof': {'found': plot_dict['options']['user_options']['system_model']['kite_dof'],
                                      'expected': 3},
                         'u_ref': {'found': plot_dict['options']['user_options']['wind']['u_ref'],
                                   'expected': 10.},
                         'wind_reference': {'found': plot_dict['options']['model']['params']['wind']['log_wind']['z0_air'],
                                            'expected': 0.0002}
                         }
    return is_this_a_verification_test(test_name, verification_dict, thresh=thresh)


def is_this_a_verification_test(test_name, verification_dict, thresh=0.01):

    this_is_a_verification_test = True
    for comparison, val_dict in verification_dict.items():
        if np.abs(val_dict['expected']) > thresh:
            normalization = np.abs(val_dict['expected'])
        else:
            normalization = 1.

        error = (val_dict['expected'] - val_dict['found']) / normalization
        if np.abs(error) > thresh:
            this_is_a_verification_test = False

    if this_is_a_verification_test:
        message = "It seems like you're trying to perform a verification test using " + test_name + "'s published induction values. We're going to proceed as though it is!"
        print_op.base_print(message, level='warning')

    return this_is_a_verification_test


def plot_induction_contour_on_kmp(plot_dict, cosmetics, fig_name, fig_num=None, direction_plotting='normal', direction_induction='normal'):

    vortex_info_exists = ('wake' in plot_dict.keys()) and (plot_dict['wake'] is not None)
    if vortex_info_exists:

        # include this once for the error message
        _ = get_kite_plane_induction_params(plot_dict, 0, suppress_wind_options_import_warning=False)

        tau_style_dict = tools.get_temporal_orientation_epigraphs_taus_and_linestyles(plot_dict)
        print_op.base_print('plotting the induction contours at taus in ' + repr(tau_style_dict.keys()))
        for tau_at_eval in tau_style_dict.keys():
            tau_rounded = np.round(tau_at_eval, 2)

            n_points_contour = plot_dict['cosmetics']['induction']['n_points_contour']
            n_points_interpolation = plot_dict['cosmetics']['interpolation']['n_points']

            print_op.base_print('generating_induction_factor_casadi_function...')
            idx_at_eval = int(np.floor((float(n_points_interpolation) -1.)  * tau_at_eval))
            a_fun = get_the_induction_factor_at_observer_function(plot_dict, cosmetics, idx_at_eval, direction=direction_induction)

            print_op.base_print('slicing the variables at the appropriate interpolated time...')
            variables_scaled = get_variables_scaled(plot_dict, cosmetics, idx_at_eval)
            scaling = plot_dict['model_variables'](plot_dict['model_scaling'])
            variables_si = plot_dict['model_variables'](struct_op.variables_scaled_to_si(plot_dict['model_variables'], variables_scaled, scaling))
            kite_plane_induction_params = get_kite_plane_induction_params(plot_dict, idx_at_eval)
            radius = kite_plane_induction_params['average_radius']
            x_center = kite_plane_induction_params['center']

            print_op.base_print('deciding the circumstances of the contour plot...')
            this_is_haas_test = is_this_a_haas2017_test(plot_dict, kite_plane_induction_params, variables_si, thresh=0.01)

            ### compute the induction factors
            print_op.base_print('finding coordinates...')
            side = get_induction_contour_side(plot_dict, idx_at_eval, direction=direction_plotting)
            n_hat, a_hat, b_hat = get_coordinate_axes(plot_dict, idx_at_eval, direction=direction_plotting)

            print_op.base_print('deciding the observation range...')
            plot_radius_scaled = kite_plane_induction_params['mu_end_by_path']
            delta_plot = plot_radius_scaled / float(n_points_contour)
            yy_scaled, zz_scaled = np.meshgrid(np.arange(-1. * plot_radius_scaled, plot_radius_scaled, delta_plot),
                                               np.arange(-1. * plot_radius_scaled, plot_radius_scaled, delta_plot))

            print_op.base_print('making the casadi function map...')
            aa = np.zeros(yy_scaled.shape)
            yy_number_entries = yy_scaled.shape[0] * yy_scaled.shape[1]
            parallelization_type = plot_dict['options']['model']['construction']['parallelization']['type']
            a_map = a_fun.map(yy_number_entries, parallelization_type)

            print_op.base_print('making induction contour plot...')
            total_progress = yy_scaled.shape[0] * yy_scaled.shape[1]

            print_op.base_print('generating observation grid...')
            observation_points_concatenated = []
            pdx = 0
            for idx in range(yy_scaled.shape[0]):
                for jdx in range(yy_scaled.shape[1]):
                    x_obs_centered = yy_scaled[idx, jdx] * radius * a_hat + zz_scaled[idx, jdx] * radius * b_hat
                    x_obs = x_obs_centered + x_center
                    observation_points_concatenated = cas.horzcat(observation_points_concatenated, x_obs)
                    print_op.print_progress(pdx, total_progress)
                    pdx += 1
            print_op.close_progress()

            print_op.base_print('computing induction factors...')
            aa_computed = a_map(observation_points_concatenated)

            print_op.base_print('reassigning computed induction factors...')
            ldx = 0
            for idx in range(yy_scaled.shape[0]):
                for jdx in range(yy_scaled.shape[0]):
                    aa[idx, jdx] = float(aa_computed[ldx])
                    print_op.print_progress(ldx, total_progress)
                    ldx += 1
            print_op.close_progress()

            ### initialize the figure
            fig_new, ax = plt.subplots(num=int(tau_rounded * 1e4), facecolor='none')

            ### draw the swept annulus
            print_op.base_print('drawing swept background...')
            draw_swept_background(ax, side, plot_dict)

            ### draw the contour
            haas_levels = [-0.05, 0., 0.2]
            haas_linestyles = ['dashdot', 'solid', 'dashed']
            haas_colors = ['k', 'k', 'k']
            general_levels = 10 #[-1.0, -0.5, -0.4, -0.3, -0.2, -0.1, -0.05, -0.01, 0., 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
            general_colors = 'k'
            general_linestyles = 'solid'
            if this_is_haas_test and ((np.any(aa < haas_levels[0])) and (np.any(aa > haas_levels[-1]))):
                cs = ax.contour(yy_scaled, zz_scaled, aa, haas_levels, colors=haas_colors, linestyles=haas_linestyles)
            else:
                cs = ax.contour(yy_scaled, zz_scaled, aa, general_levels, colors=general_colors, linestyles=general_linestyles)
            ax.clabel(cs, cs.levels, inline=True)

            ### draw the vortex positions
            parameters = plot_dict['parameters_plot']
            wake = plot_dict['wake']
            bound_wake = wake.get_substructure('bound')

            print_op.base_print('drawing wake...')
            wake.draw(ax, side, variables_scaled=variables_scaled, parameters=parameters, cosmetics=cosmetics)
            # bound_wake.draw(ax, side, variables_scaled=variables_scaled, parameters=parameters, cosmetics=cosmetics)

            ax.grid(True)
            normal_vector_model = plot_dict['options']['model']['aero']['actuator']['normal_vector_model']
            title = 'Induction factor over the kite plane \n tau = ' + str(tau_rounded)
            title += ' w. n_hat model ' + normal_vector_model
            if this_is_haas_test:
                scaled_haas_error = compute_the_scaled_haas_error(plot_dict, cosmetics, direction_induction=direction_induction)
                title += '; scaled_haas_error = ' + str(scaled_haas_error)

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

            print_op.base_print('saving figure at tau = ' + str(tau_rounded) + '...')
            fig_new.savefig('figures/' + plot_dict['name'] + '_' + direction_induction + '_induction_contour_on_' + direction_plotting + '_kmp_tau' + str(tau_rounded) + '.pdf')

        print_op.base_print('done with induction contour plotting!')

    return None


def get_induction_contour_side(plot_dict, idx_at_eval, direction='normal'):
    kite_plane_induction_params = get_kite_plane_induction_params(plot_dict, idx_at_eval)

    radius = kite_plane_induction_params['average_radius']
    x_center = kite_plane_induction_params['center']
    n_hat, a_hat, b_hat = get_coordinate_axes(plot_dict, idx_at_eval, direction=direction)

    side = (x_center, a_hat, b_hat, 1./radius)
    return side


def draw_swept_background(ax, side, plot_dict):
    for kite in plot_dict['architecture'].kite_nodes:
        for zeta in np.arange(-0.5, 0.5, 1. / 100.):
            tools.plot_path_of_wingtip(ax, side, plot_dict, kite, zeta, color='gray', alpha=0.2)
    return None

def plot_induction_contour_on_vwt_cross_sections(plot_dict, cosmetics, fig_name, fig_num=None, direction_induction='wind'):

    vortex_info_exists = ('wake' in plot_dict.keys()) and (plot_dict['wake'] is not None)
    if vortex_info_exists:

        # include this once for the error message
        _ = get_kite_plane_induction_params(plot_dict, 0, suppress_wind_options_import_warning=False)

        tau_style_dict = tools.get_temporal_orientation_epigraphs_taus_and_linestyles(plot_dict)
        print_op.base_print('plotting the induction contours at taus in ' + repr(tau_style_dict.keys()))
        for tau_at_eval in tau_style_dict.keys():
            tau_rounded = np.round(tau_at_eval, 2)

            n_points_contour = plot_dict['cosmetics']['induction']['n_points_contour']
            n_points_interpolation = plot_dict['cosmetics']['interpolation']['n_points']

            print_op.base_print('generating_induction_factor_casadi_function...')
            idx_at_eval = int(np.floor((float(n_points_interpolation) -1.)  * tau_at_eval))
            u_wind_proj_fun, _ = get_total_wind_at_observer_function(plot_dict, cosmetics, idx_at_eval=idx_at_eval, direction_induction=direction_induction)

            ### initialize the figure
            fig_new, axes = plt.subplots(num=int(tau_rounded * 1e5), facecolor='none', nrows=2, ncols=1)

            print_op.base_print('slicing the variables at the appropriate interpolated time...')
            variables_scaled = get_variables_scaled(plot_dict, cosmetics, idx_at_eval)
            scaling = plot_dict['model_variables'](plot_dict['model_scaling'])
            variables_si = struct_op.variables_scaled_to_si(plot_dict['model_variables'], variables_scaled, scaling)
            kite_plane_induction_params = get_kite_plane_induction_params(plot_dict, idx_at_eval)

            print_op.base_print('deciding the circumstances of the contour plot...')
            this_is_haas_test = is_this_a_haas2019_test(plot_dict, kite_plane_induction_params, variables_si,
                                                        thresh=0.01)

            side_axes_dict = {0: 'z', 1: 'y'}
            for adx in side_axes_dict.keys():
                ax = axes[adx]
        
                ### compute the induction factors
                print_op.base_print('finding coordinates...')
                side = 'x' + side_axes_dict[adx]

                print_op.base_print('deciding the observation range...')
                x_min = 0.
                z_min = 0.
                if this_is_haas_test:
                    x_max = 2400.
                    y_min = -400.
                    y_max = 400.
                    z_max = 800.
                    position_shift_xy_in_z = 260 * vect_op.zhat_dm()
                else:
                    x_center = kite_plane_induction_params['center']
                    diameter = 2. * kite_plane_induction_params['average_radius']
                    x_max = x_center[0] + 6.5 * diameter
                    z_max = x_center[2] + 2. * diameter
                    y_max = z_max / 2.
                    y_min = -1. * y_max
                    position_shift_xy_in_z = x_center[2] * vect_op.zhat_dm()
                ax.set_aspect('equal')

                x_hat = vect_op.xhat_dm()
                if side == 'xy':
                    a_hat = vect_op.yhat_dm()
                    a_min = y_min
                    a_max = y_max
                    position_shift = position_shift_xy_in_z
                elif side == 'xz':
                    a_hat = vect_op.zhat_dm()
                    a_min = z_min
                    a_max = z_max
                    position_shift = cas.DM.zeros((3, 1))

                ratio_points = float((x_max - x_min) / (a_max - a_min))
                n_x = int(np.round(ratio_points * n_points_contour))
                n_a = n_points_contour
                xx_mesh, aa_mesh = np.meshgrid(np.linspace(x_min, x_max, n_x),
                                               np.linspace(a_min, a_max, n_a),
                                               indexing='xy'
                                               )

                print_op.base_print('making the casadi function map...')

                xx_number_entries = xx_mesh.shape[0] * xx_mesh.shape[1]
                parallelization_type = plot_dict['options']['model']['construction']['parallelization']['type']
                u_map = u_wind_proj_fun.map(xx_number_entries, parallelization_type)
    
                print_op.base_print('making induction contour plot...')
                total_progress = xx_mesh.shape[0] * xx_mesh.shape[1]
                print_op.base_print('generating observation grid...')
                observation_points_concatenated = []
                pdx = 0
                for idx in range(xx_mesh.shape[0]):
                    for jdx in range(xx_mesh.shape[1]):
                        x_obs = xx_mesh[idx, jdx] * vect_op.xhat_dm() + aa_mesh[idx, jdx] * a_hat + position_shift
                        observation_points_concatenated = cas.horzcat(observation_points_concatenated, x_obs)
                        print_op.print_progress(pdx, total_progress)
                        pdx += 1
                print_op.close_progress()
    
                print_op.base_print('computing induction factors...')
                uu_computed = u_map(observation_points_concatenated)
    
                print_op.base_print('reassigning computed induction factors...')
                ldx = 0
                uu = np.zeros(xx_mesh.shape)
                for idx in range(xx_mesh.shape[0]):
                    for jdx in range(xx_mesh.shape[1]):
                        uu[idx, jdx] = float(uu_computed[ldx])
                        print_op.print_progress(ldx, total_progress)
                        ldx += 1
                print_op.close_progress()

                ### draw the contour
                u_min = 0
                if this_is_haas_test:
                    u_max = 15.
                else:
                    _, vec_u_max = get_total_wind_at_observer_function(plot_dict, cosmetics, idx_at_eval=0,
                                                                   direction_induction='wind',
                                                                   x_obs=z_max * vect_op.zhat_dm())
                    u_max = float(vec_u_max[0])

                general_levels = np.linspace(u_min, u_max, n_points_contour)

                cmap = 'viridis' #'YlGnBu_r'
                cs = ax.contourf(xx_mesh, aa_mesh, uu, general_levels, cmap=cmap)
                cbar = fig_new.colorbar(cs, ax=ax)
                clines = ax.contour(xx_mesh, aa_mesh, uu, general_levels, cmap=cmap)
                tick_locator = ticker.MaxNLocator(nbins=6)
                cbar.locator = tick_locator
                cbar.update_ticks()
                cbar.ax.set_ylabel('u [m/s]', rotation=90)

                ### draw the trajectory
                print_op.base_print('drawing trajectory...')
                temp_cosmetics = copy.deepcopy(cosmetics)
                temp_cosmetics['trajectory']['tethers'] = False
                temp_cosmetics['trajectory']['colors'] = ['k'] * 20
                tools.plot_trajectory_contents(ax, plot_dict, temp_cosmetics, side, plot_kites=True, linewidth=0.25, idx_at_eval=idx_at_eval)

                # title
                ax.set_xlabel(side[0] + " [m]")
                ax.set_ylabel(side[1] + " [m]")

            title = 'Instantanous (' + direction_induction + ' dir.) flow development, at tau = ' + str(tau_rounded)
            axes[0].set_title(title)

            print_op.base_print('saving figure at tau = ' + str(tau_rounded) + '...')
            fig_new.savefig('figures/' + plot_dict['name'] + '_' + direction_induction + '_induction_contour_vwt_cs' + str(tau_rounded) + '.pdf')

        print_op.base_print('done with induction contour on vwt plotting!')

    return None
