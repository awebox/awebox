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
import awebox.tools.struct_operations as struct_op
from itertools import chain
import matplotlib.colors as colors
import matplotlib.cm as cmx
import awebox.tools.vector_operations as vect_op
import awebox.opti.diagnostics as diagnostics
from awebox.logger.logger import Logger as awelogger
import awebox.tools.print_operations as print_op

def get_naca_airfoil_coordinates(s, m, p, t):

    if s < p:
        yc = m / p**2. * (2. * p * s - s**2.)

        dycdx = 2. * m / p**2. * (p - s)

    else:
        yc = m/ (1. - p)**2. * ((1. - 2. * p) + 2. * p * s - s**2.)

        dycdx = 2. * m / (1. - p) ** 2. * (p - s)

    yt = 5. * t * (0.2969 * s**0.5 - 0.1260 * s - 0.3515 * s**2. + 0.2843 * s**3. - 0.1015 * s**5.)

    theta = np.arctan(dycdx)

    xu = s - yt * np.sin(theta)
    xl = s + yt * np.sin(theta)

    yu = yc + yt * np.cos(theta)
    yl = yc - yt * np.cos(theta)

    return xu, xl, yu, yl


def get_naca_shell(chord, naca="0012", center_at_quarter_chord = True):

    m = float(naca[0]) / 100.
    p = float(naca[1]) / 10.
    t = float(naca[2:]) / 100.

    s_list = np.arange(0., 101.) / 100.

    x_upper = []
    x_lower = []

    for s in s_list:
        xu, xl, yu, yl = get_naca_airfoil_coordinates(s, m, p, t)

        new_x_upper = xu * vect_op.xhat_np() + yu * vect_op.zhat_np()
        new_x_lower = xl * vect_op.xhat_np() + yl * vect_op.zhat_np()

        if center_at_quarter_chord:
            new_x_upper = new_x_upper - vect_op.xhat_np() / 4.
            new_x_lower = new_x_lower - vect_op.xhat_np() / 4.

        x_upper = cas.vertcat(x_upper, (chord * new_x_upper.T))
        x_lower = cas.vertcat(x_lower, (chord * new_x_lower.T))

    x_upper = np.array(x_upper)
    x_lower = np.array(x_lower)[::-1]

    x = np.array(cas.vertcat(x_lower, x_upper))

    return x


def draw_lifting_surface(ax, q, r, b_ref, c_tipn, c_root, c_tipp, kite_color, side, body_cross_sections_per_meter, naca="0012", shift_tipn=0., shift_root=0., shift_tipp=0.):

    r_dcm = np.array(cas.reshape(r, (3, 3)))

    num_spanwise = np.ceil(b_ref * body_cross_sections_per_meter / 2.)

    ypos = np.arange(-1. * num_spanwise, num_spanwise + 1.) / num_spanwise / 2.

    leading_edges = []
    trailing_edges = []

    for y in ypos:

        yloc = cas.mtimes(r_dcm, vect_op.yhat_np()) * y * b_ref

        s = np.abs(y)/0.5 # 1 at tips and 0 at root
        if y < 0:
            c_side = c_tipn
            shift_side = shift_tipn
        else:
            c_side = c_tipp
            shift_side = shift_tipp

        c_local = c_root * (1. - s) + c_side * s
        shift_local = (shift_root * (1. - s) + shift_side * s) * c_root

        basic_shell = get_naca_shell(c_local, naca)

        basic_leading_ege = basic_shell[np.argmin(basic_shell[:, 0]), :]
        basic_trailing_ege = basic_shell[np.argmax(basic_shell[:, 0]), :]

        base_position = q + yloc + np.array(cas.mtimes(r_dcm, shift_local * vect_op.xhat_dm()))

        new_leading_edge = base_position + np.array(cas.mtimes(r_dcm, basic_leading_ege.T))
        new_trailing_edge = base_position + np.array(cas.mtimes(r_dcm, basic_trailing_ege.T))

        leading_edges = cas.vertcat(leading_edges, new_leading_edge.T)
        trailing_edges = cas.vertcat(trailing_edges, new_trailing_edge.T)

        horizontal_shell = []
        for idx in range(basic_shell[:, 0].shape[0]):

            new_point = base_position + np.array(cas.mtimes(r_dcm, basic_shell[idx, :].T))

            horizontal_shell = cas.vertcat(horizontal_shell, new_point.T)
        horizontal_shell = np.array(horizontal_shell)

        basic_draw(ax, side, data=horizontal_shell, color=kite_color)

    basic_draw(ax, side, data=leading_edges, color=kite_color)
    basic_draw(ax, side, data=trailing_edges, color=kite_color)

    return None

def draw_kite_fuselage(ax, q, r, length, kite_color, side, body_cross_sections_per_meter, naca="0006"):

    r_dcm = np.array(cas.reshape(r, (3, 3)))

    total_width = float(naca[2:]) / 100. * length

    num_spanwise = np.ceil(total_width * body_cross_sections_per_meter / 2.)

    ypos = np.arange(-1. * num_spanwise, num_spanwise + 1.) / num_spanwise / 2.

    for y in ypos:

        yloc = cas.mtimes(r_dcm, vect_op.yhat_np()) * y * total_width
        zloc = cas.mtimes(r_dcm, vect_op.zhat_np()) * y * total_width

        basic_shell = get_naca_shell(length, naca) * (1 - (2. * y)**2.)

        span_direction_shell = []
        up_direction_shell = []
        for idx in range(basic_shell[:, 0].shape[0]):

            new_point_spanwise = q + yloc + np.array(cas.mtimes(r_dcm, basic_shell[idx, :].T))
            span_direction_shell = cas.vertcat(span_direction_shell, new_point_spanwise.T)

            new_point_upwise = q + zloc + np.array(cas.mtimes(r_dcm, basic_shell[idx, :].T))
            up_direction_shell = cas.vertcat(up_direction_shell, new_point_upwise.T)

        span_direction_shell = np.array(span_direction_shell)
        basic_draw(ax, side, data=span_direction_shell, color=kite_color)

        up_direction_shell = np.array(up_direction_shell)
        basic_draw(ax, side, data=up_direction_shell, color=kite_color)

    return None

def draw_kite_wing(ax, q, r, b_ref, c_root, c_tip, kite_color, side, body_cross_sections_per_meter, naca="0012"):

    draw_lifting_surface(ax, q, r, b_ref, c_tip, c_root, c_tip, kite_color, side, body_cross_sections_per_meter, naca)

def draw_kite_horizontal(ax, q, r, length, height, b_ref, c_ref, kite_color, side, body_cross_sections_per_meter, naca="0012"):

    r_dcm = np.array(cas.reshape(r, (3, 3)))
    ehat_1 = np.reshape(r_dcm[:, 0], (3,1))
    ehat_3 = np.reshape(r_dcm[:, 2], (3,1))

    horizontal_space = (3. * length / 4. - c_ref / 3.) * ehat_1
    pos = q + horizontal_space + ehat_3 * height

    draw_lifting_surface(ax, pos, r_dcm, b_ref / 3., c_ref / 3., c_ref / 2., c_ref / 3., kite_color, side, body_cross_sections_per_meter, naca)


def draw_kite_vertical(ax, q, r, length, height, c_ref, kite_color, side):

    r_dcm = cas.reshape(r, (3, 3))
    ehat_1 = vect_op.columnize(r_dcm[:, 0])
    ehat_2 = vect_op.columnize(r_dcm[:, 1])
    ehat_3 = vect_op.columnize(r_dcm[:, 2])

    new_ehat1 = ehat_1
    new_ehat2 = -1. * ehat_3
    new_ehat3 = ehat_2
    r_new = np.array(cas.horzcat(new_ehat1, new_ehat2, new_ehat3))

    horizontal_space = (3. * length / 4. - c_ref / 3.) * ehat_1
    pos = q + horizontal_space + ehat_3 * height / 2.

    c_fuselage = c_ref
    c_top = c_ref / 4.
    c_root = (c_fuselage + c_top) / 2.

    shift_fuselage = 0.
    shift_top = (c_top - c_fuselage) / c_root
    shift_root = (shift_fuselage+ shift_top) / 2.

    draw_lifting_surface(ax, pos, r_new, height, c_top, c_root, c_fuselage, kite_color, side, 3./height,
                         shift_tipn=shift_fuselage, shift_root=shift_root, shift_tipp=shift_top)
    return None


def draw_kite(ax, q, r, model_options, kite_color, side, body_cross_sections_per_meter):
    # read in inputs
    geometry = model_options['geometry']
    geometry_params = model_options['params']['geometry']

    if geometry['fuselage']:
        draw_kite_fuselage(ax, q, r, geometry['length'], kite_color, side, body_cross_sections_per_meter)

    if geometry['wing']:

       if geometry['wing_profile'] is None:
            draw_kite_wing(ax, q, r, geometry_params['b_ref'], geometry['c_root'], geometry['c_tip'], kite_color, side, body_cross_sections_per_meter)
       else:
            draw_kite_wing(ax, q, r, geometry_params['b_ref'], geometry['c_root'], geometry['c_tip'], kite_color, side,
                           body_cross_sections_per_meter, geometry['wing_profile'])

    if geometry['tail']:
        draw_kite_horizontal(ax, q, r, geometry['length'], geometry['height'], geometry_params['b_ref'], geometry_params['c_ref'], kite_color, side, body_cross_sections_per_meter)
        draw_kite_vertical(ax, q, r, geometry['length'], geometry['height'], geometry_params['c_ref'], kite_color, side)

    return None


def draw_kite_aero_dcm(ax, side, plot_dict, cosmetics, index):
    b_ref = plot_dict['options']['model']['params']['geometry']['b_ref']
    dcm_colors = cosmetics['trajectory']['dcm_colors']
    visibility_scaling = b_ref

    variables_si = assemble_variable_slice_from_interpolated_data(plot_dict, index)

    architecture = plot_dict['architecture']
    for kite in architecture.kite_nodes:
        parent = architecture.parent_map[kite]

        ehat_chord = []
        ehat_span = []
        ehat_up = []
        for dim in range(3):
            local_chord = plot_dict['outputs']['aerodynamics']['ehat_chord' + str(kite)][dim][index]
            local_span = plot_dict['outputs']['aerodynamics']['ehat_span' + str(kite)][dim][index]
            local_up = plot_dict['outputs']['aerodynamics']['ehat_up' + str(kite)][dim][index]

            ehat_chord = cas.vertcat(ehat_chord, local_chord)
            ehat_span = cas.vertcat(ehat_span, local_span)
            ehat_up = cas.vertcat(ehat_up, local_up)

        ehat_dict = {'x': ehat_chord,
                     'y': ehat_span,
                     'z': ehat_up}

        x_start = variables_si['x', 'q' + str(kite) + str(parent)]

        for vec_name, vec_ehat in ehat_dict.items():
            x_end = x_start + visibility_scaling * vec_ehat

            color = dcm_colors[vec_name]
            basic_draw(ax, side, color=color, x_start=x_start, x_end=x_end, linestyle='-')
    return None


def basic_draw(ax, side, x_start=None, x_end=None, data=None, color='k', marker=None, linestyle='-', alpha=1., label=None):

    no_start = x_start is None
    no_end = x_end is None
    no_segment = no_start or no_end
    no_data = data is None

    if no_segment and no_data:
        message = 'insufficient data provided to basic_draw'
        print_op.log_and_raise_error(message)

    elif (not no_segment) and (not no_data):
        message = 'too much data provided to basic_draw'
        print_op.log_and_raise_error(message)

    elif not no_segment:
        x = [float(x_start[0]), float(x_end[0])]
        y = [float(x_start[1]), float(x_end[1])]
        z = [float(x_start[2]), float(x_end[2])]

    elif not no_data:
        if (not hasattr(data, 'shape')) or (not len(data.shape) == 2):
            message = 'data provided to basic_draw has wrong format or wrong shape'
            print_op.log_and_raise_error(message)

        if isinstance(data, cas.DM):
            data = np.array(data)

        if data.shape[0] == 3:
            pass
        elif data.shape[1] == 3:
            data = data.T
        else:
            message = 'data provided to basic_draw is not 3d-cartesian'
            print_op.log_and_raise_error(message)

        x = data[0, :]
        y = data[1, :]
        z = data[2, :]

    else:
        message = 'set of choices intended to be complete, appears not to be.'
        print_op.log_and_raise_error(message)

    if side == 'xy':
        ax.plot(x, y, marker=marker, c=color, linestyle=linestyle, alpha=alpha, label=label)
    elif side == 'xz':
        ax.plot(x, z, marker=marker, c=color, linestyle=linestyle, alpha=alpha, label=label)
    elif side == 'yz':
        ax.plot(y, z, marker=marker, c=color, linestyle=linestyle, alpha=alpha, label=label)
    elif side == 'isometric':
        ax.plot3D(x, y, z, marker=marker, c=color, linestyle=linestyle, alpha=alpha, label=label)

    return None


def draw_all_kites(ax, plot_dict, index, cosmetics, side, init_colors=bool(False)):

    options = plot_dict['options']
    architecture = plot_dict['architecture']
    kite_nodes = architecture.kite_nodes
    parent_map = architecture.parent_map
    body_cross_sections_per_meter = cosmetics['trajectory']['body_cross_sections_per_meter']

    search_name = 'interpolation' + '_' + plot_dict['cosmetics']['variables']['si_or_scaled']
    x_vals = plot_dict[search_name]['x']

    for kite in kite_nodes:

        # kite colors
        if init_colors:
            local_color = 'k'
        else:
            local_color = cosmetics['trajectory']['colors'][kite_nodes.index(kite)]

        parent = parent_map[kite]

        # kite position information
        q_kite = []
        for j in range(3):
            q_kite = cas.vertcat(q_kite, x_vals['q' + str(kite) + str(parent)][j][index])

        # dcm information
        r_dcm = []
        for j in range(3):
            r_dcm = cas.vertcat(r_dcm, plot_dict[search_name]['outputs']['aerodynamics']['ehat_chord' + str(kite)][j][index])
        for j in range(3):
            r_dcm = cas.vertcat(r_dcm, plot_dict[search_name]['outputs']['aerodynamics']['ehat_span' + str(kite)][j][index])
        for j in range(3):
            r_dcm = cas.vertcat(r_dcm, plot_dict[search_name]['outputs']['aerodynamics']['ehat_up' + str(kite)][j][index])

        # draw kite body
        draw_kite(ax, q_kite, r_dcm, options['model'], local_color, side, body_cross_sections_per_meter)

    return None


def plot_path_of_node(ax, side, plot_dict, node, ref=False, color='k', marker=None, linestyle='-', alpha=1., label=None):

    parent = plot_dict['architecture'].parent_map[node]

    if ref:
        search_name = 'ref'
    else:
        search_name = 'interpolation'
    search_name = search_name + '_' + plot_dict['cosmetics']['variables']['si_or_scaled']
    x_vals = plot_dict[search_name]['x']

    data = []
    for dim in range(3):
        local_dim = x_vals['q' + str(node) + str(parent)][dim]
        data = cas.horzcat(data, local_dim)

    basic_draw(ax, side, data=data, color=color, marker=marker, linestyle=linestyle, alpha=alpha, label=label)
    return None


def plot_all_tethers(ax, side, plot_dict, ref=False, color='k', marker=None, linestyle='-', alpha=0.2, label=None, index=-1):
    architecture = plot_dict['architecture']

    if ref:
        search_name = 'ref'
    else:
        search_name = 'interpolation'
    search_name += '_' + plot_dict['cosmetics']['variables']['si_or_scaled']
    x_vals = plot_dict[search_name]['x']


    for node in range(1, architecture.number_of_nodes):
        parent = architecture.parent_map[node]

        if parent == 0:
            x_start = cas.DM.zeros((3, 1))
        else:
            grandparent = architecture.parent_map[parent]
            x_start = []
            for dim in range(3):
                local_val = x_vals['q' + str(parent) + str(grandparent)][dim][index]
                x_start = cas.vertcat(x_start, local_val)

        x_end = []
        for dim in range(3):
            local_val = x_vals['q' + str(node) + str(parent)][dim][index]
            x_end = cas.vertcat(x_end, local_val)

        basic_draw(ax, side, x_start=x_start, x_end=x_end, color=color, marker=marker, linestyle=linestyle, alpha=alpha, label=label)

    return None


def plot_trajectory_contents(ax, plot_dict, cosmetics, side, init_colors=bool(False), plot_kites=bool(True), label=None):

    # read in inputs
    model_options = plot_dict['options']['model']
    architecture = plot_dict['architecture']
    kite_nodes = architecture.kite_nodes

    search_name = 'interpolation_' + plot_dict['cosmetics']['variables']['si_or_scaled']

    body_cross_sections_per_meter = cosmetics['trajectory']['body_cross_sections_per_meter']

    old_label = None
    for kite in kite_nodes:
        parent = architecture.parent_map[kite]
        kite_index = architecture.kite_nodes.index(kite)

        if init_colors == True:
            local_color = 'k'
        elif init_colors == False:
            local_color = cosmetics['trajectory']['colors'][kite_index]
        else:
            local_color = init_colors

        if (cosmetics['trajectory']['kite_bodies'] and plot_kites):
            local_index = 0

            q_local = []
            for dim in range(3):
                local_val = plot_dict[search_name]['x']['q' + str(kite) + str(parent)][dim][local_index]
                q_local = cas.vertcat(q_local, local_val)

            r_local = []
            for dim in range(9):
                local_val = plot_dict[search_name]['outputs']['aerodynamics']['r' + str(kite)][dim][local_index]
                r_local = cas.vertcat(r_local, local_val)

            draw_kite(ax, q_local, r_local, model_options, local_color, side, body_cross_sections_per_meter)

        if old_label == label:
            label = None

        plot_path_of_node(ax, side, plot_dict, kite, ref=False, color=local_color, label=label)
        if cosmetics['plot_ref']:
            plot_path_of_node(ax, side, plot_dict, kite, ref=True, color=local_color, label=label, linestyle='--', alpha=0.5)

        old_label = label

    plot_tether = (len(kite_nodes) == 1)
    if plot_tether:
        time_entries = plot_dict[search_name]['x']['q10'][0].shape[0]
        for index in range(time_entries):
            plot_all_tethers(ax, side, plot_dict, ref=False, index=index)
            if cosmetics['plot_ref']:
                plot_all_tethers(ax, side, plot_dict, ref=True, index=index, alpha=0.5)

    return None


def get_q_limits(plot_dict, cosmetics):
    dims = ['x', 'y', 'z']

    extrema = {}
    centers = {}
    deltas = []
    for dim in dims:
        extrema[dim] = get_q_extrema_in_dimension(dim, plot_dict, cosmetics)
        centers[dim] = np.average(extrema[dim])
        deltas = np.append(deltas, extrema[dim][1] - extrema[dim][0])

    max_dim = np.max(deltas)

    limits = {}
    signs = [-1., +1.]
    for dim in dims:
        limits[dim] = [centers[dim] + sign * 0.5 * max_dim for sign in signs]

    return limits


def get_q_extrema_in_dimension(dim, plot_dict, cosmetics):

    temp_min = 1.e5
    temp_max = -1.e5

    if dim == 'x' or dim == '0':
        jdx = 0
        dim = 'x'
    elif dim == 'y' or dim == '1':
        jdx = 1
        dim = 'y'
    elif dim == 'z' or dim == '2':
        jdx = 2
        dim = 'z'
    else:
        jdx = 0
        dim = 'x'

        message = 'selected dimension for q_limits not supported. setting dimension to x'
        awelogger.logger.warning(message)

    search_name = 'interpolation_' + plot_dict['cosmetics']['variables']['si_or_scaled']

    for name in list(plot_dict[search_name]['x'].keys()):
        if name[0] == 'q':
            temp_min = np.min(cas.vertcat(temp_min, np.min(plot_dict[search_name]['x'][name][jdx])))
            temp_max = np.max(cas.vertcat(temp_max, np.max(plot_dict[search_name]['x'][name][jdx])))

        if name[0] == 'w' and name[1] == dim and cosmetics['trajectory']['wake_nodes']:
            vals = np.array(cas.vertcat(*plot_dict[search_name]['x'][name]))
            temp_min = np.min(cas.vertcat(temp_min, np.min(vals)))
            temp_max = np.max(cas.vertcat(temp_max, np.max(vals)))

    # get margins
    margin = cosmetics['trajectory']['margin']
    lmargin = 1.0 - margin
    umargin = 1.0 + margin

    if temp_min > 0.0:
        temp_min = lmargin * temp_min
    else:
        temp_min = umargin * temp_min

    if temp_max < 0.0:
        temp_max = lmargin * temp_max
    else:
        temp_max = umargin * temp_max

    q_lim = [temp_min, temp_max]

    return q_lim


def plot_control_block_smooth(cosmetics, V_opt, plt, fig, plot_table_r, plot_table_c, idx, location, name, plot_dict, number_dim=1):

    # read in inputs
    tgrid_x = plot_dict['time_grids']['x']

    plt.subplot(plot_table_r, plot_table_c, idx)
    for jdx in range(number_dim):
        plt.plot(tgrid_x, cas.vertcat(*np.array(V_opt[location, :, :, name, jdx])), color=cosmetics['controls']['colors'][jdx])
    plt.grid(True)
    plt.title(name)

def plot_control_block(cosmetics, V_opt, plt, fig, plot_table_r, plot_table_c, idx, location, name, plot_dict, number_dim=1):

    # read in inputs
    tgrid_u = plot_dict['time_grids']['u']
    tgrid_ip = plot_dict['time_grids']['ip']

    interp_name = 'interpolation_' + plot_dict['cosmetics']['variables']['si_or_scaled']
    ref_name = 'ref_' + plot_dict['cosmetics']['variables']['si_or_scaled']

    plt.subplot(plot_table_r, plot_table_c, idx)
    for jdx in range(number_dim):
        color=cosmetics['controls']['colors'][jdx]

        if plot_dict['u_param'] == 'poly':
            plt.plot(np.array(tgrid_ip), np.array(plot_dict[interp_name]['u'][name][jdx]), color=color)
            if plot_dict['options']['visualization']['cosmetics']['plot_bounds']:
                plot_bounds(plot_dict, 'u', name, jdx, tgrid_ip, color=color)
            if plot_dict['options']['visualization']['cosmetics']['plot_ref']:
                plt.plot(plot_dict['time_grids']['ref']['ip'], plot_dict[ref_name]['u'][name][jdx],
                    linestyle='--', color=color)

        else:
            p = plt.step(tgrid_ip, plot_dict[interp_name]['u'][name][jdx], where='post', color=color)
            if plot_dict['options']['visualization']['cosmetics']['plot_bounds']:
                plot_bounds(plot_dict, 'u', name, jdx, tgrid_ip, color=color)
            if plot_dict['options']['visualization']['cosmetics']['plot_ref']:
                plt.step(plot_dict['time_grids']['ref']['ip'], plot_dict[ref_name]['u'][name][jdx],
                         where='post', linestyle='--', color=color)
    plt.grid(True)
    plt.title(name)
    plt.autoscale(enable=True, axis= 'x', tight = True)


def get_sweep_colors(number_of_trials):

    cmap = plt.get_cmap('jet')
    c_norm = colors.Normalize(vmin=0, vmax=(number_of_trials - 1))
    scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=cmap)

    color_list = []
    for trial in range(number_of_trials):
        color_list += [scalar_map.to_rgba(float(trial))]

    return color_list


def calibrate_visualization(model, nlp, name, options):
    """
    Generate plot dict with all calibration operations that only have to be performed once per trial.
    :return: plot dictionary
    """

    plot_dict = {}

    # trial information
    plot_dict['name'] = name
    plot_dict['options'] = options
    plot_dict['cosmetics'] = options['visualization']['cosmetics']

    # nlp information
    plot_dict['n_k'] = nlp.n_k
    plot_dict['discretization'] = nlp.discretization

    if nlp.discretization == 'direct_collocation':
        plot_dict['d'] = nlp.d
        plot_dict['u_param'] = options['nlp']['collocation']['u_param']
    else:
        plot_dict['u_param'] = 'zoh'

    # model information
    plot_dict['outputs_dict'] = struct_op.strip_of_contents(model.outputs_dict)
    plot_dict['model_outputs'] = struct_op.strip_of_contents(model.outputs)
    plot_dict['variables_dict'] = struct_op.strip_of_contents(model.variables_dict)
    # plot_dict['integral_variables'] = list(model.integral_outputs.keys())
    plot_dict['model_scaling'] = model.scaling.cat
    plot_dict['model_parameters'] = struct_op.strip_of_contents(model.parameters)
    plot_dict['model_variables'] = struct_op.strip_of_contents(model.variables)
    plot_dict['integral_output_names'] = model.integral_outputs.keys()
    plot_dict['architecture'] = model.architecture
    plot_dict['variable_bounds'] = model.variable_bounds
    plot_dict['global_outputs'] = nlp.global_outputs

    plot_dict['Collocation'] = nlp.Collocation

    if model.wake is not None:
        model.wake.define_model_variables_to_info_functions(model.variables, model.parameters)
    plot_dict['wake'] = model.wake

    # wind information
    u_ref = model.options['params']['wind']['u_ref']
    plot_dict['u_ref'] = float(u_ref)

    return plot_dict


def recalibrate_visualization(V_plot_scaled, plot_dict, output_vals, integral_output_vals, options, time_grids, cost, name, V_ref_scaled, global_output_vals, iterations=None, return_status_numeric=None, timings=None, n_points=None):
    """
    Recalibrate plot dict with all calibration operation that need to be perfomed once for every plot.
    :param plot_dict: plot dictionary before recalibration
    :return: recalibrated plot dictionary
    """

    # extract information
    cosmetics = options['visualization']['cosmetics']
    if n_points is not None:
        cosmetics['interpolation']['n_points'] = int(n_points)

    plot_dict['cost'] = cost

    # add V_plot to dict
    scaling = plot_dict['model_variables'](plot_dict['model_scaling'])
    plot_dict['V_plot_scaled'] = V_plot_scaled
    plot_dict['V_ref_scaled'] = V_ref_scaled
    plot_dict['V_plot_si'] = struct_op.scaled_to_si(V_plot_scaled, scaling)
    plot_dict['V_ref_si'] = struct_op.scaled_to_si(V_ref_scaled, scaling)
    plot_dict['parameters_plot'] = assemble_model_parameters(plot_dict)

    V_plot_si = plot_dict['V_plot_si']

    # get new name
    plot_dict['name'] = name

    # get new outputs
    plot_dict['output_vals'] = output_vals
    plot_dict['integral_output_vals'] = integral_output_vals
    plot_dict['global_output_vals'] = global_output_vals

    # get new time grids
    plot_dict['time_grids'] = time_grids
    if plot_dict['discretization'] == 'direct_collocation':
        plot_dict['time_grids']['coll'] = time_grids['coll'].T.reshape((plot_dict['n_k'] * plot_dict['d'], 1))

    # get new options
    plot_dict['options'] = options

    # interpolate data
    plot_dict = interpolate_data(plot_dict, cosmetics)
    if cosmetics['plot_ref']:
        plot_dict = interpolate_ref_data(plot_dict, cosmetics)

    # interations
    if iterations is not None:
        plot_dict['iterations'] = iterations

    # return status numeric
    if return_status_numeric is not None:
        plot_dict['return_status_numeric'] = return_status_numeric

    # timings
    if timings is not None:
        plot_dict['timings'] = timings

    # power and performance
    plot_dict['power_and_performance'] = diagnostics.compute_power_and_performance(plot_dict)

    # plot scaling
    plot_dict['max_x'] = np.max(np.array(V_plot_si['x', :, 'q10', 0])) * 1.2
    plot_dict['max_y'] = np.max(np.abs(np.array(V_plot_si['x', :, 'q10', 1]))) * 1.2
    plot_dict['max_z'] = np.max(np.array(V_plot_si['x', :, 'q10', 2])) * 1.2
    plot_dict['mazim'] = np.max([plot_dict['max_x'], plot_dict['max_y'], plot_dict['max_z']])
    plot_dict['scale_power'] = 1.  # e-3

    if '[x,0,l_t,0]' in V_plot_si.labels():
        plot_dict['scale_axes'] = float(V_plot_si['x', 0, 'l_t'])
    elif '[theta,l_t,0]' in V_plot_si.labels():
        plot_dict['scale_axes'] = float(V_plot_si['theta', 'l_t'])
    else:
        message = '(main) tether length could not be found in V_plot_si'
        print_op.log_and_raise_error(message)

    dashes = []
    for ldx in range(20):
        new_dash = []
        for jdx in range(4):
            new_dash += [int(np.random.randint(1, 6))]
        new_dash += [1, 1]

        dashes += [new_dash]
    plot_dict['dashes'] = dashes

    return plot_dict


def interpolate_data(plot_dict, cosmetics):
    '''
    Postprocess data from V-structure to (interpolated) data vectors
        with associated time grid
    :param plot_dict: dictionary of all relevant plot information
    :param cosmetics: dictionary of cosmetic plot choices
    :return: plot dictionary with added entries corresponding to interpolation
    '''

    # extract information
    nlp_options = cosmetics
    time_grids = plot_dict['time_grids']
    variables_dict = plot_dict['variables_dict']
    V_plot_si = plot_dict['V_plot_si']
    V_plot_scaled = plot_dict['V_plot_scaled']
    model_outputs = plot_dict['model_outputs']
    outputs_dict = plot_dict['outputs_dict']
    outputs_opt = plot_dict['output_vals']['opt']
    integral_output_names = plot_dict['integral_output_names']
    integral_outputs_opt = plot_dict['integral_output_vals']['opt']
    Collocation = plot_dict['Collocation']

    # make the interpolation
    # todo: allow the interpolation to be imported directly from the quality-check, if the interpolation options are the same
    interpolation_si = struct_op.interpolate_solution(nlp_options, time_grids, variables_dict, V_plot_si,
                                                   outputs_dict, outputs_opt, model_outputs,
                                                   integral_output_names, integral_outputs_opt,
                                                   Collocation=Collocation)

    interpolation_scaled = struct_op.interpolate_solution(nlp_options, time_grids, variables_dict, V_plot_scaled,
                                                   outputs_dict, outputs_opt, model_outputs,
                                                   integral_output_names, integral_outputs_opt,
                                                   Collocation=Collocation)

    # store the interpolation
    for name in ['interpolation_si', 'interpolation_scaled']:
        if name not in plot_dict.keys():
            plot_dict[name] = {}

    for name, value in interpolation_si.items():
        plot_dict['interpolation_si'][name] = value

    for name, value in interpolation_scaled.items():
        plot_dict['interpolation_scaled'][name] = value

    return plot_dict


def interpolate_ref_data(plot_dict, cosmetics):
    '''
    Postprocess tracking reference data from V-structure to (interpolated) data vectors
        with associated time grid
    :param plot_dict: dictionary of all relevant plot information
    :param cosmetics: dictionary of cosmetic plot choices
    :return: plot dictionary with added entries corresponding to interpolation
    '''

    # extract information
    nlp_options = cosmetics
    time_grids = plot_dict['time_grids']['ref']
    variables_dict = plot_dict['variables_dict']
    V_ref_si = plot_dict['V_ref_si']
    V_ref_scaled = plot_dict['V_ref_scaled']
    model_outputs = plot_dict['outputs']
    outputs_dict = plot_dict['outputs_dict']
    outputs_ref = plot_dict['output_vals']['ref']
    integral_output_names = plot_dict['integral_output_names']
    integral_outputs_ref = plot_dict['integral_output_vals']['ref']
    Collocation = plot_dict['Collocation']

    # make the interpolation
    interpolation_si = struct_op.interpolate_solution(nlp_options, time_grids, variables_dict, V_ref_si,
                                                   outputs_dict, outputs_ref, model_outputs,
                                                   integral_output_names, integral_outputs_ref,
                                                   Collocation=Collocation)
    interpolation_scaled = struct_op.interpolate_solution(nlp_options, time_grids, variables_dict, V_ref_scaled,
                                                   outputs_dict, outputs_ref, model_outputs,
                                                   integral_output_names, integral_outputs_ref,
                                                   Collocation=Collocation)

    # store the interpolation
    for name in ['ref_si', 'ref_scaled']:
        if name not in plot_dict.keys():
            plot_dict[name] = {}

    for name, value in interpolation_si.items():
        plot_dict['ref_si'][name] = value

    for name, value in interpolation_scaled.items():
        plot_dict['ref_scaled'][name] = value

    return plot_dict


def map_flag_to_function(flag, plot_dict, cosmetics, fig_name, plot_logic_dict):

    standard_args = (plot_dict, cosmetics, fig_name)
    if flag not in plot_logic_dict.keys():
        message = 'cannot process plot with flag (' + flag +'). skipping it, instead.'
        print_op.base_print(message, level='warning')
        return None

    additional_args = plot_logic_dict[flag][1]

    # execute function from dict
    if type(additional_args) == dict:
        plot_logic_dict[flag][0](*standard_args, **additional_args)

    elif additional_args is None:
        plot_logic_dict[flag][0](*standard_args)
    else:
        raise TypeError('Additional arguments for plot functions must be passed as a dict or as None.')

    return None


def reconstruct_comparison_labels(plot_dict):
    comparison_labels = []

    if 'actuator' in plot_dict['outputs']:
        actuator_outputs = plot_dict['outputs']['actuator']
        architecture = plot_dict['architecture']
        layers = architecture.layer_nodes
        layer_test = layers[0]

        kites = architecture.children_map[layer_test]
        kite_test = kites[0]

        idx = 0
        for label in ['qaxi', 'qasym', 'uaxi', 'uasym']:
            test_name = 'local_a_' + label + str(kite_test)
            if test_name in actuator_outputs.keys():
                idx += 1
                comparison_labels += [label]

    return comparison_labels


def set_max_and_min(y_vals, y_max, y_min):

    y_min = np.min([y_min, np.min(y_vals)])
    y_max = np.max([y_max, np.max(y_vals)])

    return y_max, y_min


def make_layer_plot_in_fig(layers, fig_num):
    nrows = len(layers)
    plt.figure(fig_num).clear()
    fig, axes = plt.subplots(nrows=nrows, ncols=1, sharex='all', num=fig_num)
    return fig, axes, nrows


def set_layer_plot_titles(axes, nrows, title):
    if nrows == 1:
        axes.set_title(title)
    else:
        axes[0].set_title(title)
    return axes


def set_layer_plot_axes(axes, nrows, xlabel, ylabel, ldx = 0):
    if nrows == 1:
        axes.set_ylabel(ylabel)
        axes.set_xlabel(xlabel)
    else:
        axes[ldx].set_ylabel(ylabel)
        axes[ldx].set_xlabel(xlabel)
    return axes


def set_layer_plot_legend(axes, nrows, ldx = 0):
    if nrows == 1:
        axes.legend()
    else:
        axes[ldx].legend()
    return axes


def set_layer_plot_scale(axes, nrows, x_min, x_max, y_min, y_max):
    if nrows == 1:
        axes.set_autoscale_on(False)
        axes.axis([x_min, x_max, y_min, y_max])
    else:
        for idx in range(nrows):
            axes[idx].set_autoscale_on(False)
            axes[idx].axis([x_min, x_max, y_min, y_max])
    return axes


def add_switching_time_epigraph(axes, nrows, tau, y_min, y_max):
    if nrows == 1:
        axes.plot([tau, tau], [y_min, y_max], 'k--')
    else:
        for idx in range(nrows):
            axes[idx].plot([tau, tau], [y_min, y_max], 'k--')
    return axes


def get_nondim_time_and_switch(plot_dict):
    time_dim = np.array(plot_dict['time_grids']['ip'])
    t_f = time_dim[-1]
    time_nondim = time_dim / t_f

    if 't_switch' in plot_dict['time_grids'].keys():
        t_switch = plot_dict['time_grids']['t_switch']
        tau = t_switch / t_f
    else:
        tau = 1.

    return time_nondim, tau


def assemble_variable_slice_from_interpolated_data(plot_dict, index, si_or_scaled=None):

    collected_vals = []

    if si_or_scaled is None:
        si_or_scaled = plot_dict['cosmetics']['variables']['si_or_scaled']

    interpolation_data = plot_dict['interpolation_' + si_or_scaled]

    model_variables = plot_dict['model_variables']
    for jdx in range(model_variables.shape[0]):
        canonical = model_variables.getCanonicalIndex(jdx)
        var_type = canonical[0]
        var_name = canonical[1]
        dim = canonical[2]

        if (var_type == 'theta'):
            category = interpolation_data['theta'][var_name]
            if category.shape == ():
                local_val = category
            else:
                local_val = category[dim]
            collected_vals = cas.vertcat(collected_vals, local_val)

        elif (var_type == 'xdot'):
            if (var_name in interpolation_data['x'].keys()):
                local_val = interpolation_data['x'][var_name][dim][index]
            else:
                # be advised: this function does not compute dynamics
                local_val = cas.DM.zeros((1,1))
            collected_vals = cas.vertcat(collected_vals, local_val)

        elif (var_type in interpolation_data.keys()) and (var_name in interpolation_data[var_type].keys()):
            local_val = interpolation_data[var_type][var_name][dim][index]
            collected_vals = cas.vertcat(collected_vals, local_val)

        else:
            message = 'unrecognized variable type or name when re-assembling a (model) variable from interpolated data.'
            print_op.log_and_raise_error(message)

    try:
        vars_si = model_variables(collected_vals)
    except:
        message = 'interpolated data does not have the recognizable structure of a (model) variable'
        print_op.log_and_raise_error(message)

    return vars_si


def assemble_model_parameters(plot_dict):

    collected_vals = []

    options_model = plot_dict['options']['model']
    options_params = plot_dict['options']['params']

    model_parameters = plot_dict['model_parameters']
    for jdx in range(model_parameters.shape[0]):
        canonical = model_parameters.getCanonicalIndex(jdx)
        var_type = canonical[0]
        kdx = canonical[-1]

        if (var_type == 'phi'):
            var_name = canonical[1]
            kdx = canonical[2]
            local_val = plot_dict['V_plot_si'][var_type, var_name, kdx]
            collected_vals = cas.vertcat(collected_vals, local_val)

        elif (var_type == 'theta0') and (kdx == 0):

            if canonical[1] in options_params.keys():
                local_val = options_params
            elif canonical[1] in options_model.keys():
                local_val = options_model
            else:
                message = 'something went wrong when assembling theta0 model parameters.'
                print_op.log_and_raise_error(message)

            # remember that the first entry of canonical is (already) 'theta0' and the last entry of canonical will be the dimension (kdx)
            for sdx in range(len(canonical)-2):
                local_val = local_val[canonical[sdx+1]]

            if hasattr(local_val, 'shape'):
                local_shape = cas.DM(local_val).shape
                local_val = cas.reshape(local_val, (local_shape[0] * local_shape[1], 1))
            collected_vals = cas.vertcat(collected_vals, local_val)

        elif (kdx == 0):
            message = 'unrecognized parameter type or name when re-assembling a (model) parameter from solution data'
            print_op.log_and_raise_error(message)

    try:
        params = model_parameters(collected_vals)
    except:
        message = 'unable to assign re-assembled interpolated data into a (model) parameters structure'
        print_op.log_and_raise_error(message)

    return params


def plot_bounds(plot_dict, var_type, name, jdx, tgrid_ip, p=None, color=None):

    if (p is None) and (color is None):
        message = 'not enough information to draw the bounds'
        print_op.log_and_raise_error(message)
    elif (p is not None):
        color = p[-1].get_color()

    if (color is None):
        message = 'something went wrong when defining the color in which to plot bounds'
        print_op.log_and_raise_error(message)

    bounds = plot_dict['variable_bounds'][var_type][name]
    scaling = plot_dict['model_variables'](plot_dict['model_scaling'])[var_type, name]

    bound_types = ['lb', 'ub']
    for type in bound_types:

        potential_bound = bounds[type]

        if isinstance(potential_bound, np.ndarray):
            local_bound = potential_bound[jdx]
        elif (isinstance(potential_bound, cas.DM)) and potential_bound.shape == (1, 1):
            local_bound = float(potential_bound)
        elif isinstance(potential_bound, cas.DM):
            local_bound = float(potential_bound[jdx])
        else:
            local_bound = potential_bound

        if scaling.shape == (1, 1):
            local_scaling = float(scaling)
        else:
            local_scaling = float(scaling[jdx])

        if np.isfinite(local_bound):
            bound_grid_ip = local_bound * local_scaling * np.ones(tgrid_ip.shape)
            plt.plot(tgrid_ip, bound_grid_ip, linestyle='dotted', color=color)

    return None


def setup_axes_for_side(cosmetics, side):
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

    return fig, ax


def test_naca_coordinates():

    naca = "0012"
    m = float(naca[0]) / 100.
    p = float(naca[1]) / 10.
    t = float(naca[2:]) / 100.

    s_le = 0.
    s_te = 1.0

    xu_le, xl_le, yu_le, yl_le = get_naca_airfoil_coordinates(s_le, m, p, t)
    xu_te, xl_te, yu_te, yl_te = get_naca_airfoil_coordinates(s_te, m, p, t)

    epsilon_small = 1.e-8
    epsilon_large = 1.e-2
    x_vals_equal = ((xu_le - xl_le)**2. < epsilon_small**2.) and ((xu_te - xl_te)**2. < epsilon_small**2.)
    le_at_origin = (xu_le**2. < epsilon_small**2.) and (xl_le**2. < epsilon_small**2.) and (yu_le**2. < epsilon_small**2.) and (yu_le**2. < epsilon_small**2.)
    chord_length_correct = ((yu_te - 1.) < epsilon_small**2.) and ((yl_te - 1.) < epsilon_small**2.)
    te_joins = ((yu_te - yl_te)**2. < epsilon_large**2.) and ((xu_te - xl_te)**2. < epsilon_small**2.)

    works_correctly = x_vals_equal and le_at_origin and chord_length_correct and te_joins
    if not works_correctly:
        message = 'something went wrong with the naca 0012 coordinate generation.'
        print_op.log_and_raise_error(message)
    return None