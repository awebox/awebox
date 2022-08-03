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
import casadi.tools as cas
import numpy as np
import matplotlib.pyplot as plt
import awebox.tools.struct_operations as struct_op
from itertools import chain
import matplotlib.colors as colors
import matplotlib.cm as cmx
import awebox.tools.vector_operations as vect_op
import awebox.opti.diagnostics as diagnostics
from awebox.logger.logger import Logger as awelogger
import awebox.mdl.aero.induction_dir.vortex_dir.tools as vortex_tools
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

    m = np.float(naca[0]) / 100.
    p = np.float(naca[1]) / 10.
    t = np.float(naca[2:]) / 100.

    s_list = np.arange(0., 101.) / 100.

    x_upper = []
    x_lower = []

    for s in s_list:
        [xu, xl, yu, yl] = get_naca_airfoil_coordinates(s, m, p, t)

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

def make_side_plot(ax, vertically_stacked_array, side, plot_color, plot_marker=' ', label=None, alpha = 1, linestyle = '-', plot_tether = False):
    vsa = np.array(cas.DM(vertically_stacked_array))

    if vsa.shape[0] == 3 and (not vsa.shape[1] == 3):
        vsa = vsa.T

    if side == 'isometric':
        ax.plot(vsa[:, 0], vsa[:, 1], zs=vsa[:, 2], color=plot_color, marker=plot_marker, label=label, alpha = alpha, linestyle = linestyle)
        if plot_tether:
            for kk in range(int(vsa.shape[0]/5)-1):
                ax.plot([0, vsa[5*kk, 0]], [0, vsa[5*kk,1]], zs=[0, vsa[5*kk, 2]], color = 'black', alpha = 0.3)
    else:
        side_num = ''
        for sdx in side:
            if sdx == 'x':
                side_num += '0'
            elif sdx == 'y':
                side_num += '1'
            elif sdx == 'z':
                side_num += '2'

        idx = int(side_num[0])
        jdx = int(side_num[1])

        ax.plot(vsa[:, idx], vsa[:, jdx], color=plot_color, marker=plot_marker, label = label, alpha = alpha, linestyle = linestyle)
        if plot_tether:
            for kk in range(int(vsa.shape[0]/5)-1):
                ax.plot([0, vsa[5*kk, idx]], [0, vsa[5*kk,jdx]], color = 'black', alpha = 0.3)

    return None

def draw_lifting_surface(ax, q, r, b_ref, c_tipn, c_root, c_tipp, kite_color, side, body_cross_sections_per_meter, naca="0012"):

    r_dcm = np.array(cas.reshape(r, (3, 3)))

    num_spanwise = np.ceil(b_ref * body_cross_sections_per_meter / 2.)

    ypos = np.arange(-1. * num_spanwise, num_spanwise + 1.) / num_spanwise / 2.

    leading_edges = []
    trailing_edges = []

    for y in ypos:

        yloc = cas.mtimes(r_dcm, vect_op.yhat_np()) * y * b_ref

        s = np.abs(y)/0.5 # 1 at tips and 0 at root
        if y < 0:
            c_local = c_root * (1. - s) + c_tipn * s
        else:
            c_local = c_root * (1. - s) + c_tipp * s

        basic_shell = get_naca_shell(c_local, naca)

        basic_leading_ege = basic_shell[np.argmin(basic_shell[:, 0]), :]
        basic_trailing_ege = basic_shell[np.argmax(basic_shell[:, 0]), :]

        new_leading_edge = q + yloc + np.array(cas.mtimes(r_dcm, basic_leading_ege.T))
        new_trailing_edge = q + yloc + np.array(cas.mtimes(r_dcm, basic_trailing_ege.T))

        leading_edges = cas.vertcat(leading_edges, new_leading_edge.T)
        trailing_edges = cas.vertcat(trailing_edges, new_trailing_edge.T)

        horizontal_shell = []
        for idx in range(basic_shell[:, 0].shape[0]):

            new_point = q + yloc + np.array(cas.mtimes(r_dcm, basic_shell[idx, :].T))

            horizontal_shell = cas.vertcat(horizontal_shell, new_point.T)
        horizontal_shell = np.array(horizontal_shell)

        make_side_plot(ax, horizontal_shell, side, kite_color)

    make_side_plot(ax, leading_edges, side, kite_color)
    make_side_plot(ax, trailing_edges, side, kite_color)

    return None

def draw_kite_fuselage(ax, q, r, length, kite_color, side, body_cross_sections_per_meter, naca="0006"):

    r_dcm = np.array(cas.reshape(r, (3, 3)))

    total_width = np.float(naca[2:]) / 100. * length

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
        make_side_plot(ax, span_direction_shell, side, kite_color)

        up_direction_shell = np.array(up_direction_shell)
        make_side_plot(ax, up_direction_shell, side, kite_color)


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

def draw_kite_vertical(ax, q, r, length, height, b_ref, c_ref, kite_color, side, body_cross_sections_per_meter, naca="0012"):

    r_dcm = np.array(cas.reshape(r, (3, 3)))
    ehat_1 = np.reshape(r_dcm[:, 0], (3, 1))
    ehat_3 = np.reshape(r_dcm[:, 2], (3, 1))

    new_ehat1 = ehat_1
    new_ehat2 = ehat_3
    new_ehat3 = np.array(vect_op.normed_cross(new_ehat1, new_ehat2))
    r_new = np.array(cas.horzcat(new_ehat1, new_ehat2, new_ehat3))

    horizontal_space = (3. * length / 4. - c_ref / 3.) * ehat_1
    pos = q + horizontal_space + ehat_3 * height / 2.

    draw_lifting_surface(ax, pos, r_new, height, c_ref, c_ref / 2., c_ref / 4., kite_color, side, body_cross_sections_per_meter, naca)

def draw_kite(ax, q, r, model_options, kite_color, side, body_cross_sections_per_meter):
    # read in inputs
    geometry = model_options['geometry']
    geometry_params = model_options['params']['geometry']

    if geometry['fuselage']:
        draw_kite_fuselage(ax, q, r, geometry['length'], kite_color, side, body_cross_sections_per_meter)

    if geometry['wing']:

        if not geometry['wing_profile'] == None:
            draw_kite_wing(ax, q, r, geometry_params['b_ref'], geometry['c_root'], geometry['c_tip'], kite_color, side,
                           body_cross_sections_per_meter, geometry['wing_profile'])
        else:
            draw_kite_wing(ax, q, r, geometry_params['b_ref'], geometry['c_root'], geometry['c_tip'], kite_color, side, body_cross_sections_per_meter)

    if geometry['tail']:
        draw_kite_horizontal(ax, q, r, geometry['length'], geometry['height'], geometry_params['b_ref'], geometry_params['c_ref'], kite_color, side, body_cross_sections_per_meter)
        draw_kite_vertical(ax, q, r, geometry['length'], geometry['height'], geometry_params['b_ref'], geometry_params['c_ref'], kite_color, side, body_cross_sections_per_meter)



def plot_output_block(plot_table_r, plot_table_c, params, output, plt, fig, idx, output_type, output_name, cosmetics, reload_dict, dim=0):

    # kite nodes
    kite_nodes = params['model']['architecture'].kite_nodes

    plt.subplot(plot_table_r, plot_table_c, idx)
    for n in kite_nodes:
        output_values, tgrid, ndim = merge_output_values(output, output_type, output_name+str(n), dim, reload_dict, cosmetics)

        idx = kite_nodes.index(n)
        plt.plot(tgrid, output_values, color=cosmetics['diagnostics']['colors'][idx])
        plt.grid('on')

    if ndim > 1:
        if dim == 0:
            dimname = 'x'
        elif dim == 1:
            dimname = 'y'
        else:
            dimname = 'z'
        plt.title(output_name + ' (' + dimname + ' hat)')
    else:
        plt.title(output_name)

def merge_output_values(output_vals, output_type, output_name, dim, plot_dict, cosmetics, ref = False):

    # read in inputs
    discretization = plot_dict['discretization']
    if  discretization == 'direct_collocation':

        scheme = plot_dict['options']['nlp']['collocation']['scheme']

        if not ref:
            tgrid_coll = plot_dict['time_grids']['coll']

            # total time points
            tgrid_u_coll = plot_dict['time_grids']['x_coll'][:-1]
        else:
            tgrid_coll = plot_dict['time_grids']['ref']['coll']

            # total time points
            tgrid_u_coll = plot_dict['time_grids']['ref']['x_coll'][:-1]  

    # interval time points
    if not ref:
        tgrid_u = plot_dict['time_grids']['u']
    else:
        tgrid_u = plot_dict['time_grids']['ref']['u']

    if discretization == 'multiple_shooting':
        # take interval values
        output_values = np.array(cas.vertcat(*output_vals['outputs',:,output_type,output_name,dim]).full())
        tgrid = tgrid_u

        ndim = output_vals['outputs',0,output_type,output_name].shape[0]

    elif discretization == 'direct_collocation':
        if scheme != 'radau':
            output_values = []
            # merge interval and node values
            for k in range(plot_dict['n_k']):
                # add interval values
                output_values = cas.vertcat(output_values, output_vals['outputs',k, output_type, output_name,dim])
                if cosmetics['plot_coll']:
                    # add node values
                    output_values = cas.vertcat(output_values, cas.vertcat(*output_vals['coll_outputs',k, :, output_type, output_name,dim]))
            output_values = np.array(output_values)
            if cosmetics['plot_coll']:
                tgrid = tgrid_u_coll
            else:
                tgrid = tgrid_u
            ndim = output_vals['outputs',0,output_type,output_name].shape[0]

        else:
            if cosmetics['plot_coll']:
                # add only node values for radau case
                output_values = np.array(struct_op.coll_slice_to_vec(output_vals['coll_outputs',:,:,output_type,output_name,dim]))
                tgrid = tgrid_coll
                ndim = output_vals['coll_outputs',0,0,output_type,output_name].shape[0]
            else:
                output_values = []
                tgrid = []
                ndim = 1


    # make list of time grid and values
    tgrid = list(chain.from_iterable(tgrid.full().tolist()))
    output_values = list(chain.from_iterable(output_values))

    return output_values, tgrid, ndim

def merge_x_values(V ,name, dim, plot_dict, cosmetics, ref = False):

    # read in inputs

    discretization = plot_dict['discretization']
    if discretization == 'direct_collocation':
        scheme = plot_dict['options']['nlp']['collocation']['scheme']

        if not ref:
            tgrid_coll = plot_dict['time_grids']['coll']
            # total time points
            tgrid_x_coll = plot_dict['time_grids']['x_coll']
        else:
            tgrid_coll = plot_dict['time_grids']['ref']['coll']
            # total time points
            tgrid_x_coll = plot_dict['time_grids']['ref']['x_coll'] 

    # interval time points
    if not ref:
        tgrid_x = plot_dict['time_grids']['x']
    else:
        tgrid_x = plot_dict['time_grids']['ref']['x']

    if discretization == 'multiple_shooting':
        # take interval values
        x_values = np.array(cas.vertcat(*V['x',:,name,dim]).full())
        tgrid = tgrid_x

    elif discretization == 'direct_collocation':
        if scheme != 'radau':
            x_values = []
            # merge interval and node values
            for k in range(plot_dict['n_k']+1):
                # add interval values
                x_values = cas.vertcat(x_values, V['x',k, name,dim])
                if (cosmetics['plot_coll'] and k < plot_dict['n_k']):
                    # add node values
                    x_values = cas.vertcat(x_values, cas.vertcat(*V['coll_var',k, :, 'x', name,dim]).full())
            x_values = np.array(x_values)
            if cosmetics['plot_coll']:
                tgrid = tgrid_x_coll
            else:
                tgrid = tgrid_x

        elif scheme == 'radau':
            if cosmetics['plot_coll']:
                # add node values
                x_values = np.array(struct_op.coll_slice_to_vec(V['coll_var',:, :, 'x', name,dim]))
                tgrid = tgrid_coll
            else:
                x_values = []
                tgrid = []

    # make list of time grid
    tgrid = list(chain.from_iterable(tgrid.full().tolist()))
    x_values = list(chain.from_iterable(x_values))

    return x_values, tgrid

def merge_z_values(V, var_type, name, dim, plot_dict, cosmetics, ref = False):

    # read in inputs
    discretization = plot_dict['discretization']
    if discretization == 'direct_collocation':
        scheme = plot_dict['options']['nlp']['collocation']['scheme']

        if not ref:
            tgrid_coll = plot_dict['time_grids']['coll']
            # total time points
            tgrid_z_coll = plot_dict['time_grids']['x_coll'][:-1]
        else:
            tgrid_coll = plot_dict['time_grids']['ref']['coll']
            # total time points
            tgrid_z_coll = plot_dict['time_grids']['ref']['x_coll'][:-1]   

    # interval time points
    if not ref:
        tgrid_z = plot_dict['time_grids']['u']
    else:
        tgrid_z = plot_dict['time_grids']['ref']['u']

    if discretization == 'multiple_shooting':
        # take interval values
        z_values = np.array(cas.vertcat(*V[var_type,:,name,dim]).full())
        tgrid = tgrid_z

    elif discretization == 'direct_collocation':
        if scheme != 'radau':
            z_values = []
            # merge interval and node values
            for k in range(plot_dict['n_k']):
                # add interval values
                z_values = cas.vertcat(z_values, V[var_type,k, name,dim])
                if cosmetics['plot_coll']:
                    # add node values
                    z_values = cas.vertcat(z_values, cas.vertcat(*V['coll_var',k, :, var_type, name,dim]))
            z_values = np.array(z_values)
            if cosmetics['plot_coll']:
                tgrid = tgrid_z_coll
            else:
                tgrid = tgrid_z

        elif scheme == 'radau':
            if cosmetics['plot_coll']:
                # add node values
                z_values = np.array(struct_op.coll_slice_to_vec(V['coll_var',:, :, var_type, name,dim]))
                tgrid = tgrid_coll
            else:
                z_values = []
                tgrid = []

    # make list of time grid and values
    tgrid = list(chain.from_iterable(tgrid.full().tolist()))
    z_values = list(chain.from_iterable(z_values))

    return z_values, tgrid

def merge_integral_output_values(int_out, name, plot_dict, cosmetics, ref = False):

    # read in inputs
    discretization = plot_dict['discretization']
    if discretization == 'direct_collocation':
        # total time points
        if not ref:
            tgrid_x_coll = plot_dict['time_grids']['x_coll']
        else:
            tgrid_x_coll = plot_dict['time_grids']['ref']['x_coll']

    # interval time points
    if not ref:
        tgrid_x = plot_dict['time_grids']['x']
    else:
        tgrid_x = plot_dict['time_grids']['ref']['x']

    if discretization == 'multiple_shooting':
        # take interval values
        output_values = np.array(cas.vertcat(*int_out['int_out',:,name]).full())
        tgrid = tgrid_x

    elif discretization == 'direct_collocation':
        output_values = []
        # merge interval and node values
        for k in range(plot_dict['n_k']+1):
            # add interval values
            output_values = cas.vertcat(output_values, int_out['int_out',k, name])
            if (cosmetics['plot_coll'] and k < plot_dict['n_k']):
                # add node values
                output_values = cas.vertcat(output_values, cas.vertcat(*int_out['coll_int_out',k, :, name]))

        if cosmetics['plot_coll']:
            tgrid = tgrid_x_coll
        else:
            tgrid = tgrid_x

    # make list of time grid and values
    tgrid = list(chain.from_iterable(tgrid.full().tolist()))

    return output_values, tgrid

def plot_trajectory_contents(ax, plot_dict, cosmetics, side, init_colors=bool(False), plot_kites=bool(True), label=None):

    # read in inputs
    model_options = plot_dict['options']['model']
    kite_nodes = plot_dict['architecture'].kite_nodes
    parent_map = plot_dict['architecture'].parent_map

    body_cross_sections_per_meter = cosmetics['trajectory']['body_cross_sections_per_meter']

    # get kite locations
    kite_locations = []
    kite_ref_locations = []
    kite_rotations = []


    for kite in kite_nodes:

        traj = []
        traj_ref = []
        rot = []

        parent = parent_map[kite]

        for dim in range(3):
            traj.append(
                cas.vertcat(plot_dict['x']['q' + str(kite) + str(parent)][dim])#,
            )
            if cosmetics['plot_ref']:
                traj_ref.append(cas.vertcat(plot_dict['ref']['x']['q' + str(kite) + str(parent)][dim]))

            for dim in range(9):
                rot.append(plot_dict['outputs']['aerodynamics']['r' + str(kite)][dim])

        kite_locations.append(traj)
        kite_ref_locations.append(traj_ref)
        kite_rotations.append(rot)

    old_label = None
    plot_tether = (len(kite_nodes) == 1)
    for kdx in range(len(kite_nodes)):


        if init_colors == True:
            local_color = 'k'
        elif init_colors == False:
            local_color = cosmetics['trajectory']['colors'][kdx]
        else:
            local_color = init_colors

        vertically_stacked_kite_locations = cas.horzcat(kite_locations[kdx][0],
                                                    kite_locations[kdx][1],
                                                    kite_locations[kdx][2])

        if (cosmetics['trajectory']['kite_bodies'] and plot_kites):

            pdx = 0

            q_local = []
            for dim in range(3):
                q_local = cas.vertcat(q_local, kite_locations[kdx][dim][pdx])

            r_local = []
            for dim in range(9):
                r_local = cas.vertcat(r_local, kite_rotations[kdx][dim][pdx])

            draw_kite(ax, q_local, r_local, model_options, local_color, side, body_cross_sections_per_meter)


        if old_label == label:
            label = None
        make_side_plot(ax, vertically_stacked_kite_locations, side, local_color, label=label, plot_tether = plot_tether)

        if cosmetics['plot_ref']:
            vertically_stacked_kite_ref_locations = cas.horzcat(kite_ref_locations[kdx][0],
                                                        kite_ref_locations[kdx][1],
                                                        kite_ref_locations[kdx][2])
            make_side_plot(ax, vertically_stacked_kite_ref_locations, side, local_color, label=label,linestyle='--', plot_tether = plot_tether)

        old_label = label

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

    for name in list(plot_dict['x'].keys()):
        if name[0] == 'q':
            temp_min = np.min(cas.vertcat(temp_min, np.min(plot_dict['x'][name][jdx])))
            temp_max = np.max(cas.vertcat(temp_max, np.max(plot_dict['x'][name][jdx])))

        if name[0] == 'w' and name[1] == dim and cosmetics['trajectory']['wake_nodes']:
            vals = np.array(cas.vertcat(*plot_dict['x'][name]))
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

    plt.subplot(plot_table_r, plot_table_c, idx)
    for jdx in range(number_dim):
        if plot_dict['u_param'] == 'poly':
            p = plt.plot(tgrid_ip, plot_dict['u'][name][jdx])
            if plot_dict['options']['visualization']['cosmetics']['plot_bounds']:
                plot_bounds(plot_dict, 'u', name, jdx, tgrid_ip, p)
            if plot_dict['options']['visualization']['cosmetics']['plot_ref']:
                plt.plot(plot_dict['time_grids']['ref']['ip'], plot_dict['ref']['u'][name][jdx],
                    linestyle= '--', color = p[-1].get_color() )

        else:
            p = plt.step(tgrid_ip, plot_dict['u'][name][jdx],where='post')
            if plot_dict['options']['visualization']['cosmetics']['plot_bounds']:
                plot_bounds(plot_dict, 'u', name, jdx, tgrid_ip, p)
            if plot_dict['options']['visualization']['cosmetics']['plot_ref']:
                plt.step(plot_dict['time_grids']['ref']['ip'], plot_dict['ref']['u'][name][jdx],where='post',
                    linestyle =  '--', color = p[-1].get_color())
    plt.grid(True)
    plt.title(name)
    plt.autoscale(enable=True, axis= 'x', tight = True)

def get_sweep_colors(number_of_trials):

    cmap = plt.get_cmap('jet')
    c_norm = colors.Normalize(vmin=0, vmax=(number_of_trials - 1))
    scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=cmap)

    color_list = []
    for trial in range(number_of_trials):
        color_list += [scalar_map.to_rgba(np.float(trial))]

    return color_list

def spline_interpolation(time_grid, values, time_grid_ip, n_points, name):
    """ Interpolate solution values with b-splines
    """

    # create interpolating function
    if all(v == 0 for v in values):
        # can't use splines if all entries zero
        values_ip = np.zeros(len(time_grid_ip))
    else:
        spline = cas.interpolant(name, 'bspline', [time_grid], values, {})
        # function map to new discretization
        spline = spline.map(n_points)
        # interpolate
        values_ip = spline(time_grid_ip).full()[0]

    return values_ip

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
    plot_dict['Collocation'] = nlp.Collocation

    # model information
    plot_dict['integral_variables'] = list(model.integral_outputs.keys())
    plot_dict['outputs_dict'] = struct_op.strip_of_contents(model.outputs_dict)
    plot_dict['architecture'] = model.architecture
    plot_dict['variables'] = struct_op.strip_of_contents(model.variables)
    plot_dict['parameters'] = struct_op.strip_of_contents(model.parameters)
    plot_dict['variables_dict'] = struct_op.strip_of_contents(model.variables_dict)
    plot_dict['scaling'] = model.scaling
    plot_dict['variable_bounds'] = model.variable_bounds

    # wind information
    u_ref = model.options['params']['wind']['u_ref']
    plot_dict['u_ref'] = float(u_ref)

    return plot_dict

def recalibrate_visualization(V_plot, plot_dict, output_vals, integral_outputs_final, options, time_grids, cost, name, V_ref, iterations=None, return_status_numeric=None, timings=None, N=None, ):
    """
    Recalibrate plot dict with all calibration operation that need to be perfomed once for every plot.
    :param plot_dict: plot dictionary before recalibration
    :return: recalibrated plot dictionary
    """

    # extract information
    cosmetics = options['visualization']['cosmetics']
    if N is not None:
        cosmetics['interpolation']['N'] = int(N)

    plot_dict['cost'] = cost

    # add V_plot to dict
    scaling = plot_dict['scaling']
    plot_dict['V_plot'] = struct_op.scaled_to_si(V_plot, scaling)
    plot_dict['V_ref'] = struct_op.scaled_to_si(V_ref, scaling)

    # get new name
    plot_dict['name'] = name

    # get new outputs
    plot_dict['output_vals'] = output_vals
    plot_dict['integral_outputs_final'] = integral_outputs_final

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
    plot_dict['max_x'] = np.max(np.array(V_plot['x', :, 'q10', 0])) * 1.2
    plot_dict['max_y'] = np.max(np.abs(np.array(V_plot['x', :, 'q10', 1]))) * 1.2
    plot_dict['max_z'] = np.max(np.array(V_plot['x', :, 'q10', 2])) * 1.2
    plot_dict['mazim'] = np.max([plot_dict['max_x'], plot_dict['max_y'], plot_dict['max_z']])
    plot_dict['scale_power'] = 1.  # e-3
    plot_dict['scale_axes'] = np.float(V_plot['x', 0, 'l_t'])


    dashes = []
    for ldx in range(20):
        new_dash = []
        for jdx in range(4):
            new_dash += [int(np.random.randint(1,6))]
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
    variables_dict = plot_dict['variables']
    outputs_dict = plot_dict['outputs_dict']
    output_vals = plot_dict['output_vals'][1]
    integral_outputs = plot_dict['integral_outputs_final']
    nlp_options = plot_dict['options']['nlp']
    V_plot = plot_dict['V_plot']
    if plot_dict['Collocation'] is not None:
        interpolator = plot_dict['Collocation'].build_interpolator(nlp_options, V_plot)
        int_interpolator = plot_dict['Collocation'].build_interpolator(nlp_options, V_plot, integral_outputs)
        u_param = plot_dict['u_param']
    else:
        u_param = 'zoh'

    # add states and outputs to plotting dict
    plot_dict['x'] = {}
    plot_dict['z'] = {}
    plot_dict['u'] = {}
    plot_dict['outputs'] = {}
    plot_dict['integral_outputs'] = {}

    # interpolating time grid
    n_points = cosmetics['interpolation']['N']

    # x-values
    for name in list(struct_op.subkeys(variables_dict, 'x')):
        plot_dict['x'][name] = []
        for j in range(variables_dict['x',name].shape[0]):
            # merge values
            values, time_grid = merge_x_values(V_plot, name, j, plot_dict, cosmetics)
            plot_dict['time_grids']['ip'] = np.linspace(time_grid[0], time_grid[-1], n_points)

            # interpolate
            if cosmetics['interpolation']['type'] == 'spline' or plot_dict['discretization'] == 'multiple_shooting':
                values_ip = spline_interpolation(time_grid, values, plot_dict['time_grids']['ip'], n_points, name)
            elif cosmetics['interpolation']['type'] == 'poly' and plot_dict['discretization'] == 'direct_collocation':
                values_ip = interpolator(plot_dict['time_grids']['ip'], name, j, 'x')
            plot_dict['x'][name] += [values_ip.full()]

    # z-values
    for var_type in set(variables_dict.keys()) - set(['x', 'u', 'xdot', 'theta']):
        for name in list(struct_op.subkeys(variables_dict,var_type)):
            plot_dict[var_type][name] = []
            for j in range(variables_dict[var_type,name].shape[0]):
                if plot_dict['discretization'] == 'direct_collocation':
                    values_ip = interpolator(plot_dict['time_grids']['ip'], name, j, var_type)
                else:
                    values, time_grid = merge_z_values(V_plot, var_type, name, j, plot_dict, cosmetics)
                    # interpolate
                    values_ip = spline_interpolation(time_grid, values, plot_dict['time_grids']['ip'], n_points, name)
                plot_dict[var_type][name] += [values_ip]

    # u-values
    for name in list(struct_op.subkeys(variables_dict,'u')):
        plot_dict['u'][name] = []
        for j in range(variables_dict['u',name].shape[0]):

            if u_param == 'zoh':
                control = plot_dict['V_plot']['u',:,name,j]
                time_grids = plot_dict['time_grids']
                values_ip = sample_and_hold_controls(time_grids, control)
            elif u_param == 'poly':
                values_ip = interpolator(plot_dict['time_grids']['ip'], name, j, 'u')
            plot_dict['u'][name] += [values_ip]

    # output values
    for output_type in list(outputs_dict.keys()):
        plot_dict['outputs'][output_type] = {}
        for name in list(outputs_dict[output_type].keys()):
            plot_dict['outputs'][output_type][name] = []
            for j in range(outputs_dict[output_type][name].shape[0]):
                # merge values
                values, time_grid, ndim = merge_output_values(output_vals, output_type, name, j, plot_dict, cosmetics)
                # inteprolate
                values_ip = spline_interpolation(time_grid, values, plot_dict['time_grids']['ip'], n_points, name)
                plot_dict['outputs'][output_type][name] += [values_ip]

    # integral outptus
    if plot_dict['discretization'] == 'direct_collocation':
        for name in plot_dict['integral_variables']:
            values_ip = int_interpolator(plot_dict['time_grids']['ip'], name, 0, 'int_out')
            plot_dict['integral_outputs'][name] = [values_ip]

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
    variables_dict = plot_dict['variables']
    nlp_options = plot_dict['options']['nlp']
    outputs_dict = plot_dict['outputs_dict']
    output_vals = plot_dict['output_vals'][2]
    V_ref = plot_dict['V_ref']

    if plot_dict['Collocation'] is not None:
        interpolator = plot_dict['Collocation'].build_interpolator(nlp_options, V_ref)
        u_param = plot_dict['u_param']
    else:
        u_param = 'zoh'

    # add states and outputs to plotting dict
    plot_dict['ref'] = {'x': {},'u':{},'z':{},'time_grids':{},'outputs':{}}

    # interpolating time grid
    n_points = plot_dict['time_grids']['ip'].shape[0]

    # x-values
    for name in list(struct_op.subkeys(variables_dict, 'x')):
        plot_dict['ref']['x'][name] = []
        for j in range(variables_dict['x',name].shape[0]):
            # merge values
            values, time_grid = merge_x_values(V_ref, name, j, plot_dict, cosmetics, ref = True)
            plot_dict['time_grids']['ref']['ip'] =  np.linspace(time_grid[0], time_grid[-1], n_points)

            # interpolate
            if cosmetics['interpolation']['type'] == 'spline' or plot_dict['discretization'] == 'multiple_shooting':
                values_ip = spline_interpolation(time_grid, values, plot_dict['time_grids']['ref']['ip'], n_points, name)
            elif cosmetics['interpolation']['type'] == 'poly' and plot_dict['discretization'] == 'direct_collocation':
                values_ip = interpolator(plot_dict['time_grids']['ref']['ip'], name, j, 'x')
            plot_dict['ref']['x'][name] += [values_ip.full()]

    # z-values
    for var_type in set(variables_dict.keys()) - set(['x', 'u', 'xdot', 'theta']):
        for name in list(struct_op.subkeys(variables_dict,var_type)):
            plot_dict['ref'][var_type][name] = []
            for j in range(variables_dict[var_type,name].shape[0]):
                if plot_dict['discretization'] == 'direct_collocation':
                    values_ip = interpolator(plot_dict['time_grids']['ref']['ip'], name, j, var_type)
                else:
                    values, time_grid = merge_z_values(V_ref, var_type, name, j, plot_dict, cosmetics, ref = True)
                    # interpolate
                    values_ip = spline_interpolation(time_grid, values, plot_dict['time_grids']['ref']['ip'], n_points, name)
                plot_dict['ref'][var_type][name] += [values_ip]

    # u-values
    for name in list(struct_op.subkeys(variables_dict,'u')):
        plot_dict['ref']['u'][name] = []
        for j in range(variables_dict['u',name].shape[0]):

            if u_param == 'zoh':
                control = plot_dict['V_ref']['u',:,name,j]
                time_grids = plot_dict['time_grids']['ref']
                values_ip = sample_and_hold_controls(time_grids, control)
            elif u_param == 'poly':
                values_ip = interpolator(plot_dict['time_grids']['ref']['ip'], name, j, 'u')
            plot_dict['ref']['u'][name] += [values_ip]

    # output values
    for output_type in list(outputs_dict.keys()):
        plot_dict['ref']['outputs'][output_type] = {}
        for name in list(outputs_dict[output_type].keys()):
            plot_dict['ref']['outputs'][output_type][name] = []
            for j in range(outputs_dict[output_type][name].shape[0]):
                # merge values
                values, time_grid, ndim = merge_output_values(output_vals, output_type, name, j, plot_dict, cosmetics, ref = True)
                # interpolate
                values_ip = spline_interpolation(time_grid, values, plot_dict['time_grids']['ref']['ip'], n_points, name)
                plot_dict['ref']['outputs'][output_type][name] += [values_ip]

    return plot_dict


def sample_and_hold_controls(time_grids, control):

    tgrid_u = time_grids['u']
    tgrid_ip = time_grids['ip']
    values_ip = np.zeros(len(tgrid_ip),)
    for index in range(len(tgrid_ip)):
        for j in range(tgrid_u.shape[0] - 1):
            if tgrid_u[j] < tgrid_ip[index] and tgrid_ip[index] < tgrid_u[j + 1]:
                values_ip[index] = control[j]
                break
        if tgrid_u[-1] < tgrid_ip[index]:
            values_ip[index] = control[-1]

    return values_ip

def map_flag_to_function(flag, plot_dict, cosmetics, fig_name, plot_logic_dict):

    standard_args = (plot_dict, cosmetics, fig_name)
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

def assemble_variable_slice_from_interpolated_data(plot_dict, index, var_type):

    variables_dict = plot_dict['variables_dict']

    if not var_type in variables_dict.keys():
        awelogger.logger.error('requested variable type does not exist.')
        return None

    else:
        local_dict = variables_dict[var_type]
        collected_vals = []

        for name in local_dict.keys():
            column_vals = plot_dict[var_type][name]
            # assume that all variables are saved in column format!!
            n_entries = len(column_vals)

            for edx in range(n_entries):
                entry_val = column_vals[edx][index]
                collected_vals = cas.vertcat(collected_vals, entry_val)

        var_slice = local_dict(collected_vals)
        return var_slice

def plot_bounds(plot_dict, var_type, name, jdx, tgrid_ip, p):

    bounds = plot_dict['variable_bounds'][var_type][name]
    scaling = plot_dict['scaling'][var_type][name]
    if type(bounds['lb']) == np.ndarray:
        lb = bounds['lb'][jdx]
    else:
        lb = bounds['lb']
    if type(bounds['ub']) == np.ndarray:
        ub = bounds['ub'][jdx]
    else:
        ub = bounds['ub']
    if lb > -np.inf:
        plt.plot(tgrid_ip, [lb*scaling]*len(tgrid_ip), linestyle='dotted', color = p[-1].get_color())
    if ub < np.inf:
        plt.plot(tgrid_ip, [ub*scaling]*len(tgrid_ip), linestyle='dotted', color = p[-1].get_color())

    return None