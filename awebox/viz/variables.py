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
import pdb

import matplotlib
import awebox.tools.vector_operations as vect_op

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import awebox.tools.struct_operations as struct_op
import awebox.tools.print_operations as print_op

from . import tools
import numpy as np
from awebox.logger.logger import Logger as awelogger
import casadi.tools as cas

def plot_states(plot_dict, cosmetics, fig_name, individual_state=None, fig_num=None):

    # read in inputs
    variables_dict = plot_dict['variables_dict']
    integral_variables = plot_dict['integral_output_names']

    if individual_state == None:
        variables_to_plot = []
        for var_name in variables_dict['x'].keys():
            if not is_wake_variable(var_name):
                variables_to_plot += [var_name]
        integral_variables_to_plot = integral_variables

    else:
        if individual_state in list(variables_dict['x'].keys()):
            variables_to_plot = [individual_state]
            integral_variables_to_plot = []
        elif individual_state in integral_variables:
            variables_to_plot = []
            integral_variables_to_plot = [individual_state]

    plot_variables_from_list(plot_dict, cosmetics, fig_name, 'x', variables_to_plot, integral_variables_to_plot, fig_num)

    return None

def is_wake_variable(name):
    is_wake_variable = (name[0] == 'w') or (name[:2] == 'dw')
    return is_wake_variable

def plot_wake_states(plot_dict, cosmetics, fig_name, individual_state=None, fig_num=None):

    # read in inputs
    variables_dict = plot_dict['variables_dict']

    if individual_state == None:
        variables_to_plot = []
        for var_name in variables_dict['x'].keys():
            if is_wake_variable(var_name):
                variables_to_plot += [var_name]

    else:
        if individual_state in list(variables_dict['x'].keys()):
            variables_to_plot = [individual_state]

    integral_variables_to_plot = []

    plot_variables_from_list(plot_dict, cosmetics, fig_name, 'x', variables_to_plot, integral_variables_to_plot, fig_num)

    return None


def include_specific_variable_solution(relevant_axes, plot_dict, var_type, var_name, var_dim, local_color):

    if plot_dict['cosmetics']['plot_ref']:
        search_name = 'ref'
    else:
        search_name = 'plot'
    search_name = "V_" + search_name + '_' + plot_dict['cosmetics']['variables']['si_or_scaled']
    V_search = plot_dict[search_name]

    base_size = 5

    if var_type in set(V_search.keys()) - set(['theta', 'xi', 'phi', 'u']):
        if var_type == 'x':
            control_times = np.array(vect_op.columnize(plot_dict['time_grids']['x']))
        else:
            control_times = np.array(vect_op.columnize(plot_dict['time_grids']['u']))
        control_series = np.array(vect_op.columnize(V_search[var_type, :, var_name, var_dim]))

        label = var_name + " [" + str(var_dim) + "] (found at control)"
        relevant_axes.plot(control_times, control_series, '*', markersize=base_size*2, color=local_color, label=label)

    if 'coll_var' in V_search.keys():
        collocation_times = np.array(vect_op.columnize(plot_dict['time_grids']['coll']))
        collocation_series = np.array(vect_op.columnize(V_search["coll_var", :, :, var_type, var_name, var_dim]))

        label = var_name + " [" + str(var_dim) + "] (found at collocation)"
        relevant_axes.plot(collocation_times, collocation_series, '*', markersize=base_size, color=local_color, label=label)

    return None


def plot_lifted(plot_dict, cosmetics, fig_name, individual_state=None, fig_num=None):

    # read in inputs
    variables_dict = plot_dict['variables_dict']
    integral_variables = plot_dict['integral_output_names']

    # check if lifted variables exist
    if 'z' not in variables_dict.keys():
        awelogger.logger.warning('Plot for lifted variables requested, but no lifted variables found. Ignoring request.')
        return None

    if individual_state == None:
        variables_to_plot = []
        for var_name in variables_dict['z'].keys():
            if not is_wake_variable(var_name):
                variables_to_plot += [var_name]
        integral_variables_to_plot = integral_variables

    else:
        if individual_state in list(variables_dict['z'].keys()):
            variables_to_plot = [individual_state]
            integral_variables_to_plot = []
        elif individual_state in integral_variables:
            variables_to_plot = []
            integral_variables_to_plot = [individual_state]

    plot_variables_from_list(plot_dict, cosmetics, fig_name, 'z', variables_to_plot, integral_variables_to_plot, fig_num)

    return None


def plot_wake_lifted(plot_dict, cosmetics, fig_name, individual_state=None, fig_num=None):

    # read in inputs
    integral_outputs = plot_dict['integral_output_vals']['opt']
    variables_dict = plot_dict['variables_dict']
    tgrid_ip = plot_dict['time_grids']['ip']

    # check if lifted variables exist
    if 'z' not in variables_dict.keys():
        awelogger.logger.warning('Plot for lifted varibles requested, but no lifted variables found. Ignoring request.')
        return None

    if individual_state == None:
        variables_to_plot = []
        for var_name in variables_dict['z'].keys():
            if is_wake_variable(var_name):
                variables_to_plot += [var_name]
    else:
        if individual_state in list(variables_dict['z'].keys()):
            variables_to_plot = [individual_state]

    integral_variables_to_plot = []

    plot_variables_from_list(plot_dict, cosmetics, fig_name, 'z', variables_to_plot, integral_variables_to_plot, fig_num)

    return None


def plot_controls(plot_dict, cosmetics, fig_name, individual_control=None, fig_num = None):

    # read in inputs
    V_plot_si = plot_dict['V_plot_si']
    variables_dict = plot_dict['variables_dict']

    if individual_control == None:
        plot_table_r = 2
        control_keys = list(variables_dict['u'].keys())
        controls_to_plot = []
        for ctrl in control_keys:
            if 'fict' not in ctrl:
                controls_to_plot.append(ctrl)
        plot_table_c = int(len(controls_to_plot) / plot_table_r) + 1 * \
                                                    (not np.mod(len(controls_to_plot), plot_table_r) == 0)
    else:
        controls_to_plot = [individual_control]
        plot_table_r = len(controls_to_plot)
        plot_table_c = 1

    # create new figure if desired
    if fig_num is not None:
        fig = plt.figure(num = fig_num)

    else:
        fig, _ = plt.subplots(nrows = plot_table_r, ncols = plot_table_c)

    pdu = 1
    for name in controls_to_plot:

        number_dim = variables_dict['u'][name].shape[0]
        tools.plot_control_block(cosmetics, V_plot_si, plt, fig, plot_table_r, plot_table_c, pdu, 'u', name, plot_dict, number_dim)
        pdu = pdu + 1

    plt.suptitle(fig_name)
    fig.canvas.draw()


def plot_invariants(plot_dict, cosmetics, fig_name):

    # read in inputs
    number_of_nodes = plot_dict['architecture'].number_of_nodes
    parent_map = plot_dict['architecture'].parent_map

    interp_name = 'interpolation_' + plot_dict['cosmetics']['variables']['si_or_scaled']
    ref_name = 'ref_' + plot_dict['cosmetics']['variables']['si_or_scaled']

    fig = plt.figure()
    fig.clf()
    legend_names = []
    tgrid_ip = plot_dict['time_grids']['ip']
    invariants = plot_dict[interp_name]['outputs']['invariants']
    if cosmetics['plot_ref']:
        ref_invariants = plot_dict[ref_name]['outputs']['invariants']
        ref_tgrid_ip = plot_dict['time_grids']['ref']['ip']

    for n in range(1, number_of_nodes):
        parent = parent_map[n]
        for prefix in ['','d', 'dd']:
            p = plt.semilogy(tgrid_ip, abs(invariants[prefix + 'c' + str(n) + str(parent)][0]), label = prefix + 'c' + str(n) + str(parent))
            if cosmetics['plot_ref']:
                plt.semilogy(ref_tgrid_ip, abs(ref_invariants[prefix + 'c' + str(n) + str(parent)][0]), linestyle = '--', color = p[-1].get_color())

    if plot_dict['options']['model']['cross_tether'] and number_of_nodes > 2:
        for l in plot_dict['architecture'].layer_nodes:
            kites = plot_dict['architecture'].kites_map[l]
            if len(kites) == 2:
                c_name = 'c{}{}'.format(kites[0], kites[1])
                for prefix in ['','d', 'dd']:
                    p = plt.semilogy(tgrid_ip, abs(invariants[prefix + c_name][0]), label = prefix + c_name)
                    if cosmetics['plot_ref']:
                        plt.semilogy(ref_tgrid_ip, abs(ref_invariants[prefix + c_name][0]), linestyle = '--', color = p[-1].get_color())
            else:
                for k in range(len(kites)):
                    c_name = 'c{}{}'.format(kites[k], kites[(k+1)%len(kites)])
                    for prefix in ['','d', 'dd']:
                        p = plt.semilogy(tgrid_ip, abs(invariants[prefix + c_name][0]), label = prefix + c_name)
                        if cosmetics['plot_ref']:
                            plt.semilogy(ref_tgrid_ip, abs(ref_invariants[prefix + c_name][0]), linestyle = '--', color = p[-1].get_color())
    plt.legend()
    plt.suptitle(fig_name)

    return None


def plot_algebraic_variables(plot_dict, cosmetics, fig_name):
    # read in inputs
    number_of_nodes = plot_dict['architecture'].number_of_nodes
    parent_map = plot_dict['architecture'].parent_map

    interp_name = 'interpolation_' + plot_dict['cosmetics']['variables']['si_or_scaled']
    ref_name = 'ref_' + plot_dict['cosmetics']['variables']['si_or_scaled']

    fig = plt.figure()
    fig.clf()
    legend_names = []
    tgrid_ip = plot_dict['time_grids']['ip']

    for n in range(1, number_of_nodes):
        parent = parent_map[n]
        lam_name = 'lambda' + str(n) + str(parent)
        lambdavec = plot_dict[interp_name]['z'][lam_name]
        p = plt.plot(tgrid_ip, lambdavec[0])
        if cosmetics['plot_bounds']:
            tools.plot_bounds(plot_dict, 'z', lam_name, 0, tgrid_ip, p=p)
        if cosmetics['plot_ref']:
            plt.plot(plot_dict['time_grids']['ref']['ip'], plot_dict[ref_name]['z'][lam_name][0],
                     linestyle='--', color=p[-1].get_color())
        legend_names.append('lambda' + str(n) + str(parent))

    if plot_dict['options']['model']['cross_tether'] and number_of_nodes > 2:
        for l in plot_dict['architecture'].layer_nodes:
            kites = plot_dict['architecture'].kites_map[l]
            if len(kites) == 2:
                lam_name = 'lambda{}{}'.format(kites[0], kites[1])
                lambdavec = plot_dict[interp_name]['z'][lam_name]
                p = plt.plot(tgrid_ip, lambdavec[0])
                if cosmetics['plot_bounds']:
                    tools.plot_bounds(plot_dict, 'z', lam_name, 0, tgrid_ip, p=p)
                if cosmetics['plot_ref']:
                    plt.plot(
                        plot_dict['time_grids']['ref']['ip'], plot_dict[ref_name]['z'][lam_name][0],
                        linestyle='--', color=p[-1].get_color()
                    )
                legend_names.append(lam_name)
            else:
                for k in range(len(kites)):
                    lam_name = 'lambda{}{}'.format(kites[k], kites[(k + 1) % len(kites)])
                    lambdavec = plot_dict[interp_name]['z'][lam_name]
                    p = plt.plot(tgrid_ip, lambdavec[0])
                    if cosmetics['plot_bounds']:
                        tools.plot_bounds(plot_dict, 'z', lam_name, 0, tgrid_ip, p=p)
                    if cosmetics['plot_ref']:
                        plt.plot(
                            plot_dict['time_grids']['ref']['ip'],
                            plot_dict[ref_name]['z'][lam_name][0],
                            linestyle='--', color=p[-1].get_color()
                        )
                    legend_names.append(lam_name)
    plt.legend(legend_names)
    plt.suptitle(fig_name)


def plot_variables_from_list(plot_dict, cosmetics, fig_name, var_type, variables_to_plot, integral_variables_to_plot, fig_num=None):

    search_name = 'interpolation_' + plot_dict['cosmetics']['variables']['si_or_scaled']

    if len(variables_to_plot + integral_variables_to_plot) > 0:

        counter = 0
        for var_name in variables_to_plot:
            if not is_wake_variable(var_name):
                counter += 1
        counter += len(integral_variables_to_plot)

        fig, axes = setup_fig_and_axes(variables_to_plot, integral_variables_to_plot, fig_num)

        counter = 0
        for var_name in variables_to_plot:
            ax = plt.axes(axes[counter])
            plot_indiv_variable(ax, plot_dict, cosmetics, var_type, var_name)
            plot_indiv_variable_at_discete_solution(ax, plot_dict, cosmetics, var_type, var_name)
            counter += 1

        for var_name in integral_variables_to_plot:
            ax = plt.axes(axes[counter])
            variable_dimensions = len(plot_dict[search_name]['integral_outputs'][var_name])
            for dim in range(variable_dimensions):
                plot_indiv_integral_variable(ax, plot_dict, cosmetics, var_name, dim=dim)
                counter += 1

        plt.subplots_adjust(wspace=0.3, hspace=2.0)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 3))
        plt.suptitle(fig_name)

    else:
        message = 'a request to plot variables of type ' + var_type + ' was passed an empty list of variable-names.' \
                                                                      ' as a result, the request was ignored.'
        awelogger.logger.warning(message)

    return None

def plot_indiv_variable(ax, plot_dict, cosmetics, var_type, var_name, first_time_through=True):

    variables_dict = plot_dict['variables_dict']

    ax = plt.axes(ax)
    number_of_dimensions = variables_dict[var_type][var_name].shape[0]

    show_ref = cosmetics['plot_ref'] and first_time_through
    if show_ref:
        search_name = 'ref'
    else:
        search_name = 'interpolation'
    search_name = search_name + '_' + plot_dict['cosmetics']['variables']['si_or_scaled']
    variables_plot = plot_dict[search_name][var_type][var_name]

    if show_ref:
        tgrid_ip = plot_dict['time_grids']['ref']['ip']
        linestyle = '--'
        first_time_through = False
    else:
        tgrid_ip = plot_dict['time_grids']['ip']
        linestyle = '-'

    for dim in range(number_of_dimensions):
        variable_data = variables_plot[dim]
        color = cosmetics['trajectory']['colors'][dim]
        p = plt.plot(tgrid_ip, variable_data, linestyle=linestyle, color=color)

        if cosmetics['plot_bounds']:
            tools.plot_bounds(plot_dict, var_type, var_name, dim, tgrid_ip, p=p)

    if show_ref:
        plot_indiv_variable(ax, plot_dict, cosmetics, var_type, var_name, first_time_through=first_time_through)

    plt.title(var_name)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.grid(True)
    ax.tick_params(axis='both', which='major')
    return None

def plot_indiv_variable_at_discete_solution(ax, plot_dict, cosmetics, var_type, var_name):

    if cosmetics['variables']['include_solution']:

        ax = plt.axes(ax)
        variables_dict = plot_dict['variables_dict']
        number_of_dimensions = variables_dict[var_type][var_name].shape[0]
        for var_dim in range(number_of_dimensions):
            color = cosmetics['trajectory']['colors'][var_dim]
            include_specific_variable_solution(ax, plot_dict, var_type, var_name, var_dim, color)

    return None


def plot_indiv_integral_variable(ax, plot_dict, cosmetics, var_name, dim=0):

    ax = plt.axes(ax)

    tgrid_out = plot_dict['time_grids']['ip']

    search_name = 'interpolation_' + plot_dict['cosmetics']['variables']['si_or_scaled']
    out_values = plot_dict[search_name]['integral_outputs'][var_name][dim]

    plt.plot(np.array(tgrid_out), np.array(out_values))

    plt.title(var_name)
    plt.autoscale(enable=True, axis = 'x', tight = True)
    plt.grid(True)
    ax.tick_params(axis='both', which='major')

    return None

def setup_fig_and_axes(variables_to_plot, integral_variables_to_plot, fig_num=None):

    counter = len(variables_to_plot) + len(integral_variables_to_plot)

    if counter == 1:
        plot_table_r = 1
        plot_table_c = 1
    elif np.mod(counter, 3) == 0:
        plot_table_r = 3
        plot_table_c = int(counter / plot_table_r)
    elif np.mod(counter, 4) == 0:
        plot_table_r = 4
        plot_table_c = int(counter / plot_table_r)
    else:
        plot_table_r = 3
        plot_table_c = int(np.ceil(float(counter) / float(plot_table_r)))

    # create new figure if desired
    if fig_num is not None:
        fig = plt.figure(num = fig_num)
        axes = fig.axes
        if len(axes) == 0: # if figure does not exist yet
            fig, axes = plt.subplots(num = fig_num, nrows = plot_table_r, ncols = plot_table_c)

    else:
        fig, axes = plt.subplots(nrows = plot_table_r, ncols = plot_table_c)

    # make vertical column array or list of all axes
    if type(axes) == np.ndarray:
        axes = axes.reshape(plot_table_r*plot_table_c,)
    elif type(axes) is not list:
        axes = [axes]

    return fig, axes