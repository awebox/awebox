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
import matplotlib.pyplot as plt
from . import tools
import numpy as np
from awebox.logger.logger import Logger as awelogger

def plot_states(plot_dict, cosmetics, fig_name, individual_state=None, fig_num=None):

    # read in inputs
    variables_dict = plot_dict['variables_dict']
    integral_variables = plot_dict['integral_variables']
    tgrid_ip = plot_dict['time_grids']['ip']

    if individual_state == None:
        variables_to_plot = list(variables_dict['xd'].keys())
        integral_variables_to_plot = integral_variables
    else:
        if individual_state in list(variables_dict['xd'].keys()):
            variables_to_plot = [individual_state]
            integral_variables_to_plot = []
        elif individual_state in integral_variables:
            variables_to_plot = []
            integral_variables_to_plot = [individual_state]

    counter = 0
    for name in variables_to_plot:
        if not is_wake_variable(name):
            counter += 1
    counter += len(integral_variables)

    if individual_state == None:
        plot_table_c = 3
        plot_table_r = int(np.ceil(np.float(counter) / np.float(plot_table_c)))
    else:
        plot_table_r = len(variables_to_plot + integral_variables_to_plot)
        plot_table_c = 1

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

    counter = 0
    for name in variables_to_plot:
        if not is_wake_variable(name):
            counter += 1
            ax = plt.axes(axes[counter-1])
            for jdx in range(variables_dict['xd'][name].shape[0]):
                p = plt.plot(tgrid_ip, plot_dict['xd'][name][jdx])
                if cosmetics['plot_bounds']:
                    tools.plot_bounds(plot_dict, 'xd', name, jdx, tgrid_ip, p)
                if cosmetics['plot_ref']:
                    plt.plot(plot_dict['time_grids']['ref']['ip'], plot_dict['ref']['xd'][name][jdx],
                        linestyle= '--', color = p[-1].get_color() )

                plt.title(name)

    for name in integral_variables_to_plot:
        counter += 1
        ax = plt.axes(axes[counter-1])
        if plot_dict['discretization'] == 'multiple_shooting':
            out_values, tgrid_out = tools.merge_integral_output_values(plot_dict['integral_outputs_final'],name, plot_dict, cosmetics)
            p = plt.plot(tgrid_out, out_values)
        else:
            p = plt.plot(tgrid_ip, plot_dict['integral_outputs'][name][0])
        plt.title(name)

    ax.tick_params(axis='both', which='major')
    plt.subplots_adjust(wspace=0.3, hspace=2.0)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 3))
    plt.suptitle(fig_name)


def is_wake_variable(name):
    is_wake_variable = (name[0] == 'w') or (name[:2] == 'dw')
    return is_wake_variable

def plot_wake_states(plot_dict, cosmetics, fig_name, individual_state=None, fig_num=None):

    # read in inputs
    variables_dict = plot_dict['variables_dict']
    integral_variables = plot_dict['integral_variables']
    tgrid_ip = plot_dict['time_grids']['ip']

    if individual_state == None:
        variables_to_plot = list(variables_dict['xd'].keys())
        integral_variables_to_plot = integral_variables
    else:
        if individual_state in list(variables_dict['xd'].keys()):
            variables_to_plot = [individual_state]
            integral_variables_to_plot = []
        elif individual_state in integral_variables:
            variables_to_plot = []
            integral_variables_to_plot = [individual_state]

    counter = 0
    for name in variables_to_plot:
        if is_wake_variable(name):
            counter += 1
    counter += len(integral_variables)

    if individual_state == None:
        plot_table_r = 4
        plot_table_c = int(np.ceil(np.float(counter) / np.float(plot_table_r)))
    else:
        plot_table_r = len(variables_to_plot + integral_variables_to_plot)
        plot_table_c = 1

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

    counter = 0
    for name in variables_to_plot:
        if is_wake_variable(name):
            counter += 1
            ax = plt.axes(axes[counter-1])
            for jdx in range(variables_dict['xd'][name].shape[0]):
                p = plt.plot(tgrid_ip, plot_dict['xd'][name][jdx])
                if cosmetics['plot_ref']:
                    plt.plot(plot_dict['time_grids']['ref']['ip'], plot_dict['ref']['xd'][name][jdx],
                        linestyle= '--', color = p[-1].get_color() )

                plt.title(name)

    for name in integral_variables_to_plot:
        counter += 1
        ax = plt.axes(axes[counter-1])
        if plot_dict['discretization'] == 'multiple_shooting':
            out_values, tgrid_out = tools.merge_integral_output_values(plot_dict['integral_outputs_final'],name, plot_dict, cosmetics)
            p = plt.plot(tgrid_out, out_values)
        else:
            p = plt.plot(tgrid_ip, plot_dict['integral_outputs'][name][0])
        plt.title(name)

    ax.tick_params(axis='both', which='major')
    plt.subplots_adjust(wspace=0.3, hspace=2.0)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 3))
    plt.suptitle(fig_name)



def plot_lifted(plot_dict, cosmetics, fig_name, individual_state=None, fig_num=None):

    # read in inputs
    integral_outputs = plot_dict['integral_outputs_final']
    variables_dict = plot_dict['variables_dict']
    integral_variables = plot_dict['integral_variables']
    tgrid_ip = plot_dict['time_grids']['ip']

    # check if lifted variables exist
    if 'xl' not in variables_dict.keys():
        awelogger.logger.warning('Plot for lifted varibles requested, but no lifted variables found. Ignoring request.')
        return None

    if individual_state == None:
        variables_to_plot = list(variables_dict['xl'].keys())
        integral_variables_to_plot = integral_variables
    else:
        if individual_state in list(variables_dict['xl'].keys()):
            variables_to_plot = [individual_state]
            integral_variables_to_plot = []
        elif individual_state in integral_variables:
            variables_to_plot = []
            integral_variables_to_plot = [individual_state]

    counter = 0
    for name in variables_to_plot:
        if not is_wake_variable(name):
            counter += 1
    counter += len(list(integral_outputs.keys()))

    if individual_state == None:
        plot_table_r = 4
        plot_table_c = int(np.ceil(np.float(counter) / np.float(plot_table_r)))
    else:
        plot_table_r = len(variables_to_plot + integral_variables_to_plot)
        plot_table_c = 1

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

    counter = 0
    for name in variables_to_plot:
        if not is_wake_variable(name):
            counter += 1
            ax = plt.axes(axes[counter-1])
            for jdx in range(variables_dict['xl'][name].shape[0]):
                p = plt.plot(tgrid_ip, plot_dict['xl'][name][jdx])
                if cosmetics['plot_bounds']:
                    tools.plot_bounds(plot_dict, 'xl', name, jdx, tgrid_ip, p)
                if cosmetics['plot_ref']:
                    plt.plot(plot_dict['time_grids']['ref']['ip'], plot_dict['ref']['xl'][name][jdx],
                        linestyle= '--', color = p[-1].get_color() )
                plt.title(name)

    for name in integral_variables_to_plot:
        counter += 1
        ax = plt.axes(axes[counter-1])
        out_values, tgrid_out = tools.merge_integral_output_values(integral_outputs,name, plot_dict, cosmetics)
        plt.plot(tgrid_out, out_values)
        plt.title(name)

    ax.tick_params(axis='both', which='major')
    plt.subplots_adjust(wspace=0.3, hspace=2.0)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 3))
    plt.suptitle(fig_name)




def plot_wake_lifted(plot_dict, cosmetics, fig_name, individual_state=None, fig_num=None):

    # read in inputs
    integral_outputs = plot_dict['integral_outputs_final']
    variables_dict = plot_dict['variables_dict']
    integral_variables = plot_dict['integral_variables']
    tgrid_ip = plot_dict['time_grids']['ip']

    # check if lifted variables exist
    if 'xl' not in variables_dict.keys():
        awelogger.logger.warning('Plot for lifted varibles requested, but no lifted variables found. Ignoring request.')
        return None

    if individual_state == None:
        variables_to_plot = list(variables_dict['xl'].keys())
        integral_variables_to_plot = integral_variables
    else:
        if individual_state in list(variables_dict['xl'].keys()):
            variables_to_plot = [individual_state]
            integral_variables_to_plot = []
        elif individual_state in integral_variables:
            variables_to_plot = []
            integral_variables_to_plot = [individual_state]

    counter = 0
    for name in variables_to_plot:
        if is_wake_variable(name):
            counter += 1
    counter += len(list(integral_outputs.keys()))

    if individual_state == None:
        plot_table_r = 4
        plot_table_c = int(np.ceil(np.float(counter) / np.float(plot_table_r)))
    else:
        plot_table_r = len(variables_to_plot + integral_variables_to_plot)
        plot_table_c = 1

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

    counter = 0
    for name in variables_to_plot:
        if is_wake_variable(name):
            counter += 1
            ax = plt.axes(axes[counter-1])
            for jdx in range(variables_dict['xl'][name].shape[0]):
                p = plt.plot(tgrid_ip, plot_dict['xl'][name][jdx])
                if cosmetics['plot_ref']:
                    plt.plot(plot_dict['time_grids']['ref']['ip'], plot_dict['ref']['xl'][name][jdx],
                        linestyle= '--', color = p[-1].get_color() )
                plt.title(name)

    for name in integral_variables_to_plot:
        counter += 1
        ax = plt.axes(axes[counter-1])
        out_values, tgrid_out = tools.merge_integral_output_values(integral_outputs,name, plot_dict, cosmetics)
        plt.plot(tgrid_out, out_values)
        plt.title(name)

    ax.tick_params(axis='both', which='major')
    plt.subplots_adjust(wspace=0.3, hspace=2.0)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 3))
    plt.suptitle(fig_name)




def plot_controls(plot_dict, cosmetics, fig_name, individual_control=None, fig_num = None):

    # read in inputs
    V_plot = plot_dict['V_plot']
    variables_dict = plot_dict['variables_dict']

    if individual_control == None:
        plot_table_r = 4
        plot_table_c = int(len(list(variables_dict['u'].keys())) / plot_table_r) + 1 * \
                                                    (not np.mod(len(list(variables_dict['u'].keys())), plot_table_r) == 0)
        controls_to_plot = list(variables_dict['u'].keys())
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
        tools.plot_control_block(cosmetics, V_plot, plt, fig, plot_table_r, plot_table_c, pdu, 'u', name, plot_dict, number_dim)
        pdu = pdu + 1

    plt.suptitle(fig_name)
    fig.canvas.draw()

def plot_invariants(plot_dict, cosmetics, fig_name):

    # read in inputs
    number_of_nodes = plot_dict['architecture'].number_of_nodes
    parent_map = plot_dict['architecture'].parent_map

    fig = plt.figure()
    fig.clf()
    legend_names = []
    tgrid_ip = plot_dict['time_grids']['ip']
    invariants = plot_dict['outputs']['tether_length']
    if cosmetics['plot_ref']:
        ref_invariants = plot_dict['ref']['outputs']['tether_length']
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

    fig = plt.figure()
    fig.clf()
    legend_names = []
    tgrid_ip = plot_dict['time_grids']['ip']

    for n in range(1, number_of_nodes):
        parent = parent_map[n]
        lam_name = 'lambda' + str(n) + str(parent)
        lambdavec = plot_dict['xa'][lam_name]
        p = plt.plot(tgrid_ip, lambdavec[0])
        if cosmetics['plot_bounds']:
            tools.plot_bounds(plot_dict, 'xa', lam_name, 0, tgrid_ip, p)
        if cosmetics['plot_ref']:
            plt.plot(plot_dict['time_grids']['ref']['ip'], plot_dict['ref']['xa'][lam_name][0],
                linestyle= '--', color = p[-1].get_color())
        legend_names.append('lambda' + str(n) + str(parent))
    
    if plot_dict['options']['model']['cross_tether'] and number_of_nodes > 2:
        for l in plot_dict['architecture'].layer_nodes:
            kites = plot_dict['architecture'].kites_map[l]
            if len(kites) == 2:
                lam_name = 'lambda{}{}'.format(kites[0], kites[1])
                lambdavec = plot_dict['xa'][lam_name]
                p = plt.plot(tgrid_ip, lambdavec[0])
                if cosmetics['plot_bounds']:
                    tools.plot_bounds(plot_dict, 'xa', lam_name, 0, tgrid_ip, p)
                if cosmetics['plot_ref']:
                    plt.plot(
                        plot_dict['time_grids']['ref']['ip'],
                        plot_dict['ref']['xa'][lam_name][0],
                        linestyle= '--', color = p[-1].get_color()
                        )
                legend_names.append(lam_name)
            else:
                for k in range(len(kites)):
                    lam_name = 'lambda{}{}'.format(kites[k], kites[(k+1)%len(kites)])
                    lambdavec = plot_dict['xa'][lam_name]
                    p = plt.plot(tgrid_ip, lambdavec[0])
                    if cosmetics['plot_bounds']:
                        tools.plot_bounds(plot_dict, 'xa', lam_name, 0, tgrid_ip, p)
                    if cosmetics['plot_ref']:
                        plt.plot(
                            plot_dict['time_grids']['ref']['ip'],
                            plot_dict['ref']['xa'][lam_name][0],
                            linestyle= '--', color = p[-1].get_color()
                            )
                    legend_names.append(lam_name)
    plt.legend(legend_names)
    plt.suptitle(fig_name)

