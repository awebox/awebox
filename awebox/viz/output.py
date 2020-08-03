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
import numpy as np
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import awebox.viz.tools as tools
from awebox.logger.logger import Logger as awelogger

import casadi.tools as cas

def plot_outputs(plot_dict, cosmetics, fig_name, output_path, fig_num = None):

    time_grid_ip = plot_dict['time_grids']['ip']
    outputs = plot_dict['outputs']

    if cosmetics['plot_ref']:
        ref_time_grid_ip = plot_dict['time_grids']['ref']['ip']
        ref_outputs = plot_dict['ref']['outputs']

    output_key_list = output_path.split(':')
    if len(output_key_list) == 1:
        output = outputs[output_key_list[0]]
        if cosmetics['plot_ref']:
            ref_output = ref_outputs[output_key_list[0]]
    elif len(output_key_list) == 2:
        output = outputs[output_key_list[0]][output_key_list[1]]
        if cosmetics['plot_ref']:
            ref_output = ref_outputs[output_key_list[0]][output_key_list[1]]
    elif len(output_key_list) == 3:
        output = outputs[output_key_list[0]][output_key_list[1]][output_key_list[2]]
        if cosmetics['plot_ref']:
            ref_output = ref_outputs[output_key_list[0]][output_key_list[1]][output_key_list[2]]
    else:
        raise ValueError('Error: Wrong recursion depth (' + str(len(output_key_list)) + ') for output plots!' + str(output_key_list))
    recursive_output_plot(output, fig_name, time_grid_ip, fig_num)
    if cosmetics['plot_ref']:
        recursive_output_plot(ref_output, fig_name, ref_time_grid_ip,  plt.gcf().number , linestyle = '--')

    return None

def recursive_output_plot(outputs, fig_name, time_grid_ip, fig_num = None, linestyle = '-'):

    try:
        for key in list(outputs.keys()):
            recursive_output_plot(outputs[key], key, time_grid_ip, fig_num, linestyle = linestyle)
    except:
        if fig_num is None:
            fig = plt.figure()
            fig.clf()
        else:
            fig = plt.figure(fig_num)

        plt.plot(time_grid_ip, outputs[0], linestyle = linestyle)
        plt.title(fig_name)



def plot_induction_factor(plot_dict, cosmetics, fig_name):

    idx = 0
    comparison_labels = tools.reconstruct_comparison_labels(plot_dict)

    plot_modelled_induction_factor_with_time(plot_dict, cosmetics, fig_name, 1000 + idx, comparison_labels)
    plot_modelled_induction_factor_cycle(plot_dict, cosmetics, fig_name, 1100 + idx, comparison_labels)

    plot_avg_induction_factor_with_time(plot_dict, cosmetics, fig_name, 1500, comparison_labels)
    plot_avg_induction_factor_cycle(plot_dict, cosmetics, fig_name, 1600, comparison_labels)


def plot_relative_radius(plot_dict, cosmetics, fig_name):
    plot_relative_radius_with_time(plot_dict, cosmetics, fig_name)
    plot_relative_radius_cycle(plot_dict, cosmetics, fig_name)




def plot_generic_actuator_output(time_or_cycle, y_var_name, y_var_sym, y_var_latex, y_is_per_kite, plot_dict, cosmetics, fig_num, comparison_labels):

    architecture = plot_dict['architecture']
    kite_nodes = architecture.kite_nodes
    layer_nodes = architecture.layer_nodes
    dashes = plot_dict['dashes']

    if y_is_per_kite:
        node_set = kite_nodes
        set_name = 'kite'
    else:
        node_set = layer_nodes
        set_name = 'layer'


    # collect all of induction values
    y_dict = {}
    for node in node_set:

        if 'actuator' in plot_dict['outputs']:
            actuator_outputs = plot_dict['outputs']['actuator']

            for label in comparison_labels:
                key_name = y_var_sym + '_' + label + str(node)

                if key_name in actuator_outputs.keys():

                    if not (node in y_dict.keys()):
                        y_dict[node] = {}
                    y_dict[node][label] = actuator_outputs[key_name][0]

        if 'vortex' in plot_dict['outputs']:
            vortex_outputs = plot_dict['outputs']['vortex']

            key_name = y_var_sym + str(node)
            if key_name in vortex_outputs.keys():

                if not (node in y_dict.keys()):
                    y_dict[node] = {}
                y_dict[node]['vort'] = vortex_outputs[key_name][0]


    ldx = 0
    if [node_name for node_name in y_dict.keys()]:

        colors = cosmetics['trajectory']['colors']
        layers = architecture.layer_nodes

        if time_or_cycle == 'time':
            x_var_name = 'non-dim. time'
            x_var_latex = r'$t/t_f$ [-]'
            x_vals, tau = tools.get_nondim_time_and_switch(plot_dict)

        elif time_or_cycle == 'cycle':
            x_var_name = 'reel-out factor'
            x_var_latex = r'$f$ [-]'
            if 'actuator' in plot_dict['outputs'].keys():
                f1 = plot_dict['outputs']['actuator']['f1']
            elif 'vortex' in plot_dict['outputs'].keys():
                f1 = plot_dict['outputs']['vortex']['f1']
            else:
                awelogger.logger.error('model not yet implemented.')
            x_vals = np.array(f1[0])

        else:
            awelogger.logger.error('model not yet implemented.')

        fig, axes, nrows = tools.make_layer_plot_in_fig(layers, fig_num)
        title = y_var_name + ' by model and ' + x_var_name
        axes = tools.set_layer_plot_titles(axes, nrows, title)

        x_min = np.min(x_vals)
        x_max = np.max(x_vals)
        y_min = 10.
        y_max = 0.

        kdx = 0
        for node in y_dict.keys():
            kdx += 1

            mdx = 0
            for model in y_dict[node].keys():
                mdx += 1

                y_vals = y_dict[node][model]
                line_label = set_name + ' ' + str(node) +', ' + model
                y_max, y_min = tools.set_max_and_min(y_vals, y_max, y_min)

                color_vals = colors[mdx]
                dash_style = dashes[kdx]
                line_style = ':'

                line, = axes.plot(x_vals, y_vals, color=color_vals, linestyle=line_style, label=line_label)
                line.set_dashes(dash_style)

        xlabel = x_var_name + ' ' + x_var_latex
        ylabel = y_var_name + ' ' + y_var_latex
        axes = tools.set_layer_plot_axes(axes, nrows, xlabel, ylabel, ldx)
        axes = tools.set_layer_plot_legend(axes, nrows, ldx)

        ldx += 1

        axes = tools.set_layer_plot_scale(axes, nrows, x_min, x_max, y_min, y_max)
        if time_or_cycle == 'time':
            axes = tools.add_switching_time_epigraph(axes, nrows, tau, y_min, y_max)

    return None

def plot_avg_induction_factor_with_time(plot_dict, cosmetics, fig_name, fig_num, comparison_labels):
    y_var_name = 'avg. induction factor'
    y_var_sym = 'a0'
    y_var_latex = r'$a_0$ [-]'
    y_is_per_kite = False

    plot_generic_actuator_output('time', y_var_name, y_var_sym, y_var_latex, y_is_per_kite, plot_dict, cosmetics, fig_num, comparison_labels)


def plot_relative_radius_with_time(plot_dict, cosmetics, fig_name):
    y_var_name = 'avg. relative radius'
    y_var_sym = 'bar_varrho'
    y_var_latex = r'$\bar{\varrho}$ [-]'
    y_is_per_kite = False

    fig_num = 2100
    comparison_labels = ['']

    plot_generic_actuator_output('time', y_var_name, y_var_sym, y_var_latex, y_is_per_kite, plot_dict, cosmetics, fig_num, comparison_labels)


def plot_modelled_induction_factor_cycle(plot_dict, cosmetics, fig_name, fig_num, comparison_labels):
    y_var_name = 'local induction factor'
    y_var_sym = 'local_a'
    y_var_latex = r'$a_k$ [-]'
    y_is_per_kite = True

    plot_generic_actuator_output('cycle', y_var_name, y_var_sym, y_var_latex, y_is_per_kite, plot_dict, cosmetics, fig_num, comparison_labels)


def plot_modelled_induction_factor_with_time(plot_dict, cosmetics, fig_name, fig_num, comparison_labels):
    y_var_name = 'local induction factor'
    y_var_sym = 'local_a'
    y_var_latex = r'$a_k$ [-]'
    y_is_per_kite = True

    plot_generic_actuator_output('time', y_var_name, y_var_sym, y_var_latex, y_is_per_kite, plot_dict, cosmetics, fig_num, comparison_labels)


def plot_relative_radius_cycle(plot_dict, cosmetics, fig_name):
    y_var_name = 'avg. relative radius'
    y_var_sym = 'bar_varrho'
    y_var_latex = r'$\bar{\varrho}$ [-]'
    y_is_per_kite = False

    fig_num = 2000
    comparison_labels = ['']

    plot_generic_actuator_output('cycle', y_var_name, y_var_sym, y_var_latex, y_is_per_kite, plot_dict, cosmetics, fig_num, comparison_labels)


def plot_avg_induction_factor_cycle(plot_dict, cosmetics, fig_name, fig_num, comparison_labels):
    y_var_name = 'avg. induction factor'
    y_var_sym = 'a0'
    y_var_latex = r'$a_0$ [-]'
    y_is_per_kite = False

    plot_generic_actuator_output('cycle', y_var_name, y_var_sym, y_var_latex, y_is_per_kite, plot_dict, cosmetics, fig_num, comparison_labels)





def plot_reduced_frequency(solution_dict, cosmetics, fig_num, reload_dict):

    outputs = solution_dict['outputs']
    V_plot = solution_dict['V_final']
    options = solution_dict['options']
    architecture = solution_dict['architecture']

    kite_nodes = options['model']['architecture'].kite_nodes
    parent_map = options['model']['architecture'].parent_map
    kite_dof = options['user_options']['system_model']['kite_dof']
    n_k = options['nlp']['n_k']
    d = options['nlp']['collocation']['d']

    plt.figure(fig_num).clear()
    # fig, axes = plt.subplots(nrows=len(kite_nodes), ncols=1, sharex='all', num=fig_num)
    fig, axes = plt.subplots(nrows=1, ncols=1, sharex='all', num=fig_num)

    width = 8
    text_loc = 1e4
    text_height = 0.

    max_fstar_def = -999.
    f_min_all = 999.
    counter = 0.

    control_surfaces = []
    if int(kite_dof) == 6:
        for kite in kite_nodes:
            parent = parent_map[kite]

            delta_name ='delta' + str(kite) + str(parent)
            control_surfaces += [delta_name]

        for control in control_surfaces:
            for dim in range(3):

                fstar_control = float(outputs['final', 'control_freq', control + '_' + str(dim)])
                fstar_control_min = 0.9 * fstar_control
                fstar_control_max = 1.1 * fstar_control

                axes.semilogx([fstar_control_min, fstar_control_max], [counter, counter], 'g', linewidth = width)
                axes.text(text_loc, counter+ text_height, 'f* ' + control + ' (' + str(dim) + ') [Hz]')
                counter += 1.

                max_fstar_def = np.max([max_fstar_def, fstar_control])
                f_min_all = np.min([f_min_all, fstar_control_min])


    fstar_traj = float(1. / V_plot['theta', 't_f'])

    fstar_traj_min = 0.9 * fstar_traj
    fstar_traj_max = 1.1 * fstar_traj

    max_fstar_def = np.max([max_fstar_def, fstar_traj])
    f_min_all = np.min([f_min_all, fstar_traj_min])

    axes.semilogx([fstar_traj_min, fstar_traj_max], [counter, counter], 'r', linewidth=width)
    axes.text(text_loc, counter + 0.1, 'f* traj [Hz]')
    counter += 1


    for kite in kite_nodes:
        windings = np.round(float(outputs['final', 'winding', 'winding' + str(kite)]))
        fstar_loop = float(windings / V_plot['theta', 't_f'])

        fstar_loop_min = 0.9 * fstar_loop
        fstar_loop_max = 1.1 * fstar_loop

        max_fstar_def = np.max([max_fstar_def, fstar_loop])
        f_min_all = np.min([f_min_all, fstar_loop_min])

        axes.semilogx([fstar_loop_min, fstar_loop_max], [counter, counter], 'r', linewidth = width)
        axes.text(text_loc, counter+ text_height, 'f* loop ' + str(kite) + ' [Hz]')
        counter += 1

    for kite in kite_nodes:
        fstar_aero = []
        for kdx in range(n_k):
            for ddx in range(d):
                local_fstar_aero = outputs['coll_outputs', kdx, ddx, 'aerodynamics', 'fstar_aero' + str(kite)]
                fstar_aero = cas.vertcat(fstar_aero, local_fstar_aero)

        fstar_aero_max = np.max(np.array(fstar_aero))
        fstar_aero_min = np.min(np.array(fstar_aero))

        f_min_all = np.min([f_min_all, fstar_aero_min])

        axes.semilogx([fstar_aero_min, fstar_aero_max], [counter, counter], 'b', linewidth = width)
        axes.text(text_loc, counter+ text_height, 'f* kite ' + str(kite) + ' [Hz]')
        counter += 1.

    layer_parents = architecture.layer_nodes
    for parent in layer_parents:
        center_x = []
        u_app_c = []

        for kdx in range(n_k):
            center_x = cas.vertcat(center_x, cas.vertcat(*outputs['coll_outputs', kdx, :, 'actuator', 'center' + str(parent), 0]))
            u_app_c = cas.vertcat(u_app_c, cas.vertcat(*outputs['coll_outputs', kdx, :, 'actuator', 'u_app' + str(parent), 0]))
        center_x = np.array(center_x)
        u_app_c = np.abs(np.array(u_app_c))
        delta_center_x = np.max(center_x) - np.min(center_x)

        fstar_act_min = np.min(u_app_c) / delta_center_x
        fstar_act_max = np.max(u_app_c) / delta_center_x

        axes.semilogx([fstar_act_min, fstar_act_max], [counter, counter], 'b', linewidth = width)
        axes.text(text_loc, counter + 0.1, 'f* actuator ' + str(parent) + ' [Hz]')
        counter += 1.

        f_min_all = np.min([f_min_all, fstar_act_min])

    plt.axhline(y = counter, color='k', linestyle='-')

    counter += 1

    fkite_max = max_fstar_def / fstar_aero_min
    fkite_min = max_fstar_def / fstar_aero_max

    axes.semilogx([fkite_min, fkite_max], [counter, counter], 'k', linewidth = width)
    axes.text(text_loc, counter+ text_height, 'f kite [-]')
    counter += 1.

    f_min_all = np.min([f_min_all, fkite_min])

    fact_max = max_fstar_def / fstar_act_min
    fact_min = max_fstar_def / fstar_act_max

    axes.semilogx([fact_min, fact_max], [counter, counter], 'k', linewidth = width)
    axes.text(text_loc, counter+ text_height, 'f actuator [-]')
    counter += 1.

    f_min_all = np.min([f_min_all, fact_min])

    axes.set_ylim([-1., counter])
    axes.set_xlim([f_min_all / 10., text_loc * 1e3])

    plt.axvline(x=1., color='k', linestyle='--')

    plt.title('reduced frequency components')
    plt.yticks([], [])

    plt.show()

def plot_energy_over_time(solution_dict, cosmetics, fig_num, reload_dict):
    outputs = solution_dict['outputs']
    options = solution_dict['options']
    architecture = solution_dict['architecture']

    tgrid_coll = np.array(reload_dict['tgrid_coll'])
    potential_energy = {}
    kinetic_energy = {}

    elements = ['groundstation']
    number_of_nodes = architecture.number_of_nodes
    parent_map = architecture.parent_map
    for node in range(1, number_of_nodes):
        elements += ['q' + str(node) + str(parent_map[node])]

    for elem in elements:
        if elem not in list(potential_energy.keys()):
            potential_energy[elem] = []

        if elem not in list(kinetic_energy.keys()):
            kinetic_energy[elem] = []

    n_k = options['nlp']['n_k']
    d_k = options['nlp']['collocation']['d']
    for n in range(n_k):
        for d in range(d_k):
            for elem in elements:
                potential_energy[elem] += [outputs['coll_outputs', n, d, 'e_potential', elem]]
                kinetic_energy[elem] += [outputs['coll_outputs', n, d, 'e_kinetic', elem]]

    fig, axes = plt.subplots(nrows=(len(elements)+1), ncols=1, sharex='all', num=fig_num)

    axes_counter = 0

    e_kin_total = 0.0
    e_pot_total = 0.0

    for elem in elements:
        e_kin = np.array(kinetic_energy[elem])
        e_pot = np.array(potential_energy[elem])

        ax = axes[axes_counter]
        ax.set_title('energy for ' + elem)
        # ax.set_xlabel('t [s]')
        ax.set_ylabel('e [J]')

        ax.stackplot(tgrid_coll.flatten(), e_kin.flatten(), e_pot.flatten(), labels=["e_kin", "e_pot"])
        ax.legend(loc = 'upper right')

        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
        ax.yaxis.set_major_locator(MaxNLocator(3))

        axes_counter += 1
        e_kin_total += e_kin
        e_pot_total += e_pot

    ax = axes[axes_counter]
    ax.set_title('energy for whole system')
    ax.set_xlabel('t [s]')
    ax.set_ylabel('e [J]')
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    ax.yaxis.set_major_locator(MaxNLocator(3))

    ax.stackplot(tgrid_coll.flatten(), e_kin_total.flatten(), e_pot_total.flatten(), labels=['e_kin','e_pot'])
    ax.legend(loc = 'upper right')

    plt.tight_layout(w_pad=1.)

    plt.show()

def plot_loyd_comparison(plot_dict, cosmetics, fig_name, fig_num=None):

    interesting_outputs = [('performance', 'phf_loyd_total'),
                           ('performance', 'loyd_factor'),
                           ('performance', 'p_loyd_total'),
                           ('performance', 'freelout')]
    plot_output(plot_dict, cosmetics, fig_name, interesting_outputs, fig_num)

def plot_aero_forces(solution_dict, cosmetics, fig_num, reload_dict):

    # read in input
    options = solution_dict['options']
    outputs = solution_dict['outputs']

    fig = plt.figure(fig_num)

    selected_outputs = [('aerodynamics','f_aero')]
    dimensions = 3

    plot_table_r = 4
    plot_table_c = int(len(selected_outputs) * dimensions / plot_table_r) + \
        1 * (not np.mod(len(selected_outputs) * dimensions, plot_table_r) == 0)

    pdu = 1
    for output_pair in selected_outputs:
        output_type = output_pair[0]
        output_name = output_pair[1]
        for dim in range(dimensions):

            tools.plot_output_block(plot_table_r, plot_table_c, options, outputs, plt, fig, pdu, output_type, output_name, cosmetics, reload_dict, dim)

            pdu = pdu + 1

    fig.canvas.draw()

# def plot_output(solution_dict, cosmetics, fig_num, reload_dict): #todo: fix output plot!
def plot_output(plot_dict, cosmetics, fig_name, interesting_outputs=[], fig_num = None):

    outputs = plot_dict['outputs']
    architecture = plot_dict['architecture']
    tgrid_ip = plot_dict['time_grids']['ip']

    options_are_not_empty = not (interesting_outputs == [])

    if options_are_not_empty:
        number_of_opts = len(interesting_outputs)

        # create new figure if desired
        if fig_num is not None:
            fig = plt.figure(num=fig_num)
            axes = fig.axes
            if len(axes) == 0:  # if figure does not exist yet
                fig, axes = plt.subplots(num=fig_num, nrows=number_of_opts, ncols=1)

        else:
            fig, axes = plt.subplots(nrows=number_of_opts, ncols=1)

        axes[-1].set_xlabel('t [s]')

        kite_nodes = architecture.kite_nodes

        for odx in range(len(interesting_outputs)):
            opt = interesting_outputs[odx]

            category = opt[0]
            base_name = opt[1]

            output_is_systemwide = base_name in outputs[category].keys()

            if output_is_systemwide:
                data = np.array(outputs[opt[0]][base_name][0])
                local_color = cosmetics['trajectory']['colors'][0]
                axes[odx].plot(tgrid_ip, data, color=local_color)

            else:
                for kite in kite_nodes:

                    data = np.array(outputs[opt[0]][base_name + str(kite)][0])
                    local_color = cosmetics['trajectory']['colors'][kite_nodes.index(kite)]
                    axes[odx].plot(tgrid_ip, data, color=local_color)

            axes[odx].set_ylabel(opt[1])

        for adx in range(len(axes)):
            axes[adx].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
            axes[adx].yaxis.set_major_locator(MaxNLocator(3))

        plt.suptitle(fig_name)
        fig.canvas.draw()


def plot_actuator_center_in_aerotime(solution_dict, cosmetics, fig_num, reload_dict):

    outputs = solution_dict['outputs']
    architecture = solution_dict['architecture']
    options = solution_dict['options']

    n_k = options['nlp']['n_k']

    fig = plt.figure(fig_num)

    ax1 = plt.subplot2grid((4, 2), (0, 0), colspan=2, rowspan=2)
    ax2 = plt.subplot2grid((4, 2), (2, 0))
    ax3 = plt.subplot2grid((4, 2), (3, 0), sharex=ax2)
    ax4 = plt.subplot2grid((4, 2), (2, 1))
    ax5 = plt.subplot2grid((4, 2), (3, 1), sharex=ax4)

    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax4.get_xticklabels(), visible=False)

    ax1.xaxis.set_ticks_position('top')
    ax1.xaxis.set_label_position('top')

    layer_parents = architecture.layer_nodes

    for parent in layer_parents:
        tgrid_coll = reload_dict['tgrid_xa_aerotime' + str(parent)]

        center_x = []
        center_z = []
        v_x = []
        v_z = []

        for kdx in range(n_k):

            center_x = cas.vertcat(center_x, cas.vertcat(*outputs['coll_outputs', kdx, :, 'actuator', 'center' + str(parent), 0]))
            center_z = cas.vertcat(center_z, cas.vertcat(*outputs['coll_outputs', kdx, :, 'actuator', 'center' + str(parent), 2]))

            v_x = cas.vertcat(v_x, cas.vertcat(*outputs['coll_outputs', kdx, :, 'actuator', 'velocity' + str(parent), 0]))
            v_z = cas.vertcat(v_z, cas.vertcat(*outputs['coll_outputs', kdx, :, 'actuator', 'velocity' + str(parent), 2]))

        center_x = np.array(center_x)
        center_z = np.array(center_z)

        v_x = np.array(v_x)
        v_z = np.array(v_z)

        avg_radius = reload_dict['avg_radius' + str(parent)]

        ax1.plot(center_x / avg_radius, center_z / avg_radius)
        ax2.plot(tgrid_coll, center_x / avg_radius)
        ax3.plot(tgrid_coll, center_z / avg_radius)
        ax4.plot(tgrid_coll, v_x / reload_dict['u_hub' + str(parent)])
        ax5.plot(tgrid_coll, v_z / reload_dict['u_hub' + str(parent)])

    ax1.axis('equal')

    ax1.yaxis.set_major_locator(MaxNLocator(4))
    ax1.xaxis.set_major_locator(MaxNLocator(4))

    ax1.set_xlabel('x_c / bar R [-]')
    ax1.set_ylabel('z_c / bar R [-]')

    ax3.set_xlabel('t u_infty / bar R [-]')
    ax2.set_ylabel('x_c / bar R [-]')
    ax3.set_ylabel('z_c / bar R [-]')
    ax2.grid(True)
    ax3.grid(True)

    ax5.set_xlabel('t u_infty / bar R [-]')
    ax4.set_ylabel('u_c / u_infty [-]')
    ax5.set_ylabel('w_c / u_infty [-]')
    ax4.grid(True)
    ax5.grid(True)

    ax2.yaxis.set_major_locator(MaxNLocator(4))
    ax2.xaxis.set_major_locator(MaxNLocator(4))
    ax3.yaxis.set_major_locator(MaxNLocator(4))
    ax3.xaxis.set_major_locator(MaxNLocator(4))
    ax4.yaxis.set_major_locator(MaxNLocator(4))
    ax4.xaxis.set_major_locator(MaxNLocator(4))
    ax5.yaxis.set_major_locator(MaxNLocator(4))
    ax5.xaxis.set_major_locator(MaxNLocator(4))

    plt.tight_layout()

    plt.show()

def plot_actuator_area_in_aerotime(solution_dict, cosmetics, fig_num, reload_dict):

    outputs = solution_dict['outputs']
    architecture = solution_dict['architecture']
    options = solution_dict['options']

    n_k = options['nlp']['n_k']

    fig = plt.figure(fig_num)

    layer_parents = architecture.layer_nodes

    for parent in layer_parents:
        tgrid_coll = reload_dict['tgrid_xa_aerotime' + str(parent)]

        area = []

        for kdx in range(n_k):
            area = cas.vertcat(area, cas.vertcat(*outputs['coll_outputs', kdx, :, 'actuator', 'area' + str(parent)]))

        area = np.array(area)

        avg_radius = reload_dict['avg_radius' + str(parent)]
        avg_area = np.pi * avg_radius**2.

        plt.plot(tgrid_coll, area / avg_area)

    plt.xlabel('t u_infty / bar R [-]')
    plt.ylabel('A / (pi bar R^2) [-]')

    plt.show()

def plot_actuator_thrust_coeff_in_aerotime(solution_dict, cosmetics, fig_num, reload_dict):

    outputs = solution_dict['outputs']
    architecture = solution_dict['architecture']
    options = solution_dict['options']

    n_k = options['nlp']['n_k']

    if 'actuator' in outputs.keys():

        fig, axes = plt.subplots(nrows=4, ncols=1, sharex='all', num=fig_num)

        layer_parents = architecture.layer_nodes

        for parent in layer_parents:
            tgrid_coll = reload_dict['tgrid_xa_aerotime' + str(parent)]

            thrust = []
            thrust1_coeff = []
            thrust2_area_coeff = []
            thrust3_coeff = []

            for kdx in range(n_k):

                thrust = cas.vertcat(thrust, cas.vertcat(*outputs['coll_outputs', kdx, :, 'actuator', 'thrust' + str(parent)]))

                thrust1_coeff = cas.vertcat(thrust1_coeff,
                                        cas.vertcat(*outputs['coll_outputs', kdx, :, 'actuator', 'thrust1_coeff' + str(parent)]))
                thrust2_area_coeff = cas.vertcat(thrust2_area_coeff,
                                        cas.vertcat(*outputs['coll_outputs', kdx, :, 'actuator', 'thrust2_area_coeff' + str(parent)]))
                thrust3_coeff = cas.vertcat(thrust3_coeff,
                                        cas.vertcat(*outputs['coll_outputs', kdx, :, 'actuator', 'thrust3_coeff' + str(parent)]))

            avg_radius = reload_dict['avg_radius' + str(parent)]
            avg_area = np.pi * avg_radius**2.

            thrust = np.array(thrust)

            # T / (1/2 rho u_infty^2 A)
            thrust1_coeff = np.array(thrust1_coeff)

            # T / (1/2 rho u_infty^2 Abar)
            thrust2_coeff = np.array(thrust2_area_coeff) / float(avg_area)

            # 4 a (cos gamma - a)
            thrust3_coeff = np.array(thrust3_coeff)

            axes[0].plot(tgrid_coll, thrust)
            axes[1].plot(tgrid_coll, thrust1_coeff)
            axes[2].plot(tgrid_coll, thrust2_coeff)
            axes[3].plot(tgrid_coll, thrust3_coeff)

        axes[-1].set_xlabel('t u_infty / bar R [-]')

        axes[0].set_ylabel('T [N]')
        axes[1].set_ylabel('CT_1 [-]')
        axes[2].set_ylabel('CT_2 [-]')
        axes[3].set_ylabel('CT_3 [-]')

        axes[0].set_title('actuator thrust and thrust coefficients')

        for adx in range(len(axes)):
            axes[adx].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
            axes[adx].yaxis.set_major_locator(MaxNLocator(3))

        plt.tight_layout(w_pad=1.)

        plt.show()

def plot_dimensionless_aero_indictors(plot_dict, cosmetics, fig_name, fig_num = None):

    interesting_outputs = [('aerodynamics', 'alpha_deg'),
                           ('aerodynamics', 'beta_deg'),
                           ('aerodynamics', 'airspeed'),
                           ('aerodynamics', 'reynolds'),
                           ('aerodynamics', 'mach')]
    plot_output(plot_dict, cosmetics, fig_name, interesting_outputs, fig_num)


def plot_constraints(plot_dict, cosmetics, fig_num, constr_type):

    outputs = plot_dict['outputs']
    constraints = plot_dict['constraints_dict'][constr_type]
    n_constr = len(list(constraints.keys())) # number of constraints
    fig, axes = plt.subplots(nrows=n_constr, ncols=1, sharex='all')

    counter = 0
    for constr_name in list(constraints.keys()):

        # plot all constraints of similar type
        for name in list(plot_dict['outputs'][constr_name].keys()):
            for idx in range(constraints[constr_name, name].shape[0]):

                # exract data
                output_vals = plot_dict['outputs'][constr_name][name][idx]
                tgrid = plot_dict['time_grids']['ip']

                # add labels
                if constraints[constr_name, name].shape[0] == 1:
                    label = name
                else:
                    label = name+'_'+str(idx)

                # plot data with label
                p = axes[counter].plot(tgrid, output_vals, label = label)

                if cosmetics['plot_ref']:
                    ref_output_vals = plot_dict['ref']['outputs'][constr_name][name][idx]
                    ref_tgrid = plot_dict['time_grids']['ref']['ip']
                    axes[counter].plot(ref_tgrid, ref_output_vals, linestyle = '--',color = p[-1].get_color())

        axes[counter].plot(tgrid, np.zeros(tgrid.shape),'k--')
        axes[counter].set_ylabel(constr_name)
        axes[counter].set_xlabel('time [s]')
        axes[counter].set_xlim([tgrid[0], tgrid[-1]])
        axes[counter].legend(loc = 'upper right')

        if counter == 0:
            axes[counter].set_title(constr_type + ' constraints')
        counter += 1


    plt.suptitle(fig_num)
