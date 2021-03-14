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

def plot_outputs(plot_dict, cosmetics, fig_name, output_top_name, fig_num=None, epigraph=None):

    interesting_outputs = []
    for cstr_name in plot_dict['outputs'][output_top_name].keys():
        interesting_outputs += [(output_top_name, cstr_name)]
    plot_output(plot_dict, cosmetics, fig_name, interesting_outputs, fig_num=fig_num, epigraph=epigraph)

    return None

def plot_output(plot_dict, cosmetics, fig_name, interesting_outputs=[], fig_num=None, epigraph=None):

    outputs = plot_dict['outputs']
    architecture = plot_dict['architecture']
    tgrid_ip = plot_dict['time_grids']['ip']

    options_are_not_empty = not (interesting_outputs == [])

    if options_are_not_empty:
        number_of_opts = len(interesting_outputs)

        if number_of_opts == 1:
            plot_table_r = 1
            plot_table_c = 1
        elif np.mod(number_of_opts, 3) == 0:
            plot_table_r = 3
            plot_table_c = int(number_of_opts / plot_table_r)
        elif np.mod(number_of_opts, 4) == 0:
            plot_table_r = 4
            plot_table_c = int(number_of_opts / plot_table_r)
        elif np.mod(number_of_opts, 5) == 0:
            plot_table_r = 5
            plot_table_c = int(number_of_opts / plot_table_r)
        else:
            plot_table_r = 3
            plot_table_c = int(np.ceil(np.float(number_of_opts) / np.float(plot_table_r)))

        # create new figure if desired
        if fig_num is not None:
            fig = plt.figure(num=fig_num)
            axes = fig.axes
            if len(axes) == 0:  # if figure does not exist yet
                fig, axes = plt.subplots(num=fig_num, nrows=plot_table_r, ncols=plot_table_c)

        else:
            fig, axes = plt.subplots(nrows=plot_table_r, ncols=plot_table_c)

        # make vertical column array or list of all axes
        if type(axes) == np.ndarray:
            axes = axes.reshape(plot_table_r * plot_table_c, )
        elif type(axes) is not list:
            axes = [axes]

        kite_nodes = architecture.kite_nodes

        for odx in range(number_of_opts):
            axes[odx].set_xlabel('t [s]')

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

                    if number_of_opts == 1:
                        axes.plot(tgrid_ip, data, color=local_color)
                    else:
                        axes[odx].plot(tgrid_ip, data, color=local_color)

            if (epigraph is not None) and (isinstance(epigraph, float)):

                if number_of_opts == 1:
                    axes.axhline(y=epigraph, color='gray', linestyle='--')
                else:
                    axes[odx].axhline(y=epigraph, color='gray', linestyle='--')

            if 't_switch' in plot_dict['time_grids'].keys():
                t_switch = float(plot_dict['time_grids']['t_switch'])

                axes[odx].axvline(x=t_switch, color='gray', linestyle='--')

            axes[odx].set_ylabel(opt[1])

        for adx in range(number_of_opts):
            axes[adx].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
            axes[adx].yaxis.set_major_locator(MaxNLocator(3))

        plt.suptitle(fig_name)
        fig.canvas.draw()

def plot_aero_validity(plot_dict, cosmetics, fig_name, fig_num = None):
    interesting_outputs = [('aerodynamics', 'alpha_deg'),
                           ('aerodynamics', 'beta_deg'),
                           ('aerodynamics', 'airspeed'),
                           ('aerodynamics', 'reynolds'),
                           ('aerodynamics', 'mach')]
    plot_output(plot_dict, cosmetics, fig_name, interesting_outputs, fig_num)

def plot_aero_coefficients(plot_dict, cosmetics, fig_name, fig_num = None):

    interesting_outputs = [('aerodynamics', 'CL'),
                           ('aerodynamics', 'CD'),
                           ('aerodynamics', 'CS'),
                           ('aerodynamics', 'LoverD')]
    plot_output(plot_dict, cosmetics, fig_name, interesting_outputs, fig_num)

def plot_model_inequalities(plot_dict, cosmetics, fig_name, fig_num=None):
    plot_outputs(plot_dict, cosmetics, fig_name, 'model_inequalities', fig_num, epigraph=0.)

def plot_model_equalities(plot_dict, cosmetics, fig_name, fig_num=None):
    plot_outputs(plot_dict, cosmetics, fig_name, 'model_equalities', fig_num, epigraph=0.)

def plot_constraints(plot_dict, cosmetics, fig_name, fig_num=None):
    if len(plot_dict['outputs']['model_inequalities'].keys()) > 0:
        plot_model_inequalities(plot_dict, cosmetics, fig_name, fig_num)

    plot_model_equalities(plot_dict, cosmetics, fig_name, fig_num)

def plot_loyd_comparison(plot_dict, cosmetics, fig_name, fig_num=None):

    interesting_outputs = [('performance', 'phf_loyd_total'),
                           ('performance', 'loyd_factor'),
                           ('performance', 'p_loyd_total'),
                           ('performance', 'freelout')]
    plot_output(plot_dict, cosmetics, fig_name, interesting_outputs, fig_num)

def plot_circulation(plot_dict, cosmetics, fig_name, fig_num=None):

    interesting_outputs = []
    for kite in plot_dict['architecture'].kite_nodes:
        interesting_outputs += [('aerodynamics', 'circulation' + str(kite))]
    # #
    # if plot_dict['architecture'].number_of_kites == 1:
    #     interesting_outputs += [('aerodynamics', 'circulation_cl' + str(kite))]

    plot_output(plot_dict, cosmetics, fig_name, interesting_outputs, fig_num)



# actuator outputs

def plot_generic_actuator_output(y_var_name, y_var_sym, y_var_latex, y_is_per_kite, plot_dict, cosmetics, fig_num, comparison_labels):

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

        x_var_name = 'non-dim. time'
        x_var_latex = r'$t/t_f$ [-]'
        x_vals, tau = tools.get_nondim_time_and_switch(plot_dict)

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
        axes = tools.add_switching_time_epigraph(axes, nrows, tau, y_min, y_max)

    return None

def plot_avg_induction_factor_with_time(plot_dict, cosmetics, fig_name, fig_num, comparison_labels):
    y_var_name = 'avg. induction factor'
    y_var_sym = 'a0'
    y_var_latex = r'$a_0$ [-]'
    y_is_per_kite = False

    plot_generic_actuator_output(y_var_name, y_var_sym, y_var_latex, y_is_per_kite, plot_dict, cosmetics, fig_num, comparison_labels)


def plot_relative_radius_with_time(plot_dict, cosmetics, fig_name):
    y_var_name = 'avg. relative radius'
    y_var_sym = 'bar_varrho'
    y_var_latex = r'$\bar{\varrho}$ [-]'
    y_is_per_kite = False

    fig_num = 2100
    comparison_labels = ['']

    plot_generic_actuator_output(y_var_name, y_var_sym, y_var_latex, y_is_per_kite, plot_dict, cosmetics, fig_num, comparison_labels)


def plot_modelled_induction_factor_with_time(plot_dict, cosmetics, fig_name, fig_num, comparison_labels):
    y_var_name = 'local induction factor'
    y_var_sym = 'local_a'
    y_var_latex = r'$a_k$ [-]'
    y_is_per_kite = True

    plot_generic_actuator_output(y_var_name, y_var_sym, y_var_latex, y_is_per_kite, plot_dict, cosmetics, fig_num, comparison_labels)

def plot_induction_factor(plot_dict, cosmetics, fig_name):

    idx = 0
    comparison_labels = tools.reconstruct_comparison_labels(plot_dict)

    plot_modelled_induction_factor_with_time(plot_dict, cosmetics, fig_name, 1000 + idx, comparison_labels)
    plot_avg_induction_factor_with_time(plot_dict, cosmetics, fig_name, 1500, comparison_labels)


def plot_relative_radius(plot_dict, cosmetics, fig_name):
    plot_relative_radius_with_time(plot_dict, cosmetics, fig_name)

