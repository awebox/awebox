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
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


import numpy as np
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as mtick
import awebox.viz.tools as tools
from awebox.logger.logger import Logger as awelogger

import casadi.tools as cas
import awebox.tools.vector_operations as vect_op
import awebox.tools.struct_operations as struct_op
import awebox.tools.print_operations as print_op


def plot_outputs(plot_dict, cosmetics, fig_name, output_top_name, fig_num=None, epigraph=None):

    interesting_outputs = []
    for cstr_name in plot_dict['outputs'][output_top_name].keys():
        interesting_outputs += [(output_top_name, cstr_name)]
    plot_output(plot_dict, cosmetics, fig_name, interesting_outputs, fig_num=fig_num, epigraph=epigraph)

    return None

def get_dictionary_with_output_dimensions(interesting_outputs, outputs, architecture):
    output_dimensions = {}
    for odx in range(len(interesting_outputs)):
        opt = interesting_outputs[odx]
        category = opt[0]
        base_name = opt[1]

        output_is_systemwide = base_name in outputs[category].keys()

        if output_is_systemwide:
            number_of_output_dims = len(outputs[opt[0]][base_name])
        else:
            kite = architecture.kite_nodes[0]
            number_of_output_dims = len(outputs[opt[0]][base_name + str(kite)])

        output_dimensions[opt] = number_of_output_dims
    return output_dimensions

def include_specific_output_solution(relevant_axes, plot_dict, output_type, output_name, output_dim, local_color):

    if ('output_vals' in plot_dict.keys()) and ('opt' in plot_dict['output_vals'].keys()):

        original_times = struct_op.get_original_time_data_for_output_interpolation(plot_dict['time_grids'])

        collocation_entries = original_times.shape[0] * original_times.shape[1]
        collocation_d = int(collocation_entries / plot_dict['n_k'])

        outputs_opt = plot_dict['output_vals']['opt']
        model_outputs = plot_dict['model_outputs']
        odx = struct_op.find_output_idx(model_outputs, output_type, output_name, output_dim)

        struct_op.sanity_check_find_output_idx(model_outputs)

        original_series = outputs_opt[odx, :].T
        if not (original_times.shape == original_series.shape):
            original_series = struct_op.get_output_series_with_duplicates_removed(original_times, original_series, collocation_d)

        label = output_name + " (found)"
        relevant_axes.plot(original_times.full(), original_series.full(), '*', color=local_color, label=label)

    return None

def plot_output(plot_dict, cosmetics, fig_name, interesting_outputs=[], fig_num=None, epigraph=None, all_together=False):

    include_solution = cosmetics['outputs']['include_solution']

    outputs = plot_dict['outputs']
    architecture = plot_dict['architecture']
    tgrid_ip = plot_dict['time_grids']['ip']

    options_are_not_empty = not (interesting_outputs == [])

    if options_are_not_empty:
        number_of_opts = len(interesting_outputs)

        output_dimensions_dict = get_dictionary_with_output_dimensions(interesting_outputs, outputs, architecture)
        number_of_individual_outputs_to_plot = number_of_opts

        number_of_individual_lines_to_draw = 0
        for value in output_dimensions_dict.values():
            number_of_individual_lines_to_draw += value

        if all_together:
            plot_table_r = 1
            plot_table_c = 1
        elif number_of_individual_outputs_to_plot == 1:
            plot_table_r = 1
            plot_table_c = 1
        elif np.mod(number_of_individual_outputs_to_plot, 3) == 0:
            plot_table_r = 3
            plot_table_c = int(number_of_individual_outputs_to_plot / plot_table_r)
        elif np.mod(number_of_individual_outputs_to_plot, 4) == 0:
            plot_table_r = 4
            plot_table_c = int(number_of_individual_outputs_to_plot / plot_table_r)
        elif np.mod(number_of_individual_outputs_to_plot, 5) == 0:
            plot_table_r = 5
            plot_table_c = int(number_of_individual_outputs_to_plot / plot_table_r)
        else:
            plot_table_r = 3
            plot_table_c = int(np.ceil(np.float(number_of_individual_outputs_to_plot) / np.float(plot_table_r)))

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

        number_of_outputs_plotted_so_far = 0
        number_of_lines_drawn_so_far = 0
        for odx in range(len(interesting_outputs)):

            if all_together or (number_of_opts == 1):
                relevant_axes = axes[0]
            else:
                number_of_lines_drawn_so_far = 0
                relevant_axes = axes[odx]

            opt = interesting_outputs[odx]
            output_type = opt[0]
            output_name = opt[1]
            number_of_output_dims = output_dimensions_dict[opt]
            output_is_systemwide = output_name in outputs[output_type].keys()

            if all_together:
                number_of_lines_expected_in_this_axes = number_of_individual_lines_to_draw
            else:
                number_of_lines_expected_in_this_axes = number_of_output_dims

            for output_dim in range(number_of_output_dims):
                cmap = plt.get_cmap('brg')
                decimal_color = float(number_of_lines_drawn_so_far) / float(number_of_lines_expected_in_this_axes)
                local_color = cmap(decimal_color)

                if output_is_systemwide:
                    data = np.array(outputs[output_type][output_name][output_dim])
                    relevant_axes.plot(tgrid_ip, data, color=local_color, label=output_name)
                    number_of_lines_drawn_so_far += 1

                    if include_solution:
                        include_specific_output_solution(relevant_axes, plot_dict, output_type, output_name, output_dim,
                                                         local_color)
                else:
                    for kite in kite_nodes:
                        local_name = output_name + str(kite)
                        if local_name in outputs[output_type].keys():
                            data = np.array(outputs[output_type][local_name][output_dim])
                            relevant_axes.plot(tgrid_ip, data, color=local_color, label=local_name)
                            number_of_lines_drawn_so_far += 1

                            if include_solution:
                                include_specific_output_solution(relevant_axes, plot_dict, output_type, output_name,
                                                                 output_dim,
                                                                 local_color)

                if (epigraph is not None) and (isinstance(epigraph, float)):
                    relevant_axes.axhline(y=epigraph, color='gray', linestyle='--')

                if 't_switch' in plot_dict['time_grids'].keys():
                    t_switch = float(plot_dict['time_grids']['t_switch'])

                    relevant_axes.axvline(x=t_switch, color='gray', linestyle='--')

                if not all_together:
                    relevant_axes.set_ylabel(output_name)

            number_of_outputs_plotted_so_far += 1

        if not (number_of_outputs_plotted_so_far == number_of_individual_outputs_to_plot):
            message = 'something went wrong when drawing the output plots, because '
            message += str(number_of_individual_outputs_to_plot) + ' were expected and '
            message += str(number_of_outputs_plotted_so_far) + ' were draw'
            print_op.log_and_raise_error(message)

        for adx in range(len(axes)):
            axes[adx].set_xlabel('t [s]')
            axes[adx].autoscale(enable=True, axis='x', tight=True)
            axes[adx].grid(True)
            axes[adx].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
            axes[adx].yaxis.set_major_locator(MaxNLocator(3))

        if all_together:
            plt.legend()

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
                           ('performance', 'f0')]
    plot_output(plot_dict, cosmetics, fig_name, interesting_outputs, fig_num)


def plot_circulation(plot_dict, cosmetics, fig_name, fig_num=None):

    interesting_outputs = []
    for kite in plot_dict['architecture'].kite_nodes:
        interesting_outputs += [('aerodynamics', 'circulation' + str(kite))]

    plot_output(plot_dict, cosmetics, fig_name, interesting_outputs, fig_num)



# induction outputs
def add_interesting_outputs_for_available_induction_model(interesting_outputs, plot_dict, search_name):
    search_types = set(['vortex', 'actuator']).intersection(set(plot_dict['outputs'].keys()))
    for output_type in search_types:
        for output_name in plot_dict['outputs'][output_type].keys():
            if search_name in output_name:
                interesting_outputs += [(output_type, output_name)]
    return interesting_outputs


def plot_annulus_average_induction_factor(plot_dict, cosmetics, fig_name, fig_num=None):
    base_name = 'a0'

    interesting_outputs = []
    interesting_outputs = add_interesting_outputs_for_available_induction_model(interesting_outputs, plot_dict, base_name)

    plot_output(plot_dict, cosmetics, fig_name, interesting_outputs, fig_num, all_together=True)


def plot_local_induction_factor(plot_dict, cosmetics, fig_name, fig_num=None):
    base_name = 'local_a'

    interesting_outputs = []
    for kite in plot_dict['architecture'].kite_nodes:
        local_name = base_name + str(kite)
        interesting_outputs = add_interesting_outputs_for_available_induction_model(interesting_outputs, plot_dict, local_name)

    plot_output(plot_dict, cosmetics, fig_name, interesting_outputs, fig_num, all_together=True)


def plot_relative_radius(plot_dict, cosmetics, fig_name, fig_num=None):
    interesting_outputs = []
    for parent in plot_dict['architecture'].layer_nodes:
        interesting_outputs += [('geometry', 'average_relative_radius' + str(parent))]

    plot_output(plot_dict, cosmetics, fig_name, interesting_outputs, fig_num, all_together=True)
