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
from . import trajectory
import numpy as np
from . import tools
from awebox.logger.logger import Logger as awelogger

def comparison_plot(plot_dict, cosmetics, fig_name, interesting_stats):

    number_of_trials = sum(len(v) for v in plot_dict.values())

    rgb_tuple_colors = tools.get_sweep_colors(number_of_trials)

    fig = plt.figure()
    fig.clf()

    plt.suptitle(fig_name)

    counter = 0

    plot_table_r = 2
    plot_table_c = 2

    for stat_name in interesting_stats:
        counter += 1
        ax = plt.subplot(plot_table_r, plot_table_c, counter)

        values, labels = get_stats_values_over_sweep(plot_dict, stat_name)
        print((values, labels))
        plot_bar_x(ax, values, labels, stat_name, rgb_tuple_colors)

def compare_tracking_cost(plot_dict, cosmetics, fig_name):

    interesting_stats = ['tracking_cost']
    comparison_plot(plot_dict, cosmetics, fig_name, interesting_stats)

def compare_landing(plot_dict, cosmetics, fig_name):

    interesting_stats = ['dq_final','tension_max','tension_avg']
    comparison_plot(plot_dict, cosmetics, fig_name, interesting_stats)

def compare_convergence(plot_dict, cosmetics, fig_name):

    interesting_stats = ['iterations', 'return_status_numeric', 'timings_setup', 'timings_optimization']
    comparison_plot(plot_dict, cosmetics, fig_name, interesting_stats)

def compare_stats(sweep_dict, cosmetics, fig_name):

    interesting_stats = ['power_output_kw', 'zeta', 'power_per_surface_area', 'loyd_factor']
    comparison_plot(sweep_dict, cosmetics, fig_name, interesting_stats)

def compare_parameters(plot_dict, cosmetics, fig_name):

    interesting_params = ['l_s', 't_f','l_t_max','cmax']
    comparison_plot(plot_dict, cosmetics, fig_name, interesting_params)

def compare_efficiency(plot_dict, cosmetics, fig_name):

    interesting_params = ['dq10_av', 'l_s','elevation','z_av']
    comparison_plot(plot_dict, cosmetics, fig_name, interesting_params)

def plot_bar_x(ax, values, trial_labels, comparison_label, rgb_tuple_colors):

    bar_width = 0.2
    index = np.arange(len(trial_labels)) * bar_width

    stripped_labels = strip_trial_labels(trial_labels)

    ax.bar(index, values, bar_width, color=rgb_tuple_colors)
    ax.set_ylabel(comparison_label)

    plt.setp(ax, xticks=index+bar_width/2., xticklabels=stripped_labels)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=70)

def strip_trial_labels(trial_labels):

    stripped = []
    for label in trial_labels:
        name = label.split('_')
        new_name = '_'.join(name[1:-2])
        stripped += [new_name]

    return stripped

def get_stats_values_over_sweep(plot_dict, stat_name):

    value_list = []
    labels = []
    for trial_name in list(plot_dict.keys()):
        for param_name in list(plot_dict[trial_name].keys()):

            trial_value = get_stats_values_from_trial(plot_dict[trial_name][param_name], stat_name)
            value_list += [trial_value]
            labels += [trial_name+'_'+param_name]

    value_list = np.array(value_list).reshape(len(value_list),)

    return value_list, labels

def get_stats_values_from_trial(plot_dict, stat_name):

    if stat_name == 'timings_construction':
        return plot_dict['timings']['construction']

    elif stat_name == 'timings_setup':
        return plot_dict['timings']['setup']

    elif stat_name == 'timings_optimization':
        return plot_dict['timings']['optimization']

    elif stat_name == 'iterations':
        return plot_dict['iterations']['optimization']

    elif stat_name == 'return_status_numeric':
        return plot_dict['return_status_numeric']['optimization']

    elif stat_name == 'loyd_factor':
        return np.mean(plot_dict['outputs']['performance']['loyd_factor'][0])
        awelogger.logger.warning('loyd factor calculation should be revisited!')
        #todo: loyd power factor calculation?

    elif stat_name == 'zeta':
        return float(plot_dict['power_and_performance']['zeta'])

    elif stat_name == 'power_output_kw':
        return plot_dict['power_and_performance']['avg_power'].full()*1e-3

    elif stat_name == 'power_per_surface_area':
        return plot_dict['power_and_performance']['power_per_surface_area'].full()*1e-3

    elif stat_name == 't_f':
        return plot_dict['power_and_performance']['time_period'].full()

    elif stat_name == 'l_s':
        no_kites = len(plot_dict['architecture'].kite_nodes)
        if no_kites > 1:
            return float(plot_dict['V_plot']['theta', 'l_s'])
        else:
            return 0.

    elif stat_name == 'diam_t':
        return float(plot_dict['V_final']['theta', 'diam_t'])

    elif stat_name == 'z_av':
        return float(plot_dict['power_and_performance']['z_av'])

    elif stat_name == 'dq10_av':
        return float(plot_dict['power_and_performance']['dq10_av'])

    elif stat_name == 'dq_final':
        return float(plot_dict['power_and_performance']['dq_final'])

    elif stat_name == 'l_t_max':
        return float(plot_dict['power_and_performance']['l_t_max'])

    elif stat_name == 'elevation':
        return float(plot_dict['power_and_performance']['elevation'])

    elif stat_name == 'cmax':
        return float(plot_dict['power_and_performance']['cmax'])

    elif stat_name == 'tension_max':
        return float(plot_dict['power_and_performance']['tension_max'])

    elif stat_name == 'tension_avg':
        return float(plot_dict['power_and_performance']['tension_avg'])

    elif stat_name[-4:] == 'cost':
        return plot_dict['cost'][stat_name]

    else:
        return -999

def plot_family_of_trajectories(sweep_dict, cosmetics, fig_num, side):

    i = 0
    number_of_trials = sum(len(v) for v in sweep_dict.values())
    rgb_tuple_colors = tools.get_sweep_colors(number_of_trials)

    for trial in list(sweep_dict.keys()):
        for param in list(sweep_dict[trial].keys()):
            color = rgb_tuple_colors[i]
            label = sweep_dict[trial][param]['name'] + '_' + param
            local_trial = sweep_dict[trial][param]
            trajectory.plot_trajectory(local_trial['V_plot'], cosmetics, fig_num, side, init_colors=color, label=label)
            i += 1
