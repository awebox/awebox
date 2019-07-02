#
#    This file is part of awebox.
#
#    awebox -- A modeling and optimization framework for multi-kite AWE systems.
#    Copyright (C) 2017-2019 Jochem De Schutter, Rachel Leuthold, Moritz Diehl,
#                            ALU Freiburg.
#    Copyright (C) 2018-2019 Thilo Bronnenmeyer, Kiteswarms Ltd.
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
##################################
# Class Visualization contains plotting functions to visualize data
# of trials and sweeps
###################################

from . import tools
from . import trajectory
from . import variables
from . import animation
from . import output

import matplotlib.pyplot as plt

from awebox.logger.logger import Logger as awelogger

#todo: compare to initial guess for all plots as option
#todo: options for saving plots

class Visualization(object):

    def __init__(self):

        self.__plot_dict = None

    def build(self, model, nlp, name, options):
        """
        Generate plot dictionary with all relevant plot information.
        :param model: system model
        :param nlp: NLP formulation
        :param visualization_options: visualization related options
        :return: None
        """

        self.__plot_dict = tools.calibrate_visualization(model, nlp, name, options)
        self.create_plot_logic_dict()
        self.__options = options

        return None

    def recalibrate(self, V_plot, plot_dict, output_vals, integral_outputs_final, parametric_options, time_grids, cost, name, V_ref):

        self.__plot_dict = tools.recalibrate_visualization(V_plot, plot_dict, output_vals, integral_outputs_final, parametric_options, time_grids, cost, name, V_ref)

        return None

    def plot(self, V_plot, parametric_options, output_vals, integral_outputs_final, flags, time_grids, cost, name, sweep_toggle, V_ref, fig_name='plot', fig_num = None, recalibrate = True):
        """
        Generate plots with given parametric and visualization options
        :param V_plot: plot data (scaled)
        :param parametric_options: values for parametric options
        :param visualization_options: visualization related options
        :return: None
        """

        # recalibrate plot_dict
        if recalibrate:
            self.recalibrate(V_plot, self.__plot_dict, output_vals, integral_outputs_final, parametric_options, time_grids, cost, name, V_ref)

        if type(flags) is not list:
            flags = [flags]

        # define special flags
        if 'all' in flags:
            flags = list(self.__plot_logic_dict.keys())
            flags.remove('animation')
            flags = [flag for flag in flags if 'outputs:' not in flag]

        if 'level_1' in flags:
            level_1 = ['states', 'controls', 'isometric']
            flags.remove('level_1')
            flags += level_1

        if 'level_2' in flags:
            level_2 = ['states', 'controls', 'isometric', 'projected_xy', 'projected_xz', 'projected_yz']
            flags.remove('level_2')
            flags += level_2

        if 'level_3' in flags:
            level_3 = ['states', 'controls', 'isometric', 'projected_xy', 'projected_xz', 'projected_yz', 'invariants', 'algebraic_variables', 'lifted_variables']
            flags.remove('level_3')
            flags += level_3

        # iterate over flags
        for flag in flags:
            if flag[:5] == 'comp_':
                logging.warning('Comparison plots are only supported for sweeps. Flag "' + flag + '" ignored.')
            else:
                self.__produce_plot(flag, fig_name, parametric_options['visualization']['cosmetics'], fig_num)

        if parametric_options['visualization']['cosmetics']['show_when_ready'] == True and sweep_toggle == False:
            plt.show()

        return None

    def create_plot_logic_dict(self):
        """
        Create a dict for selecting the correct plotting function for a given flag.
        Notation for adding entries:
        (FUNCTION, TUPLE_WITH_ADDITIONAL_ARGS/None)
        :return: dictionary for plot function selection
        """

        outputs = self.plot_dict['outputs_dict']
        variables_dict = self.plot_dict['variables_dict']
        integral_variables = self.plot_dict['integral_variables']

        plot_logic_dict = {}
        plot_logic_dict['isometric'] = (trajectory.plot_trajectory, {'side':'isometric'})
        plot_logic_dict['projected_xy'] = (trajectory.plot_trajectory, {'side':'xy'})
        plot_logic_dict['projected_yz'] = (trajectory.plot_trajectory, {'side':'yz'})
        plot_logic_dict['projected_xz'] = (trajectory.plot_trajectory, {'side':'xz'})
        plot_logic_dict['quad'] = (trajectory.plot_trajectory, {'side':'quad'})
        # plot_logic_dict['instant_xy'] = trajectory.plot_trajectory_instant(plot_dict, -1, -1, cosmetics, 'xy', fig_num)
        # plot_logic_dict['instant_yz'] = trajectory.plot_trajectory_instant(plot_dict, -1, -1, cosmetics, 'yz', fig_num)
        # plot_logic_dict['instant_xz'] = trajectory.plot_trajectory_instant(plot_dict, -1, -1, cosmetics, 'xz', fig_num)
        # plot_logic_dict['timelapse'] = trajectory.plot_trajectory_timelapse(plot_dict, cosmetics, 'isometric', fig_num)
        # plot_logic_dict['timelapse_xy'] = trajectory.plot_trajectory_timelapse(plot_dict, cosmetics, 'xy', fig_num)
        # plot_logic_dict['timelapse_yz'] = trajectory.plot_trajectory_timelapse(plot_dict, cosmetics, 'yz', fig_num)
        # plot_logic_dict['timelapse_xz'] = trajectory.plot_trajectory_timelapse(plot_dict, cosmetics, 'xz', fig_num)
        plot_logic_dict['animation'] = (animation.animate_monitor_plot, None)
        # plot_logic_dict['actuator_center'] = output.plot_actuator_center_in_aerotime(plot_dict, cosmetics, fig_num)
        # plot_logic_dict['actuator_area'] = output.plot_actuator_area_in_aerotime(plot_dict, cosmetics, fig_num)
        # plot_logic_dict['actuator_thrust_coeff'] = output.plot_actuator_thrust_coeff_in_aerotime(plot_dict, cosmetics, fig_num)
        # plot_logic_dict['induction_factor_vs_tether_reel'] = output.plot_induction_factor_vs_tether_reel(plot_dict, cosmetics, fig_num)
        # plot_logic_dict['reduced_frequency'] = output.plot_reduced_frequency(plot_dict, cosmetics, fig_num)
        # plot_logic_dict['elevation'] = trajectory.plot_trajectory_along_elevation(plot_dict, cosmetics, fig_num)
        # plot_logic_dict['loyd_comparison'] = output.plot_loyd_comparison(plot_dict, cosmetics, fig_num)
        # plot_logic_dict['aero_forces'] = output.plot_aero_forces(plot_dict, cosmetics, fig_num)
        # plot_logic_dict['output'] = output.plot_output(plot_dict, cosmetics, fig_num)
        # plot_logic_dict['energy'] = output.plot_energy_over_time(plot_dict, cosmetics, fig_num)
        # plot_logic_dict['aero_dimensionless'] = output.plot_dimensionless_aero_indictors(plot_dict, cosmetics, fig_num)
        plot_logic_dict['states'] = (variables.plot_states, None)
        for variable in list(variables_dict['xd'].keys()) + integral_variables:
            plot_logic_dict['states:' + variable] = (variables.plot_states, {'individual_state':variable})
        plot_logic_dict['controls'] = (variables.plot_controls, None)
        for control in list(variables_dict['u'].keys()):
            plot_logic_dict['controls:' + control] = (variables.plot_controls, {'individual_control':control})
        plot_logic_dict['invariants'] = (variables.plot_invariants, None)
        plot_logic_dict['algebraic_variables'] = (variables.plot_algebraic_variables, None)
        plot_logic_dict['lifted_variables'] = (variables.plot_lifted, None)
        plot_logic_dict['constraints'] = (output.plot_constraints, {'constr_type':'inequality'})
        for output_top in list(outputs.keys()):
            output_path = output_top
            plot_logic_dict['outputs:' + output_top] = (output.plot_outputs, {'output_path': output_top})
            try:
                for output_mid in list(outputs[output_top].keys()):
                    output_path = output_top + ':' + output_mid
                    plot_logic_dict['outputs:' + output_path] = (output.plot_outputs,{'output_path': output_path})
                    try:
                        for output_bottom in list(outputs[output_top][output_mid].keys()):
                            output_path = output_top + ':' + output_mid + ':' + output_bottom
                            plot_logic_dict['outputs:' + output_path] = (output.plot_outputs, {'output_path': output_path})
                    except:
                        pass
            except:
                pass
        self.__plot_logic_dict = plot_logic_dict
        self.__plot_dict['plot_logic_dict'] = plot_logic_dict

    def __produce_plot(self, flag, fig_name, cosmetics, fig_num = None):
        """
        Produce the plot for a given flag, fig_num and cosmetics.
        :param flag: string identifying the kind of plot that should be produced
        :param fig_num: number of the figure that the plot should be displayed in
        :param cosmetics: cosmetic options for the plot
        :return: updated fig_num
        """

        # map flag to function
        fig_name = self.__plot_dict['name'] + '_' + flag + '_' + fig_name

        if fig_num is not None:
            self.__plot_logic_dict[flag][1]['fig_num'] = fig_num

        tools.map_flag_to_function(flag, self.__plot_dict, cosmetics, fig_name, self.__plot_logic_dict)

        if fig_num is not None:
            del self.__plot_logic_dict[flag][1]['fig_num']

        # save figures
        if cosmetics['save_figs']:
            name_rep = self.__plot_dict['name']
            for char in ['(', ')', '_', ' ']:
                name_rep = name_rep.replace(char, '')

            plt.savefig('./figures/' + name_rep + '_' + flag + '.eps', bbox_inches='tight', format='eps', dpi=1000)
            plt.savefig('./figures/' + name_rep + '_' + flag + '.pdf', bbox_inches='tight', format='pdf', dpi=1000)

        return None

    @property
    def plot_dict(self):
        return self.__plot_dict

    @plot_dict.setter
    def plot_dict(self, value):
        self.__plot_dict = value

    @property
    def options(self):
        return self.__options

    @options.setter
    def options(self, value):
       self.__options = value

    @property
    def plot_logic_dict(self):
        return self.__plot_logic_dict

    @plot_logic_dict.setter
    def plot_logic_dict(self, value):
        print('Cannot set plot_logic_dict object.')
