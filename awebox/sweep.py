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
"""
Class sweep contains functions to manipulate multiple trials at once

@author: jochem de schutter alu-freiburg 2018
edit: rachel leuthold, alu-fr, 2020
"""

from awebox.logger.logger import Logger as awelogger
import awebox.tools.print_operations as print_op
import awebox.sweep_funcs as sweep_funcs
import copy
from collections import OrderedDict
import awebox.trial as trial
import awebox.tools.save_operations as save_op
import matplotlib.pyplot as plt
import awebox.viz.comparison as comparison
import awebox.viz.tools as tools


class Sweep:
    def __init__(self, seed, options = None, name = 'sweep'):

        print_op.log_license_info()

        if type(seed) == list:

            [trials_opts, params_opts] = sweep_funcs.process_sweep_opts(options, seed)

            self.__trials_opts = trials_opts
            self.__params_opts = params_opts
            self.__name = name
            self.__type = 'Sweep'
            self.__base_options = options
            self.__trial_dict = OrderedDict()
            self.__param_dict = OrderedDict()
            self.__sweep_dict = OrderedDict()
            self.__sweep_labels = OrderedDict()
            self.__plot_dict = OrderedDict()

        elif type(seed) == dict and options == None:

            self.__plot_dict = seed['plot_dict']
            self.__sweep_dict = seed['sweep_dict']
            self.__param_dict = seed['param_dict']
            self.__name = seed['name']
            self.__generate_plot_logic_dict()

        else:
            error_str = 'Sweep initialized with variables of wrong type. Must be either [list, options] or [dict].'
            awelogger.logger.error(error_str)
            raise TypeError(error_str)

    def build(self):

        awelogger.logger.info('Building sweep (' + self.__name + ')' )

        self._build_trial_dict()
        self._build_param_dict()
        self.__generate_plot_logic_dict()

        return None

    def __getitem__(self, key):
        return self.__trial_dict[key]




    def run(self, final_homotopy_step = 'final', warmstart_file = None, apply_sweeping_warmstart = False, debug_flags = [],
            debug_locations = []):

        awelogger.logger.info('Running sweep (' + self.__name + ') containing ' + str(len(list(self.__trial_dict.keys()))) + ' trials...')

        have_already_saved_prev_trial = False

        # for all trials, run a parametric sweep
        for trial_to_run in list(self.__trial_dict.keys()):

            # build trial once
            single_trial = self.__trial_dict[trial_to_run]
            single_trial.build(False)
            self.__sweep_dict[trial_to_run] = OrderedDict()
            self.__sweep_labels[trial_to_run] = OrderedDict()
            self.__plot_dict[trial_to_run] = OrderedDict()

            # run parametric sweep
            for param in list(self.__param_dict.keys()):

                awelogger.logger.info('Optimize trial (%s) with parametric setting (%s)',trial_to_run, param)

                if param == 'base_options':
                    # take the existing trial options for optimizing
                    param_options = single_trial.options

                else:
                    # add parametric sweep options to trial options and re-build
                    param_options = sweep_funcs.set_single_trial_options(single_trial.options, self.__param_dict[param], 'param')[0]
                    param_options.build(single_trial.model.architecture)
                    self.__trial_dict[trial_to_run].formulation.generate_parameterization_settings(param_options['formulation'])

                # optimize trial
                warmstart_file, prev_trial_save_name = sweep_funcs.make_warmstarting_decisions(self.__name,
                                                                                               user_defined_warmstarting_file=warmstart_file,
                                                                                               apply_sweeping_warmstart=apply_sweeping_warmstart,
                                                                                               have_already_saved_prev_trial=have_already_saved_prev_trial)

                single_trial.optimize(options = param_options,
                                      final_homotopy_step =
                                      final_homotopy_step, debug_flags =
                                      debug_flags, debug_locations =
                                      debug_locations, warmstart_file = warmstart_file)

                if apply_sweeping_warmstart and single_trial.return_status_numeric < 3:
                    single_trial.save(fn=prev_trial_save_name)
                    have_already_saved_prev_trial = True

                recalibrated_plot_dict = sweep_funcs.recalibrate_visualization(single_trial)
                self.__plot_dict[trial_to_run][param] = copy.deepcopy(recalibrated_plot_dict)

                # overwrite outputs to work around pickle bug
                for key in recalibrated_plot_dict['outputs']:
                    self.__plot_dict[trial_to_run][param]['outputs'][key] = copy.deepcopy(recalibrated_plot_dict['outputs'][key])

                # save result
                single_trial_solution_dict = single_trial.generate_solution_dict()
                self.__sweep_dict[trial_to_run][param] = copy.deepcopy(single_trial_solution_dict)

                # overwrite outputs to work around pickle bug
                for i in range(len(single_trial_solution_dict['output_vals'])):
                    self.__sweep_dict[trial_to_run][param]['output_vals'][i] = copy.deepcopy(single_trial_solution_dict['output_vals'][i])
                self.__sweep_labels[trial_to_run][param] = trial_to_run + '_' + param

        awelogger.logger.info('Sweep (' + self.__name +  ') completed.')

    def plot(self, flags):

        if type(flags) is not list:
            flags = [flags]
        if 'comp_all' in flags:
            flags.remove('comp_all')
            flags += list(self.__plot_logic_dict.keys())
            flags = [flag for flag in flags if 'outputs:' not in flag]
        if 'all' in flags:
            flags.remove('all')
            first_trial = list(self.__sweep_dict.keys())[0]
            first_param = list(self.__sweep_dict[first_trial].keys())[0]
            flags += list(self.__plot_dict[first_trial][first_param]['plot_logic_dict'])
            flags.remove('animation')
            flags = [flag for flag in flags if 'outputs:' not in flag]
        for flag in flags:
            if flag[:5] == 'comp_':
                if flag in list(self.__plot_logic_dict.keys()):
                    first_trial = list(self.__sweep_dict.keys())[0]
                    first_param = list(self.__sweep_dict[first_trial].keys())[0]
                    cosmetics = self.__plot_dict[first_trial][first_param]['cosmetics']
                    self.__produce_comparison_plot(self.__plot_dict, flag, cosmetics)
                else:
                    for trial_to_plot in list(self.__sweep_dict.keys()):
                        for param in list(self.__param_dict.keys()):
                            V_plot = self.__sweep_dict[trial_to_plot][param]['V_opt']
                            cost = self.__sweep_dict[trial_to_plot][param]['cost']
                            parametric_options = self.__sweep_dict[trial_to_plot][param]['options']
                            output_vals = self.__sweep_dict[trial_to_plot][param]['output_vals']
                            trial_seed = {'solution_dict': self.__sweep_dict[trial_to_plot][param], 'plot_dict': self.__plot_dict[trial_to_plot][param]}
                            seeded_trial = trial.Trial(trial_seed)
                            seeded_trial.plot([flag[5:]], V_plot=V_plot, cost=cost, parametric_options=parametric_options, output_vals = output_vals, sweep_toggle=True, fig_num = flag[5:])

            else:
                for trial_to_plot in list(self.__plot_dict.keys()):
                    for param in list(self.__param_dict.keys()):
                        V_plot = self.__sweep_dict[trial_to_plot][param]['V_opt']
                        cost = self.__sweep_dict[trial_to_plot][param]['cost']
                        parametric_options = self.__sweep_dict[trial_to_plot][param]['options']
                        output_vals = self.__sweep_dict[trial_to_plot][param]['output_vals']
                        trial_seed = {'solution_dict': self.__sweep_dict[trial_to_plot][param], 'plot_dict': self.__plot_dict[trial_to_plot][param]}
                        seeded_trial = trial.Trial(trial_seed)
                        seeded_trial.plot([flag], V_plot=V_plot, cost=cost, parametric_options=parametric_options, output_vals = output_vals, sweep_toggle=True)

    def __generate_plot_logic_dict(self):

        plot_logic_dict = {}
        plot_logic_dict['comp_stats'] = (comparison.compare_stats, None)
        plot_logic_dict['comp_efficiency'] = (comparison.compare_efficiency, None)
        plot_logic_dict['comp_parameters'] = (comparison.compare_parameters, None)
        plot_logic_dict['comp_convergence'] = (comparison.compare_convergence, None)
        plot_logic_dict['comp_landing'] = (comparison.compare_landing, None)
        plot_logic_dict['comp_tracking_cost'] = (comparison.compare_tracking_cost, None)
        # plot_logic_dict['comp_family_xy'] = (comparison.plot_family_of_trajectories, ('xy',))
        # plot_logic_dict['comp_family_xz'] = (comparison.plot_family_of_trajectories, ('xz',))
        # plot_logic_dict['comp_family_yz'] = (comparison.plot_family_of_trajectories, ('yz',))

        self.__plot_logic_dict = plot_logic_dict

    def __produce_comparison_plot(self, plot_dict, flag, cosmetics):

        # create fig_name
        fig_name = self.__name + '_' + flag + '_' + 'plot'

        # map flag to function
        tools.map_flag_to_function(flag, plot_dict, cosmetics, fig_name, self.__plot_logic_dict)

        # todo: sweep name?
        if 'name' in list(plot_dict.keys()):
            name_rep = plot_dict['name']
        else:
            name_rep = 'sweep'

        if cosmetics['save_figs']:
            for char in ['(', ')', '_', ' ']:
                name_rep = name_rep.replace(char, '')

            plt.savefig('./figures/' + name_rep + '_' + flag + '.eps', bbox_inches='tight', format='eps', dpi=1000)
            plt.savefig('./figures/' + name_rep + '_' + flag + '.pdf', bbox_inches='tight', format='pdf', dpi=1000)

        return None

    def _build_trial_dict(self):

        # make a list with all possible trial options combinations
        trial_combs = sweep_funcs.build_options_combinations(self.__trials_opts)

        # add trial for each possible combination
        if not trial_combs[0]:
            self._add_trial('base_trial', self.__base_options)
        else:
            for trial_sweep_opts in trial_combs:
                single_trial_options, name = sweep_funcs.set_single_trial_options(self.__base_options, trial_sweep_opts, 'trial')
                self._add_trial(name, single_trial_options)
        return None

    def _build_param_dict(self):

        # make a list with all possible parameter options combinations
        param_combs = sweep_funcs.build_options_combinations(self.__params_opts)

        if not param_combs[0]:
            self._add_param('base_options', [])
        else:
            for param_sweep_opts in param_combs:
                name = sweep_funcs.set_single_trial_options(self.__base_options, param_sweep_opts, 'param')[1]
                self._add_param(name, param_sweep_opts)
        return None

    def _add_trial(self, name, options):

        trial_to_add = trial.Trial(name = name, seed = options)
        self.__trial_dict[trial_to_add.name] = trial_to_add

        return None

    def _add_param(self, name, options):

        self.__param_dict[name] = options

        return None

    def save(self, saving_method = 'dict'):

        # log saving method
        awelogger.logger.info('Saving sweep ' + self.__name + ' using ' + saving_method)

        # choose correct function for saving method
        if saving_method == 'awe':
            self.save_to_awes()
        elif saving_method == 'dict':
            self.save_to_dict()
        else:
            awelogger.logger.error(saving_method + ' is not a supported saving method. Sweep ' + self.__name + ' could not be saved!')

        awelogger.logger.info('Sweep (%s) saved.', self.__name)
        awelogger.logger.info('')

        return None

    def save_to_awes(self):

        save_op.save(self, self.__name, 'awes')

        return None

    def save_to_dict(self):

        # create dict to be saved
        data_to_save = {}

        # store necessary information
        data_to_save['sweep_dict'] = self.__sweep_dict
        data_to_save['plot_dict'] = self.__plot_dict
        data_to_save['base_options'] = self.__base_options
        data_to_save['name'] = self.__name
        data_to_save['param_dict'] = self.__param_dict

        # pickle data
        save_op.save(data_to_save, self.__name, 'dict')

        return None

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, value):
        awelogger.logger.critical('Cannot set name object.')

    @property
    def trial_dict(self):
        return self.__trial_dict

    @trial_dict.setter
    def trial_dict(self, value):
        awelogger.logger.critical('Cannot set trial_dict object.')

    @property
    def sweep_dict(self):
        return self.__sweep_dict

    @sweep_dict.setter
    def sweep_dict(self, value):
        awelogger.logger.critical('Cannot set sweep_dict object.')

    @property
    def param_dict(self):
        return self.__param_dict

    @param_dict.setter
    def param_dict(self, value):
        print('Cannot set param_dict object.')

    @property
    def sweep_labels(self):
        return self.__sweep_labels

    @sweep_labels.setter
    def sweep_labels(self, value):
        print('Cannot set sweep_labels object.')


    @property
    def plot_dict(self):
        return self.__plot_dict

    @plot_dict.setter
    def plot_dict(self, value):
        print('Cannot set plot_dict object.')

    @property
    def plot_logic_dict(self):
        return self.__plot_logic_dict

    @plot_logic_dict.setter
    def plot_logic_dict(self, value):
        print('Cannot set plot_logic_dict object.')


    @property
    def type(self):
        return self.__type

    @type.setter
    def type(self, value):
        print('Cannot set type object.')
