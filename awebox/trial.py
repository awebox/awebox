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
###################################
# Class Trial contains information and methods to perform one
# optimization of a tree-structured multiple-kite system
###################################

import awebox.tools.print_operations as print_op
import awebox.trial_funcs as trial_funcs
import awebox.ocp.nlp as nlp
import awebox.opti.optimization as optimization
import awebox.mdl.model as model
import awebox.mdl.architecture as archi
import awebox.ocp.formulation as formulation
import awebox.viz.visualization as visualization
import awebox.quality as quality
import awebox.tools.save_operations as save_op
import awebox.opts.options as opts
import awebox.tools.struct_operations as struct_op
from awebox.logger.logger import Logger as awelogger

import numpy as np

import copy
import pdb

class Trial(object):
    __isfrozen = False
    def __setattr__(self, key, value):
        if self.__isfrozen and not hasattr(self, key):
            raise TypeError( "%r is a frozen class" % self )
        object.__setattr__(self, key, value)

    def _freeze(self):
        self.__isfrozen = True

    def __init__(self, seed, name='trial'):

        treat_as_filename = save_op.is_possibly_a_filename_containing_reloadable_seed(seed)
        treat_as_dict = (not treat_as_filename) and (trial_funcs.is_possibly_an_already_loaded_seed(seed))
        treat_as_options = (not treat_as_filename) and (not treat_as_dict)

        if treat_as_filename:
            self.__initialize_from_filename_seed(seed)
        elif treat_as_dict:
            self.__initialize_from_dict_seed(seed)
        elif treat_as_options:
            self.__initialize_from_options_seed(seed, name=name)
        else:
            message = 'unable to initialize an awebox trial from the given seed'
            print_op.log_and_raise_error(message)

        return None

    def __initialize_from_filename_seed(self, filename):
        seed = save_op.load_saved_data_from_dict(filename)
        self.__initialize_from_dict_seed(seed)
        return None

    def __initialize_from_dict_seed(self, seed):
        self.__solution_dict = seed['solution_dict']
        self.__options = seed['solution_dict']['options']

        self.__visualization = visualization.Visualization()
        self.__visualization.options = seed['solution_dict']['options']['visualization']

        self.__visualization.plot_dict = seed['plot_dict']
        self.__visualization.create_plot_logic_dict()
        return None

    def __initialize_from_options_seed(self, seed, name=None):
        self.__options_seed   = seed
        self.__options        = opts.Options()
        self.__options.fill_in_seed(self.__options_seed)
        self.__model          = model.Model()
        self.__formulation    = formulation.Formulation()
        self.__nlp            = nlp.NLP()
        self.__optimization   = optimization.Optimization()
        self.__visualization  = visualization.Visualization()
        self.__quality        = quality.Quality()
        self.__name           = name    #todo: names used as unique identifiers in sweep. smart?
        self.__type           = 'Trial'
        self.__status         = None
        self.__timings        = {}
        self.__solution_dict  = {}
        self.__save_flag      = False
        self.__return_status_numeric = -1
        self._freeze()
        return None

    def build(self, is_standalone_trial=True):

        if is_standalone_trial:
            print_op.log_license_info()

        if self.__options['user_options']['trajectory']['type'] == 'mpc':
            raise ValueError('Build method not supported for MPC trials. Use PMPC wrapper instead.')

        awelogger.logger.info(60*'=')
        awelogger.logger.info(12*' '+'Building trial "%s" ...', self.__name)
        awelogger.logger.info(60*'=')
        awelogger.logger.info('')

        architecture = archi.Architecture(self.__options['user_options']['system_model']['architecture'])
        self.__options.build(architecture)
        self.__model.build(self.__options['model'], architecture)
        self.__formulation.build(self.__options['formulation'], self.__model)
        self.__nlp.build(self.__options['nlp'], self.__model, self.__formulation)
        self.__optimization.build(self.__options['solver'], self.__nlp, self.__model, self.__formulation, self.__name)
        self.__visualization.build(self.__model, self.__nlp, self.__name, self.__options)
        self.__quality.build(self.__options['quality'], self.__name)
        self.set_timings('construction')
        awelogger.logger.info('Trial "%s" built.', self.__name)
        awelogger.logger.info('Trial construction time: %s',print_op.print_single_timing(self.__timings['construction']))
        awelogger.logger.info('')

    def optimize(self, options_seed = [], final_homotopy_step = 'final',
                 warmstart_file = None, debug_flags = [],
                 debug_locations = [], save_flag = False, intermediate_solve = False):

        if not options_seed:
            options = self.__options
        else:
            # regenerate nlp bounds for parametric sweeps
            options = opts.Options()
            options.fill_in_seed(options_seed)
            architecture = archi.Architecture(self.__options['user_options']['system_model']['architecture'])
            options.build(architecture)
            self.__model.generate_scaled_variable_bounds(options['model'])
            self.__nlp.generate_variable_bounds(options['nlp'], self.__model)

        # get save_flag
        self.__save_flag = save_flag

        if self.__options['user_options']['trajectory']['type'] == 'mpc':
            raise ValueError('Optimize method not supported for MPC trials. Use PMPC wrapper instead.')

        awelogger.logger.info(60*'=')
        awelogger.logger.info(12*' '+'Optimizing trial "%s" ...', self.__name)
        awelogger.logger.info(60*'=')
        awelogger.logger.info('')


        self.__optimization.solve(options['solver'], self.__nlp, self.__model,
                                  self.__formulation, self.__visualization,
                                  final_homotopy_step, warmstart_file, debug_flags = debug_flags, debug_locations =
                                  debug_locations, intermediate_solve = intermediate_solve)

        self.__solution_dict = self.generate_solution_dict()

        self.set_timings('optimization')

        self.__return_status_numeric = self.__optimization.return_status_numeric['optimization']

        if self.__optimization.solve_succeeded:
            awelogger.logger.info('Trial "%s" optimized.', self.__name)
            awelogger.logger.info('Trial optimization time: %s', print_op.print_single_timing(self.__timings['optimization']))

        else:

            awelogger.logger.info('WARNING: Optimization of Trial (%s) failed.', self.__name)

        # perform quality check
        if not intermediate_solve:
            if (self.__options['quality']['when'] == 'final') or (self.__options['quality']['when'] == 'final_success' and self.__optimization.solve_succeeded):
                self.__quality.check_quality(self)
            else:
                message = 'final solution quality was not checked!'
                print_op.base_print(message, level='warning')

        # print solution
        self.print_solution()

        # save trial if option is set
        if self.__save_flag is True or self.__options['solver']['save_trial'] == True:
            saving_method = self.__options['solver']['save_format']
            self.save(saving_method=saving_method)

        awelogger.logger.info('')

    def plot(self, flags, V_plot=None, cost=None, parametric_options=None, output_vals=None, sweep_toggle=False, fig_num = None):

        recalibrate = True

        if V_plot is None:
            V_plot = self.__solution_dict['V_opt']
            recalibrate = False
        if parametric_options is None:
            parametric_options = self.__options
        if output_vals == None:
            output_vals = self.__solution_dict['output_vals']
        if cost == None:
            cost = self.__solution_dict['cost']
        time_grids = self.__solution_dict['time_grids']
        integral_output_vals = self.__solution_dict['integral_output_vals']
        V_ref = self.__solution_dict['V_ref']
        trial_name = self.__solution_dict['name']
        global_outputs_opt = self.__solution_dict['global_outputs_opt']

        self.__visualization.plot(V_plot, parametric_options, output_vals, integral_output_vals, flags, time_grids, cost, trial_name, sweep_toggle, V_ref, global_outputs_opt, 'plot', fig_num, recalibrate=recalibrate)

        return None

    def set_timings(self, timing):
        if timing == 'construction':
            self.__timings['construction'] = self.model.timings['overall'] + self.formulation.timings['overall'] \
                                            + self.nlp.timings['overall'] + self.optimization.timings['setup']
        elif timing == 'optimization':
            self.__timings['optimization'] = self.optimization.timings['optimization']

    def print_solution(self):

        # the actual power indicators
        if 'e' in self.__model.integral_outputs.keys():
            e_final = self.__optimization.integral_outputs_final['int_out',-1,'e']
        else:
            e_final = self.__optimization.V_final['x', -1, 'e'][-1]

        time_period = self.__optimization.global_outputs_opt['time_period'].full()[0][0]
        avg_power = e_final / time_period

        parameter_label = 'Parameter or Output'
        optimal_label = 'Value at Optimal Solution'
        dimension_label = 'Dimension'

        dict_parameters = {
            0: {parameter_label: 'Average power output',
                                         optimal_label: str(avg_power/1.e3),
                                         dimension_label: 'kW'},
            1: {parameter_label: 'Time period',
                                         optimal_label: str(time_period),
                                         dimension_label: 's'}
            }
        theta_info = {
            'diam_t': ('Main tether diameter', 1e3, 'mm'),
            'diam_s': ('Secondary tether diameter', 1e3, 'mm'),
            'l_s': ('Secondary tether length', 1, 'm'),
            'l_t': ('Main tether length', 1, 'm'),
            'l_i': ('Intermediate tether length', 1, 'm'),
            'diam_i': ('Intermediate tether diameter', 1e3, 'mm'),
            'P_max': ('Peak power', 1e-3, 'kW'),
            'ell_radius': ('Ellipse radius', 1, 'm'),
            'ell_elevation': ('Ellipse elevation', 180.0/np.pi, 'deg'),
            'ell_theta': ('Ellipse division angle', 180.0/np.pi, 'deg'), 
            'a': ('Average induction', 1, '-'),
        }

        for theta in self.model.variables_dict['theta'].keys():
            if theta != 't_f':
                info = theta_info[theta]
                dict_parameters[len(dict_parameters.keys())] = {parameter_label: info[0],
                                                 optimal_label: str(round(self.__optimization.V_final_si['theta', theta].full()[0][0]*info[1],3)),
                                                 dimension_label: info[2]}

        print_op.print_dict_as_table(dict_parameters)

        return None

    def save(self, saving_method='reloadable_seed', filename=None, frequency=30., rotation_representation='euler'):

        object_to_save, file_extension = save_op.get_object_and_extension(saving_method=saving_method, trial_or_sweep=self.__type)

        # log saving method
        awelogger.logger.info('Saving the ' + object_to_save + ' of trial ' + self.__name + ' to ' + file_extension)

        # set savefile name to trial name if unspecified
        if filename is None:
            filename = self.__name

        # choose correct function for saving method
        if object_to_save == 'reloadable_seed':
            self.save_reloadable_seed(filename, file_extension)
        elif object_to_save == 'trajectory_only':
            self.write_to_csv(filename=filename, frequency=frequency, rotation_representation=rotation_representation)
        else:
            message = 'unable to save ' + object_to_save + ' object.'
            print_op.log_and_raise_error(message)

        # log that save is complete
        awelogger.logger.info('Trial (%s) saved.', self.__name)
        awelogger.logger.info('')
        awelogger.logger.info(print_op.hline('&'))
        awelogger.logger.info(print_op.hline('&'))
        awelogger.logger.info('')
        awelogger.logger.info('')

    def save_reloadable_seed(self, filename, file_extension):

        # create dict to be saved
        data_to_save = {}

        # store necessary information
        data_to_save['solution_dict'] = self.generate_solution_dict()
        data_to_save['plot_dict'] = self.__visualization.plot_dict

        # pickle data
        save_op.save(data_to_save, filename, file_extension)

    def write_to_csv(self, filename=None, frequency=30., rotation_representation='euler'):
        if filename is None:
            filename = self.name
        trial_funcs.generate_trial_data_and_write_to_csv(self, filename, frequency, rotation_representation)

        return None

    def generate_solution_dict(self):

        solution_dict = {}

        # seeding data
        solution_dict['time_grids'] = self.__optimization.time_grids
        solution_dict['name'] = self.__name

        # parametric sweep data
        solution_dict['V_opt'] = self.__optimization.V_opt
        solution_dict['V_final_si'] = self.__optimization.V_final_si
        solution_dict['V_ref'] = self.__optimization.V_ref
        solution_dict['options'] = self.__options
        solution_dict['output_vals'] = copy.deepcopy(self.__optimization.output_vals)
        solution_dict['integral_output_vals'] = self.__optimization.integral_output_vals
        solution_dict['stats'] = self.__optimization.stats
        solution_dict['iterations'] = self.__optimization.iterations
        solution_dict['timings'] = self.__optimization.timings
        cost_fun = self.__nlp.cost_components[0]
        cost = struct_op.evaluate_cost_dict(cost_fun, self.__optimization.V_opt, self.__optimization.p_fix_num)
        solution_dict['cost'] = cost
        solution_dict['global_outputs_opt'] = self.__optimization.global_outputs_opt

        # warmstart data
        solution_dict['final_homotopy_step'] = self.__optimization.final_homotopy_step
        solution_dict['Xdot_opt'] = self.__nlp.Xdot(self.__nlp.Xdot_fun(self.__optimization.V_opt))
        solution_dict['g_opt'] = self.__nlp.g(self.__nlp.g_fun(self.__optimization.V_opt, self.__optimization.p_fix_num))
        solution_dict['opt_arg'] = self.__optimization.arg

        return solution_dict

    def print_cost_information(self):

        if hasattr(self.optimization, 'solution'):
            sol = self.optimization.solution
            V_solution_scaled = self.nlp.V(sol['x'])
        else:
            V_solution_scaled = self.optimization.V_init

        p_fix_num = self.optimization.p_fix_num

        cost_fun = self.nlp.cost_components[0]
        cost_dict = struct_op.evaluate_cost_dict(cost_fun, V_solution_scaled, p_fix_num)

        message = '... cost components at solution are:'
        awelogger.logger.info(message)

        print_op.print_dict_as_table(cost_dict)

        awelogger.logger.info('')

        total_dict = {'total_cost': self.nlp.f_fun(V_solution_scaled, p_fix_num)}
        print_op.print_dict_as_table(total_dict)

        return None


    def generate_optimal_model(self, param_options = None):
        return trial_funcs.generate_optimal_model(self, param_options= param_options)

    @property
    def options_seed(self):
        return self.__options_seed

    @property
    def options(self):
        return self.__options

    @options.setter
    def options(self, value):
        print_op.log_and_raise_error('Cannot set options object.')

    @property
    def status(self):
        status_dict = {}
        status_dict['model'] = self.__model.status
        status_dict['nlp'] = self.__nlp.status
        status_dict['optimization'] = self.__optimization.status
        return status_dict

    @status.setter
    def status(self, value):
        print_op.log_and_raise_error('Cannot set status object.')

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, value):
        print_op.log_and_raise_error('Cannot set model object.')

    @property
    def nlp(self):
        return self.__nlp

    @nlp.setter
    def nlp(self, value):
        print_op.log_and_raise_error('Cannot set nlp object.')

    @property
    def optimization(self):
        return self.__optimization

    @optimization.setter
    def optimization(self, value):
        print_op.log_and_raise_error('Cannot set optimization object.')

    @property
    def formulation(self):
        return self.__formulation

    @formulation.setter
    def formulation(self, value):
        print_op.log_and_raise_error('Cannot set formulation object.')

    @property
    def type(self):
        return self.__type

    @type.setter
    def type(self, value):
        print_op.log_and_raise_error('Cannot set type object.')

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, value):
        print_op.log_and_raise_error('Cannot set name object.')

    @property
    def timings(self):
        return self.__timings

    @timings.setter
    def timings(self, value):
        print_op.log_and_raise_error('Cannot set timings object.')

    @property
    def visualization(self):
        return self.__visualization

    @visualization.setter
    def visualization(self, value):
        print_op.log_and_raise_error('Cannot set visualization object.')

    @property
    def quality(self):
        return self.__quality

    @quality.setter
    def quality(self, value):
        print_op.log_and_raise_error('Cannot set quality object.')

    @property
    def return_status_numeric(self):
        return self.__return_status_numeric

    @return_status_numeric.setter
    def return_status_numeric(self, value):
        print_op.log_and_raise_error('Cannot set return_status_numeric object.')

    @property
    def solution_dict(self):
        return self.__solution_dict

    @solution_dict.setter
    def solution_dict(self, value):
        print_op.log_and_raise_error('Cannot set solution_dict object.')

def generate_initial_state(model, V_init):
    # todo: is this being used anywhere?
    x0 = model.struct_list['x'](0.)
    for name in list(model.struct_list['x'].keys()):
        x0[name] = V_init['x',0,0,name]
    return x0
