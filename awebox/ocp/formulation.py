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
# Class Formulation contains the specifics of the problem-formulation, such as whether operation is in standard-operation, emergency landing, transition, etc...
###################################

import casadi.tools as cas

import time
import pickle
from awebox.logger.logger import Logger as awelogger

import awebox.tools.print_operations as print_op

import awebox.tools.struct_operations as struct_op
from . import var_struct

import awebox as awe

from . import operation


class Formulation(object):
    def __init__(self):
        self.__status = 'Formulation not yet built.'
        self.__outputs = None
        self.__timings = {}

    def build(self, options, model):

        awelogger.logger.info('Building formulation...')

        if self.__status == 'I am a formulation.':

            awelogger.logger.info('Formulation already built.')
            return None

        elif model.status == 'I am a model.':
            timer = time.time()

            self.determine_operation_conditions(options)
            self.generate_parameters(options)
            self.generate_variables(options)
            self.generate_variable_bounds(options)
            self.generate_parameter_bounds(options)
            self.generate_parameterization_settings(options)
            self.generate_integral_constraints(options, model)
            self.generate_outputs(options)

            self.__status = 'I am a formulation.'
            self.__timings['overall'] = time.time() - timer
            awelogger.logger.info('Formulation built.')
            awelogger.logger.info('Formulation construction time: %s', print_op.print_single_timing(self.__timings['overall']))
            awelogger.logger.info('')

        else:
            raise ValueError('Cannot build formulation without building model.')

    def determine_operation_conditions(self, options):

        [periodic, initial_conditions, param_initial_conditions, param_terminal_conditions, terminal_inequalities, integral_constraints] = operation.get_operation_conditions(options)

        self.__induction_model = options['induction']['induction_model']
        self.__traj_type = options['trajectory']['type']
        self.__tether_drag_model = options['tether_drag_model']
        self.__fix_tether_length = options['trajectory']['tracking']['fix_tether_length']

        self.__enforce_periodicity = periodic
        self.__enforce_initial_conditions = initial_conditions
        self.__enforce_param_initial_conditions = param_initial_conditions
        self.__enforce_param_terminal_conditions = param_terminal_conditions

        return None

    def generate_parameters(self, options):

        awelogger.logger.info('generate parameters...')

        self.__parameters = None

    def generate_variables(self, options):

        awelogger.logger.info('generate variables...')

        self.__variables = None

        return None

    def generate_variable_bounds(self, options):

        awelogger.logger.info('generate variable bounds...')

        self.__variable_bounds = None

        return None

    def generate_parameter_bounds(self,options):

        awelogger.logger.info('generate parameter bounds...')

        self.__parameter_bounds = None


        return None

    def __get_V_pickle(self, options, initial_or_terminal):

        parameterized_trajectory = options['trajectory']['transition'][initial_or_terminal + '_trajectory']
        if type(parameterized_trajectory) == awe.trial.Trial:
            parameterized_trial = parameterized_trajectory
            V_pickle = parameterized_trial.optimization.V_final
            plot_dict_pickle = parameterized_trial.visualization.plot_dict
        elif type(parameterized_trajectory) == str:
            relative_path = parameterized_trajectory
            if relative_path[-4:] == '.awe':
                parameterized_trial = pickle.load(open(relative_path, 'rb'))
                V_pickle = parameterized_trial.optimization.V_final
                plot_dict_pickle = parameterized_trial.visualization.plot_dict
            elif relative_path[-2:] == '.p':
                awelogger.logger.error('Error: reading in of pickled trajectories as .p files not supported anymore. Please use .awe files.')
            elif relative_path[-5:] == '.dict':
                parameterized_trial_seed = pickle.load(open(relative_path, 'rb'))
                V_pickle = parameterized_trial_seed['solution_dict']['V_final']
                plot_dict_pickle = parameterized_trial_seed['plot_dict']
            else:
                raise ValueError(initial_or_terminal.capitalize() + ' trajectory must be supplied in form of an .awe')

        return V_pickle, plot_dict_pickle

    def generate_parameterization_settings(self, options):

        [periodic, initial_conditions, param_initial_conditions, param_terminal_conditions, terminal_inequalities, integral_constraints] = operation.get_operation_conditions(options)

        xi = var_struct.get_xi_struct()
        xi_bounds = {}

        xi_bounds['xi_0'] = [0.0, 0.0]
        xi_bounds['xi_f'] = [0.0, 0.0]

        V_pickle_initial = None
        plot_dict_pickle_initial = None
        V_pickle_terminal = None
        plot_dict_pickle_terminal = None

        if param_initial_conditions:
            xi_bounds['xi_0'] = [0.0, 1.0]
            if options['trajectory']['type'] == 'compromised_landing':
                xi_0 = options['landing']['xi_0_initial']
                xi_bounds['xi_0'] = [xi_0, xi_0]
            V_pickle_initial, plot_dict_pickle_initial = self.__get_V_pickle(options, 'initial')

        if param_terminal_conditions:
            xi_bounds['xi_f'] = [0.0, 1.0]
            V_pickle_terminal, plot_dict_pickle_terminal = self.__get_V_pickle(options, 'terminal')

        if param_terminal_conditions and param_initial_conditions:
            for theta in struct_op.subkeys(V_pickle_initial, 'theta'):
                diff = V_pickle_terminal['theta',theta] - V_pickle_initial['theta',theta]
                if theta != 't_f':
                    if (float(diff) != 0.0):
                        raise ValueError('Parameters of initial and terminal trajectory are not identical.')

        xi_dict = {}
        xi_dict['V_pickle_initial'] = V_pickle_initial
        xi_dict['plot_dict_pickle_initial'] = plot_dict_pickle_initial
        xi_dict['V_pickle_terminal'] = V_pickle_terminal
        xi_dict['plot_dict_pickle_terminal'] = plot_dict_pickle_terminal
        xi_dict['xi'] = xi
        xi_dict['xi_bounds'] = xi_bounds
        self.__xi_dict = xi_dict

        return None

    def generate_outputs(self, options):

        awelogger.logger.info('generate outputs...')
        self.__outputs = {}
        if self.__traj_type == 'compromised_landing':
            self.__outputs['compromised_landing'] = {'emergency_scenario':options['compromised_landing']['emergency_scenario']}

        self.__outputs

        return None

    def generate_integral_constraints(self, options, model):

        awelogger.logger.info('generate integral constraints..')

        variables = model.variables(cas.MX.sym('variables', model.variables.cat.shape))
        parameters = model.parameters(cas.MX.sym('parameters', model.parameters.cat.shape))

        integral_constraints, integral_constraint_fun, integral_constants = operation.generate_integral_constraints(
            options, variables, parameters, model)

        self.__constraints = {'integral': integral_constraints}
        self.__constraints_fun = {'integral': integral_constraint_fun}
        self.__integral_constants = integral_constants

        return None

    @property
    def status(self):
        return self.__status

    @status.setter
    def status(self, value):
        awelogger.logger.warning('Cannot set status object.')

    @property
    def outputs(self):
        return self.__outputs

    @outputs.setter
    def outputs(self, value):
        awelogger.logger.warning('Cannot set outputs object.')

    @property
    def variables(self):
        return self.__variables

    @property
    def variable_bounds(self):
        return self.__variable_bounds

    @variable_bounds.setter
    def variable_bounds(self, value):
        awelogger.logger.warning('Cannot set variable_bounds object.')

    def parameters(self):
        return self.__parameters

    @property
    def parameter_bounds(self):
        return self.__parameter_bounds

    @property
    def constraints(self):
        return self.__constraints

    @property
    def constraints_fun(self):
        return self.__constraints_fun

    @property
    def induction_model(self):
        return self.__induction_model

    @induction_model.setter
    def induction_model(self):
        awelogger.logger.warning('Cannot set induction_model object.')

    @property
    def traj_type(self):
        return self.__traj_type

    @traj_type.setter
    def traj_type(self):
        awelogger.logger.warning('Cannot set traj_type object.')

    @property
    def tether_drag_model(self):
        return self.__tether_drag_model

    @tether_drag_model.setter
    def tether_drag_model(self):
        awelogger.logger.warning('Cannot set tether_drag_model object.')

    @property
    def xi_dict(self):
        return self.__xi_dict

    @xi_dict.setter
    def xi_dict(self):
        awelogger.logger.warning('Cannot set xi_dict object.')

    @property
    def timings(self):
        return self.__timings

    @timings.setter
    def timings(self):
        awelogger.logger.warning('Cannot set timings object.')

    @property
    def integral_constants(self):
        return self.__integral_constants

    @integral_constants.setter
    def integral_constants(self):
        awelogger.logger.warning('Cannot set integral_constants object.')

    @property
    def fix_tether_length(self):
        return self.__fix_tether_length

    @fix_tether_length.setter
    def fix_tether_length(self):
        awelogger.logger.warning('Cannot set fix_tether_length object.')
