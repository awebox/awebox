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
            # self.generate_parameterization_settings(options)
            self.generate_integral_constraints(options, model)
            self.generate_outputs(options)

            self.__status = 'I am a formulation.'
            self.__timings['overall'] = time.time() - timer

        else:
            raise ValueError('Cannot build formulation without building model.')

    def determine_operation_conditions(self, options):

        [periodic, initial_conditions, param_initial_conditions, param_terminal_conditions, terminal_inequalities, integral_constraints, terminal_conditions] = operation.get_operation_conditions(options)

        self.__induction_model = options['induction']['induction_model']
        self.__traj_type = options['trajectory']['type']
        self.__tether_drag_model = options['tether_drag_model']
        self.__fix_tether_length = options['trajectory']['tracking']['fix_tether_length']
        self.__phase_fix = options['phase_fix']

        self.__enforce_periodicity = periodic
        self.__enforce_initial_conditions = initial_conditions
        self.__enforce_param_initial_conditions = param_initial_conditions
        self.__enforce_param_terminal_conditions = param_terminal_conditions
        self.__enforce_terminal_conditions = terminal_conditions

        return None

    def generate_parameters(self, options):

        self.__parameters = None

    def generate_variables(self, options):

        self.__variables = None

        return None

    def generate_variable_bounds(self, options):

        self.__variable_bounds = None

        return None

    def generate_parameter_bounds(self,options):

        self.__parameter_bounds = None


        return None

    def generate_outputs(self, options):

        self.__outputs = {}
        if self.__traj_type == 'compromised_landing':
            self.__outputs['compromised_landing'] = {'emergency_scenario':options['compromised_landing']['emergency_scenario']}

        self.__outputs

        return None

    def generate_integral_constraints(self, options, model):

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

    @property
    def phase_fix(self):
        return self.__phase_fix

    @phase_fix.setter
    def phase_fix(self):
        awelogger.logger.warning('Cannot set phase_fix object.')
