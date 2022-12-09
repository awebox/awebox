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
####################################################
# Class Quality contains all quality check methods and
# information about quality check results
# Author: Thilo Bronnenmeyer, Kiteswarms, 2018
#####################################################
import pdb

from awebox.logger.logger import Logger as awelogger
import awebox.quality_funcs as quality_funcs
import awebox.tools.struct_operations as struct_op
import awebox.tools.print_operations as print_op

class Quality(object):

    def __init__(self):
        self.__results = {}
        self.__test_param_dict = {}
        self.__name = ''

    def build(self, options, name, test_param_dict = None):

        self.__name = name
        if test_param_dict == None:
            self.__test_param_dict = quality_funcs.generate_test_param_dict(options)
        else:
            self.__test_param_dict = test_param_dict

    def get_test_inputs(self, trial):
        time_grids = trial.optimization.time_grids
        quality_options = trial.options['quality']
        variables_dict = trial.model.variables_dict
        V_opt = trial.optimization.V_opt
        outputs_dict = struct_op.strip_of_contents(trial.model.outputs_dict)
        output_opt = trial.optimization.outputs_opt
        integral_output_names = trial.model.integral_outputs.keys()
        integral_outputs_opt = trial.optimization.integral_outputs_opt
        Collocation = trial.nlp.Collocation

        n_points = trial.options['quality']['interpolation']['n_points']
        quality_time_grid = struct_op.build_time_grid_for_interpolation(time_grids, n_points)
        time_grids['quality'] = quality_time_grid

        quality_input_values = struct_op.interpolate_solution(quality_options, time_grids, variables_dict, V_opt,
                                                              outputs_dict, output_opt,
                                                              integral_output_names, integral_outputs_opt,
                                                              Collocation=Collocation,
                                                              timegrid_label='quality')

        self.__input_values = quality_input_values
        self.__input_time_grid = time_grids

        return None

    def run_tests(self, trial):

        # prepare relevant inputs
        self.get_test_inputs(trial)

        # get relevant self params
        results = self.__results
        test_param_dict = self.__test_param_dict

        # run tests
        results = quality_funcs.test_opti_success(trial, test_param_dict, results)
        results = quality_funcs.test_numerics(trial, test_param_dict, results)
        results = quality_funcs.test_invariants(trial, test_param_dict, results, self.__input_values)
        results = quality_funcs.test_node_altitude(trial, test_param_dict, results)
        results = quality_funcs.test_power_balance(trial, test_param_dict, results, self.__input_values)
        results = quality_funcs.test_slack_equalities(trial, test_param_dict, results)
        results = quality_funcs.test_tracked_vortex_periods(trial, test_param_dict, results, self.__input_values)

        # save test results
        self.__results = results

    def check_quality(self, trial):
    
        self.run_tests(trial)
        self.__interpret_test_results()

    def __interpret_test_results(self):

        results = self.__results
        name = self.__name
        number_of_passed = sum(results.values())
        number_of_tests = len(list(results.keys()))
        awelogger.logger.warning('#################################################')
        awelogger.logger.warning('QUALITY CHECK results for ' + name + ':')
        awelogger.logger.warning(str(number_of_passed) + ' of ' + str(number_of_tests) + ' tests passed.')
        if number_of_tests == number_of_passed:
            awelogger.logger.warning('All tests passed, solution is numerically sound.')
        else:
            awelogger.logger.warning(str(number_of_tests - number_of_passed) + ' tests failed. Solution might be numerically unsound.')
        awelogger.logger.warning('For more information, use trial.quality.print_results()')
        awelogger.logger.warning('#################################################')

        self.__number_of_passed = number_of_passed
        self.__number_of_tests = number_of_tests

    def print_results(self):

        results = self.__results

        pass_label = 'PASSED'
        fail_label = 'FAILED'

        pass_fail_dict = {}
        for name, value in results.key():
            if value:
                pass_fail_dict[name] = pass_label
            else:
                pass_fail_dict[name] = fail_label

        print('########################################')
        print('QUALITY CHECK details:')
        print_op.print_dict_as_table(pass_fail_dict)
        print('#######################################')

    @property
    def results(self):
        return self.__results

    @results.setter
    def results(self, value):
        print_op.log_and_raise_error('Cannot set results object.')

    def all_tests_passed(self):
        return (self.__number_of_passed == self.__number_of_tests)
