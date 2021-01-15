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

from awebox.logger.logger import Logger as awelogger
import awebox.quality_funcs as quality_funcs

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

    def run_tests(self, trial):

        # get relevant self params
        results = self.__results
        test_param_dict = self.__test_param_dict

        # run tests
        results = quality_funcs.test_invariants(trial, test_param_dict, results)
        results = quality_funcs.test_outputs(trial, test_param_dict, results)
        results = quality_funcs.test_variables(trial, test_param_dict, results)
        results = quality_funcs.test_numerics(trial, test_param_dict, results)
        results = quality_funcs.test_power_balance(trial, test_param_dict, results)
        results = quality_funcs.test_opti_success(trial, test_param_dict, results)
        results = quality_funcs.test_slack_equalities(trial, test_param_dict, results)
        results = quality_funcs.test_tracked_vortex_periods(trial, test_param_dict, results)

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
        print('########################################')
        print('QUALITY CHECK details:')
        for key in list(results.keys()):
            if results[key]:
                result = 'PASSED'
            else:
                result = 'FAILED'
            print((key + ':  ' + result))
        print('#######################################')

    @property
    def results(self):
        return self.__results

    @results.setter
    def results(self, value):
        print('Cannot set results object.')

    def all_tests_passed(self):
        return (self.__number_of_passed == self.__number_of_tests)
