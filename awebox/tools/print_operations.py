#
#    This file is part of awebox.
#
#    awebox -- A modeling and optimization framework for multi-kite AWE systems.
#    Copyright (C) 2017-2021 Jochem De Schutter, Rachel Leuthold, Moritz Diehl,
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
'''
file to provide printing operations to the awebox,
_python-3.5 / casadi-3.4.5
- author:  jochem de schutter 2018
- edited: rachel leuthold, alu-fr 2018-2022
'''
import pdb

from awebox.logger.logger import Logger as awelogger
import pandas as pd
import os
import casadi.tools as cas
import numpy as np
import sys
import inspect
from tabulate import tabulate

def print_single_timing(timing):

    [days, hours, minutes, seconds] = get_display_timing(timing)

    timings_string = ''
    if days:
        timings_string += str(days)+'d'
    if hours:
        timings_string += str(hours)+'h'
    if minutes:
        timings_string += str(minutes)+'m'
    if seconds:
        timings_string += str(seconds)+'s'

    if timings_string == '':
        timings_string = '0.0s'

    return timings_string

def get_display_timing(timing):

    days = []
    hours = []
    minutes = []
    seconds = []

    if timing >= 24.0 * 3600.0:
        days = round(timing / (24.0*3600.0))
        timing = timing % (24.0*3600.0)
    if timing >= 3600.0:
        hours = round(timing / 3600.0)
        timing = timing % 3600.0
    if timing >= 60.0:
        minutes = round(timing / 60.0)
        timing = timing % 60.0
    if timing < 60.0:
        seconds = round(timing,1)

    return [days, hours, minutes, seconds]

def hline(charact, length=60):
    return (length * charact)

def get_awebox_license_info():
    license_info = []
    license_info += [80*'+']
    license_info += ['This is awebox, a modeling and optimization framework for multi-kite AWE systems.']
    license_info += ['awebox is free software; you can redistribute it and/or modify it under the terms']
    license_info += ['of the GNU Lesser General Public License as published by the Free Software']
    license_info += ['Foundation license. More information can be found at http://github.com/awebox.']
    license_info += [80*'+']
    return license_info

def log_license_info():
    awelogger.logger.info('')
    license_info = get_awebox_license_info()
    for line in license_info:
        awelogger.logger.info(line)
    awelogger.logger.info('')

def print_license_info():
    print('')
    license_info = get_awebox_license_info()
    for line in license_info:
        print(line)
    print('')

def make_beep_in_linux():
    duration = 1  # second
    freq = 440  # Hz
    os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (duration, freq))

def warn_about_temporary_functionality_alteration(editor='an editor', reason='improve the code'):
    location = inspect.getouterframes(inspect.currentframe(), 2)[1][1]
    message = editor + ' has temporarily altered awebox functionality, in order to ' + reason + ', at location: \n' + location
    awelogger.logger.warning(message)
    return None

def log_and_raise_error(message, suppress_error_logging=False):

    location = inspect.getouterframes(inspect.currentframe(), 2)[1][1]
    message += '\n' + location

    if not suppress_error_logging:
        awelogger.logger.error(message)

    raise Exception(message)

    return None


def print_variable_info(object_name, variable_struct):

    expected_count = variable_struct.shape[0]
    preface = '     ' + object_name + ' has ' + str(expected_count) + ' variables of the following types (and dimensions): '

    counter = 0
    message = ''
    for var_type in variable_struct.keys():
        if hasattr(variable_struct[var_type], 'shape'):
            local_count = variable_struct[var_type].shape[0]
            shape_string = str(local_count)

            counter += local_count
            message += ', ' + var_type + ' (' + shape_string + ')'

        elif isinstance(variable_struct[var_type], list) and hasattr(variable_struct[var_type, 0], 'shape'):

            shape_string = str(len(variable_struct[var_type]))
            shape_string += ' x ' + str(variable_struct[var_type, 0].shape[0])

            local_count = cas.vertcat(*variable_struct[var_type]).shape[0]
            shape_string += ' = ' + str(local_count)

            counter += local_count
            message += ', ' + var_type + ' (' + shape_string + ')'

        elif isinstance(variable_struct[var_type], list) and isinstance(variable_struct[var_type, 0], list) and hasattr(variable_struct[var_type, 0, 0], 'shape'):
            for sub_type in variable_struct[var_type, 0, 0, {}].keys():

                shape_string = str(len(variable_struct[var_type]))
                shape_string += ' x '
                shape_string += str(len(variable_struct[var_type, 0]))
                shape_string += ' x '
                shape_string += str(variable_struct[var_type, 0, 0, sub_type].shape[0])
                shape_string += ' = '

                local_count = cas.vertcat(*[y for x in variable_struct['coll_var', :, :, sub_type] for y in x]).shape[0]
                shape_string += str(local_count)

                counter += local_count
                message += ', ' + sub_type + ' ' + var_type + ' (' + shape_string + ')'

    if counter != expected_count:
        message = 'not all variables in structure of ' + object_name + ' have been found! counted: ' + str(counter) + ', expected: ' + str(expected_count)
        log_and_raise_error(message)

    message = preface + message[1:]

    awelogger.logger.debug(message)

    return None


def recursionably_make_pandas_sanitized_copy(val):

    if isinstance(val, complex) and (np.abs(np.imag(val)) < 1.e-16):
        val = np.real(val)

    if isinstance(val, str) or isinstance(val, float) or isinstance(val, int):
        return val
    elif isinstance(val, cas.DM) and val.shape == (1, 1):
        return float(val)
    elif isinstance(val, cas.DM) or isinstance(val, np.ndarray):
        return repr_g(val)
    elif isinstance(val, dict):
        local_copy = {}
        for subkey, subval in val.items():
            local_copy[subkey] = recursionably_make_pandas_sanitized_copy(subval)
        return local_copy
    elif isinstance(val, list):
        local_copy = []
        for subval in val:
            local_copy += [recursionably_make_pandas_sanitized_copy(subval)]
        return local_copy
    else:
        message = 'the handling of this object type for printing with pandas still needs to be settled. simply returning item itself'
        base_print(message, level='warning')
        return val


class Table:
    def __init__(self, input_dict=None):
        self.__dict = {}

        if (input_dict is not None) and isinstance(input_dict, dict):

            if get_depth_of_dict(input_dict) == 1:
                two_columned_version = {'item': {}, 'value':{}}
                idx = 0
                for key, val in input_dict.items():
                    if isinstance(val, dict) and not(isinstance(val, cas.DM) or isinstance(val, cas.SX) or isinstance(val, cas.MX)):
                        for subkey, subval in val.items():
                            two_columned_version['item'][idx] = key + ' ' + subkey
                            two_columned_version['value'][idx] = subval
                            idx += 1
                    else:
                        two_columned_version['item'][idx] = key
                        two_columned_version['value'][idx] = val
                        idx += 1

                self.__dict = two_columned_version

            else:
                self.__dict = input_dict

        self.__repr_dict = None

    def is_two_column_table(self):
        expected_list_of_headers = ['item', 'value']
        headers = self.get_list_of_headers()
        if headers == expected_list_of_headers:
            return True
        else:
            return False


    def sanitize_for_pandas(self):
        self.__repr_dict = recursionably_make_pandas_sanitized_copy(self.__dict)
        return None


    def to_pandas(self):

        if self.__repr_dict is None:
            self.sanitize_for_pandas()
            return self.to_pandas()

        else:
            df = pd.DataFrame(self.__repr_dict)
            return df

    def get_list_of_headers(self):
        return list(dict.fromkeys(self.__dict))

    def to_string(self, digits=4, repr_type='E', column_width=10):

        self.sanitize_for_pandas()
        if self.is_two_column_table():
            all_values_numeric = all(
                [(isinstance(val, int) or isinstance(val, float) or isinstance(val, cas.DM)) for val in
                 self.__repr_dict['value'].values()])
            if not all_values_numeric:
                for key, val in self.__repr_dict['value'].items():
                    self.__repr_dict['value'][key] = repr_g(val, digits=digits, repr_type=repr_type)

        df = self.to_pandas()

        headers = self.get_list_of_headers()
        max_header_width = np.max(np.array([len(str(header)) for header in headers]))

        float_skeleton = "%." + str(digits) + repr_type
        if self.is_two_column_table():

            key_width = int(np.max(np.array([column_size_for_dot_separated_items(), max_header_width, column_width])))
            key_skeleton = "{0:.<" + str(key_width) + "}"
            for key, val in self.__repr_dict['item'].items():
                self.__repr_dict['item'][key] = key_skeleton.format(val)

            all_values_numeric = all([(isinstance(val, int) or isinstance(val, float) or isinstance(val, cas.DM)) for val in self.__repr_dict.values()])
            if all_values_numeric:
                body_string = df.to_string(float_format=float_skeleton, header=False, index=False)
            else:
                df = self.to_pandas()
                column_width = column_size_for_dot_separated_items()
                string_skeleton = "{0:<" + str(column_width) + "}"
                body_string = df.to_string(header=False, index=False,
                                           formatters={"value": string_skeleton.format})
        else:
            body_string = df.to_string(float_format=float_skeleton, header=True, col_space=column_width, index=True)

        message = body_string + '\n'

        return message

    def to_latex(self, digits=2, repr_type='E'):
        # usethis with
        # \usepackage{booktabs, siunitx}
        # \sisetup{exponent-product=\cdot}

        df = self.to_pandas()
        skeleton = "\\num{%." + str(digits) + repr_type + "}"

        if self.is_two_column_table():
            column_format = 'rl'
            df_tex = df.to_latex(index=False, escape=False, column_format=column_format, float_format=skeleton)
        else:
            headers = self.get_list_of_headers()
            column_format = "l" + ("c" * len(headers))
            df_tex = df.to_latex(index=True, escape=False, column_format=column_format, float_format=skeleton)

        print(df_tex)

        return df_tex

    def print(self, level='info'):
        string = self.to_string()
        string_list = string.split('\n')
        for substring in string_list:
            base_print(substring, level=level)

        return None

    @property
    def repr_dict(self):
        return self.__repr_dict

    @repr_dict.setter
    def repr_dict(self, value):
        log_and_raise_error('Cannot set repr_dict object.')


    @property
    def dict(self):
        return self.__dict

    @dict.setter
    def dict(self, value):
        log_and_raise_error('Cannot set dict object.')

    @property
    def column_headers(self):
        return self.__column_headers

    @column_headers.setter
    def column_headers(self, value):
        awelogger.logger.warning('Cannot set column_headers object.')


def make_sample_two_column_dict():
    input_dict = {'int': 234,
               'float': 23.3873,
               'neg': -2.8,
               'sci': 3.431e-7,
               'cas.dm - scalar': cas.DM(8.13),
               'cas.dm - array': 4.5 * cas.DM.ones((3, 1)),
               'boolean': False,
               'string': 'apples',
               'dict': {'aa1': 3, 'bb1': 'happy', 'cc1': [1,2]}
               }
    return input_dict


def make_sample_two_column_table():
    input_dict = make_sample_two_column_dict()
    tab = Table(input_dict=input_dict)
    return tab


def make_sample_multicolumn_table():

    list_of_lists = [["Sun", 696000., 1.988435e30], ["Earth", 6371, 5.9742e24], ["Moon", 1737, 7.3477e22], ["Mars", 3390, 6.4185e23]]
    list_of_headers = ["Planet", "R (km)", "mass (kg)"]

    input_dict = {}
    for ldx in range(len(list_of_lists)):
        planet = list_of_lists[ldx][0]
        input_dict[planet] = {}
        for hdx in range(1, len(list_of_headers)):
            input_dict[planet][list_of_headers[hdx]] = list_of_lists[ldx][hdx]

    tab = Table(input_dict=input_dict)
    return tab

def test_two_column_table_to_string():
    tab = make_sample_two_column_table()
    found_string = tab.to_string(digits=3, repr_type='E')
    print(found_string)

    all_items = tab.dict['item'].values()
    all_items_included = [str(item) in found_string for item in all_items]

    all_values = tab.dict['value'].values()
    all_values_included = [repr_g(val, digits=3, repr_type='E') in found_string for val in all_values]
    for idx in range(len(all_items)):
        print(idx)
        print(tab.repr_dict['item'][idx])
        print(tab.repr_dict['value'][idx])
        print(tab.dict['value'][idx])
        print(repr_g(tab.dict['value'][idx], digits=3, repr_type='E'))
        print(repr_g(tab.dict['value'][idx], digits=3, repr_type='E') in found_string)
        print()

    example_line = 'cas.dm - array............................... DM([4.5, 4.5, 4.5])'
    example_line_included = example_line in found_string

    criteria = all_items_included and all_values_included and example_line_included
    if not criteria:
        message = 'two-column table to_string does not work as expected.'
        log_and_raise_error(message)
    return None


def test_two_column_table_to_latex():
    tab = make_sample_two_column_table()
    latex = tab.to_latex(digits=3, repr_type='E')

    opening_in_latex = '\\begin{tabular}{rl}' in latex
    header_in_latex = 'item & value \\' in latex
    ending_in_latex = '\end{tabular}' in latex

    body_lines = ['int & 234 \\',
                  'float & \\num{2.339E+01} \\',
                  'neg & \\num{-2.800E+00} \\',
                  'sci & \\num{3.431E-07} \\',
                  'cas.dm - scalar & \\num{8.130E+00} \\',
                  'cas.dm - array & [4.5, 4.5, 4.5] \\',
                  'boolean & False \\',
                  'string & apples \\',
                  'dict aa1 & 3 \\',
                  'dict bb1 & happy \\',
                  'dict cc1 & [1, 2] \\']
    all_body_included = [str(line) in latex for line in body_lines]

    criteria = opening_in_latex and header_in_latex and ending_in_latex and all_body_included
    if not criteria:
        message = 'two-column table to_latex does not work as expected.'
        log_and_raise_error(message)
    return None


def test_multicolumn_table_to_latex():
    tab = make_sample_multicolumn_table()
    test_latex = tab.to_latex(digits=2, repr_type='E')

    includes_header = ' & Sun & Earth & Moon & Mars \\' in test_latex
    includes_midrule = '\midrule' in test_latex
    test_entries = ['Sun', 'Mars', '6.96E+05', '6.42E+23', 'R (km)', 'mass (kg)']
    includes_entries = [entry in test_latex for entry in test_entries]
    includes_information = all(includes_entries)

    criteria = includes_header and includes_midrule and includes_information
    if not criteria:
        message = 'multicolumn table to_latex does not work as expected.'
        log_and_raise_error(message)
    return None


def test_multicolumn_table_to_string():

    tab = make_sample_multicolumn_table()
    test_string = tab.to_string(digits=2)
    print(test_string)

    test_entries = ['Sun', 'Earth', 'Moon', 'Mars', '6.96E+05', '6.42E+23', 'R (km)', 'mass (kg)']
    includes_entries = [entry in test_string for entry in test_entries]
    includes_information = all(includes_entries)

    criteria = includes_information
    if not criteria:
        message = 'multicolumn table to_string does not work as expected.'
        log_and_raise_error(message)

    return None



def get_depth_of_dict(dict):
    local_dict = dict
    depth = 0
    while hasattr(local_dict, 'keys'):
        depth += 1
        local_dict = [value for value in local_dict.values()][0]
    return depth

def test_depth_function():

    dict0 = 0.3
    expected = 0
    condition_0 = (get_depth_of_dict(dict0) == expected)

    dict1 = {'a': 1.}
    expected = 1
    condition_1 = (get_depth_of_dict(dict1) == expected)

    dict2 = {'a': {'b': 1}}
    expected = 2
    condition_2 = (get_depth_of_dict(dict2) == expected)

    criteria = condition_0 and condition_1 and condition_2
    if not criteria:
        message = 'something went wrong in the depth_of_dict function'
        log_and_raise_error(message)

def base_print(string, level='info'):
    if level == 'info':
        awelogger.logger.info(string)
    elif level == 'warning':
        awelogger.logger.warning(string)
    elif level == 'error':
        awelogger.logger.error(string)
    else:
        print(string)


def print_dict_as_table(dict, level='info'):
    depth = get_depth_of_dict(dict)

    if depth == 0:
        base_print(dict, level=level)

    elif depth in [1, 2]:
        tab = Table(dict)
        tab.print(level=level)

    else:
        message = 'function to print_dict_as_table is not available for dicts of depth ' + str(depth)
        log_and_raise_error(message)

    return None


def column_size_for_dot_separated_items():
    return 45

def repr_g(value, digits=4, repr_type='G'):
    if isinstance(value, str):
        return value
    elif isinstance(value, int) and (np.abs(value) < 10**digits):
        return str(value)
    elif (isinstance(value, int) or isinstance(value, float)):
        skeleton = "{:0." + str(digits) + repr_type + "}"
        message = skeleton.format(value)
        return message
    elif isinstance(value, cas.DM) and value.shape == (1, 1):
        return repr_g(float(value))
    else:
        return repr(value)


def close_progress():
    print_progress(2, 2)
    print('')
    return None


def print_progress(index, total_count):
    # warning: this does NOT log the progress, it only displays the progress, on-screen
    progress_width = 20
    progress = float(index) / float(total_count)
    int_progress = int(np.floor(progress * float(progress_width)))
    progress_message = (8 * " ") + ("[%-20s] %d%%" % ('=' * int_progress, progress * 100.))
    sys.stdout.write('\r')
    sys.stdout.write(progress_message)
    sys.stdout.flush()
    return None

def test():
    test_depth_function()
    test_multicolumn_table_to_string()
    test_multicolumn_table_to_latex()
    test_two_column_table_to_string()
    test_two_column_table_to_latex()

if __name__ == "__main__":
    test()
