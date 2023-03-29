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

def hline(charact):
    return (60 * charact)

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

def log_and_raise_error(message):
    location = inspect.getouterframes(inspect.currentframe(), 2)[1][1]
    message += '\n' + location
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

class Table:
    def __init__(self):
        self.__column_headers = set([])
        self.__dict = {}

    def add_column_headers(self, list_of_headers):
        for head in list_of_headers:
            self.__column_headers.add(head)
        return None

    def get_row_headers(self):
        return self.__dict.keys()

    def get_column_headers(self):
        for row_head, row_dict in self.__dict.items():
            self.add_column_headers(row_dict.keys())
        return self.__column_headers

    def is_row_empty(self, row_head):
        if row_head in self.__dict.keys():
            row_dict = self.__dict[row_head]
            if hasattr(row_dict, 'keys'):
                return False
            elif row_dict is None:
                return True
            else:
                message = 'unexpected entry in row'
                log_and_raise_error(message)
        else:
            return True

    def insert_empty_cell_without_overwrite(self, row_head, column_head):
        if row_head not in self.__dict.keys():
            self.__dict[row_head] = {}

        if column_head not in self.__dict[row_head].keys():
            self.__dict[row_head][column_head] = None

        return None

    def insert_empty_row_without_overwrite(self, row_head):
        for column_head in self.__column_headers:
            self.insert_empty_cell_without_overwrite(row_head, column_head)

        return None

    def update_column_headers(self):
        existing_column_headers = self.get_column_headers()
        self.add_column_headers(existing_column_headers)
        return None

    def uniformify_column_headers(self):
        self.update_column_headers()
        for row_head, row_dict in self.__dict.items():
            self.insert_empty_row_without_overwrite(row_head)


    def get_number_of_columns(self):
        self.uniformify_column_headers()
        return len(self.__column_headers)

    def get_number_of_rows(self):
        return len(self.__dict.keys())

    def append_row_with_overwrite(self, row_dict={}, row_head=None):

        if row_head is None:
            row_head = str(self.get_number_of_rows())

        if row_head not in self.__dict.keys():
            self.__dict[row_head] = {}

        for column_head, value in row_dict.items():
            self.__dict[row_head][column_head] = value

        self.uniformify_column_headers()

    def to_string(self):
        self.uniformify_column_headers()

        table = []

        for row_head, row_dict in self.__dict.items():
            local_row = row_dict.values()
            table += [local_row]

        headers = row_dict.keys()

        message = '\n' + tabulate(table, headers=headers)
        return message

    def print(self, level='info'):
        base_print(self.to_string())
        return None

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


def test_table_print():

    table = [["Sun", 696000, 1989100000], ["Earth", 6371, 5973.6], ["Moon", 1737, 73.5], ["Mars", 3390, 641.85]]
    headers = ["Planet", "R (km)", "mass (x 10^29 kg)"]
    tabulate_string = '\n' + tabulate(table, headers=headers)

    tab = Table()

    body_label = headers[0]
    radius_label = headers[1]
    mass_label = headers[2]

    for entry in table:
        local_row = {body_label:entry[0],
                     radius_label:entry[1],
                     mass_label:entry[2]
                     }
        tab.append_row_with_overwrite(local_row)
    test_string = tab.to_string()

    criteria = (test_string == tabulate_string)
    if not criteria:
        message = 'table to_string does not work as expected.'
        log_and_raise_error(message)


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

    dict1 = {'a':1.}
    expected = 1
    condition_1 = (get_depth_of_dict(dict1) == expected)

    dict2 = {'a':{'b':1}}
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
    elif depth == 1:
        print_dict_as_dot_separated_two_column_table(dict, level=level)

    elif depth == 2:
        tab = Table()
        for row_head, row_dict in dict.items():
            tab.append_row_with_overwrite(row_dict=row_dict, row_head=row_head)
        tab.print(level=level)
    else:
        message = 'function to print_dict_as_table is not available for dicts of depth ' + str(depth)
        log_and_raise_error(message)

    return None

def print_dict_as_dot_separated_two_column_table(dict, level='info'):
    for name, value in dict.items():
        print_dot_separated_info(name, value, level=level)
    base_print('', level=level)

    return None

def print_dot_separated_info(name, value, level='info'):

    if isinstance(value, complex) and (np.imag(value) == 0.0):
        value = np.real(value)

    if isinstance(value, cas.DM) and value.shape == (1, 1):
        value = float(value)

    value_is_moderate_valued_float = isinstance(value, float) and (np.log10(np.abs(value)) > -1.) and (np.log10(np.abs(value)) < 4.)

    if isinstance(value, int) or isinstance(value, str) or isinstance(value, dict):
        message = "{:.<26}: {}".format(name, value)
    elif value_is_moderate_valued_float:
        message = "{:.<26}: {:.4}".format(name, value)
    elif isinstance(value, float):
        message = "{:.<26}: {:.4E}".format(name, value)
    else:
        error_message = 'unexpected type for object value (' + repr(value) + ')'
        log_and_raise_error(error_message)

    base_print(message, level=level)
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
    test_table_print()
    test_depth_function()