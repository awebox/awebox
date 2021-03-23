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
- edited: rachel leuthold, alu-fr 2018-2021
'''

from awebox.logger.logger import Logger as awelogger
import os
import casadi.tools as cas


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
    return (80 * charact)

def get_awebox_license_info():
    license_info = []
    license_info += [hline('+')]
    license_info += ['This is awebox, a modeling and optimization framework for multi-kite AWE systems.']
    license_info += ['awebox is free software; you can redistribute it and/or modify it under the terms']
    license_info += ['of the GNU Lesser General Public License as published by the Free Software']
    license_info += ['Foundation license. More information can be found at http://github.com/awebox.']
    license_info += [hline('+')]
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

def warn_about_temporary_funcationality_removal(location='unspecified', editor='an editor'):
    awelogger.logger.warning(
        editor + ' has temporarily removed awebox functionality, in order to improve the code. location: ' + location)
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
        error = 'not all variables in structure of ' + object_name + ' have been found! counted: ' + str(counter) + ', expected: ' + str(expected_count)
        awelogger.logger.error(error)
        raise Exception(error)

    message = preface + message[1:]

    awelogger.logger.info(message)

    return None

def print_dict_as_table(dict):
    for key in dict.keys():
        if isinstance(dict[key], float):
            message = '{:>30}: {:.2e}'.format(key, dict[key])
            awelogger.logger.info(message)
        else:
            message = '{:>30}: '.format(key) + str(dict[key])
            awelogger.logger.info(message)

    return None