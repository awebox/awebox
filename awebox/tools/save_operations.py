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
Stores helper functions for saving data
-author: Thilo Bronnenmeyer, kiteswarms, 2019
-edit: Rachel Leuthold, ALU-FR, 2020
"""
import copy

import os
import pickle
import awebox.tools.print_operations as print_op
from awebox.logger.logger import Logger as awelogger
import csv
import collections
import awebox.tools.struct_operations as struct_op
import casadi.tools as cas
import awebox.tools.vector_operations as vect_op

def get_dict_of_saveable_objects_and_extensions(trial_or_sweep='Trial'):
    reloadable_seed_str = 'reloadable_seed'
    trajectory_only_str = 'trajectory_only'

    reloadable_seed_extension = 'dict'
    trajectory_only_extension = 'csv'

    saveable_dict = {reloadable_seed_str: reloadable_seed_extension}

    if trial_or_sweep == 'Trial':
        saveable_dict[trajectory_only_str] = trajectory_only_extension

    return saveable_dict

def get_object_and_extension(saving_method='reloadable_seed', trial_or_sweep='Trial'):

    saveable_dict = get_dict_of_saveable_objects_and_extensions(trial_or_sweep=trial_or_sweep)

    default_save_str = 'reloadable_seed'

    type_has_been_recognized = False
    for type, ext in saveable_dict.items():
        if (saving_method == type) or (saving_method == ext):
            object_to_save = type
            type_has_been_recognized = True

    if not type_has_been_recognized:
        object_to_save = default_save_str
        message = 'unrecognized saving_method (' + saving_method + ') for ' + trial_or_sweep + ' object. proceed to save as a ' + default_save_str
        awelogger.logger.warning(message)

    file_extension = saveable_dict[object_to_save]

    return object_to_save, file_extension

def is_possibly_a_filename_containing_reloadable_seed(filename):

    filename_is_a_string = isinstance(filename, str)
    if not filename_is_a_string:
        return False

    filename = correct_filename_if_reloadable_seed_file_extension_is_missing(filename)

    file_is_present_in_loading_folder = os.path.isfile(filename)
    if not file_is_present_in_loading_folder:
        return False

    return True

def save(data, file_name, file_type):

    file_pi = open(file_name + '.' + file_type, 'wb')
    pickle.dump(data, file_pi)
    file_pi.close()

    return None

def write_or_append_two_column_dict_to_csv(table_dict, filename):

    fieldnames = table_dict.keys()

    this_file_exists = True
    try:
        with open(filename, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            fieldnames_are_correct = set(reader.fieldnames) == set(fieldnames)
    except:
        this_file_exists = False

    if this_file_exists and not fieldnames_are_correct:
        message = 'the read fieldnames are not the same as the keys of the dict attempted to be stored. ignoring request to save'
        print_op.base_print(message, level='warning')
        return None

    if this_file_exists:
        open_style = 'a'
        add_header = False
    else:
        open_style = 'w'
        add_header = True

    with open(filename, open_style, newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if add_header:
            writer.writeheader()
        writer.writerow(table_dict)

    return None


def correct_filename_if_reloadable_seed_file_extension_is_missing(filename):
    expected_file_extension = get_dict_of_saveable_objects_and_extensions()['reloadable_seed']
    possible_given_extension = filename[-1 * len(expected_file_extension):]
    if not (possible_given_extension == expected_file_extension):
        filename += '.' + expected_file_extension
    return filename


def load_saved_data_from_dict(filename):
    filename = correct_filename_if_reloadable_seed_file_extension_is_missing(filename)
    filehandler = open(filename, 'rb')
    data = pickle.load(filehandler)
    filehandler.close()

    return data


def write_csv_data(data_dict, filename, rotation_representation='euler'):
    write_csv_dict = init_write_csv_dict(data_dict)

    # write into .csv
    with open(filename + '.csv', 'w') as point_cloud:
        pcdw = csv.DictWriter(point_cloud, delimiter=',', fieldnames=write_csv_dict)
        pcdw.writeheader()
        for k in range(data_dict['time_grids']['ip'].shape[0]):
            write_data_row(pcdw, data_dict, write_csv_dict, data_dict['time_grids']['ip'], k, rotation_representation)


def init_write_csv_dict(data_dict):
    """
    Initialize dictionary used to write into .csv
    :param data_dict: data dictionary containing all data for the .csv with the necessary structure
    :return: Empty dictionary used to write into .csv
    """

    # initialize ordered dict
    write_csv_dict = collections.OrderedDict()

    # create empty entries corresponding to the structure of plot_dict
    for variable_type in ['x', 'z', 'u', 'outputs']:
        for variable in list(data_dict[variable_type].keys()):

            # check for sub_variables in case there are some
            if type(data_dict[variable_type][variable]) is dict:
                for sub_variable in list(data_dict[variable_type][variable].keys()):
                    variable_length = len(data_dict[variable_type][variable][sub_variable])
                    for index in range(variable_length):
                        write_csv_dict[variable_type + '_' + variable + '_' + sub_variable + '_' + str(index)] = None

            # continue without sub_variables in case there are none
            else:
                variable_length = len(data_dict[variable_type][variable])
                for index in range(variable_length):
                    write_csv_dict[variable_type + '_' + variable + '_' + str(index)] = None

    # add time stamp
    write_csv_dict['time'] = None

    for variable in data_dict['theta'].keys():
        local_value = data_dict['theta'][variable]
        if hasattr(local_value, 'len'):
            variable_length = len(local_value)
        if hasattr(local_value, 'shape'):
            if local_value.shape == ():
                variable_length = 1
            else:
                variable_length = variable_length.shape[0]

        for index in range(variable_length):
            write_csv_dict['theta_' + variable + '_' + str(index)] = None

    # add architecture information
    write_csv_dict['nodes'] = None
    write_csv_dict['parent'] = None
    write_csv_dict['kites'] = None
    write_csv_dict['cross_tether'] = None

    return write_csv_dict



def write_data_row(pcdw, data_dict, write_csv_dict, tgrid_ip, k, rotation_representation):
    """
    Write one row of data into the .csv file
    :param pcdw: dictWriter object
    :param data_dict: dictionary containing trial data
    :param write_csv_dict: csv helper dict used to write the trial data into the .csv
    :param k: time step in trajectory
    :return: None
    """

    # loop over variables
    for variable_type in ['x', 'z', 'u', 'outputs']:
        for variable in list(data_dict[variable_type].keys()):

            # check whether sub_variables exist
            if type(data_dict[variable_type][variable]) == dict:
                for sub_variable in list(data_dict[variable_type][variable].keys()):
                    var = data_dict[variable_type][variable][sub_variable]
                    variable_length = len(var)
                    for index in range(variable_length):
                        write_csv_dict[variable_type + '_' + variable + '_' + sub_variable + '_' + str(index)] = str(var[index][k])

            # continue if no sub_variables exist
            else:

                # convert rotations from dcm to euler
                if variable[0] == 'r' and rotation_representation == 'euler':
                    dcm = []
                    for i in range(9):
                        dcm = cas.vertcat(dcm, data_dict[variable_type][variable][i][k])

                    var = vect_op.rotation_matrix_to_euler_angles(cas.reshape(dcm, 3, 3))

                    for index in range(3):
                        write_csv_dict[variable_type + '_' + variable + '_' + str(index)] = str(var[index])
                elif rotation_representation not in ['euler', 'dcm']:
                    message = 'Error: Only euler angles and direct cosine matrix supported.'
                    print_op.log_and_raise_error(message)

                else:
                    var = data_dict[variable_type][variable]
                    variable_length = len(var)
                    for index in range(variable_length):
                        write_csv_dict[variable_type + '_' + variable + '_' + str(index)] = str(var[index][k])

    write_csv_dict['time'] = tgrid_ip[k]

    for variable in data_dict['theta'].keys():
        local_value = data_dict['theta'][variable]
        if hasattr(local_value, 'len'):
            variable_length = len(local_value)
        if hasattr(local_value, 'shape'):
            if local_value.shape == ():
                variable_length = 1
            else:
                variable_length = variable_length.shape[0]

        for index in range(variable_length):

            if variable_length == 1:
                local_value = data_dict['theta'][variable]
            else:
                local_value = data_dict['theta'][variable][index]

            if k == 0:
                write_csv_dict['theta_' + variable + '_' + str(index)] = local_value
            else:
                write_csv_dict['theta_' + variable + '_' + str(index)] = None

    parent_map = data_dict['architecture'].parent_map
    if k < data_dict['architecture'].number_of_nodes-1:
        node = list(parent_map.keys())[k]
        write_csv_dict['nodes'] = str(node)
        write_csv_dict['parent'] = str(parent_map[node])
        if k < len(data_dict['architecture'].kite_nodes):
            write_csv_dict['kites'] = data_dict['architecture'].kite_nodes[k]
        else:
            write_csv_dict['kites'] = None
    else:
        write_csv_dict['nodes'] = None
        write_csv_dict['parent'] = None
        write_csv_dict['kites'] = None

    write_csv_dict['cross_tether'] = int(data_dict['options']['user_options']['system_model']['cross_tether'])

    # write out sorted row
    ordered_dict = collections.OrderedDict(sorted(list(write_csv_dict.items()), key=lambda t: t[0]))
    pcdw.writerow(ordered_dict)

    return None


def test():
    test_table_save()
    # todo: test variable saving? and/or join with the table-save routine?
    return None


def test_table_save():

    filename = 'save_op_test.csv'

    table_1 = {'int': 234,
               'float': 2333.33,
               'neg': -2.8,
               'sci': 3.4e-2,
               'cas.dm - scalar': cas.DM(4.),
               'cas.dm - array': 4.5 * cas.DM.ones((3, 1)),
               'boolean': False,
               'string': 'apples'}
    write_or_append_two_column_dict_to_csv(table_1, filename)

    table_2 = copy.deepcopy(table_1)
    table_2['string'] = 'pears'
    write_or_append_two_column_dict_to_csv(table_2, filename)

    table_3 = copy.deepcopy(table_1)
    table_3['other'] = 234.3
    table_3['string'] = 'peaches'
    write_or_append_two_column_dict_to_csv(table_3, filename)

    set_found_strings = set([])
    number_of_rows = 0
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        csv_file_has_correct_number_of_columns = len(reader.fieldnames) == len(table_1.keys())
        for row in reader:
            number_of_rows += 1
            set_found_strings.add(row['string'])

    only_apples_and_pears_mentioned = (table_1['string'] in set_found_strings) and (table_2['string'] in set_found_strings) and (table_3['string'] not in set_found_strings)
    csv_file_has_correct_number_of_entries = (number_of_rows == 2)

    if not csv_file_has_correct_number_of_entries:
        message = 'unexpected number of entries in test_table_save csv file'
        print_op.log_and_raise_error(message)

    if not only_apples_and_pears_mentioned:
        message = 'distinct dicts are not appended correctly to test_table_save csv file'
        print_op.log_and_raise_error(message)

    if not csv_file_has_correct_number_of_columns:
        message = 'variable information is getting lost in test_table_save csv files'
        print_op.log_and_raise_error(message)

    os.remove(filename)

    return None

# test()