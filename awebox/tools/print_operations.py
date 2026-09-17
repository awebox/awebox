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

from awebox.logger.logger import Logger as awelogger
import pandas as pd
import os
import casadi.tools as cas
import numpy as np
import sys
import inspect

def awebox_option_name():
    return 'awebox option'

def units_name():
    return 'units'

class PrintableObject():
    def __init__(self, options_object=None, name=''):
        self.__name = name
        self.__applied_parameters_dict = {}
        self.__options_object = options_object

    def add_to_applied_params_dict(self, address, value):
        address_tuple = address.split('.')
        param_name = address_tuple[-1]

        units = None
        description = ''
        if self.__options_object is not None:
            help_dict = self.__options_object.help_dict

            for idx in range(len(address_tuple)):
                if (isinstance(help_dict, dict)) and (address_tuple[idx] in help_dict.keys()):
                    help_dict = help_dict[address_tuple[idx]]
            if len(help_dict[0]) > 0:
                description = help_dict[0][0]
            if len(help_dict[0]) > 2:
                units = help_dict[0][2]

        self.__applied_parameters_dict[param_name] = {'description': description, 'value': value, units_name(): units, awebox_option_name(): address}
        return None

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, value):
        awelogger.logger.warning('Cannot set name object.')

    @property
    def applied_parameters_dict(self):
        return self.__applied_parameters_dict

    @applied_parameters_dict.setter
    def applied_parameters_dict(self, value):
        awelogger.logger.warning('Cannot set applied_parameters_dict object.')

    @property
    def options_object(self):
        return self.__options_object

    @options_object.setter
    def options_object(self, value):
        awelogger.logger.warning('Cannot set options_object object.')


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


def recursionably_make_pandas_sanitized_copy(val, repr_type='E', digits=4, to_echo_or_latex='echo'):

    if any([isinstance(val, poss_type) for poss_type in [int, float, complex, str, cas.DM, cas.SX, cas.MX, np.ndarray, list]]):
        return repr_g(val, repr_type=repr_type, digits=digits, to_echo_or_latex=to_echo_or_latex)
    elif val is None:
        return '-'
    elif isinstance(val, dict):
        local_copy = {}
        for subkey, subval in val.items():
            local_copy[subkey] = recursionably_make_pandas_sanitized_copy(subval, digits=digits, repr_type=repr_type, to_echo_or_latex=to_echo_or_latex)
        return local_copy
    # elif isinstance(val, list):
    #     local_copy = []
    #     for subval in val:
    #         local_copy += [recursionably_make_pandas_sanitized_copy(subval, digits=digits, repr_type=repr_type)]
    #     return local_copy
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

    def is_multilayer_table(self):
        return get_depth_of_dict(self.__dict) > 1


    def sanitize_for_pandas(self, digits=4, repr_type='E', to_echo_or_latex='echo'):
        self.__repr_dict = recursionably_make_pandas_sanitized_copy(self.__dict, digits=digits, repr_type=repr_type, to_echo_or_latex=to_echo_or_latex)
        return None


    def to_pandas(self, digits=4, repr_type='E', to_echo_or_latex='echo'):

        if self.__repr_dict is None:
            self.sanitize_for_pandas(digits=digits, repr_type=repr_type, to_echo_or_latex=to_echo_or_latex)
            return self.to_pandas(digits=digits)

        else:
            df = pd.DataFrame(self.__repr_dict)
            return df

    def get_list_of_headers(self):
        return list(dict.fromkeys(self.__dict))

    def from_pandas_to_string_for_two_column_table(self, df, float_skeleton, max_header_width, column_width, digits=4, repr_type='E', to_echo_or_latex='echo'):

        key_width = int(np.max(np.array([column_size_for_dot_separated_items(), max_header_width, column_width])))
        key_skeleton = "{0:.<" + str(key_width) + "}"

        col_name = 'value'
        for row_indexer in range(len(df[col_name])):
            df.loc[row_indexer, col_name] = repr_g(df.loc[row_indexer, col_name], digits=digits, repr_type=repr_type, to_echo_or_latex=to_echo_or_latex)

        col_name = 'item'
        for row_indexer in range(len(df[col_name])):
            df.loc[row_indexer, col_name] = key_skeleton.format(repr_g(df.loc[row_indexer, col_name], digits=digits, repr_type=repr_type, to_echo_or_latex=to_echo_or_latex))

        string_skeleton = "{0:<" + str(column_size_for_dot_separated_items()) + "}"
        body_string = df.to_string(header=False, index=False, float_format=float_skeleton,
                                   formatters={"value": string_skeleton.format})

        return body_string

    def convert_from_two_column_table_to_multicolumn(self):
        rearrange_table = {}
        for idx in range(len(self.__dict['item'])):
            rearrange_table[self.__dict['item'][idx]] = {'value': self.__dict['value'][idx]}
        self.__dict = rearrange_table
        return None

    def to_string(self, digits=2, repr_type='E', column_width=10, caption=None, nan_replacement='NAN', transpose=False, sort_dim=0):

        self.sanitize_for_pandas(digits=digits, repr_type=repr_type, to_echo_or_latex='echo')

        df = self.to_pandas(digits=digits)

        if df.isnull().values.any():
            df = df.replace(np.nan, nan_replacement)

        headers = self.get_list_of_headers()
        max_header_width = np.max(np.array([len(str(header)) for header in headers]))

        if sort_dim is not None:
            df = df.sort_index(axis=sort_dim)

        if transpose:
            df = df.transpose()

        float_skeleton = "%." + str(digits) + repr_type
        if self.is_two_column_table():
            body_string = self.from_pandas_to_string_for_two_column_table(df, float_skeleton, max_header_width, column_width, digits=digits, repr_type=repr_type)
        else:
            body_string = df.to_string(float_format=float_skeleton, header=True, index=True)

        message = body_string + '\n'
        if caption is not None:
            message = caption + '\n' + message

        return message

    def to_latex(self, digits=6, repr_type='f', caption=None, nan_replacement='--', inf_replacement=r'$\infty$', transpose=False, latex_dict={}, sort_dim=0, latex_symbolic_in_first_column=False, justify='lr'):
        # usethis with
        # \usepackage{booktabs, siunitx}
        # \sisetup{exponent-product=\cdot}

        if isinstance(caption, str):
            caption = caption.replace('_', ' ')

        was_originally_two_column = False
        if self.is_two_column_table():
            was_originally_two_column = True
            self.convert_from_two_column_table_to_multicolumn()

        df = self.to_pandas(digits=digits, repr_type=repr_type, to_echo_or_latex='latex').replace(np.nan, nan_replacement)
        negative_replacement = inf_replacement.replace(r'$\in', r'-$\in')
        df = df.replace('inf', inf_replacement)
        df = df.replace("-" + inf_replacement, negative_replacement)
        skeleton = "\\num{%." + str(digits) + repr_type + "}"

        if was_originally_two_column:
            df = df.transpose()

        if transpose:
            df = df.transpose()

        if sort_dim is not None:
            df = df.sort_index(axis=sort_dim)

        import re
        def replace_whole_space_word(x):
            if not isinstance(x, str):
                return x

            for old, new in latex_dict.items():
                pattern = rf'(?<!\S){re.escape(old)}(?!\S)'
                x = re.sub(pattern, lambda _: r'$' + new + r'$', x)

            return x

        df.index = df.index.map(replace_whole_space_word)
        df.columns = df.columns.map(replace_whole_space_word)

        opt_cols = [c for c in df.columns if awebox_option_name() in c]
        for col in opt_cols:
            df[col] = df[col].apply(
                lambda x: r'\aweboxOptions{' + x + '}'
            )
        units_cols = [c for c in df.columns if units_name() in c]
        for col in units_cols:
            df[col] = df[col].apply(
                lambda x: r'\unit{' + str(x) + '}'
            )

        if was_originally_two_column:
            column_format = "rl"
        else:
            column_format = justify[0] + (justify[1] * len(df.keys()))

        joined_caption_and_reference = ''
        if caption is not None:
            table_reference = r'\label{tab:' + caption.replace(' ', '_').replace(":", "") + r'}'
            joined_caption_and_reference = caption + table_reference
        df_tex = df.to_latex(index=True, escape=False, column_format=column_format, float_format=skeleton, caption=joined_caption_and_reference)

        if was_originally_two_column:
            df_tex = df_tex.replace(r'& value', r'item & value')


        df_tex = df_tex.replace(r'\midrule', r'\hline\midrule')
        df_tex = df_tex.replace(r'\begin{table}', r'\begin{table} \centering ')

        start_adjustbox = r'\begin{adjustbox}{width=\columnwidth,center}'
        end_adjustbox = r'\end{adjustbox}'
        index_of_tabular = df_tex.find(r'\begin{tabular}')
        df_tex = df_tex[:index_of_tabular] + start_adjustbox + df_tex[index_of_tabular:]
        df_tex = df_tex.replace(r'\end{table}', end_adjustbox + r'\end{table}')

        # thought is that only the single-cell-scalar inf values will be replaced above, so any inf still remaining must be inside a pmatrix
        df_tex = df_tex.replace('inf ', inf_replacement.replace("$", "") + " ")

        df_tex = df_tex.replace(r'\unit{ kg m^2 }', r'\unit{ kg ~m^2 }')

        pre_fix = '\n' + r'\begin{center}' + '\n'
        end_fix = r'\end{center}'
        latex_out = pre_fix + df_tex + end_fix
        print(latex_out)

        return latex_out

    def print(self, level='info', caption=None, nan_replacement="NAN", transpose=False, sort_dim=0, digits=4, repr_type='G'):
        string = self.to_string(nan_replacement=nan_replacement, transpose=transpose, caption=caption, sort_dim=sort_dim, digits=digits, repr_type=repr_type)
        string_list = string.split('\n')
        for substring in string_list:
            base_print(substring, level=level)
        return string

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
               'nparray': np.array([1., 2.2, 3.33]),
               'cas.dm - scalar': cas.DM(8.13),
               'cas.dm - array': 4.5 * cas.DM.ones((3, 1)),
               'cas.dm - matrix': cas.DM([[1, 2.2], [3.3, 4.4]]),
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

    example_line = 'cas.dm - array.................................... [4.5, 4.5, 4.5]'
    example_line_included = example_line in found_string

    criteria = all(all_items_included) and all(all_values_included) and example_line_included
    if not criteria:
        message = 'two-column table to_string does not work as expected.'
        log_and_raise_error(message)

    return None


def test_two_column_table_to_latex():
    tab = make_sample_two_column_table()
    latex = tab.to_latex(digits=3, repr_type='E')

    opening_in_latex = r'\begin{tabular}{rl}' in latex
    header_in_latex = r'item & value \\' in latex
    ending_in_latex = r'\end{tabular}' in latex

    body_lines = ['boolean & False \\',
            r'cas.dm - array & $\begin{pmatrix}4.5 & 4.5 & 4.5 \end{pmatrix}^\top$ \\',
            r'cas.dm - matrix & $\begin{pmatrix}1 & 2.2 \\ 3.3 & 4.4 \end{pmatrix}$ \\',
            'cas.dm - scalar & 8.130E+00 \\',
            'dict aa1 & 3 \\',
            'dict bb1 & happy \\',
            'dict cc1 & [1, 2] \\',
            'float & 2.339E+01 \\',
            'int & 234 \\',
            'neg & -2.8 \\',
            r'nparray & $\begin{pmatrix}1 & 2.2 & 3.33 \end{pmatrix}^\top$ \\',
            'sci & 3.431E-07 \\',
            'string & apples \\']
    all_body_included = [line in latex for line in body_lines]

    criteria = opening_in_latex and header_in_latex and ending_in_latex and all(all_body_included)
    if not criteria:
        message = 'two-column table to_latex does not work as expected.'
        log_and_raise_error(message)
    return None


def test_multicolumn_table_to_latex():
    tab = make_sample_multicolumn_table()
    test_latex = tab.to_latex(digits=2, repr_type='E')

    includes_header = r' & Sun & Earth & Moon & Mars \\' in test_latex
    includes_midrule = r'\midrule' in test_latex
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
        try:
            local_dict = [value for value in local_dict.values()][0]
        except:
            return depth
        depth += 1
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


def print_dict_as_table(dict, level='info', to_echo_or_latex='echo', caption=None, nan_replacement="NAN", transpose=False, latex_dict={}, sort_dim=None, digits=4, repr_type='G', latex_symbolic_in_first_column=False):
    depth = get_depth_of_dict(dict)
    out_string = ''

    if depth == 0:
        base_print(dict, level=level)
        out_string = repr(dict)

    elif depth in [1, 2]:
        tab = Table(dict)
        if to_echo_or_latex == 'latex':
            out_string = tab.to_latex(caption=caption, nan_replacement=nan_replacement, transpose=transpose, latex_dict=latex_dict, sort_dim=sort_dim, digits=digits, repr_type=repr_type, latex_symbolic_in_first_column=latex_symbolic_in_first_column)
        else:
            out_string = tab.print(level=level, caption=caption, nan_replacement=nan_replacement, transpose=transpose, sort_dim=sort_dim, digits=digits, repr_type=repr_type)

    else:
        message = 'function to print_dict_as_table is not available for dicts of depth ' + str(depth)
        log_and_raise_error(message)

    return out_string

def print_bulleted_list(list, level='info', to_echo_or_latex='echo', caption=None):

    if to_echo_or_latex == 'latex':
        caption = r'\n' + caption
        print(caption)
        print(r"\\begin{itemize}")
        for name in list:
            print(r'\item {}'.format(name).replace('_', ' '))
        print(r'\end{itemize}')
    else:
        base_print(caption, level=level)
        for name in list:
            base_print('* {}'.format(name), level=level)
        base_print('', level=level)
    return None

def column_size_for_dot_separated_items():
    return 50

def repr_g(value, digits=4, repr_type='G', to_echo_or_latex='echo'):
    if isinstance(value, str):
        return value
    elif isinstance(value, cas.SX) or isinstance(value, cas.MX):
        return str(value)
    elif isinstance(value, np.ndarray):
        return repr_g(cas.DM(value), digits=digits, repr_type=repr_type, to_echo_or_latex=to_echo_or_latex)
    elif isinstance(value, dict):
        temp_dict = {}
        for key, local_value in value.items():
            temp_dict[key] = repr_g(local_value, digits=digits, repr_type=repr_type, to_echo_or_latex=to_echo_or_latex)
        return temp_dict
    elif isinstance(value, complex) and (np.abs(np.imag(value)) < 1.e-16):
        return repr_g(np.real(value), digits=digits, repr_type=repr_type, to_echo_or_latex=to_echo_or_latex)
    elif isinstance(value, int) and (np.abs(value) < 10**digits):
        return str(value)
    elif isinstance(value, float) and np.abs(value).is_integer():
        return repr_g(int(value), digits=digits, repr_type=repr_type, to_echo_or_latex=to_echo_or_latex)
    elif isinstance(value, float) and (np.abs(value) < 10) and (np.abs(value) * 10**digits).is_integer():
        return str(value)
    elif (isinstance(value, int) or isinstance(value, float)):
        skeleton = "{:0." + str(digits) + repr_type + "}"
        message = skeleton.format(value)
        return message

    elif isinstance(value, cas.DM):
        if value.shape == (1, 1):
            return repr_g(float(value), repr_type=repr_type, digits=digits, to_echo_or_latex=to_echo_or_latex)

        elif (len(value.shape) == 2) and (to_echo_or_latex != 'latex'):
            temp_dm = value
            is_column = False
            if (value.shape[0] == 1) or (value.shape[1] == 1):
                is_column = True
                temp_dm = value.reshape((value.shape[0] * value.shape[1], 1))
            temp_string = "["
            for idx in range(temp_dm.shape[0]):
                if is_column:
                    local_val = temp_dm[idx]
                else:
                    local_val = temp_dm[idx, :]
                temp_string += repr_g(local_val, repr_type=repr_type, digits=digits, to_echo_or_latex=to_echo_or_latex) + ", "

            temp_string = temp_string[:-2] + "]"
            return temp_string

        elif (len(value.shape) == 2) and (to_echo_or_latex == 'latex'):
            temp_string = r"\begin{pmatrix}"

            is_transposed = False
            if (value.shape[0] == 1) or (value.shape[1] == 1):
                if value.shape[1] == 1:
                    is_transposed = True
                value = value.reshape((1, value.shape[0] * value.shape[1]))

            for rdx in range(value.shape[0]):
                temp_row = ''
                for cdx in range(value.shape[1]):
                    temp_row += repr_g(value[rdx, cdx], repr_type=repr_type, digits=digits, to_echo_or_latex=to_echo_or_latex) + " & "
                temp_row = temp_row[:-2] + r"\\ "
                temp_string += temp_row
            temp_string = temp_string[:-3] +  r"\end{pmatrix}"

            if is_transposed:
                temp_string += r"^\top"

            temp_string = temp_string.replace('$', '')
            temp_string = r"$" + temp_string + r"$"

            return temp_string
        else:
            return repr(value)
    elif isinstance(value, list):
        temp_string = "["
        for idx in range(len(value)):
            temp_string += repr_g(value[idx], repr_type=repr_type, digits=digits, to_echo_or_latex=to_echo_or_latex) + ", "
        temp_string = temp_string[:-2] + "]"
        return temp_string
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
