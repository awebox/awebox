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
# Class Options contains parameters, meta-parameters, and functionality decision parameters
###################################
import copy

import numpy as np
import casadi.tools as cas
from sympy.parsing.maxima import sub_dict

from . import default
from . import funcs
from ..mdl.aero.induction_dir.vortex_dir import tools
import awebox.tools.print_operations as print_op
import awebox.tools.save_operations as save_op

class Options:
    def __init__(self):

        default_user_options, help_options = default.set_default_user_options()
        default_options, help_options = default.set_default_options(default_user_options, help_options)

        self.__options_dict = default_options
        self.__help_dict = help_options
        self.__keys_list = list(self.__options_dict.keys())
        self.__flattened_dict = {}

    def __setitem__(self, key, value):
        category_key, sub_category_key, sub_sub_category_key, option_key, help_flag = get_keys(key)
        if category_key is None:
            if type(self.__options_dict[option_key]) is type(value):
                self.__options_dict[option_key] = value
            else:
                raise TypeError('Wrong type to set option' + str(option_key) + '.')
        elif sub_category_key is None:
            if type(self.__options_dict[category_key][option_key]) is type(value):
                self.__options_dict[category_key][option_key] = value
            else:
                raise TypeError('Wrong type to set option' + str(option_key) + '.')
        elif sub_sub_category_key is None:
            if type(self.__options_dict[category_key][sub_category_key][option_key]) is type(value):
                self.__options_dict[category_key][sub_category_key][option_key] = value
            else:
                raise TypeError('Wrong type to set option' + str(option_key) + '.')
        else:
            if type(self.__options_dict[category_key][sub_category_key][sub_sub_category_key][option_key]) is type(value):
                self.__options_dict[category_key][sub_category_key][sub_sub_category_key][option_key] = value
            else:
                raise TypeError('Wrong type to set option' + str(option_key) + '.')

    def __getitem__(self, item):
        category_key, sub_category_key, sub_sub_category_key, option_key, help_flag = get_keys(item)
        if help_flag is True:
            dict = self.__help_dict
            option_key = option_key[:-3]
        else:
            dict = self.__options_dict
        if category_key is None:
            return dict[option_key]
        elif sub_category_key is None:
            return dict[category_key][option_key]
        elif sub_sub_category_key is None:
            return dict[category_key][sub_category_key][option_key]
        else:
            return dict[category_key][sub_category_key][sub_sub_category_key][option_key]

    def fill_in_seed(self, seed):

        assert type(seed) == dict, 'User-provided options should be of type "dict"!'

        for key, value in seed.items():
            keys = key.split(".")
            err_msg = f'Unknown option: {key}'

            assert len(keys) in [2,3,4], err_msg
            assert keys[0] in self.__keys_list, err_msg
            assert keys[1] in self.__options_dict[keys[0]], err_msg
            if len(keys) == 2:
                self.__options_dict[keys[0]][keys[1]] = value
            elif len(keys) == 3:
                assert keys[2] in self.__options_dict[keys[0]][keys[1]], err_msg
                self.__options_dict[keys[0]][keys[1]][keys[2]] = value
            elif len(keys) == 4:
                assert keys[2] in self.__options_dict[keys[0]][keys[1]], err_msg
                if keys[3].isdigit():
                    keys[3] = int(keys[3])
                assert keys[3] in self.__options_dict[keys[0]][keys[1]][keys[2]], err_msg
                self.__options_dict[keys[0]][keys[1]][keys[2]][keys[3]] = value

        return None

    def recursively_flatten_dict(self, base_name, current_name, current_value, current_help):

        list_of_unexpected_subtypes = ['system_bounds', 'model_bounds', 'stab_derivs']
        names_to_skip = ['stab_derivs', 'architecture']

        for test_name in names_to_skip:
            if (test_name == base_name) or (str(test_name) == str(current_name)):
                return None

        if not isinstance(current_value, dict):
            self.add_entry_to_flattened_dict(base_name, current_name, current_value, current_help)
        else:
            for local_name, local_value in current_value.items():
                if (current_help is not None) and hasattr(current_help, 'keys') and (local_name in current_help.keys()):
                    local_help = current_help[local_name]
                else:
                    local_help = current_help

                if (len(current_value.keys()) < 2) or (current_name in list_of_unexpected_subtypes):
                    self.recursively_flatten_dict(str(base_name), str(current_name) + '.' + str(local_name), local_value, local_help)
                else:
                    self.recursively_flatten_dict(str(base_name) + '.' + str(current_name), local_name, local_value, local_help)

        return None

    def prepare_flattened_dict(self):

        copied_dict = copy.deepcopy(self.__options_dict)
        copied_help = copy.deepcopy(self.__help_dict)

        for base_name, base_value in copied_dict.items():
            for local_name, local_value in base_value.items():
                if base_name in copied_help.keys() and local_name in copied_help[base_name].keys():
                    local_help = copied_help[base_name][local_name]
                else:
                    local_help = None

                self.recursively_flatten_dict(base_name, local_name, local_value, local_help)

        return None

    def add_entry_to_flattened_dict(self, flattened_key, subkey, current_value, current_help):

        if (current_help is not None) and (not isinstance(current_help, dict)) and ((len(current_help) > 0) and (current_help[0] is not None) and (len(current_help[0]) > 2)):
            current_units = current_help[0][2]
        else:
            current_units = None

        if flattened_key not in self.__flattened_dict.keys():
            self.__flattened_dict[flattened_key] = {}

        if ('cost' in str(flattened_key)) and (str(subkey)[-1].isdigit()):
            digitname = str(subkey)[-1]
            subname = str(subkey)[:-2]
        else:
            digitname = None
            subname = subkey

        if subname not in self.__flattened_dict[flattened_key].keys():
            self.__flattened_dict[flattened_key][subname] = {}

        have_already_defined_units = 'units' in self.__flattened_dict[flattened_key][subname].keys()
        if not have_already_defined_units:
            self.__flattened_dict[flattened_key][subname]['units'] = current_units
        elif not ((current_units is None) and (self.__flattened_dict[flattened_key][subname]['units'] is not None)):
            self.__flattened_dict[flattened_key][subname]['units'] = current_units

        if digitname is None:
            self.__flattened_dict[flattened_key][subname]['value'] = current_value
        else:
            self.__flattened_dict[flattened_key][subname][digitname] = current_value

        return None

    def report_stability_derivatives(self, to_echo_or_latex='echo', latex_dict={}, nan_replacement='--', trial_name='', save=False):

        copied_dict = copy.deepcopy(self.__options_dict)
        if copied_dict['user_options']['system_model']['kite_dof'] == 6:

            stab_derivs = {}
            for poss_deriv in copied_dict['params']['aero'].keys():
                if poss_deriv[0] == 'C':
                    stab_derivs[poss_deriv] = copied_dict['params']['aero'][poss_deriv]

            out_table = {}
            for deriv_name in stab_derivs.keys():
                if deriv_name not in out_table.keys():
                    out_table[deriv_name] = {}

                    for input_name, deriv_stack in stab_derivs[deriv_name].items():
                        deriv_length = deriv_stack.shape[0]

                        for ldx in range(deriv_length):

                            if to_echo_or_latex == 'latex':
                                multiplier = r'' # ~ '
                                dollar = r'$'
                                space = r' '
                                unspace = r'\hspace{-1.5ex} '
                            else:
                                multiplier = r' *'
                                dollar = r''
                                space = r''
                                unspace = r''

                            subscript = input_name
                            # the really weird spacing thing, is so that print_op's print_as_table's latex replacement
                            # function will recognize the variable names individually
                            if input_name == 'alpha' and ldx > 0:
                                subscript += space + unspace + dollar + "^" + str(ldx + 1) + dollar
                            elif ldx == 1:
                                subscript += multiplier + " alpha "
                            elif ldx > 1:
                                subscript += multiplier + " alpha" + space + unspace + dollar + "^" + str(ldx) + dollar

                            out_table[deriv_name][subscript] = deriv_stack[ldx]

            caption = 'stability derivatives for ' + trial_name
            string_out = print_op.print_dict_as_table(out_table, level='info', to_echo_or_latex=to_echo_or_latex,
                                                      nan_replacement=nan_replacement, transpose=False,
                                                      caption=caption, latex_dict=latex_dict, sort_dim=0)
            if save:
                save_op.write_string_to_txt_or_tex(string_out, trial_name.replace(' ', '_'),
                                                   to_echo_or_latex=to_echo_or_latex)

            # aero validity rules are printed with model inequality constraints

        return None

    def make_report(self, to_echo_or_latex='echo', latex_dict={}, trial_name='', print_all_options=False, save=False):

        if to_echo_or_latex == 'latex':
            trial_name = trial_name.replace('_', ' ')

        if print_all_options:
            for top_level_name, subdict in self.__flattened_dict.items():
                caption = top_level_name
                if (trial_name is not None) and (trial_name != ''):
                    caption += ' for ' + trial_name
                string_out = print_op.print_dict_as_table(subdict, to_echo_or_latex=to_echo_or_latex, caption=caption, nan_replacement='--', transpose=True, latex_dict=latex_dict)
                if save:
                    save_op.write_string_to_txt_or_tex(string_out, trial_name.replace(' ', '_'),
                                                       to_echo_or_latex=to_echo_or_latex)

        if 'stab_derivs' in latex_dict.keys():
            stab_deriv_dict = latex_dict['stab_derivs']
        else:
            stab_deriv_dict = latex_dict
        self.report_stability_derivatives(to_echo_or_latex=to_echo_or_latex, nan_replacement='--', latex_dict=stab_deriv_dict, trial_name=trial_name, save=save)

        return None

    def keys(self):
        return self.__keys_list

    def build(self, architecture):
        self.__options_dict, self.__help_dict = funcs.build_options_dict(self.__options_dict, self.__help_dict, architecture)
        self.prepare_flattened_dict()

        return None

    @property
    def flattened_dict(self):
        return self.__flattened_dict

    @flattened_dict.setter
    def flattened_dict(self, value):
        print('Cannot set flattened_dict object.')

    @property
    def help_dict(self):
        return self.__help_dict

    @help_dict.setter
    def help_dict(self, value):
        print('Cannot set help_dict object.')

    @property
    def options_dict(self):
        return self.__options_dict

    @options_dict.setter
    def options_dict(self, value):
        print('Cannot set options_dict object.')


def get_keys(item):
    category_key = None
    sub_category_key = None
    sub_sub_category_key = None
    option_key = None
    help_flag = False
    item = [item]
    try:
        [category_key, sub_category_key, sub_sub_category_key, option_key] = item
    except(ValueError):
        try:
            [category_key, sub_category_key, option_key] = item
        except(ValueError):
            try:
                [category_key, option_key] = item
            except(ValueError):
                [option_key] = item
    if str(option_key[-3:]) == ' -h':
        help_flag = True

    return category_key, sub_category_key, sub_sub_category_key, option_key, help_flag

