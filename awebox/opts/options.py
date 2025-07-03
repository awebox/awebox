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

from . import default
from . import funcs

class Options:
    def __init__(self):

        default_user_options, help_options = default.set_default_user_options()
        default_options, help_options = default.set_default_options(default_user_options, help_options)

        self.__options_dict = default_options
        self.__help_dict = help_options
        self.__keys_list = list(self.__options_dict.keys())

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

    def keys(self):
        return self.__keys_list

    def build(self, architecture):
        self.__options_dict, self.__help_dict = funcs.build_options_dict(self.__options_dict, self.__help_dict, architecture)
        return None

    @property
    def help_dict(self):
        return self.__help_dict

    @help_dict.setter
    def help_dict(self, value):
        print('Cannot set help_dict object.')

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
