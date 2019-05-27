#
#    This file is part of awebox.
#
#    awebox -- A modeling and optimization framework for multi-kite AWE systems.
#    Copyright (C) 2017-2019 Jochem De Schutter, Rachel Leuthold, Moritz Diehl,
#                            ALU Freiburg.
#    Copyright (C) 2018-2019 Thilo Bronnenmeyer, Kiteswarms Ltd.
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
    def __init__(self, internal_access=False):

        default_user_options, help_options = default.set_default_user_options(internal_access)
        default_options, help_options = default.set_default_options(default_user_options, help_options)

        self.__options_dict = default_options
        self.__help_dict = help_options
        self.__keys_list = list(self.__options_dict.keys())
        self.__internal_access = internal_access

    def __setitem__(self, key, value):
        if key in self.__options_dict.keys():
            self.__options_dict[key] = value
        else:
            raise KeyError('1 The key ' + key + ' is not valid. Valid options are ' + str(self.__options_dict.keys()) + '.')

    def __getitem__(self, item):
        if item in self.__options_dict.keys():
            return self.__options_dict[item]
        else:
            raise KeyError('2 The key ' + item + ' is not valid. Valid options are ' + str(self.__options_dict.keys()) + '.')

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
