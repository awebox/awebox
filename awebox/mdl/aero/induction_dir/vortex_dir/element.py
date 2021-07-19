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
'''
object-oriented-vortex-filament-and-cylinder operations
_python-3.5 / casadi-3.4.5
- authors: rachel leuthold 2021
'''

import casadi.tools as cas
from awebox.logger.logger import Logger as awelogger
import awebox.tools.struct_operations as struct_op


class Element:
    def __init__(self, strength=None):
        self.__strength = strength
        self.__type = 'unspecified'

    def get_biot_savart_induced_velocity(self, x_obs):
        message = 'this function is undefined for a vortex element of unspecified type'
        raise Exception(message)
        return None

    @property
    def strength(self):
        return self.__strength

    @strength.setter
    def strength(self, value):
        self.__strength = value

    @property
    def type(self):
        return self.__type

    @type.setter
    def type(self, value):
        awelogger.logger.warning('Cannot set type object.')


class ElementList:
    def __init__(self):
        self.__list = []
        self.__type = 'unspecified'

    def append(self, elem):
        if isinstance(elem, Element) and (elem.type == self.__type):
            self._list += elem
        else:
            message = 'tried to append vortex element to element list. something went wrong, so append instruction was skipped.'
            awelogger.logger.warning(message)
        return None

    def expected_number_of_elements(self):
        message = 'this function is undefined for a vortex element list of unspecified type'
        raise Exception(message)
        return -9999

    def count_elements(self):
        return len(self.__list)

    def expected_element_info_length(self):
        message = 'this function is undefined for a vortex element list of unspecified type'
        raise Exception(message)
        return -9999

    def count_element_info_length(self):
        return self.__list[0].shape[0]

    def confirm_list_has_expected_dimensions(self):
        number_elements = self.count_elements()
        expected_number_elements = self.expected_number_of_elements()

        element_info_length = self.count_element_info_length()
        expected_element_info_length = self.expected_element_info_length()

        if not number_elements == expected_number_elements:
            message = 'unexpected number of vortex elements in list'
            raise Exception(message)

        if not element_info_length == expected_element_info_length:
            message = 'unexpected element info length in list'
            raise Exception(message)

        return None

    def columnize(self):
        number_elements = self.count_elements()
        element_info_length = self.count_element_info_length()

        self.confirm_list_has_expected_dimensions()

        columnized_list = cas.reshape(self.__list, (element_info_length * number_elements, 1))
        return columnized_list

    def decolumnize(self, columnized_list):

        number_elements = self.count_elements()
        element_info_length = self.count_element_info_length()

        self.confirm_list_has_expected_dimensions()

        decolumnized_list = cas.reshape(columnized_list, (element_info_length, number_elements))
        return decolumnized_list



    @property
    def eq_list(self):
        return self.__eq_list

    @eq_list.setter
    def eq_list(self, value):
        awelogger.logger.warning('Cannot set eq_list object.')

    @property
    def ineq_list(self):
        return self.__ineq_list

    @ineq_list.setter
    def ineq_list(self, value):
        awelogger.logger.warning('Cannot set ineq_list object.')

    @property
    def all_list(self):
        return self.__all_list

    @all_list.setter
    def all_list(self, value):
        awelogger.logger.warning('Cannot set all_list object.')

    @property
    def list_name(self):
        return self.__list_name

    @list_name.setter
    def list_name(self, value):
        awelogger.logger.warning('Cannot set list_name object.')