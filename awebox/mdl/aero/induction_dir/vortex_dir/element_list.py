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
object-oriented-vortex-filament-and-cylinder operations
_python-3.5 / casadi-3.4.5
- authors: rachel leuthold 2021
'''
import pdb

import casadi.tools as cas
import matplotlib.pyplot as plt
import numpy as np

import awebox.mdl.aero.induction_dir.vortex_dir.biot_savart as biot_savart
import awebox.mdl.aero.induction_dir.vortex_dir.element as vortex_element

import awebox.tools.struct_operations as struct_op
import awebox.tools.vector_operations as vect_op
import awebox.tools.print_operations as print_op

from awebox.logger.logger import Logger as awelogger

import matplotlib
matplotlib.use('TkAgg')

class ElementList:
    def __init__(self, expected_number_of_elements=None):
        self.__list = []
        self.set_element_type(None)
        self.__number_of_elements = 0
        self.__expected_number_of_elements = expected_number_of_elements
        self.__element_info_length = None
        self.__expected_element_info_length = None

    def append(self, added_elem):

        is_element_list = isinstance(added_elem, ElementList)
        is_element = isinstance(added_elem, vortex_element.Element)
        is_correct_type = (self.__element_type is None) or (self.__element_type == added_elem.element_type)
        has_correct_length = (self.__element_info_length is None) or (is_element and (self.__element_info_length == added_elem.info_length)) or (is_element_list and (self.__element_info_length == added_elem.element_info_length))

        if not is_element_list and not is_element:
            message = 'tried to append to vortex element list, but proposed addition is neither a vortex element nor an element list. append instruction was skipped'
            awelogger.logger.warning(message)

        elif is_element_list and is_correct_type:

            for indiv_elem in added_elem.list:
                self.append(indiv_elem)

        elif is_element_list and not is_correct_type:
            message = 'tried to append element list to element list, but the types were incompatible so append instruction was skipped.'
            awelogger.logger.warning(message)

        elif is_element and is_correct_type and has_correct_length:
            self.__list += [added_elem]
            self.__number_of_elements += 1

            if self.__element_type is None:
                self.set_element_type(added_elem.element_type)

            if self.__element_info_length is None:
                self.__element_info_length = added_elem.info_length
                self.set_expected_element_info_length(added_elem.expected_info_length)

        elif is_element and not is_correct_type:
            message = 'tried to append vortex element to element list, but the types were incompatible so append instruction was skipped.'
            awelogger.logger.warning(message)

        elif is_element and is_correct_type and not has_correct_length:
            message = 'tried to append vortex element to element list, but the element did not have the correct length of information so append instruction was skipped.'
            awelogger.logger.warning(message)

        else:
            message = 'tried to append vortex element to element list. something went wrong, so append instruction was skipped.'
            awelogger.logger.warning(message)
        return None

    def confirm_list_has_expected_dimensions(self):

        number_of_elements = self.number_of_elements
        expected_number_of_elements = self.expected_number_of_elements

        if (expected_number_of_elements is not None) and (not number_of_elements == expected_number_of_elements):
            message = 'unexpected number of vortex elements in list'
            raise Exception(message)

        element_info_length = self.element_info_length
        expected_element_info_length = self.expected_element_info_length

        if (expected_element_info_length is not None) and (not element_info_length == expected_element_info_length):
            message = 'unexpected info length for vortex element in list'
            raise Exception(message)

        return None



    def get_columnized_info_list(self):

        decolumnized_list = self.get_decolumnized_info_list()

        number_of_elements = self.number_of_elements
        element_info_length = self.element_info_length
        columnized_list = cas.reshape(decolumnized_list, (element_info_length * number_of_elements, 1))

        return columnized_list

    def get_decolumnized_info_list(self, period=None, wind=None, optimization_period=None):

        self.confirm_list_has_expected_dimensions()

        if (period is not None) and (wind is not None):
            python_list_of_info = [elem.get_repeated_info(period, wind, optimization_period) for elem in self.__list]
        else:
            python_list_of_info = [elem.info for elem in self.__list]

        decolumnized_list = cas.horzcat(*python_list_of_info)

        self.confirm_that_decolumnized_info_list_has_correct_shape_for_mapping(decolumnized_list)

        return decolumnized_list

    def confirm_that_decolumnized_info_list_has_correct_shape_for_mapping(self, decolumnized_list):

        number_of_elements = self.number_of_elements
        element_info_length = self.element_info_length

        list_shape_is_correct_for_mapping = (decolumnized_list.shape == (element_info_length, number_of_elements))
        if not list_shape_is_correct_for_mapping:
            message = 'something went wrong when creating the vortex element list\'s decolumnized info list'
            awelogger.logger.error(message)
            raise Exception(message)

        return None

    def get_number_of_symbolics_for_biot_savart(self, projected=False):
        number_symbolics = self.__expected_element_info_length + 3
        if projected:
            number_symbolics += 3

        return number_symbolics

    def get_biot_savart_observer_info(self, info, projected=False):
        if projected:
            x_obs = info[-6:-3]
            n_hat = info[-3:]
        else:
            x_obs = info[-3:]
            n_hat = None

        return x_obs, n_hat

    def get_decolumnized_list_with_attached_observer_info(self, x_obs=cas.DM.zeros(3, 1), n_hat=None, period=None, wind=None, optimization_period=None):
        decolumnized_list = self.get_decolumnized_info_list(period, wind, optimization_period)

        number_of_elements = self.number_of_elements
        observer_list = cas.repmat(x_obs, (1, number_of_elements))

        if (n_hat is None):
            combi = cas.vertcat(decolumnized_list, observer_list)

        elif (n_hat is not None) and (n_hat.shape == (3, 1)):
            n_hat_list = cas.repmat(n_hat, (1, number_of_elements))
            combi = cas.vertcat(decolumnized_list, observer_list, n_hat_list)

        else:
            message = 'something went wrong when appending the normal vector to the element_list. n_hat passed was:' + str(n_hat)
            awelogger.logger.error(message)
            raise Exception(message)

        return combi

    def make_symbolic_info_function(self, model_variables, model_parameters):
        for elem in self.__list:
            elem.make_symbolic_info_function(model_variables, model_parameters)

        return None


    def make_symbolic_biot_savart_function(self):

        if self.__expected_element_info_length is not None:

            projected = True

            number_symbolics = self.get_number_of_symbolics_for_biot_savart(projected)
            info_sym = cas.SX.sym('info_sym', (number_symbolics, 1))
            info_sym_unprojected = info_sym[:-3]
            x_obs, n_hat = self.get_biot_savart_observer_info(info_sym, projected)

            if (not self.__list) or (self.__element_type is 'unspecified'):
                message = 'either the vortex element list has no vortex elements, and therefore the type of vortex element in the list is unspecified; or the type of vortex element in the non-empty list - somehow - remains unspecified. without type information, the biot savart induced velocity cannot be computed.'
                awelogger.logger.error(message)
                raise Exception(message)

            info_dict = self.__list[0].unpack_info(info_sym)
            info_dict['x_obs'] = x_obs

            if self.__element_type == 'filament':
                vec_u_ind = biot_savart.filament(info_dict)
            elif self.__element_type == 'longitudinal_cylinder':
                vec_u_ind = biot_savart.longitudinal_cylinder(info_dict)
            elif self.__element_type == 'tangential_cylinder':
                vec_u_ind = biot_savart.tangential_cylinder(info_dict)

            else:
                message = 'cannot make symbolic biot savart function for vortex element list of type: ' + self.__element_type
                awelogger.logger.error(message)
                raise Exception(message)

            u_ind_projected = cas.mtimes(vec_u_ind.T, n_hat)
            biot_savart_projected_fun = cas.Function('biot_savart_projected_fun', [info_sym], [u_ind_projected])
            self.__biot_savart_projected_fun = biot_savart_projected_fun

            biot_savart_fun = cas.Function('biot_savart_fun', [info_sym_unprojected], [vec_u_ind])
            self.__biot_savart_fun = biot_savart_fun

        return None

    def evaluate_biot_savart_induction_for_all_elements(self, x_obs=cas.DM.zeros(3, 1), n_hat=None, period=None, wind=None, optimization_period=None):

        if n_hat is None:
            biot_savart_fun = self.__biot_savart_fun
        else:
            biot_savart_fun = self.__biot_savart_projected_fun

        decolumnized_list = self.get_decolumnized_list_with_attached_observer_info(x_obs, n_hat, period, wind, optimization_period)

        number_of_elements = self.number_of_elements
        biot_savart_map = biot_savart_fun.map(number_of_elements, 'openmp')
        all = biot_savart_map(decolumnized_list)

        return all

    def evalate_total_biot_savart_induction(self, x_obs=cas.DM.zeros(3, 1), n_hat=None):

        all = self.evaluate_biot_savart_induction_for_all_elements(x_obs, n_hat)
        u_ind = cas.sum2(all)
        return u_ind

    def draw(self, ax, side, variables_scaled, parameters, cosmetics):
        for elem in self.__list:
            elem.draw(ax, side, variables_scaled, parameters, cosmetics)

        return None

    def abs_strength_max(self, variables_scaled, parameters):
        all_strengths = np.array([elem.unpack_info(elem.evaluate_info(variables_scaled, parameters))['strength'] for elem in self.__list])
        return np.max(np.abs(all_strengths))

    @property
    def list(self):
        return self.__list

    @list.setter
    def list(self, value):
        awelogger.logger.error('Cannot set list object.')

    @property
    def number_of_elements(self):
        return self.__number_of_elements

    @number_of_elements.setter
    def number_of_elements(self, value):
        awelogger.logger.error('Cannot set number_of_elements object.')

    @property
    def expected_number_of_elements(self):
        return self.__expected_number_of_elements

    @expected_number_of_elements.setter
    def expected_number_of_elements(self, value):
        if self.__expected_number_of_elements is not None:
            awelogger.logger.error('Cannot set elements object.')
        else:
            self.__expected_number_of_elements = value
        return None

    @property
    def element_info_length(self):
        return self.__element_info_length

    @element_info_length.setter
    def element_info_length(self, value):
        awelogger.logger.error('Cannot set element_info_length object.')

    @property
    def biot_savart_fun(self):
        return self.__biot_savart_fun

    @biot_savart_fun.setter
    def biot_savart_fun(self, value):
        awelogger.logger.error('Cannot set biot_savart_fun object.')

    @property
    def biot_savart_projected_fun(self):
        return self.__biot_savart_projected_fun

    @biot_savart_projected_fun.setter
    def biot_savart_projected_fun(self, value):
        awelogger.logger.error('Cannot set biot_savart_projected_fun object.')


    @property
    def element_type(self):
        return self.__element_type

    @element_type.setter
    def element_type(self, value):
        awelogger.logger.error('Cannot set element_type object.')

    def set_element_type(self, value):
        self.__element_type = value

    @property
    def expected_element_info_length(self):
        return self.__expected_element_info_length

    @expected_element_info_length.setter
    def expected_element_info_length(self, value):
        awelogger.logger.error('Cannot set expected_element_info_length object.')
        return None

    def set_expected_element_info_length(self, value):
        if self.__expected_element_info_length is not None:
            awelogger.logger.error('Cannot set expected_element_info_length object.')
        else:
            self.__expected_element_info_length = value
        return None


def get_test_filament_list():
    filament_list = ElementList()

    x_start0 = 0. * vect_op.xhat_np()
    x_end0 = x_start0 + 5. * vect_op.xhat_np()
    r_core0 = 0.
    strength0 = 3.
    dict_info0 = {'x_start': x_start0,
                  'x_end': x_end0,
                  'r_core': r_core0,
                  'strength': strength0}

    fil0 = vortex_element.Filament(dict_info0)
    filament_list.append(fil0)

    x_start1 = x_end0
    x_end1 = x_start1 + 5. * vect_op.zhat_np()
    r_core1 = 0.
    strength1 = 1.
    dict_info1 = {'x_start': x_start1,
                  'x_end': x_end1,
                  'r_core': r_core1,
                  'strength': strength1}

    fil1 = vortex_element.Filament(dict_info1)
    filament_list.append(fil1)

    x_start2 = x_end1
    x_end2 = x_start2 + 5. * vect_op.yhat_np()
    r_core2 = 0.
    strength2 = -3.
    dict_info2 = {'x_start': x_start2,
                  'x_end': x_end2,
                  'r_core': r_core2,
                  'strength': strength2}

    fil2 = vortex_element.Filament(dict_info2)
    filament_list.append(fil2)

    return filament_list

def test_filament_list_columnization():
    filament_list = get_test_filament_list()
    columnized_info_list = filament_list.get_columnized_info_list()
    assert (columnized_info_list.shape == (8 * 3, 1))
    return None

def test_filament_list_biot_savart():

    filament_list = get_test_filament_list()

    x_obs = cas.DM.zeros((3, 1))
    filament_list.compute_biot_savart_induction(x_obs=x_obs, n_hat=None)
    print_op.warn_about_temporary_funcationality_removal(location='vortex_element.test_filament_list_biot_savart')
    return None

# def test_filament_list_drawing():
#     filament_list = get_test_filament_list()
#     fig = plt.figure()
#     ax = plt.axes(projection='3d')
#     filament_list.draw(ax)
#     plt.show()
#     return None

def get_test_tangential_cylinder_list():
    cylinder_list = ElementList()
    cyl = vortex_element.get_test_tangential_cylinder()
    cylinder_list.append(cyl)
    return cylinder_list

def get_test_longitudinal_cylinder_list():
    cylinder_list = ElementList()
    cyl = vortex_element.get_test_longitudinal_cylinder()
    cylinder_list.append(cyl)
    return cylinder_list