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
import numpy as np

import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.element as obj_element
import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.finite_filament as obj_finite_filament
import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.semi_infinite_tangential_cylinder as obj_semi_infinite_tangential_cylinder

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

        self.__model_induction_fun = None
        self.__model_induction_factor_fun = None
        self.set_biot_savart_fun(None)
        self.set_concatenated_biot_savart_fun(None)

    def append(self, added_elem):

        is_element_list = isinstance(added_elem, ElementList)
        is_element = isinstance(added_elem, obj_element.Element)
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

    def get_decolumnized_info_list(self):

        self.confirm_list_has_expected_dimensions()
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

    def get_number_of_symbolics_for_concatenated_biot_savart(self):
        number_symbolics = self.__expected_element_info_length + 3
        return number_symbolics

    def get_element_info_from_concatenated_inputs(self, concatenated):
        elem_info = concatenated[:-3]
        return elem_info

    def get_observer_info_from_concatenated_inputs(self, concatenated):
        obs_info = concatenated[-3:]
        return obs_info

    def get_decolumnized_list_concatenated_with_observer_info(self, x_obs=cas.DM.zeros(3, 1)):
        decolumnized_list = self.get_decolumnized_info_list()

        number_of_elements = self.number_of_elements
        observer_list = cas.repmat(x_obs, (1, number_of_elements))
        concatenated_list = cas.vertcat(decolumnized_list, observer_list)

        return concatenated_list

    def define_model_variables_to_info_function(self, model_variables, model_parameters):
        for elem in self.__list:
            if elem.info_fun is not None:
                elem.define_model_variables_to_info_function(model_variables, model_parameters)

        return None

    def define_biot_savart_induction_function(self):
        if len(self.__list) > 0:
            elem = self.__list[0]

            if elem.biot_savart_fun is None:
                elem.define_biot_savart_induction_function()

            biot_savart_fun = elem.biot_savart_fun
            self.set_biot_savart_fun(biot_savart_fun)

            number_sym = self.get_number_of_symbolics_for_concatenated_biot_savart()
            concatenated_sym = cas.SX.sym('concatenated_sym', number_sym)
            elem_info = self.get_element_info_from_concatenated_inputs(concatenated_sym)
            x_obs = self.get_observer_info_from_concatenated_inputs(concatenated_sym)

            biot_savart_output = biot_savart_fun(elem_info, x_obs)

            concatenated_biot_savart_fun = cas.Function('concatenated_biot_savart_fun', [concatenated_sym], [biot_savart_output])
            self.set_concatenated_biot_savart_fun(concatenated_biot_savart_fun)

        else:
            message = 'unable to define the biot_savart_induction_function for the vortex element list of type ' + self.element_type
            awelogger.logger.error(message)
            raise Exception(message)

        return None

    def evaluate_biot_savart_induction_for_all_elements(self, x_obs=cas.DM.zeros(3, 1)):

        print_op.warn_about_temporary_functionality_removal(location='element_list.projected_biot_savart')
        print_op.warn_about_temporary_functionality_removal(location='element_list.repeated_biot_savart')

        if self.concatenated_biot_savart_fun is None:
            self.define_biot_savart_induction_function()

        concatenated_biot_savart_fun = self.__concatenated_biot_savart_fun
        concatenated_list = self.get_decolumnized_list_concatenated_with_observer_info(x_obs)

        number_of_elements = self.number_of_elements
        concatenated_biot_savart_map = concatenated_biot_savart_fun.map(number_of_elements, 'openmp')
        all = concatenated_biot_savart_map(concatenated_list)

        return all

    def evaluate_total_biot_savart_induction(self, x_obs=cas.DM.zeros(3, 1)):
        all = self.evaluate_biot_savart_induction_for_all_elements(x_obs)
        u_ind = cas.sum2(all)
        return u_ind

    def get_max_abs_strength(self):
        if self.__number_of_elements > 0:
            all_strengths = [elem.info_dict['strength'] for elem in self.__list]
            strengths_are_numeric = [vect_op.is_numeric(strength) for strength in all_strengths]
            if all(strengths_are_numeric):
                strengths_array = np.array(all_strengths)
                return np.max(np.abs(strengths_array))

        message = 'could not compute a numeric max-abs-strength for this element_list. proceeding with a unit value'
        awelogger.logger.warning(message)
        return 1.

    def construct_fake_cosmetics(self):
        if self.__number_of_elements > 0:
            example_element = self.__list[0]
            cosmetics = example_element.construct_fake_cosmetics()

            max_abs_strength = self.get_max_abs_strength()
            cosmetics['trajectory']['circulation_max_estimate'] = max_abs_strength
            return cosmetics

        return None

    def draw(self, ax, side, variables_scaled=None, parameters=None, cosmetics=None):

        if cosmetics is None:
            cosmetics = self.construct_fake_cosmetics()

        for elem in self.__list:
            elem.draw(ax, side, variables_scaled, parameters, cosmetics)
        return None

    def abs_strength_max(self, variables_scaled, parameters):

        all_strengths = np.array([elem.unpack_info(external_info=elem.evaluate_info(variables_scaled, parameters))['strength'] for elem in self.__list])
        return np.max(np.abs(all_strengths))

    def is_element_in_list(self, query_elem, epsilon=1.e-8):

        if query_elem in self.__list:
            return True

        else:
            for elem in self.__list:
                if elem.is_equal(query_elem, epsilon):
                    return True

        return False

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
        awelogger.logger.error('Cannot set expected_number_of_elements object.')
        return None

    def set_expected_number_of_elements(self, value):
        if self.__expected_number_of_elements is not None:
            awelogger.logger.error('Cannot set expected_number_of_elements object.')
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

    def set_biot_savart_fun(self, value):
        self.__biot_savart_fun = value

    @property
    def concatenated_biot_savart_fun(self):
        return self.__concatenated_biot_savart_fun

    @concatenated_biot_savart_fun.setter
    def concatenated_biot_savart_fun(self, value):
        awelogger.logger.error('Cannot set concatenated_biot_savart_fun object.')

    def set_concatenated_biot_savart_fun(self, value):
        self.__concatenated_biot_savart_fun = value

    @property
    def model_induction_fun(self):
        return self.__model_induction_fun

    @model_induction_fun.setter
    def model_induction_fun(self, value):
        self.__model_induction_fun = value

    @property
    def model_induction_factor_fun(self):
        return self.__model_induction_factor_fun

    @model_induction_factor_fun.setter
    def model_induction_factor_fun(self, value):
        self.__model_induction_factor_fun = value

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


####### test a basic list

def construct_test_filament_list():
    filament_list = ElementList()

    r_core = 1.e-10

    x_start0 = 0. * vect_op.xhat_np()
    x_end0 = x_start0 + 5. * vect_op.xhat_np()
    strength0 = 3.
    dict_info0 = {'x_start': x_start0,
                  'x_end': x_end0,
                  'r_core': r_core,
                  'strength': strength0}

    fil0 = obj_finite_filament.FiniteFilament(dict_info0)
    filament_list.append(fil0)

    x_start1 = x_end0
    x_end1 = x_start1 + 5. * vect_op.zhat_np()
    strength1 = 1.
    dict_info1 = {'x_start': x_start1,
                  'x_end': x_end1,
                  'r_core': r_core,
                  'strength': strength1}

    fil1 = obj_finite_filament.FiniteFilament(dict_info1)
    filament_list.append(fil1)

    x_start2 = x_end1
    x_end2 = x_start2 + 5. * vect_op.yhat_np()
    strength2 = -3.
    dict_info2 = {'x_start': x_start2,
                  'x_end': x_end2,
                  'r_core': r_core,
                  'strength': strength2}

    fil2 = obj_finite_filament.FiniteFilament(dict_info2)
    filament_list.append(fil2)

    return filament_list

def test_filament_list_columnization(filament_list):
    columnized_info_list = filament_list.get_columnized_info_list()
    assert (columnized_info_list.shape == (8 * 3, 1))
    return None

def test_filament_list_biot_savart_at_default_observer(filament_list, epsilon=1.e-4):

    x_obs = cas.DM.zeros((3, 1))
    found = filament_list.evaluate_total_biot_savart_induction()

    expected = cas.DM.zeros((3, 1))
    for fdx in range(3):
        fil = filament_list.list[fdx]
        fil.define_biot_savart_induction_function()
        vec_u_ind = fil.biot_savart_fun(fil.info, x_obs)
        expected += vec_u_ind

    diff = expected - found
    criteria = (cas.mtimes(diff.T, diff) < epsilon**2.)

    if not criteria:
        message = 'vortex element list: something went wrong when computing the total induced velocity at the default observer (origin)'
        awelogger.logger.error(message)
        raise Exception(message)

    return None

def test_filament_list_biot_savart_summation(filament_list, epsilon=1.e-4):

    x_obs = 3. * vect_op.xhat_dm() + 2. * vect_op.yhat_dm()

    found = filament_list.evaluate_total_biot_savart_induction(x_obs=x_obs)

    expected = cas.DM.zeros((3, 1))
    for fdx in range(3):
        fil = filament_list.list[fdx]
        fil.define_biot_savart_induction_function()
        vec_u_ind = fil.biot_savart_fun(fil.info, x_obs)
        expected += vec_u_ind

    diff = expected - found
    criteria = (cas.mtimes(diff.T, diff) < epsilon ** 2.)

    if not criteria:
        message = 'vortex element list: something went wrong when summing the total induced velocity at a specified observer'
        awelogger.logger.error(message)
        raise Exception(message)

    return None


def test_filament_list_biot_savart_computation(filament_list, epsilon=1.e-4):

    x_obs = 3. * vect_op.xhat_dm() + 2. * vect_op.yhat_dm()

    found = filament_list.evaluate_total_biot_savart_induction(x_obs=x_obs)

    vec_u_ind_0 = 0.183723 * vect_op.zhat_np()
    vec_u_ind_1 = -0.0173158 * vect_op.xhat_np() - 0.0173158 * vect_op.yhat_np()
    vec_u_ind_2 = (+0.0343618 * vect_op.xhat_np() - 0.0137447 * vect_op.zhat_np())

    expected = vec_u_ind_0 + vec_u_ind_1 + vec_u_ind_2

    diff = expected - found
    criteria = (cas.mtimes(diff.T, diff) < epsilon ** 2.)

    if not criteria:
        message = 'vortex element list: something went wrong when computing the total induced velocity at a specified observer'
        awelogger.logger.error(message)
        raise Exception(message)

    return None



def test_filament_list():
    filament_list = construct_test_filament_list()
    test_filament_list_columnization(filament_list)
    test_filament_list_biot_savart_at_default_observer(filament_list)
    test_filament_list_biot_savart_summation(filament_list)
    test_filament_list_biot_savart_computation(filament_list)
    return None

###### appending tests

def test_appending_multiple_times():
    filament_list = ElementList()

    fil = obj_finite_filament.construct_test_object(r_core=0.01)
    filament_list.append(fil)
    filament_list.append(fil)

    fil2 = obj_finite_filament.construct_test_object(r_core=1.)
    filament_list.append(fil2)

    found = filament_list.number_of_elements
    expected = 3
    criteria = (found == expected)

    if not criteria:
        message = 'something went wrong when appending multiple vortex objects of the same type to a vortex list'
        awelogger.logger.error(message)
        raise Exception(message)


def test_is_elem_in_list():
    filament_list = ElementList()

    fil = obj_finite_filament.construct_test_object(r_core=0.01)
    filament_list.append(fil)

    fil2 = obj_finite_filament.construct_test_object(r_core=1.)

    fil3 = obj_finite_filament.construct_test_object(r_core=0.01)

    condition_1 = (filament_list.is_element_in_list(fil) == True)
    condition_2 = (filament_list.is_element_in_list(fil2) == False)
    condition_3 = (filament_list.is_element_in_list(fil3) == True)
    criteria = condition_1 and condition_2 and condition_3

    if not criteria:
        message = 'the query "is this element in the element list" does not work as expected.'
        awelogger.logger.error(message)
        raise Exception(message)

    return None

def test_that_appending_different_types_is_ignored():

    filament_list = ElementList()
    fil = obj_finite_filament.construct_test_object(r_core = 0.01)
    filament_list.append(fil)

    tan_cyl = obj_semi_infinite_tangential_cylinder.construct_test_object()
    filament_list.append(tan_cyl)

    found = filament_list.number_of_elements
    expected = 1
    criteria = (found == expected)

    if not criteria:
        message = 'something went wrong when (trying to) append multiple vortex objects of different type to a vortex list'
        awelogger.logger.error(message)
        raise Exception(message)

    return None

def test_appending():
    test_appending_multiple_times()
    test_is_elem_in_list()
    test_that_appending_different_types_is_ignored()

    return None

#############

def test():
    test_filament_list()
    test_appending()

# test()