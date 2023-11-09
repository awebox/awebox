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
import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.semi_infinite_filament as obj_semi_infinite_filament
import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.semi_infinite_tangential_right_cylinder as obj_semi_infinite_tangential_right_cylinder
import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.semi_infinite_longitudinal_right_cylinder as obj_semi_infinite_longitudinal_right_cylinder

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

        self.set_biot_savart_fun(None)
        self.set_concatenated_biot_savart_fun(None)
        self.set_biot_savart_residual_fun(None)
        self.set_concatenated_biot_savart_residual_fun(None)


    def append(self, added_elem, suppress_type_incompatibility_warning=False):

        is_element_list = isinstance(added_elem, ElementList)
        is_element = isinstance(added_elem, obj_element.Element)
        is_correct_type = (self.__element_type is None) or (self.__element_type == added_elem.element_type)
        has_correct_length = (self.__element_info_length is None) or (is_element and (self.__element_info_length == added_elem.info_length)) or (is_element_list and (self.__element_info_length == added_elem.element_info_length))

        if not is_element_list and not is_element:
            if not suppress_type_incompatibility_warning:
                message = 'tried to append to vortex element list, but proposed addition is neither a vortex element nor an element list. append instruction was skipped'
                awelogger.logger.warning(message)

        elif is_element_list and is_correct_type:
            for indiv_elem in added_elem.list:
                self.append(indiv_elem)

        elif is_element_list and not is_correct_type:
            if not suppress_type_incompatibility_warning:
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
            if not suppress_type_incompatibility_warning:
                message = 'tried to append vortex element to element list, but the types were incompatible so append instruction was skipped.'
                awelogger.logger.warning(message)

        elif is_element and is_correct_type and not has_correct_length:
            if not suppress_type_incompatibility_warning:
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
            print_op.log_and_raise_error(message)

        element_info_length = self.element_info_length
        expected_element_info_length = self.expected_element_info_length

        if (expected_element_info_length is not None) and (not element_info_length == expected_element_info_length):
            message = 'unexpected info length for vortex element in list'
            print_op.log_and_raise_error(message)

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
            print_op.log_and_raise_error(message)

        return None

    def get_number_of_symbolics_for_concatenated_biot_savart(self):
        number_symbolics = self.__expected_element_info_length + 3
        return number_symbolics

    def get_number_of_symbolics_for_concatenated_biot_savart_residual(self, biot_savart_residual_assembly='split'):
        number_symbolics = self.__expected_element_info_length + 6

        if biot_savart_residual_assembly == 'lifted':
            number_symbolics += 4

        return number_symbolics

    def get_element_info_from_concatenated_inputs(self, concatenated):
        elem_info = concatenated[:self.__expected_element_info_length]
        return elem_info

    def get_observer_info_from_concatenated_inputs(self, concatenated):
        info_dim = self.dimension_of_the_concatenated_list_that_stores_entry_info()
        if concatenated.shape[info_dim] >= self.get_number_of_symbolics_for_concatenated_biot_savart():
            obs_info = concatenated[self.__expected_element_info_length:self.__expected_element_info_length+3]
        else:
            message = 'trying to pull observer info from an unconcatenated list.'
            print_op.log_and_raise_error(message)

        return obs_info

    def get_velocity_info_from_concatenated_inputs(self, concatenated, biot_savart_residual_assembly='split'):
        info_dim = self.dimension_of_the_concatenated_list_that_stores_entry_info()
        if concatenated.shape[info_dim] >= self.get_number_of_symbolics_for_concatenated_biot_savart_residual(biot_savart_residual_assembly):
            obs_info = concatenated[self.__expected_element_info_length+3:self.__expected_element_info_length+6]
        else:
            message = 'trying to pull velocity info from either an unconcatenated list, or one onto which velocity information was not concatenate.'
            print_op.log_and_raise_error(message)

        return obs_info

    def get_velocity_numerator_info_from_concatenated_inputs(self, concatenated, biot_savart_residual_assembly='split'):
        info_dim = self.dimension_of_the_concatenated_list_that_stores_entry_info()
        if concatenated.shape[info_dim] >= self.get_number_of_symbolics_for_concatenated_biot_savart_residual(biot_savart_residual_assembly):
            obs_info = concatenated[self.__expected_element_info_length+6:self.__expected_element_info_length+9]
        else:
            message = 'trying to pull numerator info from either an unconcatenated list, or one onto which velocity information was not concatenate.'
            print_op.log_and_raise_error(message)

        return obs_info


    def get_velocity_denominator_info_from_concatenated_inputs(self, concatenated, biot_savart_residual_assembly='split'):
        info_dim = self.dimension_of_the_concatenated_list_that_stores_entry_info()
        if concatenated.shape[info_dim] >= self.get_number_of_symbolics_for_concatenated_biot_savart_residual(biot_savart_residual_assembly):
            obs_info = concatenated[self.__expected_element_info_length+9:self.__expected_element_info_length+10]
        else:
            message = 'trying to pull denominator info from either an unconcatenated list, or one onto which velocity information was not concatenate.'
            print_op.log_and_raise_error(message)

        return obs_info


    def get_decolumnized_list_concatenated_with_observer_info(self, x_obs=cas.DM.zeros(3, 1)):
        decolumnized_list = self.get_decolumnized_info_list()

        number_of_elements = self.number_of_elements
        observer_list = cas.repmat(x_obs, (1, number_of_elements))
        concatenated_list = cas.vertcat(decolumnized_list, observer_list)

        return concatenated_list

    def dimension_of_the_concatenated_list_that_stores_entries(self):
        return 1

    def dimension_of_the_concatenated_list_that_stores_entry_info(self):
        return 0

    def get_concatenated_list_concatenated_with_velocity_info(self, concatenated_list, vec_u_ind_list):

        entry_dim = self.dimension_of_the_concatenated_list_that_stores_entries()
        info_dim = self.dimension_of_the_concatenated_list_that_stores_entry_info()

        both_lists_have_the_same_number_of_entries = (concatenated_list.shape[entry_dim] == vec_u_ind_list.shape[entry_dim])
        entry_velocity_is_given_in_three_dimensions = (vec_u_ind_list.shape[info_dim] == 3)

        if both_lists_have_the_same_number_of_entries and entry_velocity_is_given_in_three_dimensions:
            concatenated_list = cas.vertcat(concatenated_list, vec_u_ind_list)
        else:
            message = 'unable to concatenate vec_u_ind_list onto a concatenated list, due to a dimension error. maybe check that the velocity list has the individual entry velocities stored in columns?'
            print_op.log_and_raise_error(message)

        return concatenated_list

    def get_concatenated_list_concatenated_with_numerator_and_denominator_info(self, concatenated_list, vec_u_ind_num_list, vec_u_ind_den_list):

        entry_dim = self.dimension_of_the_concatenated_list_that_stores_entries()
        info_dim = self.dimension_of_the_concatenated_list_that_stores_entry_info()

        num_list_has_the_same_number_of_entries = (concatenated_list.shape[entry_dim] == vec_u_ind_num_list.shape[entry_dim])
        den_list_has_the_same_number_of_entries = (
                    concatenated_list.shape[entry_dim] == vec_u_ind_den_list.shape[entry_dim])

        biot_savart_numerators_is_given_in_three_dimensions = (vec_u_ind_num_list.shape[info_dim] == 3)
        biot_savart_denominators_is_given_in_one_dimension = (vec_u_ind_den_list.shape[info_dim] == 1)

        if num_list_has_the_same_number_of_entries and biot_savart_numerators_is_given_in_three_dimensions:
            concatenated_list = cas.vertcat(concatenated_list, vec_u_ind_num_list)
        else:
            message = 'unable to concatenate vec_u_ind_num_list onto a concatenated list, due to a dimension error. maybe check that the numerator list has the individual entry velocities stored in columns?'
            print_op.log_and_raise_error(message)

        if den_list_has_the_same_number_of_entries and biot_savart_denominators_is_given_in_one_dimension:
            concatenated_list = cas.vertcat(concatenated_list, vec_u_ind_den_list)
        else:
            message = 'unable to concatenate vec_u_ind_den_list onto a concatenated list, due to a dimension error.'
            print_op.log_and_raise_error(message)

        return concatenated_list


    # def define_model_variables_to_info_function(self, model_variables, model_parameters):
    #     for elem in self.__list:
    #         if elem.info_fun is not None:
    #             elem.define_model_variables_to_info_function(model_variables, model_parameters)
    #
    #     return None



    def define_model_variables_to_info_functions(self, model_variables, model_parameters):
        for elem in self.__list:
            if elem.info_fun is None:
                elem.define_model_variables_to_info_function(model_variables, model_parameters)
        return None


    def define_biot_savart_induction_function(self):
        elem = self.get_example_element()

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

        return None


    def define_biot_savart_induction_residual_function(self, biot_savart_residual_assembly='split'):
        elem = self.get_example_element()

        if elem.biot_savart_residual_fun is None:
            elem.define_biot_savart_induction_residual_function(biot_savart_residual_assembly)

        biot_savart_residual_fun = elem.biot_savart_residual_fun
        self.set_biot_savart_residual_fun(biot_savart_residual_fun)

        number_sym = self.get_number_of_symbolics_for_concatenated_biot_savart_residual(biot_savart_residual_assembly)
        concatenated_sym = cas.SX.sym('concatenated_sym', number_sym)
        elem_info = self.get_element_info_from_concatenated_inputs(concatenated_sym)
        x_obs = self.get_observer_info_from_concatenated_inputs(concatenated_sym)
        vec_u_ind = self.get_velocity_info_from_concatenated_inputs(concatenated_sym)

        if biot_savart_residual_assembly == 'lifted':
            vec_u_ind_num = self.get_velocity_numerator_info_from_concatenated_inputs(concatenated_sym, biot_savart_residual_assembly)
            vec_u_ind_den = self.get_velocity_denominator_info_from_concatenated_inputs(concatenated_sym, biot_savart_residual_assembly)

            biot_savart_residual_output = biot_savart_residual_fun(elem_info, x_obs, vec_u_ind, vec_u_ind_num, vec_u_ind_den)

        else:
            biot_savart_residual_output = biot_savart_residual_fun(elem_info, x_obs, vec_u_ind)

        concatenated_biot_savart_residual_fun = cas.Function('concatenated_biot_savart_residual_fun', [concatenated_sym], [biot_savart_residual_output])
        self.set_concatenated_biot_savart_residual_fun(concatenated_biot_savart_residual_fun)

        return None

    def get_example_element(self):
        if len(self.__list) > 0:
            return self.__list[0]
        else:
            message = 'the current task requires an example vortex element, but we cannot retrieve an example vortex element from this element list, because this list does not yet have any elements.'
            print_op.log_and_raise_error(message)


    def evaluate_biot_savart_induction_for_all_elements(self, x_obs=cas.DM.zeros(3, 1), variables_scaled=None, parameters=None):

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

    def evaluate_biot_savart_induction_residual_for_all_elements(self, x_obs, vec_u_ind_list, vec_u_ind_num_list=None, vec_u_ind_den_list=None, biot_savart_residual_assembly='split'):

        if self.concatenated_biot_savart_residual_fun is None:
            self.define_biot_savart_induction_residual_function(biot_savart_residual_assembly)

        concatenated_biot_savart_residual_fun = self.__concatenated_biot_savart_residual_fun
        concatenated_list = self.get_decolumnized_list_concatenated_with_observer_info(x_obs)
        concatenated_list = self.get_concatenated_list_concatenated_with_velocity_info(concatenated_list, vec_u_ind_list)

        if biot_savart_residual_assembly == 'lifted':
            concatenated_list = self.get_concatenated_list_concatenated_with_numerator_and_denominator_info(concatenated_list, vec_u_ind_num_list, vec_u_ind_den_list)

        number_of_elements = self.number_of_elements
        concatenated_biot_savart_residual_map = concatenated_biot_savart_residual_fun.map(number_of_elements, 'openmp')
        all = concatenated_biot_savart_residual_map(concatenated_list)

        return all

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
        example_element = self.get_example_element()
        cosmetics = example_element.construct_fake_cosmetics()

        max_abs_strength = self.get_max_abs_strength()
        cosmetics['trajectory']['circulation_max_estimate'] = max_abs_strength
        return cosmetics

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
        print_op.log_and_raise_error('Cannot set list object.')

    @property
    def number_of_elements(self):
        return self.__number_of_elements

    @number_of_elements.setter
    def number_of_elements(self, value):
        print_op.log_and_raise_error('Cannot set number_of_elements object.')

    @property
    def expected_number_of_elements(self):
        return self.__expected_number_of_elements

    @expected_number_of_elements.setter
    def expected_number_of_elements(self, value):
        print_op.log_and_raise_error('Cannot set expected_number_of_elements object.')
        return None

    def set_expected_number_of_elements(self, value):
        if self.__expected_number_of_elements is not None:
            print_op.log_and_raise_error('Cannot set expected_number_of_elements object.')
        else:
            self.__expected_number_of_elements = value
        return None


    @property
    def element_info_length(self):
        return self.__element_info_length

    @element_info_length.setter
    def element_info_length(self, value):
        print_op.log_and_raise_error('Cannot set element_info_length object.')

    @property
    def biot_savart_fun(self):
        return self.__biot_savart_fun

    @biot_savart_fun.setter
    def biot_savart_fun(self, value):
        print_op.log_and_raise_error('Cannot set biot_savart_fun object.')

    def set_biot_savart_fun(self, value):
        self.__biot_savart_fun = value

    @property
    def concatenated_biot_savart_fun(self):
        return self.__concatenated_biot_savart_fun

    @concatenated_biot_savart_fun.setter
    def concatenated_biot_savart_fun(self, value):
        print_op.log_and_raise_error('Cannot set concatenated_biot_savart_fun object.')

    def set_concatenated_biot_savart_fun(self, value):
        self.__concatenated_biot_savart_fun = value

    @property
    def biot_savart_residual_fun(self):
        return self.__biot_savart_residual_fun

    @biot_savart_residual_fun.setter
    def biot_savart_residual_fun(self, value):
        print_op.log_and_raise_error('Cannot set biot_savart_residual_fun object.')

    def set_biot_savart_residual_fun(self, value):
        self.__biot_savart_residual_fun = value

    @property
    def concatenated_biot_savart_residual_fun(self):
        return self.__concatenated_biot_savart_residual_fun

    @concatenated_biot_savart_residual_fun.setter
    def concatenated_biot_savart_residual_fun(self, value):
        print_op.log_and_raise_error('Cannot set concatenated_biot_savart_residual_fun object.')

    def set_concatenated_biot_savart_residual_fun(self, value):
        self.__concatenated_biot_savart_residual_fun = value

    @property
    def element_type(self):
        return self.__element_type

    @element_type.setter
    def element_type(self, value):
        print_op.log_and_raise_error('Cannot set element_type object.')

    def set_element_type(self, value):
        self.__element_type = value

    @property
    def expected_element_info_length(self):
        return self.__expected_element_info_length

    @expected_element_info_length.setter
    def expected_element_info_length(self, value):
        print_op.log_and_raise_error('Cannot set expected_element_info_length object.')
        return None

    def set_expected_element_info_length(self, value):
        if self.__expected_element_info_length is not None:
            print_op.log_and_raise_error('Cannot set expected_element_info_length object.')
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
        print_op.log_and_raise_error(message)

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
        print_op.log_and_raise_error(message)

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
        print_op.log_and_raise_error(message)

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
        print_op.log_and_raise_error(message)

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
        print_op.log_and_raise_error(message)

    return None

def test_that_appending_different_types_is_ignored():

    filament_list = ElementList()
    fil = obj_finite_filament.construct_test_object(r_core = 0.01)
    filament_list.append(fil)

    tan_cyl = obj_semi_infinite_tangential_right_cylinder.construct_test_object()
    filament_list.append(tan_cyl, suppress_type_incompatibility_warning=True)

    found = filament_list.number_of_elements
    expected = 1
    criteria = (found == expected)

    if not criteria:
        message = 'something went wrong when (trying to) append multiple vortex objects of different type to a vortex list'
        print_op.log_and_raise_error(message)

    return None

def test_appending():
    test_appending_multiple_times()
    test_is_elem_in_list()
    test_that_appending_different_types_is_ignored()

    return None

#########

def construct_test_semi_infinite_filament_list():
    filament_list = ElementList()

    fil0 = obj_semi_infinite_filament.construct_test_object()
    filament_list.append(fil0)

    x_start1 = 5. * vect_op.zhat_np()
    l_hat = vect_op.yhat_dm()
    r_core = 1.e-10
    strength1 = 1.
    dict_info1 = {'x_start': x_start1,
                  'l_hat': l_hat,
                  'r_core': r_core,
                  'strength': strength1}

    fil1 = obj_semi_infinite_filament.SemiInfiniteFilament(dict_info1)
    filament_list.append(fil1)

    return filament_list

def construct_test_semi_infinite_tangential_right_cylinder_list():
    cyl_list = ElementList()

    cyl0 = obj_semi_infinite_tangential_right_cylinder.construct_test_object()
    cyl_list.append(cyl0)

    x_center = cas.DM([1., 1., 1.])
    radius = 4.
    l_start = -2.
    l_hat = vect_op.zhat_np()

    epsilon_m = 10. ** (-5.)
    epsilon_r = 1.

    strength = 3.
    unpacked = {'x_center': x_center,
                'l_hat': l_hat,
                'radius': radius,
                'l_start': l_start,
                'epsilon_m': epsilon_m,
                'epsilon_r': epsilon_r,
                'strength': strength
                }

    cyl1 = obj_semi_infinite_tangential_right_cylinder.SemiInfiniteTangentialRightCylinder(unpacked)
    cyl_list.append(cyl1)

    return cyl_list

def construct_test_semi_infinite_longitudinal_right_cylinder_list():
    cyl_list = ElementList()

    cyl0 = obj_semi_infinite_longitudinal_right_cylinder.construct_test_object()
    cyl_list.append(cyl0)

    x_center = cas.DM([1., 1., 1.])
    radius = 4.
    l_start = -2.
    l_hat = vect_op.zhat_np()

    epsilon_m = 10. ** (-5.)
    epsilon_r = 1.

    strength = 3.
    unpacked = {'x_center': x_center,
                'l_hat': l_hat,
                'radius': radius,
                'l_start': l_start,
                'epsilon_m': epsilon_m,
                'epsilon_r': epsilon_r,
                'strength': strength
                }

    cyl1 = obj_semi_infinite_longitudinal_right_cylinder.SemiInfiniteLongitudinalRightCylinder(unpacked)
    cyl_list.append(cyl1)

    return cyl_list


def test_that_biot_savart_function_evaluates_differently_for_different_elements_for_given_list(element_list, epsilon=1.e-4):

    number_of_elements = element_list.number_of_elements

    if number_of_elements < 2:
        message = 'cannot test that the biot-savart function actually evaluates differently on different elements, because the given element list has less than two elements.'
        print_op.log_and_raise_error(message)

    element_is_different = [not element_list.list[edx].is_equal(element_list.list[0]) for edx in range(1, number_of_elements)]
    if not any(element_is_different):
        message = 'cannot test that the biot-savart function actually evaluates differently on different elements, because the elements of the given list are not actually different.'
        print_op.log_and_raise_error(message)

    x_obs = cas.DM([1., 2., 3.])
    all_vec_u = element_list.evaluate_biot_savart_induction_for_all_elements(x_obs)
    vec_u_is_different = [cas.mtimes((all_vec_u[:, edx] - all_vec_u[:, 0]).T, (all_vec_u[:, edx] - all_vec_u[:, 0])) > epsilon**2. for edx in range(1, number_of_elements)]
    condition = any(vec_u_is_different)

    criteria = condition

    if not criteria:
        message = 'something went wrong when building the biot-savart function for a ' + element_list.element_type + ' element list: the biot-savart function does not evaluate differently for different elements'
        print_op.log_and_raise_error(message)

    return None

def test_that_biot_savart_function_evaluates_differently_for_different_elements(epsilon=1.e-4):

    filament_list = construct_test_filament_list()
    test_that_biot_savart_function_evaluates_differently_for_different_elements_for_given_list(filament_list, epsilon)

    semi_infinite_filament_list = construct_test_semi_infinite_filament_list()
    test_that_biot_savart_function_evaluates_differently_for_different_elements_for_given_list(semi_infinite_filament_list, epsilon)

    si_long_right_cylinder_list = construct_test_semi_infinite_longitudinal_right_cylinder_list()
    test_that_biot_savart_function_evaluates_differently_for_different_elements_for_given_list(si_long_right_cylinder_list, epsilon)

    si_tan_right_cylinder_list = construct_test_semi_infinite_tangential_right_cylinder_list()
    test_that_biot_savart_function_evaluates_differently_for_different_elements_for_given_list(si_tan_right_cylinder_list, epsilon)

    return None

#############

def test(epsilon=1.e-4):
    test_filament_list()
    test_appending()
    test_that_biot_savart_function_evaluates_differently_for_different_elements(epsilon)

# test()