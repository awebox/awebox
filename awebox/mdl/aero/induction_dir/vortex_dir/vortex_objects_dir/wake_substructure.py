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
- authors: rachel leuthold 2022
'''

import casadi.tools as cas
import numpy as np

import awebox.tools.struct_operations as struct_op
import awebox.tools.vector_operations as vect_op
import awebox.tools.print_operations as print_op

import awebox.mdl.aero.induction_dir.vortex_dir.tools as vortex_tools

import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.element as obj_element
import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.element_list as obj_element_list
import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.finite_filament as obj_fin_fli
import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.semi_infinite_longitudinal_right_cylinder as obj_si_long_right_cyl

from awebox.logger.logger import Logger as awelogger

import matplotlib
from awebox.viz.plot_configuration import DEFAULT_MPL_BACKEND
matplotlib.use(DEFAULT_MPL_BACKEND)

class WakeSubstructure():
    def __init__(self, substructure_type=None):

        self.__substructure_type = substructure_type

        self.__finite_filament_list = None
        self.__semi_infinite_filament_list = None
        self.__semi_infinite_tangential_right_cylinder_list = None
        self.__semi_infinite_longitudinal_right_cylinder_list = None

        self.__mapped_biot_savart_fun_dict = {}
        self.__mapped_biot_savart_residual_fun_dict = {}
        for element_type in self.get_accepted_element_types():
            self.set_mapped_biot_savart_fun(element_type, None)
            self.set_mapped_biot_savart_residual_fun(element_type, None)

    def mapped_biot_savart_function_is_defined_for_initialized_lists(self):
        return all([(self.get_mapped_biot_savart_fun(elem_type) is not None) for elem_type in self.get_initialized_element_types()])

    def mapped_biot_savart_residual_function_is_defined_for_initialized_lists(self):
        return all([(self.get_mapped_biot_savart_residual_fun(elem_type) is not None) for elem_type in self.get_initialized_element_types()])

    def define_biot_savart_induction_functions(self):

        x_obs = cas.SX.sym('x_obs', (3, 1))

        initialized_types = self.get_initialized_element_types()
        for element_type in initialized_types:
            if self.get_mapped_biot_savart_fun(element_type) is None:
                self.get_list(element_type).define_biot_savart_induction_function()
                all = self.get_list(element_type).evaluate_biot_savart_induction_for_all_elements(x_obs)

                mapped_biot_savart_fun = cas.Function('mapped_biot_savart_fun', [x_obs], [all])
                self.set_mapped_biot_savart_fun(element_type, mapped_biot_savart_fun)

        return None

    def define_biot_savart_induction_residual_functions(self, degree_of_induced_velocity_lifting=3):

        x_obs = cas.SX.sym('x_obs', (3, 1))

        initialized_types = self.get_initialized_element_types()
        for element_type in initialized_types:
            if self.get_mapped_biot_savart_residual_fun(element_type) is None:
                number_of_elements = self.get_list(element_type).number_of_elements
                vec_u_ind_list = cas.SX.sym('vec_u_ind_list', (3, number_of_elements))

                self.get_list(element_type).define_biot_savart_induction_residual_function(degree_of_induced_velocity_lifting)

                if degree_of_induced_velocity_lifting == 1:
                    return None

                elif degree_of_induced_velocity_lifting == 2:
                    all = self.get_list(element_type).evaluate_biot_savart_induction_residual_for_all_elements(x_obs, vec_u_ind_list, degree_of_induced_velocity_lifting)
                    mapped_biot_savart_residual_fun = cas.Function('mapped_biot_savart_residual_fun', [x_obs, vec_u_ind_list], [all], {"allow_free": True})

                elif degree_of_induced_velocity_lifting == 3:
                    vec_u_ind_num_list = cas.SX.sym('vec_u_ind_num_list', (3, number_of_elements))
                    vec_u_ind_den_list = cas.SX.sym('vec_u_ind_den_list', (1, number_of_elements))

                    all = self.get_list(element_type).evaluate_biot_savart_induction_residual_for_all_elements(x_obs, vec_u_ind_list,
                                                                                                               vec_u_ind_num_list,
                                                                                                               vec_u_ind_den_list,
                                                                                                               degree_of_induced_velocity_lifting)
                    mapped_biot_savart_residual_fun = cas.Function('mapped_biot_savart_residual_fun',
                                                                   [x_obs, vec_u_ind_list, vec_u_ind_num_list, vec_u_ind_den_list], [all], {"allow_free": True})

                self.set_mapped_biot_savart_residual_fun(element_type, mapped_biot_savart_residual_fun)

        return None


    def define_model_variables_to_info_functions(self, model_variables, model_parameters):
        initalized_types = self.get_initialized_element_types()
        for element_type in initalized_types:
            self.get_list(element_type).define_model_variables_to_info_functions(model_variables, model_parameters)
        return None


    def evaluate_total_biot_savart_induction(self, x_obs=cas.DM.zeros(3, 1)):
        vec_u_ind = cas.DM.zeros((3, 1))
        local = cas.DM.zeros((3, 1)) #just in case there are no elements,
        # ie. far-wake = not_in_use and wake_nodes = 1 and number_of_kites = 1

        for element_type in self.get_initialized_element_types():
            elem_list = self.get_list(element_type)
            local = elem_list.evaluate_total_biot_savart_induction(x_obs)
        vec_u_ind += local
        return vec_u_ind


    def calculate_total_biot_savart_at_x_obs(self, variables_scaled, parameters, x_obs=cas.DM.zeros((3, 1))):
        vec_u_ind = cas.DM.zeros((3, 1))
        for element_type in self.get_initialized_element_types():
            elem_list = self.get_list(element_type)
            number_of_elements = elem_list.number_of_elements
            for edx in range(number_of_elements):
                elem = elem_list.list[edx]
                unpacked, cosmetics = elem.prepare_to_draw(variables_scaled, parameters, {})
                value, _, _ = elem.calculate_biot_savart_induction(unpacked, x_obs)
                vec_u_ind += value

        return vec_u_ind


    def construct_biot_savart_residual_at_kite(self, model_options, variables_si, kite_obs, architecture):
        resi = []
        for element_type in self.get_initialized_element_types():
            local_resi = self.construct_biot_savart_residual_at_kite_by_element_type(element_type, model_options, variables_si, kite_obs, architecture)
            resi = cas.horzcat(resi, local_resi)
        return resi


    def construct_biot_savart_residual_at_kite_by_element_type(self, element_type, model_options, variables_si, kite_obs, architecture):

        parent_obs = architecture.parent_map[kite_obs]

        degree_of_induced_velocity_lifting = model_options['aero']['vortex']['degree_of_induced_velocity_lifting']

        if not self.mapped_biot_savart_residual_function_is_defined_for_initialized_lists():
            self.define_biot_savart_induction_residual_functions(degree_of_induced_velocity_lifting)

        x_obs = struct_op.get_variable_from_model_or_reconstruction(variables_si, 'x', 'q' + str(kite_obs) + str(parent_obs))

        vec_u_ind_list = []
        vec_u_ind_num_list = []
        vec_u_ind_den_list = []
        number_of_elements = self.get_list(element_type).number_of_elements
        for edx in range(number_of_elements):

            # doesn't matter what values these defaults take, as long as they clearly do not satisfy residuals
            # but we need this here, otherwise the number_of_elements in the function map breaks.
            local_var = cas.DM.ones((3, 1)) * 999.
            local_num = cas.DM.ones((3, 1)) * 888.
            local_den = cas.DM.ones((1, 1)) * 777.

            if vortex_tools.not_bound_and_shed_is_obs(model_options, self.substructure_type, element_type, edx, kite_obs, architecture):
                local_var = vortex_tools.get_element_induced_velocity_si(variables_si, self.substructure_type, element_type, edx, kite_obs)

                if degree_of_induced_velocity_lifting == 3:
                    local_num = vortex_tools.get_element_induced_velocity_numerator_si(variables_si, self.substructure_type,
                                                                             element_type, edx, kite_obs)
                    local_den = vortex_tools.get_element_induced_velocity_denominator_si(variables_si, self.substructure_type,
                                                                             element_type, edx, kite_obs)

            vec_u_ind_list = cas.horzcat(vec_u_ind_list, local_var)
            vec_u_ind_num_list = cas.horzcat(vec_u_ind_num_list, local_num)
            vec_u_ind_den_list = cas.horzcat(vec_u_ind_den_list, local_den)

        if degree_of_induced_velocity_lifting == 1:
            resi_si = None
        if degree_of_induced_velocity_lifting == 2:
            resi_si = self.get_mapped_biot_savart_residual_fun(element_type)(x_obs, vec_u_ind_list)
        elif degree_of_induced_velocity_lifting == 3:
            resi_si = self.get_mapped_biot_savart_residual_fun(element_type)(x_obs, vec_u_ind_list, vec_u_ind_num_list, vec_u_ind_den_list)
        else:
            message = 'unexpected degree_of_induced_velocity_lifting (' + str(degree_of_induced_velocity_lifting) + ') found'
            print_op.log_and_raise_error(message)

        return resi_si

    def append(self, added_elem):

        is_list = isinstance(added_elem, list)
        if is_list:
            all_items_in_list_are_element_lists = all(
                [isinstance(elem, obj_element_list.ElementList) for elem in added_elem])
            all_items_in_list_are_elements = all(
                [isinstance(elem, obj_element.Element) for elem in added_elem])
            if not (all_items_in_list_are_element_lists or all_items_in_list_are_elements):
                message = 'the only type of list that can be appended to vortex wake_substructure objects are ElementLists, lists of Elements or lists of ElementLists.'
                print_op.log_and_raise_error(message)

        is_element_list = isinstance(added_elem, obj_element_list.ElementList)
        is_element = isinstance(added_elem, obj_element.Element)

        if is_element or is_element_list:
            element_type = added_elem.element_type
            if element_type is not None:
                self.initialize_list_on_first_use(element_type)
                self.get_list(element_type).append(added_elem)

        elif is_list and (all_items_in_list_are_elements or all_items_in_list_are_element_lists):
            for local_elem in added_elem:
                self.append(local_elem)
        else:
            message = 'something went wrong when trying to append to this vortex wake_substructure object.'
            print_op.log_and_raise_error(message)

        return None


    def initialize_list_on_first_use(self, element_type):

        if self.get_list(element_type) is None:
            if element_type == 'finite_filament':
                self.__finite_filament_list = obj_element_list.ElementList()

            elif element_type == 'semi_infinite_filament':
                self.__semi_infinite_filament_list = obj_element_list.ElementList()

            elif element_type == 'semi_infinite_tangential_right_cylinder':
                self.__semi_infinite_tangential_right_cylinder_list = obj_element_list.ElementList()

            elif element_type == 'semi_infinite_longitudinal_right_cylinder':
                self.__semi_infinite_longitudinal_right_cylinder_list = obj_element_list.ElementList()

            else:
                message = 'cannot recognize vortex object list for ' + element_type + ' objects'
                print_op.log_and_raise_error(message)

        return None

    def set_expected_number_of_elements(self, element_type, val):
        if self.get_list(element_type) is not None:
            self.get_list(element_type).set_expected_number_of_elements(val)
        return None

    def set_expected_number_of_elements_from_dict(self, dict):
        for elem_type, val in dict.items():
            self.set_expected_number_of_elements(elem_type, val)
        return None

    def has_at_least_one_element(self):
        initialized_types = self.get_initialized_element_types()
        for element_type in initialized_types:
            if self.get_list(element_type).number_of_elements > 0:
                return True
        else:
            return False

    def get_total_number_of_elements(self):
        initialized_types = self.get_initialized_element_types()
        total_number_of_elements = 0
        for element_type in initialized_types:
            total_number_of_elements += self.get_list(element_type).number_of_elements()

        return total_number_of_elements

    def confirm_all_lists_have_expected_dimensions(self, types_expected_to_be_initialized):

        initialized_types = self.get_initialized_element_types()
        criteria = (set(types_expected_to_be_initialized) == set(initialized_types))

        if not criteria:
            message = 'vortex wake: the set of vortex element types that are expected to be initialized is not the same as the set of the vortex element types that *are* initialized.'
            print_op.log_and_raise_error(message)

        for element_type in types_expected_to_be_initialized:
            self.get_list(element_type).confirm_list_has_expected_dimensions()

        return None

    def get_accepted_element_types(self):
        accepted_types = {'finite_filament',
                          'semi_infinite_filament',
                          'semi_infinite_tangential_right_cylinder',
                          'semi_infinite_longitudinal_right_cylinder'
                          }
        return accepted_types

    def has_initialized_element_types(self):
        initialized_types = self.get_initialized_element_types()
        return (len(initialized_types) > 0)

    def get_initialized_element_types(self):
        accepted_types = self.get_accepted_element_types()
        initialized_types = []
        for elem_type in accepted_types:
            if self.get_list(elem_type) is not None:
                initialized_types += [elem_type]

        return initialized_types

    def get_list(self, element_type):
        if element_type == 'finite_filament':
            return self.__finite_filament_list

        elif element_type == 'semi_infinite_filament':
            return self.__semi_infinite_filament_list

        elif element_type == 'semi_infinite_tangential_right_cylinder':
            return self.__semi_infinite_tangential_right_cylinder_list

        elif element_type == 'semi_infinite_longitudinal_right_cylinder':
            return self.__semi_infinite_longitudinal_right_cylinder_list

        else:
            message = 'unable to interpret the element type (' + element_type + '). maybe check your spelling.'
            print_op.log_and_raise_error(message)

    def get_max_abs_strength(self):
        if self.has_initialized_element_types():
            initialized_types = self.get_initialized_element_types()

            all_strengths = [self.get_list(elem_type).get_max_abs_strength() for elem_type in initialized_types]
            strengths_are_numeric = [vect_op.is_numeric(strength) for strength in all_strengths]
            if all(strengths_are_numeric):
                strengths_array = np.array(all_strengths)
                return np.max(np.abs(strengths_array))

        message = 'could not compute a numeric max-abs-strength for this wake_substructure. proceeding with a unit value'
        awelogger.logger.warning(message)
        return 1.

    def construct_fake_cosmetics(self):
        if self.has_initialized_element_types():
            initialized_types = self.get_initialized_element_types()

            example_element_type = initialized_types[0]
            example_list = self.get_list(example_element_type).list
            example_element = example_list[0]
            cosmetics = example_element.construct_fake_cosmetics()

            max_abs_strength = self.get_max_abs_strength()
            cosmetics['trajectory']['circulation_max_estimate'] = max_abs_strength
            return cosmetics

        return None

    def draw(self, ax, side, variables_scaled=None, parameters=None, cosmetics=None):

        if cosmetics is None:
            cosmetics = self.construct_fake_cosmetics()

        initialized_types = self.get_initialized_element_types()
        for elem_type in initialized_types:
            self.get_list(elem_type).draw(ax, side, variables_scaled, parameters, cosmetics)
        return None

    def get_mapped_biot_savart_fun(self, element_type):
        return self.__mapped_biot_savart_fun_dict[element_type]

    def set_mapped_biot_savart_fun(self, element_type, value):
        self.__mapped_biot_savart_fun_dict[element_type] = value

    def get_mapped_biot_savart_residual_fun(self, element_type):
        return self.__mapped_biot_savart_residual_fun_dict[element_type]

    def set_mapped_biot_savart_residual_fun(self, element_type, value):
        self.__mapped_biot_savart_residual_fun_dict[element_type] = value

    @property
    def substructure_type(self):
        return self.__substructure_type

    @substructure_type.setter
    def substructure_type(self, value):
        message = 'Cannot set substructure_type object.'
        print_op.log_and_raise_error(message)


def construct_test_object():
    local_substructure = WakeSubstructure()

    fil = obj_fin_fli.construct_test_object()
    local_substructure.append(fil)
    local_substructure.append(fil)

    long_cyl = obj_si_long_right_cyl.construct_test_object()
    local_substructure.append(long_cyl)

    return local_substructure

def test_append():
    local_substructure = construct_test_object()
    condition_1 = (local_substructure.get_list('finite_filament').number_of_elements == 2)
    condition_2 = (local_substructure.get_list('semi_infinite_filament') is None)
    condition_3 = (local_substructure.get_list('semi_infinite_tangential_right_cylinder') is None)
    condition_4 = (local_substructure.get_list('semi_infinite_longitudinal_right_cylinder').number_of_elements == 1)

    criteria = (condition_1 and condition_2 and condition_3 and condition_4)

    if not criteria:
        message = 'something went wrong when appending vortex objects to a wake_substructure object'
        print_op.log_and_raise_error(message)

    return None

def test_check_expected_dimensions():
    local_substructure = construct_test_object()
    number_of_elements_dict = {'finite_filament':2,
                               'semi_infinite_longitudinal_right_cylinder':1
                               }
    local_substructure.set_expected_number_of_elements_from_dict(number_of_elements_dict)

    initialized_types = number_of_elements_dict.keys()
    local_substructure.confirm_all_lists_have_expected_dimensions(initialized_types)

    return None

def test_mapped_biot_savart():
    local_substructure = construct_test_object()
    local_substructure.define_biot_savart_induction_functions()

    x_obs = cas.SX.sym('x_obs', (3, 1))

    initialized_types = local_substructure.get_initialized_element_types()
    not_initialized_types = set(local_substructure.get_accepted_element_types()) - set(initialized_types)

    conditions = {}
    total_conditions = 0
    for elem_type in initialized_types:
        mapped_biot_savart_fun = local_substructure.get_mapped_biot_savart_fun(elem_type)
        vec_u_ind_all = mapped_biot_savart_fun(x_obs)
        found_dimensions = vec_u_ind_all.shape
        expected_dimension = (3, local_substructure.get_list(elem_type).number_of_elements)

        local_condition = (found_dimensions == expected_dimension)
        conditions[elem_type] = local_condition
        total_conditions += local_condition

    for elem_type in not_initialized_types:
        mapped_biot_savart_fun = local_substructure.get_mapped_biot_savart_fun(elem_type)
        local_condition = (mapped_biot_savart_fun is None)
        conditions[elem_type] = local_condition
        total_conditions += local_condition

    criteria = (total_conditions == len(local_substructure.get_accepted_element_types()))

    if not criteria:
        message = 'something went wrong when trying to make mapped biot-savart functions for the wake_substructure object'
        print_op.log_and_raise_error(message)

    return None

def test_mapped_biot_savart_residual(epsilon=1.e-4):
    local_substructure = construct_test_object()

    for degree_of_induced_velocity_lifting in [2]:
        local_substructure.define_biot_savart_induction_functions()
        local_substructure.define_biot_savart_induction_residual_functions(degree_of_induced_velocity_lifting=degree_of_induced_velocity_lifting)

        x_obs = cas.DM([1., 2., 3.])

        conditions = {}
        total_conditions = 0
        initialized_types = local_substructure.get_initialized_element_types()
        for elem_type in initialized_types:
            mapped_biot_savart_fun = local_substructure.get_mapped_biot_savart_fun(elem_type)
            vec_u_ind_list = mapped_biot_savart_fun(x_obs)

            mapped_biot_savart_residual_fun = local_substructure.get_mapped_biot_savart_residual_fun(elem_type)
            resi = mapped_biot_savart_residual_fun(x_obs, vec_u_ind_list)

            resi_columnized = vect_op.columnize(resi)

            local_condition = all(local_resi**2. < epsilon**2. for local_resi in np.array(resi_columnized))
            conditions[elem_type] = local_condition
            total_conditions += local_condition

        criteria = (total_conditions == len(initialized_types))

        if not criteria:
            message = 'something went wrong with the biot-savart or biot-savart residuals or their mapping: the output of the mapped biot-savart function does not satisfy the mapped biot-savart residual function'
            print_op.log_and_raise_error(message)

    return None


def test(epsilon=1.e-4):
    test_append()
    test_check_expected_dimensions()
    test_mapped_biot_savart()
    test_mapped_biot_savart_residual(epsilon)

# test()