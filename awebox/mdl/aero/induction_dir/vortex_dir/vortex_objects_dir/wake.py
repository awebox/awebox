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

import copy
import pdb

import casadi.tools as cas
import numpy as np

import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.vortex_object_structure as obj_structure

import awebox.tools.struct_operations as struct_op
import awebox.tools.vector_operations as vect_op
import awebox.tools.print_operations as print_op

from awebox.logger.logger import Logger as awelogger

import matplotlib
matplotlib.use('TkAgg')

class Wake(obj_structure.VortexObjectStructure):
    def __init__(self):
        accepted_types = self.get_accepted_substructure_types()

        self.__wake_dict = {}
        for substructure_type in accepted_types:
            self.__wake_dict[substructure_type] = None

    def get_accepted_substructure_types(self):
        accepted_types = {'bound',
                          'near',
                          'far'
                          }
        return accepted_types

    def get_initialized_substructure_types_with_at_least_one_element(self):
        initialized_types = self.get_initialized_substructure_types()
        types_with_elements = []
        for substructure_type in initialized_types:
            if self.__wake_dict[substructure_type].has_at_least_one_element():
                types_with_elements += [substructure_type]
        return types_with_elements

    def get_initialized_substructure_types(self):
        accepted_types = self.get_accepted_substructure_types()
        initialized_types = []
        for substructure_type in accepted_types:
            if self.__wake_dict[substructure_type] is not None:
                initialized_types += [substructure_type]

        return initialized_types

    def has_initialized_substructure_types(self):
        initialized_types = self.get_initialized_substructure_types()
        return (len(initialized_types) > 0)

    def get_substructure(self, substucture_type):
        if substucture_type in self.get_accepted_substructure_types():
            return self.__wake_dict[substucture_type]
        else:
            message = 'the substructure type (' + substucture_type + ') is not among the recognized wake substructure types'
            print_op.log_and_raise_error(message)

        return None

    def set_substructure(self, substructure):
        substructure_type = substructure.substructure_type
        if self.__wake_dict[substructure_type] is None:
            self.__wake_dict[substructure_type] = substructure
        else:
            self.log_and_raise_overwrite_error(substructure_type)

    def log_and_raise_overwrite_error(self, substructure_type):
        message = 'there is already a wake substructure stored as the ' + substructure_type + ' wake substructure.'
        print_op.log_and_raise_error(message)
        return None

    def get_max_abs_strength(self):
        if self.has_initialized_substructure_types():
            initialized_types = self.get_initialized_substructure_types()

            all_strengths = [self.get_substructure(substructure_type).get_max_abs_strength() for substructure_type in initialized_types]
            strengths_are_numeric = [vect_op.is_numeric(strength) for strength in all_strengths]
            if all(strengths_are_numeric):
                strengths_array = np.array(all_strengths)
                return np.max(np.abs(strengths_array))

        message = 'could not compute a numeric max-abs-strength for this wake. proceeding with a unit value'
        awelogger.logger.warning(message)
        return 1.

    def calculate_total_biot_savart_residual_at_x_obs(self, variables_scaled, parameters, x_obs=cas.DM.zeros((3, 1))):

        vec_u_ind = cas.DM.zeros((3, 1))
        initialized_types = self.get_initialized_substructure_types_with_at_least_one_element()
        for substructure_type in initialized_types:
            local_u_ind = self.get_substructure(substructure_type).calculate_total_biot_savart_residual_at_x_obs(variables_scaled, parameters, x_obs)
            vec_u_ind += local_u_ind

        return vec_u_ind


    def evaluate_total_biot_savart_induction(self, x_obs=cas.DM.zeros(3, 1)):
        vec_u_ind = cas.DM.zeros((3, 1))
        initialized_types = self.get_initialized_substructure_types_with_at_least_one_element()
        for substructure_type in initialized_types:
            local = self.get_substructure(substructure_type).evaluate_total_biot_savart_induction(x_obs)
        vec_u_ind += local
        return vec_u_ind


    def construct_fake_cosmetics(self):
        if self.has_initialized_substructure_types():
            initialized_types = self.get_initialized_substructure_types()

            example_substructure_type = initialized_types[0]
            example_substructure = self.get_substructure(example_substructure_type)
            example_element_type = example_substructure.get_initialized_element_types()[0]
            example_list = example_substructure.get_list(example_element_type).list
            example_element = example_list[0]
            cosmetics = example_element.construct_fake_cosmetics()

            max_abs_strength = self.get_max_abs_strength()
            cosmetics['trajectory']['circulation_max_estimate'] = max_abs_strength
            return cosmetics

        return None

    def draw(self, ax, side, variables_scaled=None, parameters=None, cosmetics=None):

        if cosmetics is None:
            cosmetics = self.construct_fake_cosmetics()

        initialized_types = self.get_initialized_substructure_types()
        for substructure_type in initialized_types:
            self.get_substructure(substructure_type).draw(ax, side, variables_scaled, parameters, cosmetics)
        return None

    def mapped_biot_savart_function_is_defined_for_initialized_substructures(self):
        return all([self.get_substructure(substructure_type).mapped_biot_savart_function_is_defined_for_initialized_lists() for substructure_type in self.get_initialized_substructure_types()])

    def mapped_biot_savart_residual_function_is_defined_for_initialized_substructures(self):
        return all([self.get_substructure(substructure_type).mapped_biot_savart_residual_function_is_defined_for_initialized_lists() for substructure_type in self.get_initialized_substructure_types()])

    def define_biot_savart_induction_functions(self):
        initialized = self.get_initialized_substructure_types()
        for substructure_type in initialized:
            if not self.get_substructure(substructure_type).mapped_biot_savart_function_is_defined_for_initialized_lists():
                self.get_substructure(substructure_type).define_biot_savart_induction_functions()
        return None

    def define_biot_savart_induction_residual_functions(self, biot_savart_residual_assembly='split'):
        initialized = self.get_initialized_substructure_types()
        for substructure_type in initialized:
            if not self.get_substructure(substructure_type).mapped_biot_savart_residual_function_is_defined_for_initialized_lists():
                self.get_substructure(substructure_type).define_biot_savart_induction_residual_functions(biot_savart_residual_assembly)
        return None

    def define_model_variables_to_info_functions(self, model_variables, model_parameters):
        initialized = self.get_initialized_substructure_types()
        for substructure_type in initialized:
            self.get_substructure(substructure_type).define_model_variables_to_info_functions(model_variables, model_parameters)
        return None

def test():
    return None

