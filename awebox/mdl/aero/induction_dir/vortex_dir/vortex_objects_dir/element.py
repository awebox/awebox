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
import copy
import pdb

import casadi.tools as cas
import matplotlib.pyplot as plt
import numpy as np

import awebox.mdl.aero.induction_dir.vortex_dir.biot_savart as biot_savart

import awebox.tools.struct_operations as struct_op
import awebox.tools.vector_operations as vect_op
import awebox.tools.print_operations as print_op

from awebox.logger.logger import Logger as awelogger

import matplotlib
matplotlib.use('TkAgg')

class Element:
    def __init__(self, info_dict):
        self.__info_dict = info_dict
        self.set_element_type('element')
        self.set_info_order(None)
        self.set_biot_savart_fun(None)

    def set_info(self, packed_info):
        self.__info = packed_info

    def set_expected_info_length(self):
        if self.__info_order is not None:
            expected_info_length = 0
            for val_pair in self.__info_order.values():
                local_length = val_pair[1]
                expected_info_length += local_length

            self.__expected_info_length = expected_info_length
        else:
            message = 'cannot determine expected info length for vortex element, because info order has not yet been set'
            awelogger.logger.error(message)
            raise Exception(message)

        return None

    def number_of_info_values(self):
        if self.__info_order is not None:
            return len(self.__info_order.keys())

        else:
            message = 'cannot determine number of info values for vortex element, because info order has not yet been set'
            awelogger.logger.error(message)
            raise Exception(message)

        return None


    def unpack_info(self, packed_info):

        if self.__info_order is not None:
            local_index = 0
            number_of_positions = self.number_of_info_values()

            unpacked = {}
            for local_position in range(number_of_positions):
                local_pair = self.__info_order[local_position]
                local_name = local_pair[0]
                local_length = local_pair[1]

                start_index = local_index
                end_index = start_index + local_length

                unpacked[local_name] = packed_info[start_index:end_index]
            return unpacked

        else:
            message = 'cannot unpack the info vector for vortex element, because info order has not yet been set'
            awelogger.logger.error(message)
            raise Exception(message)

        return None


    def pack_info(self, dict_info):

        if self.__info_order is not None:
            number_of_positions = self.number_of_info_values()

            packed = []
            for local_position in range(number_of_positions):
                local_pair = self.__info_order[local_position]
                local_name = local_pair[0]
                local_length = local_pair[1]
                local_val = cas.reshape(dict_info[local_name], (local_length, 1))

                packed = cas.vertcat(packed, local_val)

            return packed

        else:
            message = 'cannot pack the info dictionary for vortex element, because info order has not yet been set'
            awelogger.logger.error(message)
            raise Exception(message)

        return None

    def get_strength_color(self, strength_val, cosmetics):

        strength_max = cosmetics['trajectory']['circulation_max_estimate']
        strength_min = -1. * strength_max

        if strength_val > strength_max:
            message = 'reported vortex element strength ' + str(strength_val) + ' is larger than the maximum expected ' \
                'vortex element strength ' + str(strength_max) + '. we recommend re-calculating the expected strength range bounds.'
            awelogger.logger.warning(message)

        if strength_val < strength_min:
            message = 'reported vortex element strength ' + str(strength_val) + ' is smaller than the minimum expected ' \
                'vortex element strength ' + str(strength_min)  + '. we recommend re-calculating the expected strength range bounds'
            awelogger.logger.warning(message)

        cmap = plt.get_cmap('seismic')
        strength_scaled = float((strength_val - strength_min) / (strength_max - strength_min))
        color = cmap(strength_scaled)
        return color

    def make_symbolic_info_function(self, model_variables, model_parameters):
        self.__info_fun = cas.Function('info_fun', [model_variables, model_parameters], [self.__info])
        return None

    def evaluate_info(self, variables, parameters):
        return self.__info_fun(variables, parameters)

    def draw(self, ax, side, variables_scaled, parameters, cosmetics):
        message = 'draw function does not exist for this vortex object, because the object type ' + self.__element_type + ' is insufficiently specific'
        awelogger.logger.error(message)
        raise Exception(message)

    def basic_draw(self, ax, side, strength, x_start, x_end, cosmetics):
        color = self.get_strength_color(strength, cosmetics)
        x = [x_start[0], x_end[0]]
        y = [x_start[1], x_end[1]]
        z = [x_start[2], x_end[2]]

        if side == 'xy':
            ax.plot(x, y, c=color)
        elif side == 'xz':
            ax.plot(x, z, c=color)
        elif side == 'yz':
            ax.plot(y, z, c=color)
        elif side == 'isometric':
            ax.plot3D(x, y, z, c=color)

    def get_repeated_info(self, period, wind, optimization_period):
        repeated_dict = copy.deepcopy(self.__info_dict)

        for name in repeated_dict.keys():
            if name[0] == 'x':
                local_x = repeated_dict[name]
                local_u = wind.get_velocity(local_x[2])
                shifted_x = local_x + local_u * optimization_period
                repeated_dict[name] = shifted_x

        return repeated_dict

    def define_biot_savart_induction_function(self):
        message = 'cannot define the biot savart induction function for this vortex object, because the object type ' + self.__element_type + ' is insufficiently specific'
        awelogger.logger.error(message)
        raise Exception(message)

    def test_basic_criteria(self, expected_object_type='element'):
        self.test_object_type(expected_object_type)
        self.test_has_expected_info_length()
        return None

    def test_object_type(self, expected_object_type='element'):
        criteria = (self.__element_type == expected_object_type)

        if not criteria:
            message = 'object of type ' + self.__element_type + ' is not of the expected object type ' + expected_object_type
            awelogger.logger.error(message)
            raise Exception(message)

        return None

    def test_has_expected_info_length(self):

        if self.__info_order is not None:
            criteria = (self.info_length == self.expected_info_length)

            if not criteria:
                message = 'vortex object with info length ' + str(self.__info_length) + ' does not have the expected info length ' + str(self.__expected_info_length)
                awelogger.logger.error(message)
                raise Exception(message)

        else:
            message = 'cannot confirm that this element has the correct info length, because info order has not yet been set'
            awelogger.logger.error(message)
            raise Exception(message)

        return None

    def test_biot_savart(self):
        message = 'cannot test the biot-savart induction from this vortex object, because the object type ' + self.__element_type + ' is insufficiently specific'
        awelogger.logger.error(message)
        raise Exception(message)

    @property
    def info_fun(self):
        return self.__info_fun

    @info_fun.setter
    def info_fun(self, value):
        awelogger.logger.error('Cannot set info_fun object.')

    @property
    def info(self):
        return self.__info

    @info.setter
    def info(self, value):
        awelogger.logger.error('Cannot set info object.')

    @property
    def info_order(self):
        return self.__info_order

    @info_order.setter
    def info_order(self, value):
        self.__info_order = value

    def set_info_order(self, value):
        self.__info_order = value

    @property
    def info_dict(self):
        return self.__info_dict

    @info_dict.setter
    def info_dict(self, value):
        awelogger.logger.error('Cannot set info_dict object.')

    @property
    def info_length(self):
        return self.__info.shape[0] * self.__info.shape[1]

    @property
    def element_type(self):
        return self.__element_type

    @element_type.setter
    def element_type(self, value):
        awelogger.logger.error('Cannot set element_type object.')

    def set_element_type(self, value):
        self.__element_type = value

    @property
    def expected_info_length(self):
        return self.__expected_info_length

    @expected_info_length.setter
    def expected_info_length(self, value):
        awelogger.logger.error('Cannot set info_length object.')

    @property
    def biot_savart_fun(self):
        return self.__biot_savart_fun

    @biot_savart_fun.setter
    def biot_savart_fun(self, value):
        awelogger.logger.error('Cannot set biot_savart_fun object.')

    def set_biot_savart_fun(self, value):
        self.__biot_savart_fun = value
