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

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import copy

import casadi.tools as cas
import numpy as np

import awebox.tools.vector_operations as vect_op
import awebox.tools.print_operations as print_op

from awebox.logger.logger import Logger as awelogger

import matplotlib
matplotlib.use('TkAgg')

class Element():
    def __init__(self, info_dict, info_order=None):

        for key, value in info_dict.items():
            info_dict[key] = vect_op.columnize(value)

        self.__info_dict = info_dict
        self.set_element_type('element')
        self.set_biot_savart_fun(None)
        self.set_biot_savart_residual_fun(None)
        self.set_info_fun(None)

        if info_order is not None:
            self.set_info_order(info_order)
            packed_info = self.pack_info()
            self.set_info(packed_info)

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
            print_op.log_and_raise_error(message)

        return None

    def number_of_info_values(self):
        if self.__info_order is not None:
            return len(self.__info_order.keys())

        else:
            message = 'cannot determine number of info values for vortex element, because info order has not yet been set'
            print_op.log_and_raise_error(message)

        return None


    def unpack_info(self, external_info=None):

        if external_info is not None:
            packed_info = external_info
        else:
            packed_info = self.info

        if self.__info_order is not None:
            local_index = 0
            number_of_positions = self.number_of_info_values()

            unpacked = {}
            for local_position in range(number_of_positions):
                local_pair = self.info_order[local_position]
                local_name = local_pair[0]
                local_length = local_pair[1]

                start_index = local_index
                end_index = start_index + local_length

                unpacked[local_name] = packed_info[start_index:end_index]

                local_index = end_index

            return unpacked

        else:
            message = 'cannot unpack the info vector for vortex element, because info order has not yet been set'
            print_op.log_and_raise_error(message)

        return None


    def pack_info(self, external_info=None):

        if external_info is not None:
            info_dict = external_info
        else:
            info_dict = self.info_dict

        if self.info_order is not None:
            number_of_positions = self.number_of_info_values()

            packed = []
            for local_position in range(number_of_positions):
                local_pair = self.info_order[local_position]
                local_name = local_pair[0]
                local_length = local_pair[1]
                local_val = cas.reshape(info_dict[local_name], (local_length, 1))

                packed = cas.vertcat(packed, local_val)

            return packed

        else:
            message = 'cannot pack the info dictionary for vortex element, because info order has not yet been set'
            print_op.log_and_raise_error(message)

        return None

    def get_strength_color(self, strength_val, cosmetics):

        if vect_op.is_numeric_scalar(strength_val):
            strength_val = float(strength_val)

        strength_max = cosmetics['trajectory']['circulation_max_estimate']
        strength_min = -1. * strength_max
        strength_range = 2. * strength_max

        if strength_val > strength_max:
            message = 'reported vortex element strength ' + str(strength_val) + ' is larger than the maximum expected ' \
                'vortex element strength ' + str(strength_max) + '. we recommend re-calculating the expected strength range bounds.'
            awelogger.logger.warning(message)

        if strength_val < strength_min:
            message = 'reported vortex element strength ' + str(strength_val) + ' is smaller than the minimum expected ' \
                'vortex element strength ' + str(strength_min)  + '. we recommend re-calculating the expected strength range bounds'
            awelogger.logger.warning(message)

        cmap = plt.get_cmap('seismic')
        strength_scaled = float(strength_val - strength_min) / float(strength_range)

        color = cmap(strength_scaled)
        return color

    def calculate_biot_savart_induction(self, unpacked_sym, x_obs):
        message = 'cannot calculate the biot savart induction for this vortex object, because the object type ' + self.__element_type + ' is insufficiently specific'
        print_op.log_and_raise_error(message)
        return None

    def define_biot_savart_induction_function(self):
        expected_info_length = self.expected_info_length
        packed_sym = cas.SX.sym('packed_sym', (expected_info_length, 1))
        unpacked_sym = self.unpack_info(external_info=packed_sym)

        x_obs = cas.SX.sym('x_obs', (3, 1))

        value, _, _ = self.calculate_biot_savart_induction(unpacked_sym, x_obs)

        biot_savart_fun = cas.Function('biot_savart_fun', [packed_sym, x_obs], [value])
        self.set_biot_savart_fun(biot_savart_fun)

        return None


    def define_biot_savart_induction_residual_function(self, degree_of_induced_velocity_lifting=2):
        expected_info_length = self.expected_info_length
        packed_sym = cas.SX.sym('packed_sym', (expected_info_length, 1))
        unpacked_sym = self.unpack_info(external_info=packed_sym)

        x_obs = cas.SX.sym('x_obs', (3, 1))
        vec_u_ind = cas.SX.sym('vec_u_ind', (3, 1))

        value, num, den = self.calculate_biot_savart_induction(unpacked_sym, x_obs)

        if degree_of_induced_velocity_lifting == 1:
            biot_savart_residual_fun = None

        elif degree_of_induced_velocity_lifting == 2:
            resi = vec_u_ind - value
            biot_savart_residual_fun = cas.Function('biot_savart_residual_fun', [packed_sym, x_obs, vec_u_ind], [resi])

        elif degree_of_induced_velocity_lifting == 3:
            num_sym = cas.SX.sym('vec_u_ind_num', (3, 1))
            den_sym = cas.SX.sym('vec_u_ind_den', (1, 1))

            resi_value = vec_u_ind * den_sym - num_sym
            resi_num = num_sym - num
            resi_den = den_sym - den

            resi = cas.vertcat(resi_value, resi_num, resi_den)

            biot_savart_residual_fun = cas.Function('biot_savart_residual_fun', [packed_sym, x_obs, vec_u_ind, num_sym, den_sym], [resi])

        else:
            message = 'unexpected degree_of_induced_velocity_lifting (' + str(degree_of_induced_velocity_lifting) + ')'
            print_op.log_and_raise_error(message)

        self.set_biot_savart_residual_fun(biot_savart_residual_fun)

        return None

    def define_model_variables_to_info_function(self, model_variables, model_parameters):
        info_fun = cas.Function('info_fun', [model_variables, model_parameters], [self.__info])
        self.set_info_fun(info_fun)

        return None

    def evaluate_info(self, variables_scaled, parameters):
        return self.__info_fun(variables_scaled, parameters)


    def draw(self, ax, side, variables_scaled=None, parameters=None, cosmetics=None):
        message = 'draw function does not exist for this vortex object, because the object type ' + self.__element_type + ' is insufficiently specific'
        print_op.log_and_raise_error(message)


    def construct_fake_cosmetics(self, unpacked=None):
        cosmetics = {}
        cosmetics['trajectory'] = {}
        cosmetics['trajectory']['cylinder_s_length'] = 3.
        cosmetics['trajectory']['filament_s_length'] = cosmetics['trajectory']['cylinder_s_length']
        cosmetics['trajectory']['cylinder_n_theta'] = 30
        cosmetics['trajectory']['cylinder_n_s'] = 8
        cosmetics['trajectory']['vortex_vec_u_ref'] = vect_op.xhat_np()

        if (unpacked is not None) and (vect_op.is_numeric(unpacked['strength'])):
            local_strength = unpacked['strength']
        elif (unpacked is None) and (vect_op.is_numeric(self.info_dict['strength'])):
            local_strength = self.info_dict['strength']
        else:
            local_strength = 1.

        # notice, in this case, every element 'above' this threshhold will be dark-red,
        # and every element below will be white
        epsilon_draw_nonwhite = 1.e-4
        if local_strength**2. < epsilon_draw_nonwhite**2.:
            # prevent divide by zero errors when plotting color is selected;
            local_strength = 1.

        cosmetics['trajectory']['circulation_max_estimate'] = local_strength

        return cosmetics

    def prepare_to_draw(self, variables_scaled=None, parameters=None, cosmetics=None):
        passed_information = (variables_scaled is not None) and (parameters is not None)
        if passed_information:
            evaluated = self.evaluate_info(variables_scaled, parameters)
            unpacked = self.unpack_info(external_info=evaluated)
        else:
            unpacked = self.info_dict

        if cosmetics is None:
            cosmetics = self.construct_fake_cosmetics(unpacked)

        return unpacked, cosmetics

    def basic_draw(self, ax, side, strength, x_start, x_end, cosmetics):
        if cosmetics is None:
            cosmetics = self.construct_fake_cosmetics()

        color = self.get_strength_color(strength, cosmetics)
        x = [float(x_start[0]), float(x_end[0])]
        y = [float(x_start[1]), float(x_end[1])]
        z = [float(x_start[2]), float(x_end[2])]

        marker = None
        linestyle = '-'

        if side == 'xy':
            ax.plot(x, y, marker=marker, c=color, linestyle=linestyle)
        elif side == 'xz':
            ax.plot(x, z, marker=marker, c=color, linestyle=linestyle)
        elif side == 'yz':
            ax.plot(y, z, marker=marker, c=color, linestyle=linestyle)
        elif side == 'isometric':
            ax.plot3D(x, y, z, marker=marker, c=color, linestyle=linestyle)

        return None


    def get_repeated_info(self, period, wind, optimization_period):
        repeated_dict = copy.deepcopy(self.__info_dict)

        for name in repeated_dict.keys():
            if name[0] == 'x':
                local_x = repeated_dict[name]
                local_u = wind.get_velocity(local_x[2])
                shifted_x = local_x + local_u * optimization_period
                repeated_dict[name] = shifted_x

        return repeated_dict

    def is_equal(self, query_element, epsilon=1.e-6):

        if not (self.__element_type == query_element.element_type):
            return False

        for name, val in self.__info_dict.items():
            query_val = query_element.info_dict[name]

            local_is_casadi_symbolic = isinstance(val, cas.SX) or isinstance(val, cas.MX)
            query_is_casadi_symbolic = isinstance(query_val, cas.SX) or isinstance(query_val, cas.MX)
            only_one_is_symbolic = (local_is_casadi_symbolic and not query_is_casadi_symbolic) or (query_is_casadi_symbolic and not local_is_casadi_symbolic)
            both_are_symbolic = local_is_casadi_symbolic and query_is_casadi_symbolic

            if only_one_is_symbolic:
                return False

            elif both_are_symbolic:
                return cas.is_equal(val, query_val)

            else:
                diff = (val - query_val)
                if isinstance(diff, float) or isinstance(diff, np.ndarray):
                    diff = cas.DM(diff)
                diff = diff / vect_op.smooth_norm(cas.DM(val), epsilon=0.01 * epsilon)

                criteria = (cas.mtimes(diff.T, diff) < epsilon**2.)
                if not criteria:
                    return False

        return True

    def test_calculated_biot_savart_induction_satisfies_residual(self, epsilon=1.e-6):
        for degree_of_induced_velocity_lifting in [1, 2, 3]:
            for x_obs in [cas.DM.ones((3, 1)), 5. * np.random.rand(3, 1)]:
                self.test_calculated_biot_savart_induction_satisfies_residual_at_specific_point_and_assembly(x_obs=x_obs,
                                                                             degree_of_induced_velocity_lifting=degree_of_induced_velocity_lifting,
                                                                             epsilon=epsilon)
        return None


    def test_calculated_biot_savart_induction_satisfies_residual_at_specific_point_and_assembly(self, x_obs=cas.DM.ones((3, 1)), degree_of_induced_velocity_lifting=3, epsilon=1.e-6):
        self.define_biot_savart_induction_residual_function(degree_of_induced_velocity_lifting)
        biot_savart_residual_fun = self.biot_savart_residual_fun

        packed_info = self.pack_info()
        value, num, den = self.calculate_biot_savart_induction(self.info_dict, x_obs)

        if degree_of_induced_velocity_lifting == 3:
            residual = biot_savart_residual_fun(packed_info, x_obs, value, num, den)
        elif degree_of_induced_velocity_lifting == 2:
            residual = biot_savart_residual_fun(packed_info, x_obs, value)
        elif degree_of_induced_velocity_lifting == 1:
            residual = cas.DM.zeros((1, 1))
        else:
            message = 'unexpected degree_of_induced_velocity_lifting (' + str(degree_of_induced_velocity_lifting) + ")"
            print_op.log_and_raise_error(message)

        condition = cas.mtimes(residual.T, residual) < epsilon**2.

        if not condition:
            message = 'biot-savart residual function does not work as expected for element of type ('
            message += self.__element_type
            message += ') and degree_of_induced_velocity_lifting ('
            message += degree_of_induced_velocity_lifting + ')'
            print_op.log_and_raise_error(message)

        return None

    def test_basic_criteria(self, expected_object_type='element', epsilon=1.e-6):
        self.test_object_type(expected_object_type)
        self.test_info_length()
        return None

    def test_object_type(self, expected_object_type='element'):
        criteria = (self.__element_type == expected_object_type)

        if not criteria:
            message = 'object of type ' + self.__element_type + ' is not of the expected object type ' + expected_object_type
            print_op.log_and_raise_error(message)

        return None

    def test_info_length(self):

        if self.info_order is not None:

            criteria = (self.info_length == self.expected_info_length)

            if not criteria:
                message = 'vortex object with info length ' + str(self.__info_length) + ' does not have the expected info length ' + str(self.__expected_info_length)
                print_op.log_and_raise_error(message)
        else:
            message = 'cannot confirm that this element has the correct info length, because info order has not yet been set'
            print_op.log_and_raise_error(message)

        return None

    def test_draw(self, test_includes_visualization=False):

        if test_includes_visualization:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            self.draw(ax, 'isometric')

        return None

    @property
    def info_fun(self):
        return self.__info_fun

    @info_fun.setter
    def info_fun(self, value):
        message = 'Cannot set info_fun object.'
        print_op.log_and_raise_error(message)

    def set_info_fun(self, value):
        self.__info_fun = value

    @property
    def info(self):
        return self.__info

    @info.setter
    def info(self, value):
        message = 'Cannot set info object.'
        print_op.log_and_raise_error(message)

    @property
    def info_order(self):
        return self.__info_order

    @info_order.setter
    def info_order(self, value):
        self.__info_order = value

    def set_info_order(self, value):
        if value is not None:
            self.__info_order = value
            self.set_expected_info_length()

        return None

    @property
    def info_dict(self):
        return self.__info_dict

    @info_dict.setter
    def info_dict(self, value):
        message = 'Cannot set info_dict object.'
        print_op.log_and_raise_error(message)

    @property
    def info_length(self):
        return self.__info.shape[0] * self.__info.shape[1]

    @property
    def element_type(self):
        return self.__element_type

    @element_type.setter
    def element_type(self, value):
        print_op.log_and_raise_error('Cannot set element_type object.')

    def set_element_type(self, value):
        self.__element_type = value

    @property
    def expected_info_length(self):
        return self.__expected_info_length

    @expected_info_length.setter
    def expected_info_length(self, value):
        print_op.log_and_raise_error('Cannot set info_length object.')

    @property
    def biot_savart_fun(self):
        return self.__biot_savart_fun

    @biot_savart_fun.setter
    def biot_savart_fun(self, value):
        print_op.log_and_raise_error('Cannot set biot_savart_fun object.')

    def set_biot_savart_fun(self, value):
        self.__biot_savart_fun = value

    @property
    def biot_savart_residual_fun(self):
        return self.__biot_savart_residual_fun

    @biot_savart_residual_fun.setter
    def biot_savart_residual_fun(self, value):
        print_op.log_and_raise_error('Cannot set biot_savart_residual_fun object.')

    def set_biot_savart_residual_fun(self, value):
        self.__biot_savart_residual_fun = value

def construct_test_object(alpha=1.1):

    info_order = {0: ('alpha', 1),
                  1: ('beta', 1),
                  2: ('gamma', 1)
                  }

    beta = 2.2
    gamma = 3.3
    info_dict = {'alpha': alpha,
                 'beta': beta,
                 'gamma': gamma
                 }

    elem = Element(info_dict, info_order=info_order)
    return elem

def test_number_of_info_values(elem):
    number_of_info_values = elem.number_of_info_values()
    criteria = (number_of_info_values == 3)

    if not criteria:
        message = 'vortex element: wrong number of info values being recorded'
        print_op.log_and_raise_error(message)

def test_pack_internal_info(elem):
    packed = elem.pack_info()
    expected = cas.DM([1.1, 2.2, 3.3])
    diff = vect_op.norm(packed - expected)
    criteria = (diff == 0.)

    if not criteria:
        message = 'vortex element: internal info is being packed incorrectly'
        print_op.log_and_raise_error(message)

def test_unpack_internal_info(elem):
    unpacked = elem.unpack_info()
    expected = {'alpha': 1.1, 'beta': 2.2, 'gamma': 3.3}

    criteria = True
    for key, value in expected.items():
        criteria = (criteria and (unpacked[key] == expected[key]))

    if not criteria:
        message = 'vortex element: internal info is being unpacked incorrectly'
        print_op.log_and_raise_error(message)

def test_pack_external_info(elem):

    info_dict = {'alpha': 4.1, 'beta': 5.2, 'gamma': 6.3}
    info_packed = cas.DM([4.1, 5.2, 6.3])

    packed = elem.pack_info(external_info=info_dict)

    diff = vect_op.norm(packed - info_packed)
    criteria = (diff == 0.)

    if not criteria:
        message = 'vortex element: external info is being packed incorrectly'
        print_op.log_and_raise_error(message)


def test_unpack_external_info(elem):
    info_dict = {'alpha': 4.1, 'beta': 5.2, 'gamma': 6.3}
    info_packed = cas.DM([4.1, 5.2, 6.3])

    unpacked = elem.unpack_info(external_info=info_packed)

    criteria = True
    for key, value in info_dict.items():
        criteria = (criteria and (unpacked[key] == info_dict[key]))


    if not criteria:
        message = 'vortex element: external info is being unpacked incorrectly'
        print_op.log_and_raise_error(message)

def test_is_equal():
    obj1 = construct_test_object(alpha=1.1)
    obj2 = construct_test_object(alpha=cas.DM(1.1))
    obj3 = construct_test_object(alpha=2.2)
    obj4 = construct_test_object(alpha=cas.SX.sym('alpha'))

    obj5 = construct_test_object(alpha=1.1 + 1.e-12)
    obj6 = construct_test_object(alpha=1.1 + 1.e-3)
    obj7 = construct_test_object(alpha=cas.DM(1.1 + 1.e-12))
    obj8 = construct_test_object(alpha=cas.DM(1.1 + 1.e-3))

    con1 = (obj1.is_equal(obj1) == True)
    con2 = (obj1.is_equal(obj2) == True)
    con3 = (obj1.is_equal(obj3) == False)
    con4 = (obj1.is_equal(obj4) == False)

    con5 = (obj1.is_equal(obj5) == True)
    con6 = (obj1.is_equal(obj6) == False)
    con7 = (obj1.is_equal(obj7) == True)
    con8 = (obj1.is_equal(obj8) == False)

    criteria = con1 and con2 and con3 and con4 and con5 and con6 and con7 and con8

    if not criteria:
        message = 'the function checking whether another vortex element is the same as this vortex element, does not behave as expected'
        print_op.log_and_raise_error(message)

    return None

def test():
    elem = construct_test_object()
    elem.test_basic_criteria(expected_object_type='element')
    test_number_of_info_values(elem)
    test_unpack_internal_info(elem)
    test_pack_internal_info(elem)
    test_unpack_external_info(elem)
    test_pack_external_info(elem)
    test_is_equal()

# test()