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
- authors: rachel leuthold 2021-2022
'''
import copy
import pdb

import casadi.tools as cas
import numpy as np

import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.element as obj_element

import awebox.tools.struct_operations as struct_op
import awebox.tools.vector_operations as vect_op
import awebox.tools.print_operations as print_op

from awebox.logger.logger import Logger as awelogger

import matplotlib
import awebox.mdl.aero.induction_dir.vortex_dir.tools as vortex_tools

matplotlib.use('TkAgg')


class SemiInfiniteFilament(obj_element.Element):
    def __init__(self, info_dict):
        super().__init__(info_dict)
        self.set_element_type('semi_infinite_filament')
        self.define_info_order()
        packed_info = self.pack_info()
        self.set_info(packed_info)

    def define_info_order(self):
        order = {0: ('x_start', 3),
                 1: ('l_hat', 3),
                 2: ('r_core', 1),
                 3: ('strength', 1)
                 }
        self.set_info_order(order)
        return None

    def calculate_biot_savart_induction(self, unpacked_sym, x_obs):

        x_0 = unpacked_sym['x_start']
        l_hat = unpacked_sym['l_hat']
        r_core = unpacked_sym['r_core']
        strength = unpacked_sym['strength']

        vec_0 = x_0 - x_obs

        r_squared_0 = cas.mtimes(vec_0.T, vec_0)
        r_0 = r_squared_0**0.5

        factor = strength/(4. * np.pi)
        num1 = vect_op.cross(vec_0, l_hat)
        den1 = r_squared_0
        den2 = r_0 * cas.mtimes(l_hat.T, vec_0)
        den3 = r_core**2.

        num = factor * num1
        den = den1 + den2 + den3

        value = num / den

        return value, num, den


    def draw(self, ax, side, variables_scaled=None, parameters=None, cosmetics=None):

        unpacked, cosmetics = self.prepare_to_draw(variables_scaled, parameters, cosmetics)

        x_start = unpacked['x_start']
        l_hat = unpacked['l_hat']

        if parameters is None:
            s_length = cosmetics['trajectory']['filament_s_length']
        else:
            vortex_far_convection_time = cosmetics['trajectory']['vortex_far_wake_convection_time']
            vec_u_ref = cosmetics['trajectory']['vortex_vec_u_ref']
            s_length = vortex_far_convection_time * vec_u_ref

        x_end = x_start + l_hat * s_length

        super().basic_draw(ax, side, unpacked['strength'], x_start, x_end, cosmetics)

        return None

def construct_test_object(r_core=cas.DM(0.)):
    x_start = 0. * vect_op.xhat_dm()
    l_hat = vect_op.xhat_dm()
    strength = cas.DM(4. * np.pi)
    dict_info = {'x_start': x_start,
                 'l_hat': l_hat,
                 'r_core': r_core,
                 'strength': strength}

    fil = SemiInfiniteFilament(dict_info)
    fil.define_biot_savart_induction_function()

    return fil

def test_biot_savart_infinitely_far_away(fil, epsilon=1.e-4):
    x_obs = 1.e7 * vect_op.zhat_dm()

    packed_info = fil.info
    biot_savart_fun = fil.biot_savart_fun
    vec_u_ind = biot_savart_fun(packed_info, x_obs)

    test_val = vect_op.norm(vec_u_ind)
    criteria = (test_val < epsilon)

    if not criteria:
        message = 'vortex semi-infinite filament: influence of the vortex does not vanish far from the vortex'
        print_op.log_and_raise_error(message)

def test_biot_savart_right_hand_rule(fil, epsilon=1.e-4):
    x_obs = 1. * vect_op.zhat_dm()

    packed_info = fil.info
    biot_savart_fun = fil.biot_savart_fun
    vec_u_ind = biot_savart_fun(packed_info, x_obs)

    test_val = vect_op.norm(vect_op.normalize(vec_u_ind) + vect_op.yhat_dm())
    criteria = (test_val < epsilon)

    if not criteria:

        message = 'vortex semi-infinite filament: direction of induced velocity does not satisfy the right-hand-rule'
        print_op.log_and_raise_error(message)

def test_biot_savart_inverse_radius_behavior(fil, epsilon=1.e-4):
    x_obs = 1. * vect_op.zhat_dm()
    scale = 2.

    packed_info = fil.info
    biot_savart_fun = fil.biot_savart_fun
    vec_u_ind = biot_savart_fun(packed_info, x_obs)
    vec_u_ind_2 = biot_savart_fun(packed_info, scale * x_obs)

    num = cas.mtimes(vec_u_ind.T, vect_op.yhat_dm())
    den = cas.mtimes(vec_u_ind_2.T, vect_op.yhat_dm())
    test_val = num/den - scale
    criteria = (-1. * epsilon < test_val) and (test_val < epsilon)

    if not criteria:
        message = 'vortex semi-infinite filament: the inverse-radius relationship is not satisfied'
        print_op.log_and_raise_error(message)

def test_biot_savart_unregularized_singularity_removed(fil, epsilon=1.e-4):
    x_obs = -1. * vect_op.xhat_dm()

    packed_info = fil.info
    biot_savart_fun = fil.biot_savart_fun
    vec_u_ind = biot_savart_fun(packed_info, x_obs)

    test_val = vect_op.norm(vec_u_ind)
    criteria = (test_val < epsilon)

    if not criteria:
        message = 'vortex semi-infinite filament: singularities DO occur on the axis, off of the vortex filament'
        print_op.log_and_raise_error(message)

def test_biot_savart_regularized_singularity_removed(fil_with_nonzero_core_radius, epsilon=1.e-4):
    x_obs = 1. * vect_op.xhat_dm()

    packed_info = fil_with_nonzero_core_radius.info
    biot_savart_fun = fil_with_nonzero_core_radius.biot_savart_fun
    vec_u_ind = biot_savart_fun(packed_info, x_obs)

    test_val = vect_op.norm(vec_u_ind)
    criteria = (test_val < epsilon)

    if not criteria:
        message = 'vortex semi-infinite filament: regularization does not remove the singularities on the vortex filament'
        print_op.log_and_raise_error(message)

def test_biot_savart_off_axis_values(fil, epsilon=1.e-4):
    x_obs = 10. * vect_op.xhat_dm() + 4.59612 * (vect_op.yhat_dm() + vect_op.zhat_dm())

    packed_info = fil.info
    biot_savart_fun = fil.biot_savart_fun
    vec_u_ind = biot_savart_fun(packed_info, x_obs)

    expected = 0.2 * (-1.*vect_op.yhat_dm() + vect_op.zhat_dm())
    diff = vec_u_ind - expected

    test_val = vect_op.norm(diff)
    criteria = (test_val < epsilon)

    if not criteria:
        message = 'vortex filament: computation gives unreasonable values at off-axis position'
        print_op.log_and_raise_error(message)

def test(test_includes_visualization=False, epsilon=1.e-6):

    fil = construct_test_object(r_core=0.)
    fil.test_basic_criteria(expected_object_type='semi_infinite_filament')
    fil.test_calculated_biot_savart_induction_satisfies_residual(epsilon=epsilon)

    test_biot_savart_infinitely_far_away(fil, epsilon)
    test_biot_savart_right_hand_rule(fil, epsilon)
    test_biot_savart_inverse_radius_behavior(fil, epsilon=1.e-3)
    test_biot_savart_unregularized_singularity_removed(fil, epsilon)
    test_biot_savart_off_axis_values(fil, epsilon)

    fil_with_nonzero_core_radius = construct_test_object(r_core=1.)
    test_biot_savart_regularized_singularity_removed(fil_with_nonzero_core_radius, epsilon)

    fil.test_draw(test_includes_visualization)

    return None

# test()