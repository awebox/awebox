#
#    This file is part of awebox.
#
#    awebox -- A modeling and optimization framework for multi-kite AWE systems.
#    Copyright (C) 2017-2022 Jochem De Schutter, Rachel Leuthold, Moritz Diehl,
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
object-oriented vortex finite filament
_python-3.5 / casadi-3.4.5
- authors: rachel leuthold 2021-2022
'''

import casadi.tools as cas
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np

import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.element as obj_element

import awebox.tools.struct_operations as struct_op
import awebox.tools.vector_operations as vect_op
import awebox.tools.print_operations as print_op

from awebox.logger.logger import Logger as awelogger

import matplotlib
from awebox.viz.plot_configuration import DEFAULT_MPL_BACKEND
matplotlib.use(DEFAULT_MPL_BACKEND)
import awebox.mdl.aero.induction_dir.vortex_dir.tools as vortex_tools




class FiniteFilament(obj_element.Element):
    def __init__(self, info_dict):
        super().__init__(info_dict)
        self.set_element_type('finite_filament')
        self.define_info_order()
        packed_info = self.pack_info()
        self.set_info(packed_info)

    def define_info_order(self):
        order = {0: ('x_start', 3),
                 1: ('x_end', 3),
                 2: ('r_core', 1),
                 3: ('strength', 1)
                 }
        self.set_info_order(order)

        return None

    def calculate_biot_savart_induction(self, unpacked_sym, x_obs):

        x_0 = unpacked_sym['x_start']
        x_1 = unpacked_sym['x_end']
        r_core = unpacked_sym['r_core']
        strength = unpacked_sym['strength']

        vec_0 = x_0 - x_obs
        vec_1 = x_1 - x_obs

        r_squared_0 = cas.mtimes(vec_0.T, vec_0)
        r_squared_1 = cas.mtimes(vec_1.T, vec_1)
        r_0 = r_squared_0 ** 0.5
        r_1 = r_squared_1 ** 0.5

        # notice, that we're using the cut-off model as described in
        # https: // openfast.readthedocs.io / en / main / source / user / aerodyn - olaf / OLAFTheory.html  # regularization
        # which is the unit-consistent version what's used in
        # A. van Garrel. Development of a Wind Turbine Aerodynamics Simulation Module. Technical report,
        # Energy research Centre of the Netherlands. ECN-C–03-079, aug 2003
        length = vect_op.norm(x_1 - x_0)
        epsilon_vortex = r_core ** 2. * length ** 2.

        factor = strength / (4. * np.pi)
        num1 = r_0 + r_1
        num2 = vect_op.cross(vec_0, vec_1)
        den1 = r_squared_0 * r_squared_1
        den2 = r_0 * r_1 * cas.mtimes(vec_0.T, vec_1)
        den3 = epsilon_vortex

        num = factor * num1 * num2
        den = den1 + den2 + den3

        value = num / den

        return value, num, den


    def draw(self, ax, side, variables_scaled=None, parameters=None, cosmetics=None):
        unpacked, cosmetics = self.prepare_to_draw(variables_scaled, parameters, cosmetics)

        x_start = unpacked['x_start']
        x_end = unpacked['x_end']
        super().basic_draw(ax, side, unpacked['strength'], x_start, x_end, cosmetics)

        return None


def construct_test_object(r_core=cas.DM(0.)):
    x_start = -1. * vect_op.xhat_dm()
    x_end = 1. * vect_op.xhat_dm()
    strength = cas.DM(4.) * np.pi
    dict_info = {'x_start': x_start,
                 'x_end': x_end,
                 'r_core': r_core,
                 'strength': strength
                 }

    fil = FiniteFilament(dict_info)
    fil.define_biot_savart_induction_function()
    return fil

def test_biot_savart_infinitely_far_away(fil, epsilon=1.e-4):
    x_obs = 1.e4 * vect_op.zhat_dm()

    packed_info = fil.info
    biot_savart_fun = fil.biot_savart_fun
    vec_u_ind = biot_savart_fun(packed_info, x_obs)

    test_val = vect_op.norm(vec_u_ind)
    criteria = (test_val < epsilon)

    if not criteria:
        message = 'vortex finite filament: influence of the vortex does not vanish far from the vortex'
        print_op.log_and_raise_error(message)

def test_biot_savart_right_hand_rule(fil, epsilon=1.e-4):
    x_obs = 1. * vect_op.zhat_dm()

    packed_info = fil.info
    biot_savart_fun = fil.biot_savart_fun
    vec_u_ind = biot_savart_fun(packed_info, x_obs)

    test_val = vect_op.norm(vect_op.normalize(vec_u_ind) + vect_op.yhat_dm())
    criteria = (test_val < epsilon)

    if not criteria:

        message = 'vortex finite filament: direction of induced velocity does not satisfy the right-hand-rule'
        print_op.log_and_raise_error(message)

def test_biot_savart_2D_behavior(fil, epsilon=1.e-4):

    unpacked = fil.info_dict
    x_center = 0.5 * (unpacked['x_start'] + unpacked['x_end'])
    length = vect_op.norm(unpacked['x_end'] - unpacked['x_start'])

    r_hat = vect_op.zhat_dm()
    radius = length * 1.e-3
    scale = 2.

    x_obs = x_center + radius * r_hat
    x_obs_2r = x_center + scale * radius * r_hat

    # u_ind(2 r) = u_0 / (2 r)
    # u_ind( r) = u_0 / r
    # r u_ind(r) = 2 r u_ind(2 r)
    # u_ind(r) / u_ind(2 r) = 2

    packed_info = fil.info
    biot_savart_fun = fil.biot_savart_fun
    vec_u_ind = biot_savart_fun(packed_info, x_obs)
    vec_u_ind_2r = biot_savart_fun(packed_info, x_obs_2r)

    u_theta = cas.mtimes(vec_u_ind.T, vect_op.yhat_dm())
    u_theta_2r = cas.mtimes(vec_u_ind_2r.T, vect_op.yhat_dm())

    found = u_theta/u_theta_2r
    expected = scale

    diff = found - expected
    criteria = (diff**2. < epsilon**2.)

    if not criteria:
        message = 'vortex finite filament: in an approximately 2D situation, the inverse-radius relationship is not satisfied'
        print_op.log_and_raise_error(message)

def test_biot_savart_point_vortex_behavior(fil, epsilon=1.e-4):
    x_obs = 1.e2 * vect_op.zhat_dm()
    scale = 2.

    packed_info = fil.info
    biot_savart_fun = fil.biot_savart_fun
    vec_u_ind = biot_savart_fun(packed_info, x_obs)
    vec_u_ind_2 = biot_savart_fun(packed_info, scale * x_obs)

    num = cas.mtimes(vec_u_ind.T, vect_op.yhat_dm())
    den = cas.mtimes(vec_u_ind_2.T, vect_op.yhat_dm())
    test_val = num / den - scale**2.
    criteria = (-1. * epsilon < test_val) and (test_val < epsilon)

    if not criteria:
        message = 'vortex finite filament: in an approximately point-vortex situation, the inverse-squared-radius relationship is not satisfied'
        print_op.log_and_raise_error(message)

def test_biot_savart_unregularized_singularity_removed(fil, epsilon=1.e-4):
    x_obs = 1.5 * vect_op.xhat_dm()

    packed_info = fil.info
    biot_savart_fun = fil.biot_savart_fun
    vec_u_ind = biot_savart_fun(packed_info, x_obs)

    test_val = vect_op.norm(vec_u_ind)
    criteria = (test_val < epsilon)

    if not criteria:
        message = 'vortex finite filament: singularities DO occur on the axis, outside of the endpoints'
        print_op.log_and_raise_error(message)

def test_biot_savart_regularized_singularity_removed(fil_with_nonzero_core_radius, epsilon=1.e-4):
    x_obs = fil_with_nonzero_core_radius.info_dict['x_start']

    packed_info = fil_with_nonzero_core_radius.info
    biot_savart_fun = fil_with_nonzero_core_radius.biot_savart_fun
    vec_u_ind = biot_savart_fun(packed_info, x_obs)

    test_val = vect_op.norm(vec_u_ind)
    criteria = (test_val < epsilon)

    if not criteria:
        message = 'vortex finite filament: regularization does not remove the singularities on the vortex finite filament'
        print_op.log_and_raise_error(message)

def test_biot_savart_off_axis_values(fil, epsilon=1.e-4):
    x_obs = 1.5 * vect_op.xhat_dm() + 0.112325 * (vect_op.yhat_dm() + vect_op.zhat_dm())

    packed_info = fil.info
    biot_savart_fun = fil.biot_savart_fun
    vec_u_ind = biot_savart_fun(packed_info, x_obs)

    expected = 0.2 * (-1.*vect_op.yhat_dm() + vect_op.zhat_dm())
    diff = vec_u_ind - expected

    test_val = vect_op.norm(diff)
    criteria = (test_val < epsilon)

    if not criteria:
        message = 'vortex finite filament: computation gives unreasonable values at off-axis position'
        print_op.log_and_raise_error(message)

def test(test_includes_visualization=False, epsilon=1.e-6):

    fil = construct_test_object(r_core=0.)

    fil.test_basic_criteria(expected_object_type='finite_filament')
    fil.test_calculated_biot_savart_induction_satisfies_residual(epsilon=epsilon)

    test_biot_savart_infinitely_far_away(fil, epsilon)
    test_biot_savart_right_hand_rule(fil, epsilon)
    test_biot_savart_2D_behavior(fil, epsilon=1.e-3)
    test_biot_savart_point_vortex_behavior(fil, epsilon=1.e-3)
    test_biot_savart_unregularized_singularity_removed(fil, epsilon)
    test_biot_savart_off_axis_values(fil, epsilon)

    fil_with_nonzero_core_radius = construct_test_object(r_core=1.)
    test_biot_savart_regularized_singularity_removed(fil_with_nonzero_core_radius, epsilon)

    fil.test_draw(test_includes_visualization)

    return None

# test()
