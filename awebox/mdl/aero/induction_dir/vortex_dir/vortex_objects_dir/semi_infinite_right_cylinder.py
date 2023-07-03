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
import awebox.mdl.aero.induction_dir.vortex_dir.tools as vortex_tools
import awebox.tools.struct_operations as struct_op
import awebox.tools.vector_operations as vect_op
import awebox.tools.print_operations as print_op

from awebox.logger.logger import Logger as awelogger

import matplotlib
import awebox.mdl.aero.induction_dir.general_dir.tools as general_tools
matplotlib.use('TkAgg')

class SemiInfiniteRightCylinder(obj_element.Element):
    def __init__(self, info_dict, approximation_order_for_elliptic_integrals=6):
        super().__init__(info_dict)
        self.set_element_type('semi_infinite_right_cylinder')
        self.define_info_order()
        packed_info = self.pack_info()
        self.set_info(packed_info)
        self.set_approximation_order_for_elliptic_integrals(approximation_order_for_elliptic_integrals)

    def define_info_order(self):
        order = {0: ('x_center', 3),
                 1: ('l_hat', 3),
                 2: ('radius', 1),
                 3: ('l_start', 1),
                 4: ('epsilon_m', 1),
                 5: ('epsilon_r', 1),
                 6: ('strength', 1)
                 }
        self.set_info_order(order)
        return None


    def construct_biot_savart_reference_object(self, model_options, parameters, wind, inputs={}):

        properties = vortex_tools.get_biot_savart_reference_object_properties(model_options, parameters=parameters, inputs=inputs)

        x_kite_obs = properties['x_kite_obs']

        x_center = properties['x_center']
        l_hat = properties['l_hat']
        radius = properties['radius']
        l_start = properties['far_wake_l_start']
        strength = properties['filament_strength'] / (2. * np.pi)

        epsilon_m = general_tools.get_option_from_possible_dicts(model_options, 'vortex_epsilon_m', 'vortex')
        epsilon_r = general_tools.get_option_from_possible_dicts(model_options, 'vortex_epsilon_r', 'vortex')

        unpacked_ref = {'x_center': x_center,
                        'l_hat': l_hat,
                        'radius': radius,
                        'l_start': l_start,
                        'epsilon_m': epsilon_m,
                        'epsilon_r': epsilon_r,
                        'strength': strength}

        return unpacked_ref, x_kite_obs


    def get_r_obs(self, unpacked, x_obs):

        x_center = vect_op.columnize(unpacked['x_center'])
        l_hat = vect_op.columnize(unpacked['l_hat'])

        x_axis_2 = x_center + l_hat
        r_obs = vect_op.get_altitude(x_center - x_obs, x_axis_2 - x_obs)

        return r_obs

    def get_z_obs(self, unpacked, x_obs):

        l_start = unpacked['l_start']
        l_hat = unpacked['l_hat']
        x_center = unpacked['x_center']

        x_start = x_center + l_start * l_hat

        z_obs = cas.mtimes((x_obs - x_start).T, l_hat)

        return z_obs

    def get_observational_axes(self, unpacked, x_obs):
        x_center = unpacked['x_center']
        l_hat = unpacked['l_hat']
        epsilon_r = unpacked['epsilon_r']
        epsilon_m = unpacked['epsilon_m']

        direction_assumed_to_be_orthogonal_to_longitude = vect_op.zhat_dm()

        diff = x_obs - x_center

        r_hat_base = vect_op.smooth_normalize(diff - l_hat * cas.mtimes(l_hat.T, diff), epsilon=epsilon_r)
        r_hat = vect_op.normalize(r_hat_base + epsilon_m * direction_assumed_to_be_orthogonal_to_longitude)

        theta_hat = vect_op.normed_cross(l_hat, r_hat)

        return r_hat, theta_hat, l_hat

    def get_regularized_elliptic_m_from_r_and_z(self, unpacked, r_obs, z_obs):

        r_cyl = unpacked['radius']
        epsilon_r = unpacked['epsilon_r']

        m_num = 4. * r_obs * r_cyl
        m_den = (r_cyl + r_obs)**2. + z_obs**2 + epsilon_r**2.
        m = m_num/m_den
        return m

    def get_regularized_elliptic_m(self, unpacked, x_obs):
        r_obs = self.get_r_obs(unpacked, x_obs)
        z_obs = self.get_z_obs(unpacked, x_obs)

        m = self.get_regularized_elliptic_m_from_r_and_z(unpacked, r_obs, z_obs)
        return m

    def get_regularized_elliptic_m_zero_from_r(self, unpacked, r_obs):
        z_obs = cas.DM.zeros((1,1))
        m = self.get_regularized_elliptic_m_from_r_and_z(unpacked, r_obs, z_obs)
        return m

    def get_regularized_elliptic_m_zero(self, unpacked, x_obs):
        r_obs = self.get_r_obs(unpacked, x_obs)
        m0 = self.get_regularized_elliptic_m_zero_from_r(unpacked, r_obs)
        return m0

    @property
    def approximation_order_for_elliptic_integrals(self):
        return self.__approximation_order_for_elliptic_integrals

    @approximation_order_for_elliptic_integrals.setter
    def approximation_order_for_elliptic_integrals(self, value):
        print_op.log_and_raise_error('Cannot set approximation_order_for_elliptic_integrals object.')

    def set_approximation_order_for_elliptic_integrals(self, value):
        self.__approximation_order_for_elliptic_integrals = value
        return None



def calculate_radius_and_l_start(x_start, x_center, l_hat):
    vec_dist = (x_start - x_center)
    l_start = cas.mtimes(vec_dist.T, l_hat)
    vec_radial = vec_dist - l_start * l_hat
    radius = vect_op.smooth_norm(vec_radial)
    return radius, l_start

def construct_test_object(regularized=True):
    x_center = np.array([0., 0., 0.])
    radius = 1.
    l_start = 0.
    l_hat = vect_op.xhat_np()

    if regularized:
        epsilon_m = 10.**(-5.)
        epsilon_r = 1.
    else:
        epsilon_m = 10.**(-10)
        epsilon_r = 10.**(-10)

    strength = 1.
    unpacked = {'x_center': x_center,
                'l_hat': l_hat,
                'radius': radius,
                'l_start': l_start,
                'epsilon_m': epsilon_m,
                'epsilon_r': epsilon_r,
                'strength': strength
                }

    cyl = SemiInfiniteRightCylinder(unpacked)
    return cyl

def test_r_val_on_axis(cyl, epsilon=1.e-4):
    expected = 0.
    x_obs = 10. * vect_op.xhat_dm() + expected * vect_op.zhat_dm()

    found = cyl.get_r_obs(cyl.info_dict, x_obs)
    diff = found - expected

    test_val = vect_op.norm(diff)
    criteria = (test_val < epsilon)

    if not criteria:
        message = 'vortex semi-infinite cylinder: computation does not find correct radius on axis'
        print_op.log_and_raise_error(message)

def test_r_val_off_axis(cyl, epsilon=1.e-4):
    expected = 3.
    x_obs = 10. * vect_op.xhat_dm() + expected * vect_op.zhat_dm()

    found = cyl.get_r_obs(cyl.info_dict, x_obs)
    diff = found - expected

    test_val = vect_op.norm(diff)
    criteria = (test_val < epsilon)

    if not criteria:
        message = 'vortex semi-infinite cylinder: computation does not find correct radius off axis'
        print_op.log_and_raise_error(message)

def test_z_val_before_cylinder(cyl, epsilon=1.e-4):
    expected = -10.
    x_obs = expected * vect_op.xhat_dm() + 6. * vect_op.zhat_dm()

    found = cyl.get_z_obs(cyl.info_dict, x_obs)
    diff = found - expected

    test_val = vect_op.norm(diff)
    criteria = (test_val < epsilon)

    if not criteria:
        message = 'vortex semi-infinite cylinder: computation does not find correct longitude prior to cylinder'
        print_op.log_and_raise_error(message)

def test_z_val_at_start(cyl, epsilon=1.e-4):
    expected = 0.
    x_obs = expected * vect_op.xhat_dm() + 6. * vect_op.zhat_dm()

    found = cyl.get_z_obs(cyl.info_dict, x_obs)
    diff = found - expected

    test_val = vect_op.norm(diff)
    criteria = (test_val < epsilon)

    if not criteria:
        message = 'vortex semi-infinite cylinder: computation does not find correct longitude on start of cylinder'
        print_op.log_and_raise_error(message)

def test_z_val_on_cylinder(cyl, epsilon=1.e-4):
    expected = 10.
    x_obs = expected * vect_op.xhat_dm() + 6. * vect_op.zhat_dm()

    found = cyl.get_z_obs(cyl.info_dict, x_obs)
    diff = found - expected

    test_val = vect_op.norm(diff)
    criteria = (test_val < epsilon)

    if not criteria:
        message = 'vortex semi-infinite cylinder: computation does not find correct longitude on cylinder'
        print_op.log_and_raise_error(message)

def test_regularized_m_value_at_critical_point(cyl, epsilon=1.e-4):
    unpacked = cyl.unpack_info()
    r_cyl = unpacked['radius']

    r_obs = r_cyl
    z_obs = 0.
    found = cyl.get_regularized_elliptic_m_from_r_and_z(cyl.info_dict, r_obs, z_obs)

    expected = (4. * r_cyl**2.) / (4. * r_cyl**2. + 1.)
    diff = found - expected

    criteria = (diff ** 2. < epsilon ** 2.)

    if not criteria:
        message = 'vortex semi-infinite cylinder: elliptic_m regularization does not give expected value at critical point (on cylinder starting circle)'
        print_op.log_and_raise_error(message)

def test_regularized_m_value_does_not_reach_one_at_critical_point(cyl, epsilon=1.e-4):
    unpacked = cyl.unpack_info()
    r_cyl = unpacked['radius']

    r_obs = r_cyl
    z_obs = 0.
    found = cyl.get_regularized_elliptic_m_from_r_and_z(cyl.info_dict, r_obs, z_obs)

    distance = (found - 1.)

    criteria = (distance**2. > epsilon**2.)

    if not criteria:
        message = 'vortex semi-infinite cylinder: elliptic_m regularization does not work as intended, at critical point (on cylinder starting circle)'
        print_op.log_and_raise_error(message)

def test_regularized_m_value_reaches_zero_on_axis(cyl, epsilon=1.e-4):

    r_obs = 0.
    z_obs = 0.
    found = cyl.get_regularized_elliptic_m_from_r_and_z(cyl.info_dict, r_obs, z_obs)

    criteria = (found ** 2. < epsilon ** 2.)

    if not criteria:
        message = 'vortex semi-infinite cylinder: elliptic_m regularization does not work as intended, on cylinder axis'
        print_op.log_and_raise_error(message)

def test_regularized_m_value_approaches_zero_at_large_radius(cyl, epsilon=1.e-4):
    r_obs = 10.**8
    z_obs = 10.
    found = cyl.get_regularized_elliptic_m_from_r_and_z(cyl.info_dict, r_obs, z_obs)

    criteria = (found ** 2. < epsilon ** 2.)

    if not criteria:
        message = 'vortex semi-infinite cylinder: elliptic_m regularization does not work as intended, at large radius'
        print_op.log_and_raise_error(message)

def test_regularized_m_value_approaches_zero_far_downstream(cyl, epsilon=1.e-4):
    unpacked = cyl.unpack_info()
    r_cyl = unpacked['radius']

    r_obs = 1.2 * r_cyl
    z_obs = 10.**8.
    found = cyl.get_regularized_elliptic_m_from_r_and_z(cyl.info_dict, r_obs, z_obs)

    criteria = (found ** 2. < epsilon ** 2.)

    if not criteria:
        message = 'vortex semi-infinite cylinder: elliptic_m regularization does not work as intended, far downstream'
        print_op.log_and_raise_error(message)

def test_regularized_m_value_approaches_zero_far_upstream(cyl, epsilon=1.e-4):
    unpacked = cyl.unpack_info()
    r_cyl = unpacked['radius']

    r_obs = 1.2 * r_cyl
    z_obs = -1. * 10.**8.
    found = cyl.get_regularized_elliptic_m_from_r_and_z(cyl.info_dict, r_obs, z_obs)

    criteria = (found ** 2. < epsilon ** 2.)

    if not criteria:
        message = 'vortex semi-infinite cylinder: elliptic_m regularization does not work as intended, far upstream'
        print_op.log_and_raise_error(message)

###### test axes

def test_axes_when_observer_is_on_x_hat(cyl, epsilon=1.e-6):

    x_obs = 3. * vect_op.xhat_dm()

    r_hat, theta_hat, l_hat = cyl.get_observational_axes(cyl.info_dict, x_obs)

    expected_r_hat = vect_op.zhat_dm()
    expected_theta_hat = -1. * vect_op.yhat_dm()
    expected_l_hat = vect_op.xhat_dm()

    r_diff = r_hat - expected_r_hat
    theta_diff = theta_hat - expected_theta_hat
    l_diff = l_hat - expected_l_hat

    offset = cas.mtimes(r_diff.T, r_diff) + cas.mtimes(theta_diff.T, theta_diff) + cas.mtimes(l_diff.T, l_diff)

    criteria = (offset < epsilon**2.)

    if not criteria:
        message = 'vortex semi-infinite cylinder: axes generation does not work as intended, on x_hat axis'
        print_op.log_and_raise_error(message)

    return None

def test_axes_when_observer_is_on_z_hat(cyl, epsilon=1.e-6):

    x_obs = 3. * vect_op.zhat_dm()

    r_hat, theta_hat, l_hat = cyl.get_observational_axes(cyl.info_dict, x_obs)

    expected_r_hat = vect_op.zhat_dm()
    expected_theta_hat = -1. * vect_op.yhat_dm()
    expected_l_hat = vect_op.xhat_dm()

    r_diff = r_hat - expected_r_hat
    theta_diff = theta_hat - expected_theta_hat
    l_diff = l_hat - expected_l_hat

    offset = cas.mtimes(r_diff.T, r_diff) + cas.mtimes(theta_diff.T, theta_diff) + cas.mtimes(l_diff.T, l_diff)

    criteria = (offset < epsilon**2.)

    if not criteria:
        message = 'vortex semi-infinite cylinder: axes generation does not work as intended, on z_hat axis'
        print_op.log_and_raise_error(message)

    return None

def test_axes_when_observer_is_on_y_hat(cyl, epsilon=1.e-4):

    x_obs = 3. * vect_op.yhat_dm()

    r_hat, theta_hat, l_hat = cyl.get_observational_axes(cyl.info_dict, x_obs)

    expected_r_hat = vect_op.yhat_dm()
    expected_theta_hat = vect_op.zhat_dm()
    expected_l_hat = vect_op.xhat_dm()

    r_diff = r_hat - expected_r_hat
    theta_diff = theta_hat - expected_theta_hat
    l_diff = l_hat - expected_l_hat

    offset = cas.mtimes(r_diff.T, r_diff) + cas.mtimes(theta_diff.T, theta_diff) + cas.mtimes(l_diff.T, l_diff)

    criteria = (offset < epsilon**2.)

    if not criteria:
        message = 'vortex semi-infinite cylinder: axes generation does not work as intended, on y_hat axis'
        print_op.log_and_raise_error(message)

    return None


def test():
    cyl = construct_test_object()
    cyl.test_basic_criteria(expected_object_type='semi_infinite_right_cylinder')

    test_r_val_on_axis(cyl)
    test_r_val_off_axis(cyl)
    test_z_val_before_cylinder(cyl)
    test_z_val_on_cylinder(cyl)
    test_z_val_on_cylinder(cyl)

    test_axes_when_observer_is_on_x_hat(cyl)
    test_axes_when_observer_is_on_z_hat(cyl)
    test_axes_when_observer_is_on_y_hat(cyl)

    test_regularized_m_value_does_not_reach_one_at_critical_point(cyl)
    test_regularized_m_value_reaches_zero_on_axis(cyl)
    test_regularized_m_value_approaches_zero_at_large_radius(cyl)
    test_regularized_m_value_approaches_zero_far_downstream(cyl)
    test_regularized_m_value_approaches_zero_far_upstream(cyl)
    test_regularized_m_value_at_critical_point(cyl)

    return None

# test()