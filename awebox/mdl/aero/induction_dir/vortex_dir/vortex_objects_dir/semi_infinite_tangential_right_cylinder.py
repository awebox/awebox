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
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import copy
import pdb

import casadi.tools as cas
import numpy as np
import scipy.special

import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.semi_infinite_right_cylinder as obj_semi_infinite_right_cylinder
import awebox.mdl.aero.induction_dir.general_dir.tools as general_tools

import awebox.tools.struct_operations as struct_op
import awebox.tools.vector_operations as vect_op
import awebox.tools.print_operations as print_op

from awebox.logger.logger import Logger as awelogger

import matplotlib
import scipy.special as special

matplotlib.use('TkAgg')


class SemiInfiniteTangentialRightCylinder(obj_semi_infinite_right_cylinder.SemiInfiniteRightCylinder):
    # Branlard, Emmanuel & Gaunaa, Mac.(2014). Cylindrical vortex wake model: Right cylinder. Wind Energy. 524. 10.1002/we.1800.

    def __init__(self, info_dict, approximation_order_for_elliptic_integrals=6):
        super().__init__(info_dict, approximation_order_for_elliptic_integrals)
        self.set_element_type('semi_infinite_tangential_right_cylinder')


    ##### radial induction parts

    def get_regularized_biot_savart_induction_radial_component_strength_part(self, unpacked):

        strength = unpacked['strength']

        part_1 = -1. * strength / (2. * np.pi)

        return part_1

    def get_regularized_biot_savart_induction_radial_component_middle_part(self, unpacked, r_obs, z_obs):

        r_cyl = unpacked['radius']
        epsilon_r = unpacked['epsilon_r']

        part_2a = ((r_cyl + r_obs) ** 2. + z_obs ** 2. + epsilon_r**2.)**0.5
        part_2b = r_obs + epsilon_r

        num = part_2a
        den = part_2b
        part_2 = num / den

        return part_2, num, den

    def get_regularized_biot_savart_induction_radial_component_elliptic_part(self, unpacked, elliptic_m):

        approximation_order_for_elliptic_integrals = self.approximation_order_for_elliptic_integrals

        part_3a = 1. - 0.5 * elliptic_m
        part_3b = vect_op.elliptic_k(m=elliptic_m, approximation_order_for_elliptic_integrals=approximation_order_for_elliptic_integrals)
        part_3c = vect_op.elliptic_e(m=elliptic_m, approximation_order_for_elliptic_integrals=approximation_order_for_elliptic_integrals)
        part_3 = part_3a * part_3b - part_3c

        return part_3

    def get_regularized_biot_savart_induction_radial_component(self, unpacked, r_obs, z_obs, elliptic_m):

        part_1 = self.get_regularized_biot_savart_induction_radial_component_strength_part(unpacked)
        _, num2, den2 = self.get_regularized_biot_savart_induction_radial_component_middle_part(unpacked, r_obs, z_obs)
        part_3 = self.get_regularized_biot_savart_induction_radial_component_elliptic_part(unpacked, elliptic_m)

        num = part_1 * num2 * part_3
        den = den2
        found = num / den

        return found, num, den

    ##### longitudinal induction parts

    def get_regularized_biot_savart_induction_longitudinal_component_strength_part(self, unpacked):

        strength = unpacked['strength']

        part_1 = strength / 2.

        return part_1

    def get_regularized_biot_savart_induction_longitudinal_component_middle_part(self, unpacked, r_obs):

        r_cyl = unpacked['radius']
        epsilon_r = unpacked['epsilon_r']

        difference = r_cyl - r_obs
        signed = vect_op.sign(difference, eps=epsilon_r)

        # on surface = 0.5; outside = 0.; inside = 1.
        part_2 = (signed + 1.) / 2.

        return part_2

    def get_regularized_biot_savart_induction_longitudinal_component_elliptic_part(self, unpacked, r_obs, z_obs, elliptic_m0, elliptic_m):

        r_cyl = unpacked['radius']
        epsilon_r = unpacked['epsilon_r']

        approximation_order_for_elliptic_integrals = self.approximation_order_for_elliptic_integrals

        num_1 = z_obs / np.pi
        num_2 = vect_op.elliptic_k(m=elliptic_m, approximation_order_for_elliptic_integrals=approximation_order_for_elliptic_integrals)

        num_3a = (r_cyl - r_obs)
        den_3a = (r_cyl + r_obs + epsilon_r)
        num_3b = vect_op.elliptic_pi(n=elliptic_m0, m=elliptic_m, approximation_order_for_elliptic_integrals=approximation_order_for_elliptic_integrals)
        num_3 = num_3a * num_3b

        num = num_1 * (num_2 * den_3a + num_3)

        den_4 = ( (r_cyl + r_obs)**2. + z_obs**2. + epsilon_r**2)**0.5
        den = den_3a * den_4

        part_3 = num / den

        return part_3, num, den

    def get_regularized_biot_savart_induction_longitudinal_component(self, unpacked, r_obs, z_obs, elliptic_m0, elliptic_m):

        part_1 = self.get_regularized_biot_savart_induction_longitudinal_component_strength_part(unpacked)
        part_2 = self.get_regularized_biot_savart_induction_longitudinal_component_middle_part(unpacked, r_obs)
        _, num3, den3 = self.get_regularized_biot_savart_induction_longitudinal_component_elliptic_part(unpacked, r_obs, z_obs, elliptic_m0, elliptic_m)

        num = part_1 * (part_2 * den3 + num3)
        den = den3
        found = num / den

        return found, num, den

    ########### together

    def calculate_biot_savart_induction(self, unpacked_sym, x_obs):

        r_obs = self.get_r_obs(unpacked_sym, x_obs)
        z_obs = self.get_z_obs(unpacked_sym, x_obs)

        elliptic_m = self.get_regularized_elliptic_m_from_r_and_z(unpacked_sym, r_obs, z_obs)
        elliptic_m0 = self.get_regularized_elliptic_m_zero_from_r(unpacked_sym, r_obs)

        _, num_rad, den_rad = self.get_regularized_biot_savart_induction_radial_component(unpacked=unpacked_sym, r_obs=r_obs, z_obs=z_obs, elliptic_m=elliptic_m)
        _, num_long, den_long = self.get_regularized_biot_savart_induction_longitudinal_component(unpacked=unpacked_sym, r_obs=r_obs, z_obs=z_obs, elliptic_m0=elliptic_m0, elliptic_m=elliptic_m)

        r_hat, _, l_hat = self.get_observational_axes(unpacked_sym, x_obs)

        num = num_rad * r_hat * den_long + num_long * l_hat * den_rad
        den = den_rad * den_long

        found = num / den

        return found, num, den


    def get_biot_savart_reference_denominator(self, model_options, parameters, wind):

        b_ref = parameters['theta0', 'geometry', 'b_ref']
        varrho_ref = general_tools.get_option_from_possible_dicts(model_options, 'varrho_ref', 'vortex')
        r_ref = varrho_ref * b_ref

        u_ref = wind.get_speed_ref()
        den_ref = r_ref**3. * u_ref
        return den_ref

    ###### draw

    def draw(self, ax, side, variables_scaled=None, parameters=None, cosmetics=None):

        unpacked, cosmetics = self.prepare_to_draw(variables_scaled, parameters, cosmetics)

        x_center = np.reshape(unpacked['x_center'], (3, 1))
        l_hat = unpacked['l_hat']
        r_cyl = unpacked['radius']
        l_start = unpacked['l_start']

        x_obs = vect_op.zhat_np()
        a_hat, b_hat, _ = self.get_observational_axes(unpacked, x_obs)

        n_theta = cosmetics['trajectory']['cylinder_n_theta']
        n_s = cosmetics['trajectory']['cylinder_n_s']

        s_start = l_start
        delta_s = cosmetics['trajectory']['cylinder_s_length'] / float(n_s - 1)
        s_end = s_start + delta_s

        for sdx in range(n_s):

            ss = s_start + float(sdx) * s_end

            for tdx in range(n_theta):

                theta_start = 2. * np.pi / float(n_theta-1) * float(tdx)
                theta_end = 2. * np.pi / float(n_theta-1) * float(tdx + 1.)

                x_long = x_center + l_hat * ss

                x_start = x_long + r_cyl * (np.sin(theta_start) * a_hat + np.cos(theta_start) * b_hat)
                x_end = x_long + r_cyl * (np.sin(theta_end) * a_hat + np.cos(theta_end) * b_hat)

                super().basic_draw(ax, side, unpacked['strength'], x_start, x_end, cosmetics)

        return None

def construct_test_object(regularized=True):
    cyl = obj_semi_infinite_right_cylinder.construct_test_object(regularized)
    unpacked = cyl.info_dict
    tan_cyl = SemiInfiniteTangentialRightCylinder(unpacked)
    tan_cyl.define_biot_savart_induction_function()
    return tan_cyl

##### pre-test for these (local) tests

def pretest_strength_and_radius(cyl, epsilon=1.e-6):

    unpacked = cyl.info_dict
    strength = unpacked['strength']
    r_cyl = unpacked['radius']

    strength_desired = 1.
    r_cyl_desired = 1.

    cylinder_radius_is_as_expected = ((r_cyl - r_cyl_desired) ** 2 < epsilon ** 2.)
    strength_is_as_expected = ((strength - strength_desired) ** 2 < epsilon ** 2.)
    if not (cylinder_radius_is_as_expected and strength_is_as_expected):
        message = 'something went wrong in a vortex ' + cyl.element_type + ' test. check that test cylinder has radius = ' + str(r_cyl_desired)
        message += ' and strength = ' + str(strength_desired)
        print_op.log_and_raise_error(message)

    return None

def pretest_is_regularized_cylinder(cyl, epsilon=1.e-6):

    pretest_strength_and_radius(cyl, epsilon)

    unpacked = cyl.info_dict
    epsilon_r = unpacked['epsilon_r']
    epsilon_r_desired = 1.

    epsilon_r_is_as_expected = ((epsilon_r - epsilon_r_desired) ** 2 < epsilon ** 2.)
    if not epsilon_r_is_as_expected:
        message = 'something went wrong in a vortex ' + cyl.element_type + ' test. check that test cylinder has espilon_r = '
        message += str(epsilon_r_desired)
        print_op.log_and_raise_error(message)

    return None

def pretest_is_unregularized_cylinder(cyl, epsilon=1.e-6):

    pretest_strength_and_radius(cyl, epsilon)

    unpacked = cyl.info_dict
    epsilon_r = unpacked['epsilon_r']
    epsilon_r_desired = 0.

    epsilon_r_is_as_expected = ((epsilon_r - epsilon_r_desired) ** 2 < epsilon ** 2.)
    if not epsilon_r_is_as_expected:
        message = 'something went wrong in a vortex ' + cyl.element_type + ' test. check that test cylinder has espilon_r = '
        message += str(epsilon_r_desired)
        print_op.log_and_raise_error(message)

    return None


##### radial tests

def test_regularized_biot_savart_induction_radial_component_on_axis(cyl_regularized, epsilon=1.e-4):

    pretest_is_regularized_cylinder(cyl_regularized, epsilon)

    unpacked = cyl_regularized.unpack_info()

    r_obs = 0.
    z_obs = 100.

    elliptic_m = cyl_regularized.get_regularized_elliptic_m_from_r_and_z(unpacked, r_obs, z_obs)

    found, _, _ = cyl_regularized.get_regularized_biot_savart_induction_radial_component(unpacked, r_obs, z_obs, elliptic_m)
    expected = 0.

    diff = found - expected
    criteria = (diff**2. < epsilon**2.)

    if not criteria:
        message = 'vortex ' + cyl_regularized.element_type + ': regularized biot-savart induction, radial-component, computation does not behave as expected on the axis'
        print_op.log_and_raise_error(message)

def test_regularized_biot_savart_induction_radial_component_at_large_radius(cyl_regularized, epsilon=1.e-4):

    pretest_is_regularized_cylinder(cyl_regularized, epsilon)

    unpacked = cyl_regularized.unpack_info()

    r_obs = 10.**8.
    z_obs = 10.

    elliptic_m = cyl_regularized.get_regularized_elliptic_m_from_r_and_z(unpacked, r_obs, z_obs)

    found, _, _ = cyl_regularized.get_regularized_biot_savart_induction_radial_component(unpacked, r_obs, z_obs, elliptic_m)
    expected = 0.

    diff = found - expected
    criteria = (diff ** 2. < epsilon ** 2.)

    if not criteria:
        message = 'vortex ' + cyl_regularized.element_type + ': regularized biot-savart induction, radial-component, computation does not behave as expected at large radius'
        print_op.log_and_raise_error(message)

def test_biot_savart_induction_radial_component_middle_part_at_critical_point(cyl_unregularized, epsilon=1.e-4):

    pretest_is_unregularized_cylinder(cyl_unregularized, epsilon)

    unpacked = cyl_unregularized.unpack_info()
    r_cyl = unpacked['radius']

    pretest_is_unregularized_cylinder(cyl_unregularized, epsilon=epsilon)

    r_obs = r_cyl
    z_obs = 0.

    found, _, _ = cyl_unregularized.get_regularized_biot_savart_induction_radial_component_middle_part(unpacked, r_obs, z_obs)
    # expected = np.sqrt(5.)/2. if regularized
    expected = 2.

    diff = found - expected
    criteria = (diff ** 2. < epsilon ** 2.)

    if not criteria:
        message = 'vortex ' + cyl_unregularized.element_type + ': biot-savart induction, radial-component, middle part, computation does not behave as expected at criticial point'
        print_op.log_and_raise_error(message)


def test_regularized_biot_savart_induction_radial_component_elliptical_part_at_critical_point(cyl_regularized, epsilon=1.e-4):

    pretest_is_regularized_cylinder(cyl_regularized, epsilon=epsilon)

    unpacked = cyl_regularized.unpack_info()
    r_cyl = unpacked['radius']

    r_obs = r_cyl
    z_obs = 0.

    elliptic_m = cyl_regularized.get_regularized_elliptic_m_from_r_and_z(unpacked, r_obs, z_obs)
    found = cyl_regularized.get_regularized_biot_savart_induction_radial_component_elliptic_part(unpacked, elliptic_m)
    expected = 0.175833

    diff = found - expected
    criteria = (diff ** 2. < epsilon ** 2.)

    if not criteria:
        message = 'vortex ' + cyl_regularized.element_type + ': regularized biot-savart induction, radial-component, elliptic part, computation does not behave as expected at criticial point'
        print_op.log_and_raise_error(message)

def test_biot_savart_induction_radial_component_middle_part_far_upstream(cyl_unregularized, epsilon=1.e-4):

    pretest_is_unregularized_cylinder(cyl_unregularized, epsilon)

    unpacked = cyl_unregularized.unpack_info()
    r_cyl = unpacked['radius']

    r_obs = 1.2 * r_cyl
    z_obs = -1. * 10.**5.

    found, _, _ = cyl_unregularized.get_regularized_biot_savart_induction_radial_component_middle_part(unpacked, r_obs, z_obs)
    # expected = 45454.5 if regularized

    part_2a = vect_op.abs(z_obs)
    part_2b = r_obs
    expected = part_2a / part_2b

    diff = found - expected
    criteria = ( (diff/expected) ** 2. < epsilon ** 2.)

    if not criteria:
        message = 'vortex ' + cyl_unregularized.element_type + ': biot-savart induction, radial-component, middle part, computation does not behave as expected far upstream'
        print_op.log_and_raise_error(message)

def test_regularized_biot_savart_induction_radial_component_elliptical_part_far_upstream(cyl_regularized, epsilon=1.e-4):

    pretest_is_regularized_cylinder(cyl_regularized, epsilon)

    unpacked = cyl_regularized.unpack_info()
    r_cyl = unpacked['radius']

    r_obs = 1.2 * r_cyl
    z_obs = -1. * 10.**5.

    elliptic_m = cyl_regularized.get_regularized_elliptic_m_from_r_and_z(unpacked, r_obs, z_obs)
    found = cyl_regularized.get_regularized_biot_savart_induction_radial_component_elliptic_part(unpacked, elliptic_m)
    expected = 0.

    diff = found - expected
    criteria = (diff ** 2. < epsilon ** 2.)

    if not criteria:
        message = 'vortex ' + cyl_regularized.element_type + ': regularized biot-savart induction, radial-component, elliptic part, computation does not behave as expected far-upstream'
        print_op.log_and_raise_error(message)

def test_regularized_biot_savart_induction_radial_component_far_upstream(cyl_regularized, epsilon=1.e-4):

    pretest_is_regularized_cylinder(cyl_regularized)

    unpacked = cyl_regularized.unpack_info()
    r_cyl = unpacked['radius']

    r_obs = 1.2 * r_cyl
    z_obs = -1. * 10.**5.

    elliptic_m = cyl_regularized.get_regularized_elliptic_m_from_r_and_z(unpacked, r_obs, z_obs)

    found, _, _ = cyl_regularized.get_regularized_biot_savart_induction_radial_component(unpacked, r_obs, z_obs, elliptic_m)
    expected = 0.

    diff = found - expected
    criteria = (diff ** 2. < epsilon ** 2.)

    if not criteria:
        message = 'vortex ' + cyl_regularized.element_type + ': regularized biot-savart induction, radial-component, computation does not behave as expected far-upstream'
        print_op.log_and_raise_error(message)

def test_biot_savart_induction_radial_component_middle_part_far_downstream(cyl_unregularized, epsilon=1.e-4):

    pretest_is_unregularized_cylinder(cyl_unregularized)

    unpacked = cyl_unregularized.unpack_info()
    r_cyl = unpacked['radius']

    r_obs = 0.8 * r_cyl
    z_obs = 10.**5.

    found, _, _ = cyl_unregularized.get_regularized_biot_savart_induction_radial_component_middle_part(unpacked, r_obs, z_obs)
    # expected = 55555.555 if regularized

    part_2a = z_obs
    part_2b = r_obs
    expected = part_2a / part_2b

    diff = found - expected
    criteria = ( (diff/expected) ** 2. < epsilon ** 2.)

    if not criteria:
        message = 'vortex ' + cyl_unregularized.element_type + ': biot-savart induction, radial-component, middle part, computation does not behave as expected far downstream'
        print_op.log_and_raise_error(message)

def test_regularized_biot_savart_induction_radial_component_elliptical_part_far_downstream(cyl_regularized, epsilon=1.e-4):

    pretest_is_regularized_cylinder(cyl_regularized, epsilon)

    unpacked = cyl_regularized.unpack_info()
    r_cyl = unpacked['radius']

    r_obs = 0.8 * r_cyl
    z_obs = 10.**5.

    elliptic_m = cyl_regularized.get_regularized_elliptic_m_from_r_and_z(unpacked, r_obs, z_obs)
    found = cyl_regularized.get_regularized_biot_savart_induction_radial_component_elliptic_part(unpacked, elliptic_m)
    expected = 0.

    diff = found - expected
    criteria = (diff ** 2. < epsilon ** 2.)

    if not criteria:
        message = 'vortex ' + cyl_regularized.element_type + ': regularized biot-savart induction, radial-component, elliptic part, computation does not behave as expected far-downstream'
        print_op.log_and_raise_error(message)

def test_regularized_biot_savart_induction_radial_component_far_downstream(cyl_regularized, epsilon=1.e-4):

    pretest_is_regularized_cylinder(cyl_regularized)

    unpacked = cyl_regularized.unpack_info()
    r_cyl = unpacked['radius']

    r_obs = 0.8 * r_cyl
    z_obs = 10.**5.

    elliptic_m = cyl_regularized.get_regularized_elliptic_m_from_r_and_z(unpacked, r_obs, z_obs)

    found, _, _ = cyl_regularized.get_regularized_biot_savart_induction_radial_component(unpacked, r_obs, z_obs, elliptic_m)
    expected = 0.

    diff = found - expected
    criteria = (diff ** 2. < epsilon ** 2.)

    if not criteria:
        message = 'vortex ' + cyl_regularized.element_type + ': regularized biot-savart induction, radial-component, computation does not behave as expected far-downstream'
        print_op.log_and_raise_error(message)


def test_regularized_biot_savart_induction_radial_component(cyl_regularized, cyl_unregularized):
    test_regularized_biot_savart_induction_radial_component_on_axis(cyl_regularized)

    test_biot_savart_induction_radial_component_middle_part_at_critical_point(cyl_unregularized)
    test_regularized_biot_savart_induction_radial_component_elliptical_part_at_critical_point(cyl_regularized)

    test_regularized_biot_savart_induction_radial_component_at_large_radius(cyl_regularized)

    test_biot_savart_induction_radial_component_middle_part_far_upstream(cyl_unregularized)
    test_regularized_biot_savart_induction_radial_component_elliptical_part_far_upstream(cyl_regularized)
    test_regularized_biot_savart_induction_radial_component_far_upstream(cyl_regularized)

    test_biot_savart_induction_radial_component_middle_part_far_downstream(cyl_unregularized)
    test_regularized_biot_savart_induction_radial_component_elliptical_part_far_downstream(cyl_regularized)
    test_regularized_biot_savart_induction_radial_component_far_downstream(cyl_regularized)

    return None

##### longitudinal tests

def test_biot_savart_induction_longitudinal_component_on_axis(cyl_unregularized, epsilon=1.e-4):

    pretest_is_unregularized_cylinder(cyl_unregularized)

    unpacked = cyl_unregularized.info_dict
    strength = unpacked['strength']
    r_cyl = unpacked['radius']

    r_obs = 0.
    z_obs = 100.

    elliptic_m0 = cyl_unregularized.get_regularized_elliptic_m_zero_from_r(unpacked, r_obs)
    elliptic_m = cyl_unregularized.get_regularized_elliptic_m_from_r_and_z(unpacked, r_obs, z_obs)
    found, _, _ = cyl_unregularized.get_regularized_biot_savart_induction_longitudinal_component(unpacked, r_obs, z_obs, elliptic_m0, elliptic_m)

    expected = (strength / 2.) * ( 1. + z_obs / ( (r_cyl**2. + z_obs**2.)**0.5 ) )

    diff = found - expected
    criteria = (diff ** 2. < epsilon ** 2.)

    if not criteria:
        message = 'vortex ' + cyl_unregularized.element_type + ': biot-savart induction, longitudinal-component, computation does not behave as expected on-axis'
        print_op.log_and_raise_error(message)

    return None

def test_biot_savart_induction_longitudinal_component_inside_cylinder_at_start(cyl_unregularized, epsilon=1.e-4):

    pretest_is_unregularized_cylinder(cyl_unregularized)

    unpacked = cyl_unregularized.info_dict
    r_cyl = unpacked['radius']
    strength = unpacked['strength']

    r_obs = r_cyl / 3.
    z_obs = 0.

    elliptic_m0 = cyl_unregularized.get_regularized_elliptic_m_zero_from_r(unpacked, r_obs)
    elliptic_m = cyl_unregularized.get_regularized_elliptic_m_from_r_and_z(unpacked, r_obs, z_obs)
    found, _, _ = cyl_unregularized.get_regularized_biot_savart_induction_longitudinal_component(unpacked, r_obs, z_obs, elliptic_m0, elliptic_m)

    expected = strength / 2.

    diff = found - expected
    criteria = (diff ** 2. < epsilon ** 2.)

    if not criteria:
        message = 'vortex ' + cyl_unregularized.element_type + ': biot-savart induction, longitudinal-component, computation does not behave inside cylinder at start'
        print_op.log_and_raise_error(message)

    return None


def test_biot_savart_induction_longitudinal_component_outside_cylinder_at_start(cyl_unregularized, epsilon=1.e-4):

    pretest_is_unregularized_cylinder(cyl_unregularized)

    unpacked = cyl_unregularized.info_dict
    r_cyl = unpacked['radius']

    r_obs = 2. * r_cyl
    z_obs = 0.

    elliptic_m0 = cyl_unregularized.get_regularized_elliptic_m_zero_from_r(unpacked, r_obs)
    elliptic_m = cyl_unregularized.get_regularized_elliptic_m_from_r_and_z(unpacked, r_obs, z_obs)
    found, _, _ = cyl_unregularized.get_regularized_biot_savart_induction_longitudinal_component(unpacked, r_obs, z_obs, elliptic_m0, elliptic_m)

    expected = 0.

    diff = found - expected
    criteria = (diff ** 2. < epsilon ** 2.)

    if not criteria:
        message = 'vortex ' + cyl_unregularized.element_type + ': biot-savart induction, longitudinal-component, computation does not behave outside cylinder at start'
        print_op.log_and_raise_error(message)

    return None


def test_biot_savart_induction_longitudinal_component_inside_cylinder_far_upstream(cyl_unregularized, epsilon=1.e-4):

    pretest_is_unregularized_cylinder(cyl_unregularized, epsilon)

    unpacked = cyl_unregularized.info_dict
    r_cyl = unpacked['radius']

    r_obs = r_cyl / 3.
    z_obs = -1. * 10**6.

    elliptic_m0 = cyl_unregularized.get_regularized_elliptic_m_zero_from_r(unpacked, r_obs)
    elliptic_m = cyl_unregularized.get_regularized_elliptic_m_from_r_and_z(unpacked, r_obs, z_obs)
    found, _, _ = cyl_unregularized.get_regularized_biot_savart_induction_longitudinal_component(unpacked, r_obs, z_obs, elliptic_m0, elliptic_m)

    expected = 0.

    diff = found - expected
    criteria = (diff ** 2. < epsilon ** 2.)

    if not criteria:
        message = 'vortex ' + cyl_unregularized.element_type + ': biot-savart induction, longitudinal-component, computation does not behave far-upstream'
        print_op.log_and_raise_error(message)

    return None

def test_biot_savart_induction_longitudinal_component_inside_cylinder_far_downstream(cyl_unregularized, epsilon=1.e-4):

    pretest_is_unregularized_cylinder(cyl_unregularized)

    unpacked = cyl_unregularized.info_dict
    strength = unpacked['strength']
    r_cyl = unpacked['radius']

    r_obs = r_cyl / 3.
    z_obs = 10**4.

    elliptic_m0 = cyl_unregularized.get_regularized_elliptic_m_zero_from_r(unpacked, r_obs)
    elliptic_m = cyl_unregularized.get_regularized_elliptic_m_from_r_and_z(unpacked, r_obs, z_obs)
    found, _, _ = cyl_unregularized.get_regularized_biot_savart_induction_longitudinal_component(unpacked, r_obs, z_obs, elliptic_m0, elliptic_m)

    expected = strength

    diff = found - expected
    criteria = (diff ** 2. < epsilon ** 2.)

    if not criteria:
        message = 'vortex ' + cyl_unregularized.element_type + ': biot-savart induction, longitudinal-component, computation does not behave far-downstream'
        print_op.log_and_raise_error(message)

    return None

def test_regularized_biot_savart_induction_longitudinal_component_outside_cylinder_far_downstream(cyl_regularized, epsilon=1.e-4):

    pretest_is_regularized_cylinder(cyl_regularized)

    unpacked = cyl_regularized.info_dict
    r_cyl = unpacked['radius']
    epsilon_r = unpacked['epsilon_r']
    strength = unpacked['strength']

    r_obs = 10. * r_cyl
    z_obs = 10.**8.

    elliptic_m0 = cyl_regularized.get_regularized_elliptic_m_zero_from_r(unpacked, r_obs)
    elliptic_m = cyl_regularized.get_regularized_elliptic_m_from_r_and_z(unpacked, r_obs, z_obs)
    found, _, _ = cyl_regularized.get_regularized_biot_savart_induction_longitudinal_component(unpacked, r_obs, z_obs, elliptic_m0, elliptic_m)

    elliptic_k = (4. * r_cyl * r_obs / ((r_cyl + r_obs)**2. + z_obs**2. + epsilon_r**2.))**0.5
    elliptic_k0 = (4. * r_cyl * r_obs / ((r_cyl + r_obs)**2. + 0.**2. + epsilon_r**2.))**0.5
    part_1 = strength/2.
    signed = vect_op.sign((r_cyl - r_obs), eps=epsilon_r)
    part_2 = (signed + 1.) / 2.
    part_3a = z_obs * elliptic_k / (2. * np.pi * (r_obs * r_cyl)**0.5)
    part_3b = np.pi / 2. # (elliptic_k(m = 0.))
    elliptic_pi_of_m0_and_0 = np.pi / (2. * (1. - elliptic_k0**2.)**0.5)
    part_3c = (r_cyl - r_obs) / (r_cyl + r_obs + epsilon_r) * elliptic_pi_of_m0_and_0
    part_3 = part_3a * (part_3b + part_3c)
    expected = part_1 * (part_2 + part_3)

    diff = found - expected
    criteria = (diff ** 2. < epsilon ** 2.)

    if not criteria:
        message = 'vortex ' + cyl_regularized.element_type + ': regularized biot-savart induction, longitudinal-component, computation does not outside cylinder, far-downstream'
        print_op.log_and_raise_error(message)

    return None



def test_regularized_biot_savart_induction_longitudinal_component_on_surface(cyl_regularized, epsilon=1.e-4):

    pretest_is_regularized_cylinder(cyl_regularized)

    unpacked = cyl_regularized.info_dict
    strength = unpacked['strength']
    r_cyl = unpacked['radius']
    epsilon_r = unpacked['epsilon_r']

    r_obs = r_cyl
    z_obs = 10.

    elliptic_m0 = cyl_regularized.get_regularized_elliptic_m_zero_from_r(unpacked, r_obs)
    elliptic_m = cyl_regularized.get_regularized_elliptic_m_from_r_and_z(unpacked, r_obs, z_obs)
    found, _, _ = cyl_regularized.get_regularized_biot_savart_induction_longitudinal_component(unpacked, r_obs, z_obs, elliptic_m0, elliptic_m)

    elliptic_k = (4. / (4. + z_obs**2. + epsilon_r**2.))**0.5
    part_1 = strength/2.
    part_2 = 0.5
    part_3a = z_obs * elliptic_k / (2. * np.pi)
    part_3b = special.ellipk(elliptic_k**2.)
    part_3c = 0.
    part_3 = part_3a * (part_3b + part_3c)
    expected = part_1 * (part_2 + part_3)

    diff = found - expected
    criteria = (diff ** 2. < epsilon ** 2.)

    if not criteria:
        message = 'vortex ' + cyl_regularized.element_type + ': regularized biot-savart induction, longitudinal-component, computation does not behave on cylinder surface'
        print_op.log_and_raise_error(message)

    return None


def test_regularized_biot_savart_induction_longitudinal_component_at_critical_point(cyl_regularized, epsilon=1.e-4):

    pretest_is_regularized_cylinder(cyl_regularized)

    unpacked = cyl_regularized.info_dict
    strength = unpacked['strength']
    r_cyl = unpacked['radius']

    r_obs = r_cyl
    z_obs = 0.

    elliptic_m0 = cyl_regularized.get_regularized_elliptic_m_zero_from_r(unpacked, r_obs)
    elliptic_m = cyl_regularized.get_regularized_elliptic_m_from_r_and_z(unpacked, r_obs, z_obs)
    found, _, _ = cyl_regularized.get_regularized_biot_savart_induction_longitudinal_component(unpacked, r_obs, z_obs, elliptic_m0, elliptic_m)

    part_1 = strength/2.
    part_2 = 0.5
    part_3 = 0.
    expected = part_1 * (part_2 + part_3)

    diff = found - expected
    criteria = (diff ** 2. < epsilon ** 2.)

    if not criteria:
        message = 'vortex ' + cyl_regularized.element_type + ': regularized biot-savart induction, longitudinal-component, computation does not behave at critical point'
        print_op.log_and_raise_error(message)

    return None



def test_regularized_biot_savart_induction_longitudinal_component(cyl_regularized, cyl_unregularized):
    test_biot_savart_induction_longitudinal_component_on_axis(cyl_unregularized)
    test_biot_savart_induction_longitudinal_component_inside_cylinder_at_start(cyl_unregularized)
    test_biot_savart_induction_longitudinal_component_outside_cylinder_at_start(cyl_unregularized)
    test_biot_savart_induction_longitudinal_component_inside_cylinder_far_upstream(cyl_unregularized)
    test_biot_savart_induction_longitudinal_component_inside_cylinder_far_downstream(cyl_unregularized)

    test_regularized_biot_savart_induction_longitudinal_component_outside_cylinder_far_downstream(cyl_regularized)
    test_regularized_biot_savart_induction_longitudinal_component_on_surface(cyl_regularized)
    test_regularized_biot_savart_induction_longitudinal_component_at_critical_point(cyl_regularized)

    return None

###### test joined biot savart function

def test_biot_savart_function(cyl_unregularized, epsilon=1.e-4):

    pretest_is_unregularized_cylinder(cyl_unregularized)

    unpacked = cyl_unregularized.info_dict
    strength = unpacked['strength']
    x_center = unpacked['x_center']
    r_cyl = unpacked['radius']

    r_obs = 2.
    z_obs = 1.

    l_hat = vect_op.xhat_dm()
    r_hat = vect_op.normalize(vect_op.yhat_dm() + vect_op.zhat_dm())
    x_obs = x_center + z_obs * l_hat + r_obs * r_hat

    elliptic_m = 4 * r_obs * r_cyl / ((r_obs + r_cyl)**2. + z_obs**2.)
    elliptic_m0 = 4 * r_obs * r_cyl / ((r_obs + r_cyl)**2.)
    elliptic_k = (elliptic_m)**0.5

    rad_part_1 = -1. * strength / (2. * np.pi) * (r_cyl**0.5 / r_obs**0.5)
    rad_part_2 = ((2. - elliptic_m) / elliptic_k) * special.ellipk(elliptic_m)
    rad_part_3 = 2. / elliptic_k * special.ellipe(elliptic_m)
    radial_component = rad_part_1 * (rad_part_2 - rad_part_3)

    long_part_1 = strength / 2.
    long_part_2 = ((r_cyl - r_obs) + np.abs(r_cyl - r_obs)) / (2. * np.abs(r_cyl - r_obs))
    long_part_3a = z_obs * elliptic_k / (2. * np.pi * (r_obs * r_cyl)**0.5)
    long_part_3b = special.ellipk(elliptic_m)
    long_part_3c = (r_cyl - r_obs) / (r_cyl + r_obs) * vect_op.elliptic_pi(n=elliptic_m0, m=elliptic_m)
    long_part_3 = long_part_3a * (long_part_3b + long_part_3c)
    longitudinal_component = long_part_1 * (long_part_2 + long_part_3)

    expected = radial_component * r_hat + longitudinal_component * l_hat

    packed_info = cyl_unregularized.info
    biot_savart_fun = cyl_unregularized.biot_savart_fun
    found = biot_savart_fun(packed_info, x_obs)

    diff = expected - found

    criteria = (cas.mtimes(diff.T, diff) < epsilon**2.)

    if not criteria:
        message = 'vortex ' + cyl_unregularized.element_type + ': biot-savart function does not work as intended.'
        print_op.log_and_raise_error(message)

    return None


####### concatenate tests

def test(test_includes_visualization=False):

    cyl_regularized = construct_test_object(regularized=True)
    cyl_regularized.test_basic_criteria(expected_object_type='semi_infinite_tangential_right_cylinder')

    cyl_unregularized = construct_test_object(regularized=False)
    cyl_unregularized.test_basic_criteria(expected_object_type='semi_infinite_tangential_right_cylinder')

    test_regularized_biot_savart_induction_radial_component(cyl_regularized, cyl_unregularized)
    test_regularized_biot_savart_induction_longitudinal_component(cyl_regularized, cyl_unregularized)

    test_biot_savart_function(cyl_unregularized)

    cyl_regularized.test_draw(test_includes_visualization)

    return None

# test()