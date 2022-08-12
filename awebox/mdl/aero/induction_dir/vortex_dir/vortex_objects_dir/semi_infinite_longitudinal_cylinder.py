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
import matplotlib.pyplot as plt
import scipy.special as special

import numpy as np

import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.semi_infinite_cylinder as obj_semi_infinite_cylinder

import awebox.tools.struct_operations as struct_op
import awebox.tools.vector_operations as vect_op
import awebox.tools.print_operations as print_op

from awebox.logger.logger import Logger as awelogger

import matplotlib
matplotlib.use('TkAgg')


class SemiInfiniteLongitudinalCylinder(obj_semi_infinite_cylinder.SemiInfiniteCylinder):
    # Branlard, Emmanuel & Gaunaa, Mac.(2014). Cylindrical vortex wake model: Right cylinder. Wind Energy. 524. 10.1002/we.1800.

    def __init__(self, info_dict):
        super().__init__(info_dict)
        self.set_element_type('semi_infinite_longitudinal_cylinder')

    ##### tangential induction parts

    def get_regularized_biot_savart_induction_tangential_component_strength_part(self, r_obs):

        unpacked = self.info_dict
        strength = unpacked['strength']
        r_cyl = unpacked['radius']
        epsilon_r = unpacked['epsilon_r']

        base = strength * r_cyl/ 2.
        cut_off = r_obs / (r_obs**2. + epsilon_r**2.)
        part_1 = base * cut_off

        return part_1

    def get_regularized_biot_savart_induction_tangential_component_middle_part(self, r_obs):

        unpacked = self.info_dict

        r_cyl = unpacked['radius']
        epsilon_r = unpacked['epsilon_r']

        difference = r_cyl - r_obs
        signed = vect_op.sign(difference, eps=epsilon_r)

        # on surface = 0.5; outside = 1.; inside = 0.
        part_2 = (1. - signed) / 2.

        return part_2

    def get_regularized_biot_savart_induction_tangential_component_elliptic_part(self, r_obs, z_obs, elliptic_m0, elliptic_m):

        unpacked = self.info_dict
        r_cyl = unpacked['radius']
        epsilon_r = unpacked['epsilon_r']

        num_1 = z_obs / np.pi
        num_2 = vect_op.elliptic_k(m=elliptic_m)
        num_3a = (r_cyl - r_obs) / (r_cyl + r_obs + epsilon_r)
        num_3b = vect_op.elliptic_pi(n=elliptic_m0, m=elliptic_m)
        num_3 = num_3a * num_3b

        num = num_1 * (num_2 - num_3)

        den = ( (r_cyl + r_obs)**2. + z_obs**2. + epsilon_r**2)**0.5

        part_3 = num/den

        return part_3


    def get_regularized_biot_savart_induction_tangential_component(self, r_obs, z_obs, elliptic_m0, elliptic_m):

        part_1 = self.get_regularized_biot_savart_induction_tangential_component_strength_part(r_obs)
        part_2 = self.get_regularized_biot_savart_induction_tangential_component_middle_part(r_obs)
        part_3 = self.get_regularized_biot_savart_induction_tangential_component_elliptic_part(r_obs, z_obs, elliptic_m0, elliptic_m)

        found = part_1 * (part_2 + part_3)

        return found

    ########### together

    def define_biot_savart_induction_function(self):

        expected_info_length = self.expected_info_length
        packed_sym = cas.SX.sym('packed_sym', (expected_info_length, 1))

        x_obs = cas.SX.sym('x_obs', (3, 1))

        r_obs = self.get_r_obs(x_obs)
        z_obs = self.get_z_obs(x_obs)

        elliptic_m = self.get_regularized_elliptic_m_from_r_and_z(r_obs, z_obs)
        elliptic_m0 = self.get_regularized_elliptic_m_zero_from_r(r_obs)

        tangential_component = self.get_regularized_biot_savart_induction_tangential_component(r_obs=r_obs, z_obs=z_obs, elliptic_m0=elliptic_m0, elliptic_m=elliptic_m)

        _, theta_hat, _ = self.get_observational_axes(x_obs)

        value = tangential_component * theta_hat

        biot_savart_fun = cas.Function('biot_savart_fun', [packed_sym, x_obs], [value])
        self.set_biot_savart_fun(biot_savart_fun)

        return None


    ##### draw

    def draw(self, ax, side, variables_scaled=None, parameters=None, cosmetics=None):

        unpacked, cosmetics = self.prepare_to_draw(variables_scaled, parameters, cosmetics)

        x_center = np.reshape(unpacked['x_center'], (3, 1))
        l_hat = unpacked['l_hat']
        r_cyl = unpacked['radius']
        l_start = unpacked['l_start']

        x_obs = vect_op.zhat_np()
        a_hat, b_hat, _ = self.get_observational_axes(x_obs)

        s_start = l_start
        s_end = s_start + cosmetics['trajectory']['cylinder_s_length']

        n_s = cosmetics['trajectory']['cylinder_n_s']

        for sdx in range(n_s):
            theta = 2. * np.pi * float(sdx) / float(n_s)

            x_angular = r_cyl * (np.sin(theta) * a_hat + np.cos(theta) * b_hat)

            x_start = x_center + l_hat * s_start + x_angular
            x_end = x_center + l_hat * s_end + x_angular

            super().basic_draw(ax, side, unpacked['strength'], x_start, x_end, cosmetics)

        return None



def construct_test_object(regularized=True):
    cyl = obj_semi_infinite_cylinder.construct_test_object(regularized)
    unpacked = cyl.info_dict
    long_cyl = SemiInfiniteLongitudinalCylinder(unpacked)
    long_cyl.define_biot_savart_induction_function()
    return long_cyl

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
        awelogger.logger.error(message)
        raise Exception(message)

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
        awelogger.logger.error(message)
        raise Exception(message)

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
        awelogger.logger.error(message)
        raise Exception(message)

    return None

##### tangential tests

def test_biot_savart_induction_tangential_component_on_axis(cyl_unregularized, epsilon=1.e-4):

    pretest_is_unregularized_cylinder(cyl_unregularized)

    r_obs = 0.
    z_obs = 100.

    elliptic_m0 = cyl_unregularized.get_regularized_elliptic_m_zero_from_r(r_obs)
    elliptic_m = cyl_unregularized.get_regularized_elliptic_m_from_r_and_z(r_obs, z_obs)
    found = cyl_unregularized.get_regularized_biot_savart_induction_tangential_component(r_obs, z_obs, elliptic_m0, elliptic_m)

    expected = 0.

    diff = found - expected
    criteria = (diff ** 2. < epsilon ** 2.)

    if not criteria:
        message = 'vortex ' + cyl_unregularized.element_type + ': biot-savart induction, tangential-component, computation does not behave as expected on-axis'
        awelogger.logger.error(message)
        raise Exception(message)

    return None


def test_biot_savart_induction_tangential_component_inside_cylinder_at_start(cyl_unregularized, epsilon=1.e-4):

    pretest_is_unregularized_cylinder(cyl_unregularized)

    unpacked = cyl_unregularized.info_dict
    r_cyl = unpacked['radius']

    r_obs = r_cyl / 3.
    z_obs = 0.

    elliptic_m0 = cyl_unregularized.get_regularized_elliptic_m_zero_from_r(r_obs)
    elliptic_m = cyl_unregularized.get_regularized_elliptic_m_from_r_and_z(r_obs, z_obs)
    found = cyl_unregularized.get_regularized_biot_savart_induction_tangential_component(r_obs, z_obs, elliptic_m0,
                                                                                         elliptic_m)
    expected = 0.

    diff = found - expected
    criteria = (diff ** 2. < epsilon ** 2.)

    if not criteria:
        message = 'vortex ' + cyl_unregularized.element_type + ': biot-savart induction, tangential-component, computation does not behave inside cylinder at start'
        awelogger.logger.error(message)
        raise Exception(message)

    return None


def test_biot_savart_induction_tangential_component_outside_cylinder_at_start(cyl_unregularized, epsilon=1.e-4):

    pretest_is_unregularized_cylinder(cyl_unregularized)

    unpacked = cyl_unregularized.info_dict
    r_cyl = unpacked['radius']
    strength = unpacked['strength']

    r_obs = 2. * r_cyl
    z_obs = 0.

    elliptic_m0 = cyl_unregularized.get_regularized_elliptic_m_zero_from_r(r_obs)
    elliptic_m = cyl_unregularized.get_regularized_elliptic_m_from_r_and_z(r_obs, z_obs)
    found = cyl_unregularized.get_regularized_biot_savart_induction_tangential_component(r_obs, z_obs, elliptic_m0,
                                                                                         elliptic_m)
    expected = strength / 2. * (r_cyl / r_obs)

    diff = found - expected
    criteria = (diff ** 2. < epsilon ** 2.)

    if not criteria:
        message = 'vortex ' + cyl_unregularized.element_type + ': biot-savart induction, tangential-component, computation does not behave outside cylinder at start'
        awelogger.logger.error(message)
        raise Exception(message)

    return None

def test_regularized_biot_savart_induction_tangential_component_inside_cylinder_far_upstream(cyl_regularized, epsilon=1.e-4):

    pretest_is_regularized_cylinder(cyl_regularized, epsilon)

    unpacked = cyl_regularized.info_dict
    strength = unpacked['strength']
    r_cyl = unpacked['radius']
    epsilon_r = unpacked['epsilon_r']

    r_obs = r_cyl / 3.
    z_obs = -1. * 10**6.

    elliptic_m0 = cyl_regularized.get_regularized_elliptic_m_zero_from_r(r_obs)
    elliptic_m = cyl_regularized.get_regularized_elliptic_m_from_r_and_z(r_obs, z_obs)
    found = cyl_regularized.get_regularized_biot_savart_induction_tangential_component(r_obs, z_obs, elliptic_m0, elliptic_m)

    elliptic_k = (4. * r_cyl * r_obs / ((r_cyl + r_obs)**2. + z_obs**2. + epsilon_r**2.))**0.5
    elliptic_k0 = (4. * r_cyl * r_obs / ((r_cyl + r_obs)**2. + 0.**2. + epsilon_r**2.))**0.5
    part_1 = strength/2. * r_cyl * (r_obs / (r_obs**2. + epsilon_r**2.))
    signed = vect_op.sign((r_cyl - r_obs), eps=epsilon_r)
    part_2 = (1. - signed) / 2.
    part_3a = z_obs * elliptic_k / (2. * np.pi * (r_obs * r_cyl)**0.5)
    part_3b = np.pi / 2. # (elliptic_k(m = 0.))
    elliptic_pi_of_m0_and_0 = np.pi / (2. * (1. - elliptic_k0**2.)**0.5)
    part_3c = (r_cyl - r_obs) / (r_cyl + r_obs + epsilon_r) * elliptic_pi_of_m0_and_0
    part_3 = part_3a * (part_3b - part_3c)
    expected = part_1 * (part_2 + part_3)

    diff = found - expected
    criteria = (diff ** 2. < epsilon ** 2.)

    if not criteria:
        message = 'vortex ' + cyl_regularized.element_type + ': regularized biot-savart induction, tangential-component, computation does not behave far-upstream, inside cylinder'
        awelogger.logger.error(message)
        raise Exception(message)

    return None

def test_biot_savart_induction_tangential_component_inside_cylinder_far_upstream(cyl_unregularized, epsilon=1.e-4):

    pretest_is_unregularized_cylinder(cyl_unregularized, epsilon)

    unpacked = cyl_unregularized.info_dict
    r_cyl = unpacked['radius']

    r_obs = r_cyl / 3.
    z_obs = -1. * 10**6.

    elliptic_m0 = cyl_unregularized.get_regularized_elliptic_m_zero_from_r(r_obs)
    elliptic_m = cyl_unregularized.get_regularized_elliptic_m_from_r_and_z(r_obs, z_obs)
    found = cyl_unregularized.get_regularized_biot_savart_induction_tangential_component(r_obs, z_obs, elliptic_m0, elliptic_m)

    expected = 0.

    diff = found - expected
    criteria = (diff ** 2. < epsilon ** 2.)

    if not criteria:
        message = 'vortex ' + cyl_unregularized.element_type + ': biot-savart induction, tangential-component, computation does not behave far-upstream, inside cylinder'
        awelogger.logger.error(message)
        raise Exception(message)

    return None

def test_regularized_biot_savart_induction_tangential_component_inside_cylinder_far_downstream(cyl_regularized, epsilon=1.e-4):

    pretest_is_regularized_cylinder(cyl_regularized, epsilon)

    unpacked = cyl_regularized.info_dict
    strength = unpacked['strength']
    r_cyl = unpacked['radius']
    epsilon_r = unpacked['epsilon_r']

    r_obs = r_cyl / 3.
    z_obs = 1. * 10**6.

    elliptic_m0 = cyl_regularized.get_regularized_elliptic_m_zero_from_r(r_obs)
    elliptic_m = cyl_regularized.get_regularized_elliptic_m_from_r_and_z(r_obs, z_obs)
    found = cyl_regularized.get_regularized_biot_savart_induction_tangential_component(r_obs, z_obs, elliptic_m0, elliptic_m)

    elliptic_k = (4. * r_cyl * r_obs / ((r_cyl + r_obs)**2. + z_obs**2. + epsilon_r**2.))**0.5
    elliptic_k0 = (4. * r_cyl * r_obs / ((r_cyl + r_obs)**2. + 0.**2. + epsilon_r**2.))**0.5
    part_1 = strength/2. * r_cyl * (r_obs / (r_obs**2. + epsilon_r**2.))
    signed = vect_op.sign((r_cyl - r_obs), eps=epsilon_r)
    part_2 = (1. - signed) / 2.
    part_3a = z_obs * elliptic_k / (2. * np.pi * (r_obs * r_cyl)**0.5)
    part_3b = np.pi / 2. # (elliptic_k(m = 0.))
    elliptic_pi_of_m0_and_0 = np.pi / (2. * (1. - elliptic_k0**2.)**0.5)
    part_3c = (r_cyl - r_obs) / (r_cyl + r_obs + epsilon_r) * elliptic_pi_of_m0_and_0
    part_3 = part_3a * (part_3b - part_3c)
    expected = part_1 * (part_2 + part_3)

    diff = found - expected
    criteria = (diff ** 2. < epsilon ** 2.)

    if not criteria:
        message = 'vortex ' + cyl_regularized.element_type + ': regularized biot-savart induction, tangential-component, computation does not behave far-downstream, inside cylinder'
        awelogger.logger.error(message)
        raise Exception(message)

    return None

def test_biot_savart_induction_tangential_component_inside_cylinder_far_downstream(cyl_unregularized, epsilon=1.e-4):

    pretest_is_unregularized_cylinder(cyl_unregularized, epsilon)

    unpacked = cyl_unregularized.info_dict
    r_cyl = unpacked['radius']

    r_obs = r_cyl / 3.
    z_obs = 1. * 10**6.

    elliptic_m0 = cyl_unregularized.get_regularized_elliptic_m_zero_from_r(r_obs)
    elliptic_m = cyl_unregularized.get_regularized_elliptic_m_from_r_and_z(r_obs, z_obs)
    found = cyl_unregularized.get_regularized_biot_savart_induction_tangential_component(r_obs, z_obs, elliptic_m0, elliptic_m)

    expected = 0.

    diff = found - expected
    criteria = (diff ** 2. < epsilon ** 2.)

    if not criteria:
        message = 'vortex ' + cyl_unregularized.element_type + ': biot-savart induction, tangential-component, computation does not behave far-downstream, inside cylinder'
        awelogger.logger.error(message)
        raise Exception(message)

    return None


def test_regularized_biot_savart_induction_tangential_component_outside_cylinder_far_upstream(cyl_regularized, epsilon=1.e-4):

    pretest_is_regularized_cylinder(cyl_regularized, epsilon)

    unpacked = cyl_regularized.info_dict
    strength = unpacked['strength']
    r_cyl = unpacked['radius']
    epsilon_r = unpacked['epsilon_r']

    r_obs = 2. * r_cyl
    z_obs = -1. * 10**6.

    elliptic_m0 = cyl_regularized.get_regularized_elliptic_m_zero_from_r(r_obs)
    elliptic_m = cyl_regularized.get_regularized_elliptic_m_from_r_and_z(r_obs, z_obs)
    found = cyl_regularized.get_regularized_biot_savart_induction_tangential_component(r_obs, z_obs, elliptic_m0, elliptic_m)

    elliptic_k = (4. * r_cyl * r_obs / ((r_cyl + r_obs)**2. + z_obs**2. + epsilon_r**2.))**0.5
    elliptic_k0 = (4. * r_cyl * r_obs / ((r_cyl + r_obs)**2. + 0.**2. + epsilon_r**2.))**0.5
    part_1 = strength/2. * r_cyl * (r_obs / (r_obs**2. + epsilon_r**2.))
    signed = vect_op.sign((r_cyl - r_obs), eps=epsilon_r)
    part_2 = (1. - signed) / 2.
    part_3a = z_obs * elliptic_k / (2. * np.pi * (r_obs * r_cyl)**0.5)
    part_3b = np.pi / 2. # (elliptic_k(m = 0.))
    elliptic_pi_of_m0_and_0 = np.pi / (2. * (1. - elliptic_k0**2.)**0.5)
    part_3c = (r_cyl - r_obs) / (r_cyl + r_obs + epsilon_r) * elliptic_pi_of_m0_and_0
    part_3 = part_3a * (part_3b - part_3c)
    expected = part_1 * (part_2 + part_3)

    diff = found - expected
    criteria = (diff ** 2. < epsilon ** 2.)

    if not criteria:
        message = 'vortex ' + cyl_regularized.element_type + ': regularized biot-savart induction, tangential-component, computation does not behave far-upstream, outside cylinder'
        awelogger.logger.error(message)
        raise Exception(message)

    return None


def test_biot_savart_induction_tangential_component_outside_cylinder_far_upstream(cyl_unregularized, epsilon=1.e-4):

    pretest_is_unregularized_cylinder(cyl_unregularized, epsilon)

    unpacked = cyl_unregularized.info_dict
    r_cyl = unpacked['radius']

    r_obs = 2. * r_cyl
    z_obs = -1. * 10**6.

    elliptic_m0 = cyl_unregularized.get_regularized_elliptic_m_zero_from_r(r_obs)
    elliptic_m = cyl_unregularized.get_regularized_elliptic_m_from_r_and_z(r_obs, z_obs)
    found = cyl_unregularized.get_regularized_biot_savart_induction_tangential_component(r_obs, z_obs, elliptic_m0, elliptic_m)

    expected = 0.

    diff = found - expected
    criteria = (diff ** 2. < epsilon ** 2.)

    if not criteria:
        message = 'vortex ' + cyl_unregularized.element_type + ': biot-savart induction, tangential-component, computation does not behave far-upstream, outside cylinder'
        awelogger.logger.error(message)
        raise Exception(message)

    return None


def test_regularized_biot_savart_induction_tangential_component_outside_cylinder_far_downstream(cyl_regularized, epsilon=1.e-4):

    pretest_is_regularized_cylinder(cyl_regularized, epsilon)

    unpacked = cyl_regularized.info_dict
    strength = unpacked['strength']
    r_cyl = unpacked['radius']
    epsilon_r = unpacked['epsilon_r']

    r_obs = 2. * r_cyl
    z_obs = 1. * 10**6.

    elliptic_m0 = cyl_regularized.get_regularized_elliptic_m_zero_from_r(r_obs)
    elliptic_m = cyl_regularized.get_regularized_elliptic_m_from_r_and_z(r_obs, z_obs)
    found = cyl_regularized.get_regularized_biot_savart_induction_tangential_component(r_obs, z_obs, elliptic_m0, elliptic_m)

    elliptic_k = (4. * r_cyl * r_obs / ((r_cyl + r_obs)**2. + z_obs**2. + epsilon_r**2.))**0.5
    elliptic_k0 = (4. * r_cyl * r_obs / ((r_cyl + r_obs)**2. + 0.**2. + epsilon_r**2.))**0.5
    part_1 = strength/2. * r_cyl * (r_obs / (r_obs**2. + epsilon_r**2.))
    signed = vect_op.sign((r_cyl - r_obs), eps=epsilon_r)
    part_2 = (1. - signed) / 2.
    part_3a = z_obs * elliptic_k / (2. * np.pi * (r_obs * r_cyl)**0.5)
    part_3b = np.pi / 2. # (elliptic_k(m = 0.))
    elliptic_pi_of_m0_and_0 = np.pi / (2. * (1. - elliptic_k0**2.)**0.5)
    part_3c = (r_cyl - r_obs) / (r_cyl + r_obs + epsilon_r) * elliptic_pi_of_m0_and_0
    part_3 = part_3a * (part_3b - part_3c)
    expected = part_1 * (part_2 + part_3)

    diff = found - expected
    criteria = (diff ** 2. < epsilon ** 2.)

    if not criteria:
        message = 'vortex ' + cyl_regularized.element_type + ': regularized biot-savart induction, tangential-component, computation does not behave far-downstream, outside cylinder'
        awelogger.logger.error(message)
        raise Exception(message)

    return None

def test_biot_savart_induction_tangential_component_outside_cylinder_far_downstream(cyl_unregularized, epsilon=1.e-4):

    pretest_is_unregularized_cylinder(cyl_unregularized, epsilon)

    unpacked = cyl_unregularized.info_dict
    r_cyl = unpacked['radius']
    strength = unpacked['strength']

    r_obs = 2. * r_cyl
    z_obs = 1. * 10**6.

    elliptic_m0 = cyl_unregularized.get_regularized_elliptic_m_zero_from_r(r_obs)
    elliptic_m = cyl_unregularized.get_regularized_elliptic_m_from_r_and_z(r_obs, z_obs)
    found = cyl_unregularized.get_regularized_biot_savart_induction_tangential_component(r_obs, z_obs, elliptic_m0, elliptic_m)

    expected = strength * r_cyl / r_obs

    diff = found - expected
    criteria = (diff ** 2. < epsilon ** 2.)

    if not criteria:
        message = 'vortex ' + cyl_unregularized.element_type + ': biot-savart induction, tangential-component, computation does not behave far-downstream, outside cylinder'
        awelogger.logger.error(message)
        raise Exception(message)

    return None

def test_regularized_biot_savart_induction_tangential_component_on_surface(cyl_regularized, epsilon=1.e-4):

    pretest_is_regularized_cylinder(cyl_regularized, epsilon)

    unpacked = cyl_regularized.info_dict
    strength = unpacked['strength']
    r_cyl = unpacked['radius']
    epsilon_r = unpacked['epsilon_r']

    r_obs = r_cyl
    z_obs = 10.

    elliptic_m0 = cyl_regularized.get_regularized_elliptic_m_zero_from_r(r_obs)
    elliptic_m = cyl_regularized.get_regularized_elliptic_m_from_r_and_z(r_obs, z_obs)
    found = cyl_regularized.get_regularized_biot_savart_induction_tangential_component(r_obs, z_obs, elliptic_m0, elliptic_m)

    elliptic_k = (4. / (4. + z_obs**2. + epsilon_r**2.))**0.5
    elliptic_k0 = (4. / (4. + 0.**2. + epsilon_r**2.))**0.5
    part_1 = strength / (2. * (r_obs**2. + epsilon_r**2.))
    part_2 = 0.5
    part_3a = z_obs * elliptic_k / (2. * np.pi)
    part_3b = special.ellipk(elliptic_k**2.)
    part_3c = (r_cyl - r_obs) / (r_cyl + r_obs + epsilon_r) * vect_op.elliptic_pi(elliptic_k0**2., elliptic_k**2.)
    part_3 = part_3a * (part_3b - part_3c)
    expected = part_1 * (part_2 + part_3)

    diff = found - expected
    criteria = (diff ** 2. < epsilon ** 2.)

    if not criteria:
        message = 'vortex ' + cyl_regularized.element_type + ': regularized biot-savart induction, tangential-component, computation does not behave on surface'
        awelogger.logger.error(message)
        raise Exception(message)

    return None

def test_regularized_biot_savart_induction_tangential_component_at_critical_point(cyl_regularized, epsilon=1.e-4):

    pretest_is_regularized_cylinder(cyl_regularized, epsilon)

    unpacked = cyl_regularized.info_dict
    strength = unpacked['strength']
    r_cyl = unpacked['radius']
    epsilon_r = unpacked['epsilon_r']

    r_obs = r_cyl
    z_obs = 0.

    elliptic_m0 = cyl_regularized.get_regularized_elliptic_m_zero_from_r(r_obs)
    elliptic_m = cyl_regularized.get_regularized_elliptic_m_from_r_and_z(r_obs, z_obs)
    found = cyl_regularized.get_regularized_biot_savart_induction_tangential_component(r_obs, z_obs, elliptic_m0, elliptic_m)

    part_1 = strength / (2. * (1. + epsilon_r**2.))
    part_2 = 0.5
    part_3 = 0.
    expected = part_1 * (part_2 + part_3)

    diff = found - expected
    criteria = (diff ** 2. < epsilon ** 2.)

    if not criteria:
        message = 'vortex ' + cyl_regularized.element_type + ': regularized biot-savart induction, tangential-component, computation does not behave at critical point'
        awelogger.logger.error(message)
        raise Exception(message)

    return None



def test_regularized_biot_savart_induction_tangential_component(cyl_regularized, cyl_unregularized):
    test_biot_savart_induction_tangential_component_on_axis(cyl_unregularized)
    test_biot_savart_induction_tangential_component_inside_cylinder_at_start(cyl_unregularized)
    test_biot_savart_induction_tangential_component_outside_cylinder_at_start(cyl_unregularized)

    test_regularized_biot_savart_induction_tangential_component_inside_cylinder_far_upstream(cyl_regularized)
    test_regularized_biot_savart_induction_tangential_component_inside_cylinder_far_downstream(cyl_regularized)

    test_biot_savart_induction_tangential_component_inside_cylinder_far_upstream(cyl_unregularized)
    test_biot_savart_induction_tangential_component_inside_cylinder_far_downstream(cyl_unregularized)

    test_regularized_biot_savart_induction_tangential_component_outside_cylinder_far_upstream(cyl_regularized)
    test_regularized_biot_savart_induction_tangential_component_outside_cylinder_far_downstream(cyl_regularized)

    test_biot_savart_induction_tangential_component_outside_cylinder_far_upstream(cyl_unregularized)
    test_biot_savart_induction_tangential_component_outside_cylinder_far_downstream(cyl_unregularized)

    test_regularized_biot_savart_induction_tangential_component_on_surface(cyl_regularized)
    test_regularized_biot_savart_induction_tangential_component_at_critical_point(cyl_regularized)

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

    theta_hat = vect_op.normed_cross(l_hat, r_hat)

    elliptic_m = 4 * r_obs * r_cyl / ((r_obs + r_cyl)**2. + z_obs**2.)
    elliptic_m0 = 4 * r_obs * r_cyl / ((r_obs + r_cyl)**2.)
    elliptic_k = (elliptic_m)**0.5

    tan_part_1 = strength / 2. * (r_cyl / r_obs)
    tan_part_2 = ((r_obs - r_cyl) + np.abs(r_cyl - r_obs)) / (2. * np.abs(r_cyl - r_obs))
    tan_part_3a = z_obs * elliptic_k / (2. * np.pi * (r_obs * r_cyl)**0.5)
    tan_part_3b = special.ellipk(elliptic_m)
    tan_part_3c = (r_cyl - r_obs) / (r_cyl + r_obs) * vect_op.elliptic_pi(n=elliptic_m0, m=elliptic_m)
    tan_part_3 = tan_part_3a * (tan_part_3b - tan_part_3c)
    tangential_component = tan_part_1 * (tan_part_2 + tan_part_3)

    expected = tangential_component * theta_hat

    packed_info = cyl_unregularized.info
    biot_savart_fun = cyl_unregularized.biot_savart_fun
    found = biot_savart_fun(packed_info, x_obs)

    diff = expected - found

    criteria = (cas.mtimes(diff.T, diff) < epsilon**2.)

    if not criteria:
        message = 'vortex ' + cyl_unregularized.element_type + ': biot-savart function does not work as intended.'
        awelogger.logger.error(message)
        raise Exception(message)

    return None



def test():
    cyl_regularized = construct_test_object(regularized=True)
    cyl_regularized.test_basic_criteria(expected_object_type='semi_infinite_longitudinal_cylinder')

    cyl_unregularized = construct_test_object(regularized=False)
    cyl_unregularized.test_basic_criteria(expected_object_type='semi_infinite_longitudinal_cylinder')

    test_regularized_biot_savart_induction_tangential_component(cyl_regularized, cyl_unregularized)
    test_biot_savart_function(cyl_unregularized)

    cyl_regularized.test_draw()

    return None

# test()