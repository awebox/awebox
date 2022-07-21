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
import numpy as np

import awebox.mdl.aero.induction_dir.vortex_dir.biot_savart as biot_savart
import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.semi_infinite_cylinder as vortex_cylinder

import awebox.tools.struct_operations as struct_op
import awebox.tools.vector_operations as vect_op
import awebox.tools.print_operations as print_op

from awebox.logger.logger import Logger as awelogger

import matplotlib
matplotlib.use('TkAgg')


class SemiInfiniteTangentialCylinder(vortex_cylinder.SemiInfiniteCylinder):
    # Branlard, Emmanuel & Gaunaa, Mac.(2014). Cylindrical vortex wake model: Right cylinder. Wind Energy. 524. 10.1002/we.1800.

    def __init__(self, info_dict):
        super().__init__(info_dict)
        self.set_element_type('semi_infinite_tangential_cylinder')


    ##### radial induction parts

    def get_regularized_biot_savart_induction_radial_component_strength_part(self):

        unpacked = self.info_dict
        strength = unpacked['strength']

        part_1 = -1. * strength / (2. * np.pi)

        return part_1

    def get_regularized_biot_savart_induction_radial_component_middle_part(self, r_obs, z_obs):

        unpacked = self.info_dict

        r_cyl = unpacked['radius']
        epsilon_r = unpacked['epsilon_r']

        part_2a = ((r_cyl + r_obs) ** 2. + z_obs ** 2. + epsilon_r**2.)**0.5
        part_2b = r_obs + epsilon_r
        part_2 = part_2a / part_2b

        return part_2

    def get_regularized_biot_savart_induction_radial_component_elliptic_part(self, elliptic_m):

        part_3a = 1. - 0.5 * elliptic_m
        part_3b = vect_op.elliptic_k(m=elliptic_m)
        part_3c = vect_op.elliptic_e(m=elliptic_m)
        part_3 = part_3a * part_3b - part_3c

        return part_3

    def get_regularized_biot_savart_induction_radial_component(self, r_obs, z_obs, elliptic_m):

        part_1 = self.get_regularized_biot_savart_induction_radial_component_strength_part()
        part_2 = self.get_regularized_biot_savart_induction_radial_component_middle_part(r_obs, z_obs)
        part_3 = self.get_regularized_biot_savart_induction_radial_component_elliptic_part(elliptic_m)

        found = part_1 * part_2 * part_3

        return found


    ##### longitudinal induction parts

    def get_regularized_biot_savart_induction_longitudinal_component_strength_part(self):

        unpacked = self.info_dict
        strength = unpacked['strength']

        part_1 = strength / (4. * np.pi**2.)

        return part_1

    def get_regularized_biot_savart_induction_longitudinal_component_middle_part(self, r_obs):

        unpacked = self.info_dict

        r_cyl = unpacked['radius']

        difference = r_cyl - r_obs
        signed = vect_op.smooth_sign(difference)

        part_2 = np.pi * (signed + 1.)

        return part_2

    def get_regularized_biot_savart_induction_longitudinal_component_elliptic_part(self, r_obs, z_obs, elliptic_m0, elliptic_m):

        unpacked = self.info_dict
        r_cyl = unpacked['radius']
        epsilon_r = unpacked['epsilon_r']

        num_1 = 2. * z_obs
        num_2 = vect_op.elliptic_k(m=elliptic_m)
        num_3a = (r_cyl - r_obs) / (r_cyl + r_obs + epsilon_r)
        num_3b = vect_op.elliptic_pi(n=elliptic_m0, m=elliptic_m)
        num_3 = num_3a * num_3b

        num = num_1 * (num_2 + num_3)

        den = ( (r_cyl + r_obs)**2. + z_obs**2. + epsilon_r**2)**0.5

        part_3 = num/den

        return part_3

    def get_regularized_biot_savart_induction_longitudinal_component(self, r_obs, z_obs, elliptic_m0, elliptic_m):

        part_1 = self.get_regularized_biot_savart_induction_longitudinal_component_strength_part()
        part_2 = self.get_regularized_biot_savart_induction_longitudinal_component_middle_part(r_obs)
        part_3 = self.get_regularized_biot_savart_induction_longitudinal_component_elliptic_part(r_obs, z_obs, elliptic_m0, elliptic_m)

        found = part_1 * (part_2 + part_3)

        return found



    def define_biot_savart_induction_function(self):
        expected_info_length = self.expected_info_length
        packed_sym = cas.SX.sym('packed_sym', (expected_info_length, 1))
        unpacked_sym = self.unpack_info(external_info=packed_sym)

        x_obs = cas.SX.sym('x_obs', (3, 1))

        x_center = unpacked_sym['x_center']
        l_hat = unpacked_sym['l_hat']
        radius = unpacked_sym['radius']
        l_start = unpacked_sym['l_start']
        epsilon = unpacked_sym['epsilon']
        strength = unpacked_sym['strength']

        r_obs = self.get_r_obs(x_obs)
        z_obs = self.get_z_obs(x_obs)

        vec_0 = x_0 - x_obs
        vec_1 = x_1 - x_obs

        r_squared_0 = cas.mtimes(vec_0.T, vec_0)
        r_squared_1 = cas.mtimes(vec_1.T, vec_1)
        r_0 = vect_op.smooth_sqrt(r_squared_0)
        r_1 = vect_op.smooth_sqrt(r_squared_1)

        # notice, that we're using the cut-off model as described in
        # https: // openfast.readthedocs.io / en / main / source / user / aerodyn - olaf / OLAFTheory.html  # regularization
        # which is the unit-consistent version what's used in
        # A. van Garrel. Development of a Wind Turbine Aerodynamics Simulation Module. Technical report,
        # Energy research Centre of the Netherlands. ECN-C–03-079, aug 2003
        length = vect_op.norm(x_1 - x_0)
        epsilon_vortex = r_core**2. * length**2.

        factor = strength/(4. * np.pi)
        num1 = r_0 + r_1
        num2 = vect_op.cross(vec_0, vec_1)
        den1 = r_squared_0 * r_squared_1
        den2 = r_0 * r_1 * cas.mtimes(vec_0.T, vec_1)
        den3 = epsilon_vortex

        value = factor * num1 * num2 / (den1 + den2 + den3)

        biot_savart_fun = cas.Function('biot_savart_fun', [packed_sym, x_obs], [value])
        self.set_biot_savart_fun(biot_savart_fun)

        return None


    def draw(self, ax, side, variables_scaled, parameters, cosmetics):
        evaluated = self.evaluate_info(variables_scaled, parameters)
        unpacked = self.unpack_info(external_info=evaluated)

        x_center = unpacked['x_center']
        l_hat = unpacked['l_hat']
        r_cyl = unpacked['radius']
        l_start = unpacked['l_start']

        a_hat, b_hat, _ = biot_savart.get_cylinder_axes(unpacked)

        s_start = l_start
        s_end = s_start + cosmetics['trajectory']['cylinder_s_length']

        n_theta = cosmetics['trajectory']['cylinder_n_theta']
        n_s = cosmetics['trajectory']['cylinder_n_s']

        for sdx in range(n_s):

            s = s_start + (s_end - s_start) / float(n_s) * float(sdx)

            for tdx in range(n_theta):
                theta_start = 2. * np.pi / float(n_theta) * float(tdx)
                theta_end = 2. * np.pi / float(n_theta) * (tdx + 1.)

                x_start = x_center + l_hat * s + r_cyl * np.sin(theta_start) * a_hat + r_cyl * np.cos(theta_start) * b_hat
                x_end = x_center + l_hat * s + r_cyl * np.sin(theta_end) * a_hat + r_cyl * np.cos(theta_end) * b_hat
                super().basic_draw(ax, side, unpacked['strength'], x_start, x_end, cosmetics)

        return None

def construct_test_object():
    cyl = vortex_cylinder.construct_test_object()
    unpacked = cyl.info_dict
    tan_cyl = SemiInfiniteTangentialCylinder(unpacked)
    return tan_cyl

##### radial tests

def test_regularized_biot_savart_induction_radial_component_on_axis(cyl, epsilon=1.e-4):

    r_obs = 0.
    z_obs = 100.

    elliptic_m = cyl.get_regularized_elliptic_m_from_r_and_z(r_obs, z_obs)

    found = cyl.get_regularized_biot_savart_induction_radial_component(r_obs, z_obs, elliptic_m)
    expected = 0.

    diff = found - expected
    criteria = (diff**2. < epsilon**2.)

    if not criteria:
        message = 'vortex semi-infinite tangential cylinder: regularized biot-savart induction, radial-component, computation does not behave as expected on the axis'
        awelogger.logger.error(message)
        raise Exception(message)


def test_regularized_biot_savart_induction_radial_component_at_large_radius(cyl, epsilon=1.e-4):
    r_obs = 10.**8.
    z_obs = 10.

    elliptic_m = cyl.get_regularized_elliptic_m_from_r_and_z(r_obs, z_obs)

    found = cyl.get_regularized_biot_savart_induction_radial_component(r_obs, z_obs, elliptic_m)
    expected = 0.

    diff = found - expected
    criteria = (diff ** 2. < epsilon ** 2.)

    if not criteria:
        message = 'vortex semi-infinite tangential cylinder: regularized biot-savart induction, radial-component, computation does not behave as expected at large radius'
        awelogger.logger.error(message)
        raise Exception(message)


def test_regularized_biot_savart_induction_radial_component_middle_part_at_critical_point(cyl, epsilon=1.e-4):

    unpacked = cyl.unpack_info()
    r_cyl = unpacked['radius']
    epsilon_r = unpacked['epsilon_r']

    cylinder_radius_is_as_expected = ((r_cyl - 1.)**2 < epsilon**2.)
    epsilon_r_is_as_expected = ((epsilon_r - 1.)**2 < epsilon**2.)
    if not (cylinder_radius_is_as_expected and epsilon_r_is_as_expected):
        message = 'something went wrong in a vortex semi-infinite tangential cylinder test. check that test cylinder has radius = 1 and espilon_r = 1'
        awelogger.logger.error(message)
        raise Exception(message)

    r_obs = r_cyl
    z_obs = 0.

    found = cyl.get_regularized_biot_savart_induction_radial_component_middle_part(r_obs, z_obs)
    expected = np.sqrt(5.)/2.

    diff = found - expected
    criteria = (diff ** 2. < epsilon ** 2.)

    if not criteria:
        message = 'vortex semi-infinite tangential cylinder: regularized biot-savart induction, radial-component, middle part, computation does not behave as expected at criticial point'
        awelogger.logger.error(message)
        raise Exception(message)



def test_regularized_biot_savart_induction_radial_component_elliptical_part_at_critical_point(cyl, epsilon=1.e-4):

    unpacked = cyl.unpack_info()
    r_cyl = unpacked['radius']
    epsilon_r = unpacked['epsilon_r']

    cylinder_radius_is_as_expected = ((r_cyl - 1.)**2 < epsilon**2.)
    epsilon_r_is_as_expected = ((epsilon_r - 1.)**2 < epsilon**2.)
    if not (cylinder_radius_is_as_expected and epsilon_r_is_as_expected):
        message = 'something went wrong in a vortex semi-infinite tangential cylinder test. check that test cylinder has radius = 1 and espilon_r = 1'
        awelogger.logger.error(message)
        raise Exception(message)

    r_obs = r_cyl
    z_obs = 0.

    elliptic_m = cyl.get_regularized_elliptic_m_from_r_and_z(r_obs, z_obs)
    found = cyl.get_regularized_biot_savart_induction_radial_component_elliptic_part(elliptic_m)
    expected = 0.175833

    diff = found - expected
    criteria = (diff ** 2. < epsilon ** 2.)

    if not criteria:
        message = 'vortex semi-infinite tangential cylinder: regularized biot-savart induction, radial-component, elliptic part, computation does not behave as expected at criticial point'
        awelogger.logger.error(message)
        raise Exception(message)


def test_regularized_biot_savart_induction_radial_component_middle_part_far_upstream(cyl, epsilon=1.e-4):

    unpacked = cyl.unpack_info()
    r_cyl = unpacked['radius']

    r_obs = 1.2 * r_cyl
    z_obs = -1. * 10.**5.

    found = cyl.get_regularized_biot_savart_induction_radial_component_middle_part(r_obs, z_obs)
    expected = 45454.5

    diff = found - expected
    criteria = ( (diff/expected) ** 2. < epsilon ** 2.)

    if not criteria:
        message = 'vortex semi-infinite tangential cylinder: regularized biot-savart induction, radial-component, middle part, computation does not behave as expected far upstream'
        awelogger.logger.error(message)
        raise Exception(message)

def test_regularized_biot_savart_induction_radial_component_elliptical_part_far_upstream(cyl, epsilon=1.e-4):

    unpacked = cyl.unpack_info()
    r_cyl = unpacked['radius']

    r_obs = 1.2 * r_cyl
    z_obs = -1. * 10.**5.

    elliptic_m = cyl.get_regularized_elliptic_m_from_r_and_z(r_obs, z_obs)
    found = cyl.get_regularized_biot_savart_induction_radial_component_elliptic_part(elliptic_m)
    expected = 0.

    diff = found - expected
    criteria = (diff ** 2. < epsilon ** 2.)

    if not criteria:
        message = 'vortex semi-infinite tangential cylinder: regularized biot-savart induction, radial-component, elliptic part, computation does not behave as expected far-upstream'
        awelogger.logger.error(message)
        raise Exception(message)


def test_regularized_biot_savart_induction_radial_component_far_upstream(cyl, epsilon=1.e-4):

    unpacked = cyl.unpack_info()
    r_cyl = unpacked['radius']

    r_obs = 1.2 * r_cyl
    z_obs = -1. * 10.**5.

    elliptic_m = cyl.get_regularized_elliptic_m_from_r_and_z(r_obs, z_obs)

    found = cyl.get_regularized_biot_savart_induction_radial_component(r_obs, z_obs, elliptic_m)
    expected = 0.

    diff = found - expected
    criteria = (diff ** 2. < epsilon ** 2.)

    if not criteria:
        message = 'vortex semi-infinite tangential cylinder: regularized biot-savart induction, radial-component, computation does not behave as expected far-upstream'
        awelogger.logger.error(message)
        raise Exception(message)


def test_regularized_biot_savart_induction_radial_component_middle_part_far_downstream(cyl, epsilon=1.e-4):

    unpacked = cyl.unpack_info()
    r_cyl = unpacked['radius']

    r_obs = 0.8 * r_cyl
    z_obs = 10.**5.

    found = cyl.get_regularized_biot_savart_induction_radial_component_middle_part(r_obs, z_obs)
    expected = 55555.555

    diff = found - expected
    criteria = ( (diff/expected) ** 2. < epsilon ** 2.)

    if not criteria:
        message = 'vortex semi-infinite tangential cylinder: regularized biot-savart induction, radial-component, middle part, computation does not behave as expected far downstream'
        awelogger.logger.error(message)
        raise Exception(message)

def test_regularized_biot_savart_induction_radial_component_elliptical_part_far_downstream(cyl, epsilon=1.e-4):

    unpacked = cyl.unpack_info()
    r_cyl = unpacked['radius']

    r_obs = 0.8 * r_cyl
    z_obs = 10.**5.

    elliptic_m = cyl.get_regularized_elliptic_m_from_r_and_z(r_obs, z_obs)
    found = cyl.get_regularized_biot_savart_induction_radial_component_elliptic_part(elliptic_m)
    expected = 0.

    diff = found - expected
    criteria = (diff ** 2. < epsilon ** 2.)

    if not criteria:
        message = 'vortex semi-infinite tangential cylinder: regularized biot-savart induction, radial-component, elliptic part, computation does not behave as expected far-downstream'
        awelogger.logger.error(message)
        raise Exception(message)


def test_regularized_biot_savart_induction_radial_component_far_downstream(cyl, epsilon=1.e-4):

    unpacked = cyl.unpack_info()
    r_cyl = unpacked['radius']

    r_obs = 0.8 * r_cyl
    z_obs = 10.**5.

    elliptic_m = cyl.get_regularized_elliptic_m_from_r_and_z(r_obs, z_obs)

    found = cyl.get_regularized_biot_savart_induction_radial_component(r_obs, z_obs, elliptic_m)
    expected = 0.

    diff = found - expected
    criteria = (diff ** 2. < epsilon ** 2.)

    if not criteria:
        message = 'vortex semi-infinite tangential cylinder: regularized biot-savart induction, radial-component, computation does not behave as expected far-downstream'
        awelogger.logger.error(message)
        raise Exception(message)


def test_regularized_biot_savart_induction_radial_component(cyl):
    test_regularized_biot_savart_induction_radial_component_on_axis(cyl)

    test_regularized_biot_savart_induction_radial_component_middle_part_at_critical_point(cyl)
    test_regularized_biot_savart_induction_radial_component_elliptical_part_at_critical_point(cyl)

    test_regularized_biot_savart_induction_radial_component_at_large_radius(cyl)

    test_regularized_biot_savart_induction_radial_component_middle_part_far_upstream(cyl)
    test_regularized_biot_savart_induction_radial_component_elliptical_part_far_upstream(cyl)
    test_regularized_biot_savart_induction_radial_component_far_upstream(cyl)

    test_regularized_biot_savart_induction_radial_component_middle_part_far_downstream(cyl)
    test_regularized_biot_savart_induction_radial_component_elliptical_part_far_downstream(cyl)
    test_regularized_biot_savart_induction_radial_component_far_downstream(cyl)

    return None

##### longitudinal tests

def test_regularized_biot_savart_induction_longitudinal_component_on_axis(cyl, epsilon=1.e-4):

    unpacked = cyl.info_dict
    strength = unpacked['strength']
    r_cyl = unpacked['radius']

    r_obs = 0.
    z_obs = 100.

    elliptic_m0 = cyl.get_regularized_elliptic_m_zero_from_r(r_obs)
    elliptic_m = cyl.get_regularized_elliptic_m_from_r_and_z(r_obs, z_obs)
    found = cyl.get_regularized_biot_savart_induction_longitudinal_component(r_obs, z_obs, elliptic_m0, elliptic_m)

    expected = (strength / 2.) * ( 1. + z_obs / ( (r_cyl**2. + z_obs**2.)**0.5 ) )

    diff = found - expected
    criteria = (diff ** 2. < epsilon ** 2.)

    if not criteria:
        message = 'vortex semi-infinite tangential cylinder: regularized biot-savart induction, longitudinal-component, computation does not behave as expected on-axis'
        awelogger.logger.error(message)
        raise Exception(message)

    return None

def test_regularized_biot_savart_induction_longitudinal_component_inside_cylinder_at_start(cyl, epsilon=1.e-4):

    unpacked = cyl.info_dict
    strength = unpacked['strength']
    r_cyl = unpacked['radius']

    r_obs = r_cyl / 3.
    z_obs = 0.

    elliptic_m0 = cyl.get_regularized_elliptic_m_zero_from_r(r_obs)
    elliptic_m = cyl.get_regularized_elliptic_m_from_r_and_z(r_obs, z_obs)
    found = cyl.get_regularized_biot_savart_induction_longitudinal_component(r_obs, z_obs, elliptic_m0, elliptic_m)

    expected = (strength / 2.)

    diff = found - expected
    criteria = (diff ** 2. < epsilon ** 2.)

    if not criteria:
        message = 'vortex semi-infinite tangential cylinder: regularized biot-savart induction, longitudinal-component, computation does not behave inside cylinder at start'
        awelogger.logger.error(message)
        raise Exception(message)

    return None

def test_regularized_biot_savart_induction_longitudinal_component_inside_cylinder_far_upstream(cyl, epsilon=1.e-4):

    unpacked = cyl.info_dict
    strength = unpacked['strength']
    r_cyl = unpacked['radius']

    r_obs = r_cyl / 3.
    z_obs = -1. * 10**4.

    elliptic_m0 = cyl.get_regularized_elliptic_m_zero_from_r(r_obs)
    elliptic_m = cyl.get_regularized_elliptic_m_from_r_and_z(r_obs, z_obs)
    found = cyl.get_regularized_biot_savart_induction_longitudinal_component(r_obs, z_obs, elliptic_m0, elliptic_m)

    expected = 0.

    diff = found - expected
    criteria = (diff ** 2. < epsilon ** 2.)

    if not criteria:
        message = 'vortex semi-infinite tangential cylinder: regularized biot-savart induction, longitudinal-component, computation does not behave far-upstream'
        awelogger.logger.error(message)
        raise Exception(message)

    return None

def test_regularized_biot_savart_induction_longitudinal_component_inside_cylinder_far_downstream(cyl, epsilon=1.e-4):

    unpacked = cyl.info_dict
    strength = unpacked['strength']
    r_cyl = unpacked['radius']

    r_obs = r_cyl / 3.
    z_obs = 10**4.

    elliptic_m0 = cyl.get_regularized_elliptic_m_zero_from_r(r_obs)
    elliptic_m = cyl.get_regularized_elliptic_m_from_r_and_z(r_obs, z_obs)
    found = cyl.get_regularized_biot_savart_induction_longitudinal_component(r_obs, z_obs, elliptic_m0, elliptic_m)

    expected = strength

    diff = found - expected
    criteria = (diff ** 2. < epsilon ** 2.)

    if not criteria:
        message = 'vortex semi-infinite tangential cylinder: regularized biot-savart induction, longitudinal-component, computation does not behave far-downstream'
        awelogger.logger.error(message)
        raise Exception(message)

    return None

def test_regularized_biot_savart_induction_longitudinal_component(cyl):
    test_regularized_biot_savart_induction_longitudinal_component_on_axis(cyl)
    test_regularized_biot_savart_induction_longitudinal_component_inside_cylinder_at_start(cyl)
    test_regularized_biot_savart_induction_longitudinal_component_inside_cylinder_far_upstream()
    test_regularized_biot_savart_induction_longitudinal_component_inside_cylinder_far_downstream(cyl)

    return None

####### concatenate tests

def test():
    cyl = construct_test_object()
    cyl.test_basic_criteria(expected_object_type='semi_infinite_tangential_cylinder')

    test_regularized_biot_savart_induction_radial_component(cyl)
    test_regularized_biot_savart_induction_longitudinal_component(cyl)

    return None

test()