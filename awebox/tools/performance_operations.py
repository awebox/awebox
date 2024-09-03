#
#    This file is part of awebox.
#
#    awebox -- A modeling and optimization framework for multi-kite AWE systems.
#    Copyright (C) 2017-2020 Jochem De Schutter, Rachel Leuthold, Moritz Diehl,
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
file to provide operations related to the system performance, to the awebox,
_python-3.5 / casadi-3.4.5
- author: rachel leuthold alu-fr 2020
'''

import casadi.tools as cas
import numpy as np
from awebox.logger.logger import Logger as awelogger

import awebox.tools.print_operations as print_op
import awebox.tools.vector_operations as vect_op

def get_loyd_power(power_density, CL, CD, s_ref, elevation_angle=0.):
    phf = get_loyd_phf(CL, CD, elevation_angle)
    p_loyd = power_density * s_ref * phf
    return p_loyd

def get_loyd_phf(CL, CD, elevation_angle=0.):
    epsilon = 1.e-6

    interior = CD**2. / (CL**2 + epsilon**2.)
    CR = CL * (1. + interior)**0.5

    phf = 4. / 27. * CR * (CR / CD) ** 2. * cas.cos(elevation_angle) ** 3.
    return phf

def get_reelout_factor_with_respect_to_wind_at_position(variables_si, position, wind):
    if 'dl_t' in variables_si['x'].keys():
        dl_t = variables_si['x']['dl_t']
    else:
        dl_t = cas.DM.zeros((1, 1))

    vec_u_infty = wind.get_velocity(position[2])
    f_val = dl_t / vect_op.smooth_norm(vec_u_infty)
    return f_val


def determine_if_periodic(options):

    enforce_periodicity = bool(True)
    if options['trajectory']['type'] in ['transition', 'compromised_landing', 'nominal_landing', 'launch','mpc']:
         enforce_periodicity = bool(False)

    return enforce_periodicity

def get_elevation_angle(q10):
    length_along_ground = (q10[0] ** 2. + q10[1] ** 2.) ** 0.5
    elevation_angle = cas.arctan2(q10[2], length_along_ground)
    return elevation_angle

def test_elevation_angle(epsilon=1.e-4):
    expected_angle = np.pi/4.
    q10 = 1. * vect_op.xhat_np() + 1. * vect_op.zhat_np()
    found_angle = get_elevation_angle(q10)

    criteria = ((expected_angle - found_angle)**2. < epsilon**2.)
    if not criteria:
        message = 'elevation angle computation does not work as expected'
        print_op.log_and_raise_error(message)

    return None


def get_cone_angle(position_kite, position_parent, position_grandparent):
    ehat_tether = vect_op.normalize(position_parent - position_grandparent)
    altitude = vect_op.norm(vect_op.dot(position_kite - position_parent, ehat_tether))
    hypotenuse = vect_op.norm(position_kite - position_parent)
    cone_angle = cas.arccos(altitude/hypotenuse)
    return cone_angle

def test_cone_angle(epsilon=1.e-3):

    ehat_tether = vect_op.xhat_np()
    ehat_normal = vect_op.zhat_np()

    position_parent = 0. * ehat_tether
    position_grandparent = -1. * ehat_tether

    expected_cone_angle = np.pi/4.
    altitude = 1.
    leg = 1.

    position_kite = position_parent + altitude * ehat_tether + leg * ehat_normal
    found_cone_angle = get_cone_angle(position_kite, position_parent, position_grandparent)

    criteria = ((expected_cone_angle - found_cone_angle)**2 < epsilon**2.)

    if not criteria:
        message = 'cone angle computation does not work as expected'
        print_op.log_and_raise_error(message)

    return None

def test():
    test_elevation_angle()
    test_cone_angle()

# test()