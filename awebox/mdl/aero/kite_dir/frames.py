#
#    This file is part of awebox.
#
#    awebox -- A modeling and optimization framework for multi-kite AWE systems.
#    Copyright (C) 2017-2019 Jochem De Schutter, Rachel Leuthold, Moritz Diehl,
#                            ALU Freiburg.
#    Copyright (C) 2018-2019 Thilo Bronnenmeyer, Kiteswarms Ltd.
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
reference frames and conversion methods, specifically for the kite forces
_python-3.5 / casadi-3.4.5
- edited: rachel leuthold, jochem de schutter alu-fr 2017-20
'''

import casadi.tools as cas
import numpy as np

import awebox.mdl.aero.induction_dir.tools_dir.path_based_geom as path_based_geom
import awebox.tools.vector_operations as vect_op
from awebox.logger.logger import Logger as awelogger



def get_wind_dcm(vec_u, kite_dcm):
    ehat_span = kite_dcm[:, 1]

    Dhat = vect_op.smooth_normalize(vec_u)
    Lhat = vect_op.smooth_normed_cross(vec_u, ehat_span)
    Shat = vect_op.smooth_normed_cross(Lhat, Dhat)

    wind_dcm = cas.horzcat(Dhat, Shat, Lhat)
    return wind_dcm



def from_earth_to_body(kite_dcm, vector):
    dcm_inv = cas.inv(kite_dcm)
    transformed = cas.mtimes(dcm_inv, vector)
    return transformed

def from_body_to_earth(kite_dcm, vector):
    transformed = cas.mtimes(kite_dcm, vector)
    return transformed

def from_wind_to_earth(vec_u, kite_dcm, vector):
    wind_dcm = get_wind_dcm(vec_u, kite_dcm)
    transformed = cas.mtimes(wind_dcm, vector)
    return transformed

def from_earth_to_wind(vec_u, kite_dcm, vector):
    wind_dcm = get_wind_dcm(vec_u, kite_dcm)
    wind_dcm_inv = cas.inv(wind_dcm)
    transformed = cas.mtimes(wind_dcm_inv, vector)
    return transformed

def from_body_to_wind(vec_u, kite_dcm, vector):
    in_earth = from_body_to_earth(kite_dcm, vector)
    in_wind = from_earth_to_wind(vec_u, kite_dcm, in_earth)
    return in_wind

def from_wind_to_body(vec_u, kite_dcm, vector):
    in_earth = from_wind_to_earth(vec_u, kite_dcm, vector)
    in_body = from_earth_to_body(kite_dcm, in_earth)
    return in_body

def test_conversions(epsilon=1.e-10):

    test_horizontal_body_earth(epsilon)
    test_vertical_body_earth(epsilon)
    test_level_body_wind(epsilon)
    test_right_body_wind(epsilon)

    return None


def test_horizontal_body_earth(epsilon):
    name = 'horizontal'

    xhat = vect_op.xhat_np()
    yhat = vect_op.yhat_np()
    zhat = vect_op.zhat_np()

    ehat1_k = vect_op.xhat_np()
    ehat2_k = vect_op.yhat_np()
    ehat3_k = vect_op.zhat_np()

    ehat_chord = xhat
    ehat_span = zhat
    ehat_up = -1. * yhat
    kite_dcm = cas.horzcat(ehat_chord, ehat_span, ehat_up)

    dir = 'chord body->earth'
    transformed = from_body_to_earth(kite_dcm, ehat1_k)
    reference = ehat_chord
    diff = transformed - reference
    resi = cas.mtimes(diff.T, diff)
    if resi > epsilon:
        awelogger.logger.error(
            'kite frame transformation test (' + name + ' ' + dir + ') gives error of size: ' + str(resi))

    dir = 'span body->earth'
    transformed = from_body_to_earth(kite_dcm, ehat2_k)
    reference = ehat_span
    diff = transformed - reference
    resi = cas.mtimes(diff.T, diff)
    if resi > epsilon:
        awelogger.logger.error(
            'kite frame transformation test (' + name + ' ' + dir + ') gives error of size: ' + str(resi))

    dir = 'up body->earth'
    transformed = from_body_to_earth(kite_dcm, ehat3_k)
    reference = ehat_up
    diff = transformed - reference
    resi = cas.mtimes(diff.T, diff)
    if resi > epsilon:
        awelogger.logger.error(
            'kite frame transformation test (' + name + ' ' + dir + ') gives error of size: ' + str(resi))

    dir = 'chord earth->body'
    transformed = from_earth_to_body(kite_dcm, ehat_chord)
    reference = ehat1_k
    diff = transformed - reference
    resi = cas.mtimes(diff.T, diff)
    if resi > epsilon:
        awelogger.logger.error(
            'kite frame transformation test (' + name + ' ' + dir + ') gives error of size: ' + str(resi))

    dir = 'span earth->body'
    transformed = from_earth_to_body(kite_dcm, ehat_span)
    reference = ehat2_k
    diff = transformed - reference
    resi = cas.mtimes(diff.T, diff)
    if resi > epsilon:
        awelogger.logger.error(
            'kite frame transformation test (' + name + ' ' + dir + ') gives error of size: ' + str(resi))

    dir = 'up earth->body'
    transformed = from_earth_to_body(kite_dcm, ehat_up)
    reference = ehat3_k
    diff = transformed - reference
    resi = cas.mtimes(diff.T, diff)
    if resi > epsilon:
        awelogger.logger.error(
            'kite frame transformation test (' + name + ' ' + dir + ') gives error of size: ' + str(resi))

    return None


def test_vertical_body_earth(epsilon):
    name = 'vertical'

    xhat = vect_op.xhat_np()
    yhat = vect_op.yhat_np()
    zhat = vect_op.zhat_np()

    ehat1_k = vect_op.xhat_np()
    ehat2_k = vect_op.yhat_np()
    ehat3_k = vect_op.zhat_np()

    ehat_chord = -1. * zhat
    ehat_span = -1. * xhat
    ehat_up = yhat
    kite_dcm = cas.horzcat(ehat_chord, ehat_span, ehat_up)

    dir = 'chord body->earth'
    transformed = from_body_to_earth(kite_dcm, ehat1_k)
    reference = ehat_chord
    diff = transformed - reference
    resi = cas.mtimes(diff.T, diff)
    if resi > epsilon:
        awelogger.logger.error(
            'kite frame transformation test (' + name + ' ' + dir + ') gives error of size: ' + str(resi))

    dir = 'span body->earth'
    transformed = from_body_to_earth(kite_dcm, ehat2_k)
    reference = ehat_span
    diff = transformed - reference
    resi = cas.mtimes(diff.T, diff)
    if resi > epsilon:
        awelogger.logger.error(
            'kite frame transformation test (' + name + ' ' + dir + ') gives error of size: ' + str(resi))

    dir = 'up body->earth'
    transformed = from_body_to_earth(kite_dcm, ehat3_k)
    reference = ehat_up
    diff = transformed - reference
    resi = cas.mtimes(diff.T, diff)
    if resi > epsilon:
        awelogger.logger.error(
            'kite frame transformation test (' + name + ' ' + dir + ') gives error of size: ' + str(resi))

    dir = 'chord earth->body'
    transformed = from_earth_to_body(kite_dcm, ehat_chord)
    reference = ehat1_k
    diff = transformed - reference
    resi = cas.mtimes(diff.T, diff)
    if resi > epsilon:
        awelogger.logger.error(
            'kite frame transformation test (' + name + ' ' + dir + ') gives error of size: ' + str(resi))

    dir = 'span earth->body'
    transformed = from_earth_to_body(kite_dcm, ehat_span)
    reference = ehat2_k
    diff = transformed - reference
    resi = cas.mtimes(diff.T, diff)
    if resi > epsilon:
        awelogger.logger.error(
            'kite frame transformation test (' + name + ' ' + dir + ') gives error of size: ' + str(resi))

    dir = 'up earth->body'
    transformed = from_earth_to_body(kite_dcm, ehat_up)
    reference = ehat3_k
    diff = transformed - reference
    resi = cas.mtimes(diff.T, diff)
    if resi > epsilon:
        awelogger.logger.error(
            'kite frame transformation test (' + name + ' ' + dir + ') gives error of size: ' + str(resi))

    return None




def get_test_wind(alpha, beta, kite_dcm):

    denom = np.sqrt( np.tan(alpha)**2. + (1./np.cos(beta))**2. )
    x_val = 1. / denom
    y_val = np.tan(beta) / denom
    z_val = np.tan(alpha) / denom

    ehat1 = kite_dcm[:, 0]
    ehat2 = kite_dcm[:, 1]
    ehat3 = kite_dcm[:, 2]

    u_wind = x_val * ehat1 + y_val * ehat2 + z_val * ehat3

    return u_wind


def test_level_body_wind(epsilon):

    name = 'level wind'
    kite_dcm = cas.DM.eye(3)

    # CA = CD, CY = CS, CN = CL
    alpha = 0.
    beta = 0.
    u_test = get_test_wind(alpha, beta, kite_dcm)

    dir = 'x body->wind'
    test = vect_op.xhat_np()
    reference = vect_op.xhat_np()
    transformed = from_body_to_wind(u_test, kite_dcm, test)
    diff = transformed - reference
    resi = cas.mtimes(diff.T, diff)
    if resi > epsilon:
        awelogger.logger.error(
            'kite frame transformation test (' + name + ' ' + dir + ') gives error of size: ' + str(resi))

    dir = 'y body->wind'
    test = vect_op.yhat_np()
    reference = vect_op.yhat_np()
    transformed = from_body_to_wind(u_test, kite_dcm, test)
    diff = transformed - reference
    resi = cas.mtimes(diff.T, diff)
    if resi > epsilon:
        awelogger.logger.error(
            'kite frame transformation test (' + name + ' ' + dir + ') gives error of size: ' + str(resi))

    dir = 'z body->wind'
    test = vect_op.zhat_np()
    reference = vect_op.zhat_np()
    transformed = from_body_to_wind(u_test, kite_dcm, test)
    diff = transformed - reference
    resi = cas.mtimes(diff.T, diff)
    if resi > epsilon:
        awelogger.logger.error(
            'kite frame transformation test (' + name + ' ' + dir + ') gives error of size: ' + str(resi))

    return None



def test_right_body_wind(epsilon):

    name = 'right wind'
    kite_dcm = cas.DM.eye(3)

    # CA = CD, CY = CS, CN = CL
    alpha = np.pi / 2.
    beta = 0.
    u_test = get_test_wind(alpha, beta, kite_dcm)

    dir = 'x body->wind'
    test = vect_op.xhat_np()
    reference = -1. * vect_op.zhat_np()
    transformed = from_body_to_wind(u_test, kite_dcm, test)
    diff = transformed - reference
    resi = cas.mtimes(diff.T, diff)
    if resi > epsilon:
        awelogger.logger.error(
            'kite frame transformation test (' + name + ' ' + dir + ') gives error of size: ' + str(resi))

    dir = 'y body->wind'
    test = vect_op.yhat_np()
    reference = vect_op.yhat_np()
    transformed = from_body_to_wind(u_test, kite_dcm, test)
    diff = transformed - reference
    resi = cas.mtimes(diff.T, diff)
    if resi > epsilon:
        awelogger.logger.error(
            'kite frame transformation test (' + name + ' ' + dir + ') gives error of size: ' + str(resi))

    dir = 'z body->wind'
    test = vect_op.zhat_np()
    reference = vect_op.xhat_np()
    transformed = from_body_to_wind(u_test, kite_dcm, test)
    diff = transformed - reference
    resi = cas.mtimes(diff.T, diff)
    if resi > epsilon:
        awelogger.logger.error(
            'kite frame transformation test (' + name + ' ' + dir + ') gives error of size: ' + str(resi))

    dir = 'x wind->body'
    test = vect_op.xhat_np()
    reference = vect_op.zhat_np()
    transformed = from_wind_to_body(u_test, kite_dcm, test)
    diff = transformed - reference
    resi = cas.mtimes(diff.T, diff)
    if resi > epsilon:
        awelogger.logger.error(
            'kite frame transformation test (' + name + ' ' + dir + ') gives error of size: ' + str(resi))

    dir = 'y wind->body'
    test = vect_op.yhat_np()
    reference = vect_op.yhat_np()
    transformed = from_wind_to_body(u_test, kite_dcm, test)
    diff = transformed - reference
    resi = cas.mtimes(diff.T, diff)
    if resi > epsilon:
        awelogger.logger.error(
            'kite frame transformation test (' + name + ' ' + dir + ') gives error of size: ' + str(resi))

    dir = 'z wind->body'
    test = vect_op.zhat_np()
    reference = -1. * vect_op.xhat_np()
    transformed = from_wind_to_body(u_test, kite_dcm, test)
    diff = transformed - reference
    resi = cas.mtimes(diff.T, diff)
    if resi > epsilon:
        awelogger.logger.error(
            'kite frame transformation test (' + name + ' ' + dir + ') gives error of size: ' + str(resi))

    return None