#
#    This file is part of awebox.
#
#    awebox -- A modeling and optimization framework for multi-kite AWE systems.
#    Copyright (C) 2017-2020 Jochem De Schutter, Rachel Leuthold, Moritz Diehl,
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
reference frames and conversion methods
_python-3.5 / casadi-3.4.5
- edited: rachel leuthold, jochem de schutter alu-fr 2017-20
'''

import awebox.tools.vector_operations as vect_op
import casadi.tools as cas
import numpy as np
from awebox.logger.logger import Logger as awelogger


def get_body_axes(q_upper, q_lower):

    tether = q_upper - q_lower

    yhat = vect_op.yhat()

    ehat_z = vect_op.normalize(tether)
    ehat_x = vect_op.normed_cross(yhat, tether)
    ehat_y = vect_op.normed_cross(ehat_z, ehat_x)

    return ehat_x, ehat_y, ehat_z

def from_earth_to_body(vector, q_upper, q_lower):

    [ehat_x, ehat_y, ehat_z] = get_body_axes(q_upper, q_lower)

    r_from_body_to_ef = cas.horzcat(ehat_x, ehat_y, ehat_z)
    r_from_ef_to_body = cas.inv(r_from_body_to_ef)

    transformed = cas.mtimes(r_from_ef_to_body, vector)

    return transformed

def from_body_to_earth(vector, q_upper, q_lower):

    [ehat_x, ehat_y, ehat_z] = get_body_axes(q_upper, q_lower)

    r_from_body_to_ef = cas.horzcat(ehat_x, ehat_y, ehat_z)
    transformed = cas.mtimes(r_from_body_to_ef, vector)

    return transformed


def test_transforms():

    test_horizontal()
    test_vertical()

    test_transform_from_earth()
    test_transform_from_body()

    return None


def test_horizontal():

    name = 'horizontal'

    xhat = vect_op.xhat_np()
    yhat = vect_op.yhat_np()
    zhat = vect_op.zhat_np()

    ehat_x = vect_op.xhat_np()
    ehat_y = vect_op.yhat_np()
    ehat_z = vect_op.zhat_np()

    q_upper = 5. * xhat
    q_lower = 2. * xhat

    dir = 'x'
    transformed = from_earth_to_body(xhat, q_upper, q_lower)
    reference = ehat_z
    diff = transformed - reference
    resi = cas.mtimes(diff.T, diff)
    epsilon = 1.e-12
    if resi > epsilon:
        awelogger.logger.error('tether frame transformation test (' + name + ' ' + dir + ') gives error of size: ' + str(resi))


    dir = 'y'
    transformed = from_earth_to_body(yhat, q_upper, q_lower)
    reference = ehat_y
    diff = transformed - reference
    resi = cas.mtimes(diff.T, diff)
    epsilon = 1.e-12
    if resi > epsilon:
        awelogger.logger.error('tether frame transformation test (' + name + ' ' + dir + ') gives error of size: ' + str(resi))
        



    dir = 'z'
    transformed = from_earth_to_body(zhat, q_upper, q_lower)
    reference = -1. * ehat_x
    diff = transformed - reference
    resi = cas.mtimes(diff.T, diff)
    epsilon = 1.e-12
    if resi > epsilon:
        awelogger.logger.error('tether frame transformation test (' + name + ' ' + dir + ') gives error of size: ' + str(resi))
        

    return None







def test_vertical():

    name = 'vertical'

    xhat = vect_op.xhat_np()
    yhat = vect_op.yhat_np()
    zhat = vect_op.zhat_np()

    ehat_x = vect_op.xhat_np()
    ehat_y = vect_op.yhat_np()
    ehat_z = vect_op.zhat_np()


    q_upper = 15. * zhat
    q_lower = 8. * zhat

    dir = 'x'
    transformed = from_earth_to_body(xhat, q_upper, q_lower)
    reference = ehat_x
    diff = transformed - reference
    resi = cas.mtimes(diff.T, diff)
    epsilon = 1.e-12
    if resi > epsilon:
        awelogger.logger.error('tether frame transformation test (' + name + ' ' + dir + ') gives error of size: ' + str(resi))
        


    dir = 'y'
    transformed = from_earth_to_body(yhat, q_upper, q_lower)
    reference = ehat_y
    diff = transformed - reference
    resi = cas.mtimes(diff.T, diff)
    epsilon = 1.e-12
    if resi > epsilon:
        awelogger.logger.error('tether frame transformation test (' + name + ' ' + dir + ') gives error of size: ' + str(resi))
        



    dir = 'z'
    transformed = from_earth_to_body(zhat, q_upper, q_lower)
    reference = ehat_z
    diff = transformed - reference
    resi = cas.mtimes(diff.T, diff)
    epsilon = 1.e-12
    if resi > epsilon:
        awelogger.logger.error('tether frame transformation test (' + name + ' ' + dir + ') gives error of size: ' + str(resi))
        

    return None





def test_transform_from_earth():

    xhat = vect_op.xhat_np()
    yhat = vect_op.yhat_np()
    zhat = vect_op.zhat_np()

    qx_test = np.random.random() * 1000.
    qy_test = np.random.random() * 1000.
    qz_test = np.random.random() * 1000.

    q_upper = qx_test * xhat + qy_test * yhat + qz_test * zhat
    q_lower = q_upper / 2.

    x_test = np.random.random() * 100.
    y_test = np.random.random() * 100.
    z_test = np.random.random() * 100.
    test_vec = x_test * xhat + y_test * yhat + z_test * zhat
    test_mag = vect_op.norm(test_vec)

    trans_vec = from_earth_to_body(test_vec, q_upper, q_lower)
    trans_mag = vect_op.norm(trans_vec)
    norm_error = vect_op.norm(trans_mag - test_mag)

    reformed_vec = from_body_to_earth(trans_vec, q_upper, q_lower)
    vector_diff = reformed_vec - test_vec
    vector_error = cas.mtimes(vector_diff.T, vector_diff)

    tot_error = norm_error + vector_error
    epsilon = 1.e-12
    if tot_error > epsilon:
        awelogger.logger.error('tether frame transformation test gives error of size: ' + str(tot_error))
        

    return None

def test_transform_from_body():

    xhat = vect_op.xhat_np()
    yhat = vect_op.yhat_np()
    zhat = vect_op.zhat_np()

    qx_test = np.random.random() * 1000.
    qy_test = np.random.random() * 1000.
    qz_test = np.random.random() * 1000.

    q_upper = qx_test * xhat + qy_test * yhat + qz_test * zhat
    q_lower = q_upper / 2.

    x_test = np.random.random() * 100.
    y_test = np.random.random() * 100.
    z_test = np.random.random() * 100.
    test_vec = x_test * xhat + y_test * yhat + z_test * zhat
    test_mag = vect_op.norm(test_vec)

    trans_vec = from_body_to_earth(test_vec, q_upper, q_lower)
    trans_mag = vect_op.norm(trans_vec)
    norm_error = vect_op.norm(trans_mag - test_mag)

    reformed_vec = from_earth_to_body(trans_vec, q_upper, q_lower)
    vector_diff = reformed_vec - test_vec
    vector_error = cas.mtimes(vector_diff.T, vector_diff)

    tot_error = norm_error + vector_error
    epsilon = 1.e-12
    if tot_error > epsilon:
        awelogger.logger.error('tether frame transformation test gives error of size: ' + str(tot_error))
        

    return None




def get_inverse_equivalence_matrix(tether_length):
    # equivalent forces at upper node = [a, b, c]
    # equivalent forces at lower node = [d, e, f]
    # total forces = [Fx, Fy, Fz]
    # total moment = [Mx, My, 0]

    # a + d = Fx
    # b + e = Fy
    # c + f = Fz
    # L (a - d) = My
    # L (a - e) = Mx
    # c - f = 0

    # A [a, b, c, d, e, f].T = [Fx, Fy, Fz, Mx, My, 0].T
    # [a, b, c, d, e, f].T = Ainv [Fx, Fy, Fz, Mx, My, 0].T

    ell = tether_length
    over = 1./ell
    nver = -1. / ell
    half = 0.5
    nalf = -0.5

    Ainv_row1 = cas.horzcat(half, 0., 0., over, 0., 0.)
    Ainv_row2 = cas.horzcat(0., half, 0., 0., over, 0.)
    Ainv_row3 = cas.horzcat(0., 0., half, 0., 0., half)
    Ainv_row4 = cas.horzcat(half, 0., 0., nver, 0., 0.)
    Ainv_row5 = cas.horzcat(0., half, 0., 0., nver, 0.)
    Ainv_row6 = cas.horzcat(0., 0., half, 0., 0., nalf)

    Ainv = cas.vertcat(Ainv_row1,
                       Ainv_row2,
                       Ainv_row3,
                       Ainv_row4,
                       Ainv_row5,
                       Ainv_row6
                       )

    return Ainv
