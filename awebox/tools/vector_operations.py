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
file to provide vector operations to the awebox,
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, jochem de schutter alu-fr 2017-19
'''

import matplotlib.pylab as plt
import scipy
import scipy.io
import scipy.special as special
import scipy.sparse as sps

import awebox.tools.print_operations as print_op

import casadi.tools as cas
import numpy as np
from awebox.logger.logger import Logger as awelogger


def cross(a, b):
    vi = xhat() * (a[1] * b[2] - a[2] * b[1])
    vj = yhat() * (a[0] * b[2] - a[2] * b[0])
    vk = zhat() * (a[0] * b[1] - a[1] * b[0])

    v = vi - vj + vk

    return v

def norm(a):
    # norm = (a[0] ** 2.0 + a[1] ** 2.0 + a[2] ** 2.0) ** 0.5
    norm = smooth_norm(a, 0.)

    return norm

def smooth_norm(a, epsilon=1e-8):
    if type(a) == np.ndarray:
        dot_product = np.matmul(a.T, a)
    else:
        dot_product = cas.mtimes(a.T, a)

    norm = smooth_sqrt(dot_product, epsilon)

    return norm

def abs(a):
    abs = smooth_abs(a, 0.)
    return abs

def smooth_abs(arg, epsilon=1e-8):

    if hasattr(arg, 'shape') and (arg.shape == (1,1)):
        abs = smooth_sqrt(arg ** 2., epsilon)

    elif hasattr(arg, 'shape') and (len(arg.shape) > 0):
        abs = []
        for idx in range(arg.shape[0]):
            local = smooth_sqrt(arg[idx] ** 2., epsilon)
            abs = cas.vertcat(abs, local)

    elif isinstance(arg, list):
        abs = []
        for idx in range(len(arg)):
            local = smooth_sqrt(arg[idx] ** 2., epsilon)
            abs += [local]

    else:
        abs = smooth_sqrt(arg ** 2., epsilon)

    return abs

def smooth_sqrt(arg, epsilon=1e-4):
    sqrt = (arg + epsilon ** 2.) ** 0.5
    return sqrt

def normalize(a):
    normed = a / norm(a)

    return normed

def smooth_normalize(a, epsilon=1e-8):
    normed = a / smooth_norm(a, epsilon)
    return normed

def normed_cross(a, b):
    temp = cross(a, b)
    vhat = normalize(temp)

    return vhat

def smooth_normed_cross(a, b, epsilon=1e-8):
    temp = cross(a, b)
    vhat = smooth_normalize(temp, epsilon)

    return vhat

def dot(a, b):
    v = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    return v

def angle_between(a, b):
    theta = np.arctan2(norm(cross(a, b)), dot(a, b))

    return theta

def angle_between_resi(a, b, theta):
    resi = np.tan(theta) * dot(a, b) - norm(cross(a, b))
    return resi

def zeros_mx(shape):
    return cas.MX.zeros(shape[0], shape[1])

def zeros_sx(shape):
    return cas.SX.zeros(shape[0], shape[1])

def ones_mx(shape):
    return cas.MX.ones(shape[0], shape[1])

def ones_sx(shape):
    return cas.SX.ones(shape[0], shape[1])


def xhat():
    return xhat_np()

def yhat():
    return yhat_np()

def zhat():
    return zhat_np()

def xhat_sx():
    xhat_sx = cas.SX.eye(3)[:, 0]
    return xhat_sx

def yhat_sx():
    yhat_sx = cas.SX.eye(3)[:, 1]
    return yhat_sx

def zhat_sx():
    zhat_sx = cas.SX.eye(3)[:, 2]
    return zhat_sx

def xhat_mx():
    xhat_mx = cas.MX.eye(3)[:, 0]
    return xhat_mx

def yhat_mx():
    yhat_mx = cas.MX.eye(3)[:, 1]
    return yhat_mx

def zhat_mx():
    zhat_mx = cas.MX.eye(3)[:, 2]
    return zhat_mx

def xhat_np():
    xhat_np = np.array(cas.vertcat(1., 0., 0.))
    return xhat_np

def yhat_np():
    yhat_np = np.array(cas.vertcat(0., 1., 0.))
    return yhat_np

def zhat_np():
    zhat_np = np.array(cas.vertcat(0., 0., 1.))
    return zhat_np

def xhat_dm():
    return cas.DM(xhat_np())

def yhat_dm():
    return cas.DM(yhat_np())

def zhat_dm():
    return cas.DM(zhat_np())

def spy(matrix, tol=0.1, color=True, title=''):
    fig = plt.figure()
    fig.clf()

    matrix = sps.csr_matrix(matrix)

    elements = matrix.shape[0]
    markersize = (1./float(elements)) * 500.

    if color:
        matrix_dense = matrix.todense()
        plt.imshow(matrix_dense, interpolation='none', cmap='binary')
        plt.colorbar()
    else:
        plt.spy(matrix, precision=tol, markersize=markersize)

    plt.title(title)


def skew(vec):
    " creates skew-symmetric matrix"
    a = vec[0]
    b = vec[1]
    c = vec[2]
    vecskew = cas.blockcat([[0., -c, b],
                        [c, 0., -a],
                        [-b, a, 0.]])
    return vecskew

def unskew(A):
    "Unskew matrix to vector"

    B = 0.5*cas.vertcat(
        A[2,1]-A[1,2],
        A[0,2]-A[2,0],
        A[1,0]-A[0,1]
    )
    return B

def rotation(R, A):
    "Rotation operator as defined in Gros2013b"
    return  unskew(cas.mtimes(R.T,A))

def jacobian_dcm(expr, xd_si, variables_scaled, kite, parent):
    """ Differentiate expression w.r.t. kite direct cosine matrix"""

    dcm_si = xd_si['r{}{}'.format(kite, parent)]
    dcm_scaled = variables_scaled['xd', 'r{}{}'.format(kite, parent)]

    jac_dcm = rotation(
            cas.reshape(dcm_si, (3,3)),
            cas.reshape(cas.jacobian(expr, dcm_scaled), (3,3))
    ).T
    return jac_dcm

def upper_triangular_inclusive(matrix):

    matrix_resquared = resquare(matrix)

    elements = []
    for r in range(matrix_resquared.shape[0]):
        for c in range(matrix_resquared.shape[1]):
            if c >= r:
                elements = cas.vertcat(elements, matrix_resquared[r, c])
    return elements

def lower_triangular_exclusive(matrix):

    matrix_resquared = resquare(matrix)

    elements = []
    for r in range(matrix_resquared.shape[0]):
        for c in range(matrix_resquared.shape[1]):
            if c < r:
                elements = cas.vertcat(elements, matrix_resquared[r, c])
    return elements

def lower_triangular_inclusive(matrix):

    matrix_resquared = resquare(matrix)

    elements = []
    for r in range(matrix_resquared.shape[0]):
        for c in range(matrix_resquared.shape[1]):
            if c <= r:
                elements = cas.vertcat(elements, matrix_resquared[r, c])
    return elements

def columnize(matrix):
    # only allows 2D matrices for variable

    [counted_rows, counted_columns] = matrix.shape
    number_elements = counted_rows * counted_columns

    column_var = cas.reshape(matrix, (number_elements, 1))

    return column_var

def resquare(column):

    entries = column.shape[0] * column.shape[1]
    guess_side_dim = np.sqrt(float(entries))
    can_be_resquared = (np.floor(guess_side_dim) **2. == float(entries))

    if can_be_resquared:
        side = int(guess_side_dim)
        return cas.reshape(column, (side, side))
    else:
        message = 'column matrix cannot be re-squared. inappropriate number of entries: ' + str(entries)
        awelogger.logger.error(message)
        return column

def sign(val, eps=1e-8):
    sign = 2. * unitstep(val, eps) - 1.
    return sign

def find_zero_rows(matrix, tol):
    mask = (matrix > tol) + (matrix < -1. * tol)
    zero_rows = np.where(np.sum(mask, axis=1) == 0)[0]
    return zero_rows

def find_zero_cols(matrix, tol):
    mask = (matrix > tol) + (matrix < -1. * tol)
    zero_cols = np.where(np.sum(mask, axis=0) == 0)[0]
    return zero_cols

def unitstep(val, eps=1e-8):
    heavi = np.arctan(val / eps) / np.pi + 0.5
    return heavi

def step_in_out(number, step_in, step_out, eps=1e-4):
    step_in = unitstep(number - step_in, eps)
    step_out = unitstep(number - step_out, eps)
    step = step_in - step_out
    return step

def sum(all_array):
    sum = cas.sum1(all_array)
    return sum

def smooth_max(all_array):
    exp_array = np.exp(all_array)
    sum_exp = sum(exp_array)
    log_sum_exp = np.log(sum_exp)

    return log_sum_exp

def smooth_min(all_array):

    negative_array = -1. * all_array
    maxed = smooth_max(negative_array)
    mined = -1. * maxed

    return mined

def smallest_nonunity_factor(n):

    n_float = float(n)

    sqrt_int = np.int(n_float**0.5)

    for num in range(2, sqrt_int):
        if np.mod(n_float, float(num)) == 0:
            return n

    return -999.

def estimate_1d_covariance(x, k):

    N = x.shape[0] - k
    dot_prod = cas.mtimes(x[:-k].T, x[k:])
    cov = dot_prod / N

    return cov

def pisarenko_harmonic_decomposition(x):
    r1 = estimate_1d_covariance(x, 1)
    r2 = estimate_1d_covariance(x, 2)
    a = (r2 + (r2 ** 2. + 8. * r1 ** 2.)**0.5) / 4. / r1

    smoothed_a = (-1.) * step_in_out(a, -cas.inf, -1.) + step_in_out(a, -1., 1.) * a + step_in_out(a, 1., cas.inf)
    acos = np.arccos(smoothed_a)
    return acos

def estimate_1d_frequency(x, sample_step=1, dt=1.0):
    # http://tkf.github.io/2010/10/03/estimate-frequency-using-numpy.html
    # TODO: weigh mean with different time constants in case of phase fixing!
    mean = sum(x) / x.shape[0]
    zero_mean = x - mean

    omega = pisarenko_harmonic_decomposition(zero_mean[::sample_step])
    return omega / 2.0 / np.pi / sample_step / dt


# Checks if a matrix is a valid rotation matrix.
def is_rotation_matrix(dcm, thresh=1.e-1):

    diff = cas.DM.eye(3) - cas.mtimes(dcm.T, dcm)
    diff_vert = cas.reshape(diff, (9, 1))
    resi = norm(diff_vert)**0.5

    return resi < thresh

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotation_matrix_to_euler_angles(dcm):

    if not is_rotation_matrix(dcm):
        awelogger.logger.warning('given rotation matrix is not a member of SO(3).')

    sy = np.math.sqrt(dcm[0, 0] * dcm[0, 0] + dcm[1, 0] * dcm[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = np.math.atan2(dcm[2, 1], dcm[2, 2])
        y = np.math.atan2(-dcm[2, 0], sy)
        z = np.math.atan2(dcm[1, 0], dcm[0, 0])
    else:
        x = np.math.atan2(-dcm[1, 2], dcm[1, 1])
        y = np.math.atan2(-dcm[2, 0], sy)
        z = 0

    return np.array([x, y, z])

# element-wise multiplication
def mtimes_elementwise(a, b):
    return np.diag(np.array(cas.mtimes(a, b.T)))

def elliptic_k(k=None, m=None):

    if (k is not None) and (m is None):
        m = k**2.
    elif (m is None) and (k is None):
        message = 'no acceptable argument given for elliptic_k approximation'
        awelogger.logger.error(message)
        raise Exception(message)
    elif (m is not None) and (k is not None):
        message = 'too many arguments given for elliptic_k approximation'
        awelogger.logger.error(message)
        raise Exception(message)

    if (isinstance(m, float)) or (isinstance(m, int)) or (isinstance(m, cas.DM)):
        if not (m >= 0 or m < 1):
            message = 'm argument of elliptic integral K(m) is outside of acceptable range.'
            awelogger.logger.error(message)
            raise Exception(message)
    # else:
        # be advised: as the argument of elliptic integral K(m) is a casadi symbolic, cannot automatically check that m is within acceptable range of 0 <= m < 1

    a1 = 2.78187
    a2 = -5.25143
    a3 = 2.97986

    part_1 = a1 * m + a2 * m**2. + a3 * m**3.
    part_2 = cas.log(1./(1. - m))

    found = np.pi / 2. + part_1 * part_2

    return found

def elliptic_k_approximation_max_error():
    return 0.05

def test_elliptic_k():

    # boundary case
    m = 0.
    found = elliptic_k(m=m)
    expected = special.ellipk(m)
    error_bound = elliptic_k_approximation_max_error() * expected

    error = found - expected
    if (error**2. > error_bound**2.):
        message = '(boundary) elliptic integral K(m) approximation did not work as expected'
        awelogger.logger.error(message)
        raise Exception(message)


    # intermediate value
    m = 0.5
    found = elliptic_k(m=m)
    expected = special.ellipk(m)
    error_bound = elliptic_k_approximation_max_error() * expected

    error = found - expected
    if (error ** 2. > error_bound ** 2.):
        message = '(intermediate) elliptic integral K(m) approximation did not work as expected'
        awelogger.logger.error(message)
        raise Exception(message)

    return None

def elliptic_pi(n=None, alpha=None):
    part_1 = (1. + n**2.38175 / cas.sqrt(1. - n))
    part_2 = elliptic_k(m = alpha)
    found = part_1 * part_2
    return found

def elliptic_pi_approximation_max_error():
    return 0.3

def test_elliptic_pi():
    # origin case
    n = 0.
    alpha = 0.
    found = elliptic_pi(n=n, alpha=alpha)
    expected = np.pi/2.
    error_bound = elliptic_pi_approximation_max_error() * expected

    error = found - expected
    if (error**2. > error_bound**2.):
        message = '(origin) elliptic integral Pi(n|alpha) approximation did not work as expected'
        awelogger.logger.error(message)
        raise Exception(message)

    # center case
    n = 0.5
    alpha = 0.5
    found = elliptic_pi(n=n, alpha=alpha)
    expected = 2.70129
    error_bound = elliptic_pi_approximation_max_error() * expected

    error = found - expected
    if (error**2. > error_bound**2.):
        message = '(center) elliptic integral Pi(n|alpha) approximation did not work as expected'
        awelogger.logger.error(message)
        raise Exception(message)


    return None

def elliptic_e(m=None):
    found = 0.166667 * (9.42478 - 2.823 * m + 1.646 * m**2. - 2.24778 * m**3.)
    return found

def elliptic_e_approximation_max_error():
    return 0.009

def test_elliptic_e():

    # boundary case
    m = 0.
    found = elliptic_e(m=m)
    expected = special.ellipe(m)
    error_bound = elliptic_e_approximation_max_error() * expected

    error = found - expected
    if (error**2. > error_bound**2.):
        message = '(boundary 0) elliptic integral E(m) approximation did not work as expected'
        awelogger.logger.error(message)
        raise Exception(message)


    # intermediate value
    m = 0.5
    found = elliptic_e(m=m)
    expected = special.ellipe(m)
    error_bound = elliptic_e_approximation_max_error() * expected

    error = found - expected
    if (error ** 2. > error_bound ** 2.):
        message = '(intermediate) elliptic integral K(m) approximation did not work as expected'
        awelogger.logger.error(message)
        raise Exception(message)

    # boundary case
    m = 1.
    found = elliptic_e(m=m)
    expected = special.ellipe(m)
    error_bound = elliptic_e_approximation_max_error() * expected

    error = found - expected
    if (error**2. > error_bound**2.):
        message = '(boundary 1) elliptic integral E(m) approximation did not work as expected'
        awelogger.logger.error(message)
        raise Exception(message)

    return None

def get_altitude(vec_1, vec_2):
    vec_a = cross(vec_1, vec_2)
    altitude = smooth_norm(vec_a) / smooth_norm(vec_1 - vec_2)
    return altitude

def test_altitude():

    expected = 1.
    x_obs = expected * zhat_dm()

    # right triangle
    x_1 = 0. * xhat_dm()
    x_2 = 1. * xhat_dm()
    difference = get_altitude(x_1 - x_obs, x_2 - x_obs) - expected
    thresh = 1.e-6
    if thresh < difference**2.:
        message = 'biot-savart right-triangle altitude test gives error of size: ' + str(difference)
        awelogger.logger.error(message)
        raise Exception(message)

    # obtuse triangle
    x_1 = 1. * xhat_np()
    x_2 = 2. * xhat_np()
    difference = get_altitude(x_1 - x_obs, x_2 - x_obs) - expected
    thresh = 1.e-6
    if thresh < difference**2.:
        message = 'biot-savart obtuse-triangle altitude test gives error of size: ' + str(difference)
        awelogger.logger.error(message)
        raise Exception(message)

    return None
