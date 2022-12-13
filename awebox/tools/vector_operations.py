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
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import pdb

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

    norm = smooth_sqrt(dot_product, epsilon**2.)

    return norm

def abs(a):
    abs = smooth_abs(a, 0.)
    return abs

def smooth_abs(arg, epsilon=1e-8):

    if hasattr(arg, 'shape') and (arg.shape == (1,1)):
        abs = smooth_sqrt(arg ** 2., epsilon**2.)

    elif hasattr(arg, 'shape') and (len(arg.shape) > 0):
        abs = []
        for idx in range(arg.shape[0]):
            local = smooth_sqrt(arg[idx] ** 2., epsilon**2.)
            abs = cas.vertcat(abs, local)

    elif isinstance(arg, list):
        abs = []
        for idx in range(len(arg)):
            local = smooth_sqrt(arg[idx] ** 2., epsilon**2.)
            abs += [local]

    else:
        abs = smooth_sqrt(arg ** 2., epsilon**2.)

    return abs

def smooth_sqrt(arg, epsilon=1e-8):
    sqrt = (arg + epsilon) ** 0.5
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
    theta = cas.arctan2(norm(cross(a, b)), dot(a, b))

    return theta

def angle_between_resi(a, b, theta):
    resi = cas.tan(theta) * dot(a, b) - norm(cross(a, b))
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

def jacobian_dcm(expr, x_si, variables_scaled, kite, parent):
    """ Differentiate expression w.r.t. kite direct cosine matrix"""

    dcm_si = x_si['r{}{}'.format(kite, parent)]
    dcm_scaled = variables_scaled['x', 'r{}{}'.format(kite, parent)]

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

    if not hasattr(matrix, 'shape'):
        message = 'the columnize function is not yet available for objects that do not have a shape attribute'
        print_op.log_and_raise_error(message)

    shape = matrix.shape

    if len(shape) == 1:
        if isinstance(matrix, np.ndarray):
            matrix = cas.DM(matrix)
        else:
            message = 'the columnize function is not yet available for 1D objects that are not numpy ndarrays.'
            print_op.log_and_raise_error(message)

    # only procede with 2D matrices for variable
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
        print_op.log_and_raise_error(message)
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
    heavi = cas.arctan(val / eps) / np.pi + 0.5
    return heavi

def step_in_out(number, step_in, step_out, eps=1e-4):
    step_in = unitstep(number - step_in, eps)
    step_out = unitstep(number - step_out, eps)
    step = step_in - step_out
    return step

def sum(all_array):

    array_columnized = columnize(all_array)
    ones = cas.DM.ones(array_columnized.shape)
    sum = cas.mtimes(array_columnized.T, ones)

    # sum = cas.sum1(all_array)
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

def elliptic_k(approximation_order_for_elliptic_integrals=6, k=None, m=None):

    m = get_elliptic_m_from_m_or_k(k=k, m=m)

    if approximation_order_for_elliptic_integrals == 0:
        aa = cas.DM([np.pi / 2.])
        bb = cas.DM([0.429204])

    elif approximation_order_for_elliptic_integrals == 1:
        aa = cas.DM([1.40832, 0.162481])
        bb = cas.DM([0.495222, 0.0608882])

    elif approximation_order_for_elliptic_integrals == 2:
        aa = cas.DM([1.38807, 0.130564, 0.0521589])
        bb = cas.DM([0.499693, 0.110545, 0.0173617])

    elif approximation_order_for_elliptic_integrals == 3:
        aa = cas.DM([1.38642, 0.103929, 0.0608376, 0.0196065])
        bb = cas.DM([0.499981, 0.122766, 0.0485613, 0.00581447])

    elif approximation_order_for_elliptic_integrals == 4:
        aa = cas.DM([1.3863, 0.0976991, 0.0435461, 0.0355245, 0.00772332])
        bb = cas.DM([0.499999, 0.12472, 0.0644774, 0.0236746, 0.00208671])

    elif approximation_order_for_elliptic_integrals == 5:
        aa = cas.DM([1.38629, 0.0967156, 0.0342165, 0.0296768, 0.0207886, 0.00310389])
        bb = cas.DM([0.5, 0.124969, 0.0691274, 0.0389237, 0.0117541, 0.000777829])

    elif approximation_order_for_elliptic_integrals == 6:
        aa = cas.DM([1.38629, 0.0965894, 0.0315497, 0.0209059, 0.0224215, 0.0117732, 0.00126104])
        bb = cas.DM([0.5, 0.124997, 0.0701137, 0.0459352, 0.0240693, 0.00581644, 0.000296896])

    else:
        message = 'elliptic_k approximation of order ' + str(approximation_order_for_elliptic_integrals) + ' is not yet available.'
        print_op.log_and_raise_error(message)


    # correct for rounding errors in the coefficients at initial condition (m=0)
    if aa.shape[0] > 1:
        aa[-1] = scipy.special.ellipk(0.) - sum(aa[:-1])

    one_minus_m = 1. - m

    part_1 = 0.
    part_2 = 0.
    for ndx in range(approximation_order_for_elliptic_integrals + 1):
        part_1 += aa[ndx] * one_minus_m**ndx
        part_2 += bb[ndx] * one_minus_m**ndx

    part_2 = part_2 * cas.log(1./one_minus_m)

    found = part_1 + part_2

    return found

def elliptic_k_approximation_max_abs_error(approximation_order_for_elliptic_integrals):
    if approximation_order_for_elliptic_integrals == 0:
        max_abs_error = 0.141593

    elif approximation_order_for_elliptic_integrals == 1:
        max_abs_error = 0.00955592

    elif approximation_order_for_elliptic_integrals == 2:
        max_abs_error = 0.000614075

    elif approximation_order_for_elliptic_integrals == 3:
        max_abs_error = 0.0000389314

    elif approximation_order_for_elliptic_integrals == 4:
        max_abs_error = 2.45442e-6

    elif approximation_order_for_elliptic_integrals == 5:
        max_abs_error = 1.54298e-7

    elif approximation_order_for_elliptic_integrals == 6:
        max_abs_error = 7.67115e-7

    else:
        message = 'elliptic_k approximation of order ' + str(approximation_order_for_elliptic_integrals) + ' is not yet available.'
        print_op.log_and_raise_error(message)

    return max_abs_error

def get_available_approximation_orders_for_elliptic_integrals():
    return range(7)

def test_elliptic_k_at_position(approximation_order_for_elliptic_integrals, elliptic_m, epsilon=1.e-5):

    found = elliptic_k(approximation_order_for_elliptic_integrals=approximation_order_for_elliptic_integrals, m=elliptic_m)
    expected = special.ellipk(elliptic_m)
    error_bound = elliptic_k_approximation_max_abs_error(approximation_order_for_elliptic_integrals) + epsilon

    diff = found - expected
    error = diff / expected
    if (abs(error) > error_bound):
        message = 'the error of the elliptic integral K(m) approximation (N=' + str(approximation_order_for_elliptic_integrals) + ') was not within the expected range, at m = ' + str(elliptic_m)
        print_op.log_and_raise_error(message)

    return None

def test_elliptic_k(epsilon=1.e-5):
    for approximation_order_for_elliptic_integrals in get_available_approximation_orders_for_elliptic_integrals():
        test_elliptic_k_at_position(approximation_order_for_elliptic_integrals, elliptic_m=0., epsilon=epsilon)
        test_elliptic_k_at_position(approximation_order_for_elliptic_integrals, elliptic_m=0.5, epsilon=epsilon)
        delta = 1.e-4
        test_elliptic_k_at_position(approximation_order_for_elliptic_integrals, elliptic_m=(1.0-delta), epsilon=epsilon)
    return None

def elliptic_pi(approximation_order_for_elliptic_integrals=6, n=None, m=None):

    one_minus_n = 1. - n
    sqrt_one_minus_n = cas.sqrt(one_minus_n)

    psi = cas.atan2(m-n, m+n)

    pin_at_psi_0 = psi
    pin_at_psi_positive_quarter = (psi - np.pi/4.)
    pin_at_psi_negative_quarter = (psi + np.pi/4.)

    pinned_a1_loc = pin_at_psi_0 * pin_at_psi_positive_quarter
    pinned_a1_expr = np.pi/(2. * sqrt_one_minus_n)

    pinned_a2_loc = pin_at_psi_negative_quarter * pin_at_psi_positive_quarter
    pinned_a2_expr = elliptic_e(approximation_order_for_elliptic_integrals=approximation_order_for_elliptic_integrals, m=n) / one_minus_n

    pinned_a3_loc = pin_at_psi_0 * pin_at_psi_negative_quarter
    pinned_a3_expr = elliptic_k(approximation_order_for_elliptic_integrals=approximation_order_for_elliptic_integrals, m=m)

    pinned = 8. / np.pi**2. * (pinned_a1_loc * pinned_a1_expr - 2. * pinned_a2_loc * pinned_a2_expr + pinned_a3_loc * pinned_a3_expr)

    a1 = -4.96273
    a2 = 18.6521
    a3 = -18.2879
    epsilon_m = 1.e-6

    m_squared_plus_n_squared = m**2. + n**2.
    diff_part_1 = 4. * m * n * (m**2. - n**2.)
    diff_part_2 = a3 * m_squared_plus_n_squared + a2 * cas.sqrt(m_squared_plus_n_squared) + a1
    diff_part_3 = cas.log(1./ (1. - cas.sqrt(m_squared_plus_n_squared)/np.sqrt(2.)))

    diff = diff_part_1 * diff_part_2 * diff_part_3 / (m_squared_plus_n_squared + epsilon_m)

    found = pinned + diff
    return found

def test_elliptic_pi(epsilon=1.e-4):

    error_bound = epsilon

    approximation_order_for_elliptic_integrals=6

    # origin case
    n = 0.
    m = 0.
    found = elliptic_pi(approximation_order_for_elliptic_integrals=approximation_order_for_elliptic_integrals, n=n, m=m)
    expected = np.pi/2.

    error = (found - expected) / expected
    if (error**2. > error_bound**2.):
        message = '(origin) elliptic integral Pi(n|alpha) approximation did not work as expected'
        print_op.log_and_raise_error(message)

    # m = 0 case
    n = 0.5
    m = 0.
    found = elliptic_pi(approximation_order_for_elliptic_integrals=approximation_order_for_elliptic_integrals, n=n, m=m)
    expected = np.pi / (2. * (1. - n)**0.5)

    error = (found - expected) / expected
    if (error**2. > error_bound**2.):
        message = '(m=0 case) elliptic integral Pi(n|m) approximation did not work as expected'
        print_op.log_and_raise_error(message)


    # n = 0 case
    n = 0.
    m = 0.5
    found = elliptic_pi(approximation_order_for_elliptic_integrals=approximation_order_for_elliptic_integrals, n=n, m=m)
    expected = special.ellipk(m=m)

    error = (found - expected) / expected
    if (error**2. > error_bound**2.):
        message = '(n=0 case) elliptic integral Pi(n|m) approximation did not work as expected'
        print_op.log_and_raise_error(message)

    # center case
    n = 0.5
    m = 0.5
    found = elliptic_pi(approximation_order_for_elliptic_integrals=approximation_order_for_elliptic_integrals, n=n, m=m)
    expected = 2.70129

    error = (found - expected) / expected
    if (error**2. > error_bound**2.):
        message = '(center) elliptic integral Pi(n|alpha) approximation did not work as expected'
        print_op.log_and_raise_error(message)

    return None

def get_elliptic_m_from_m_or_k(k=None, m=None):
    if (k is not None) and (m is None):
        m = k**2.
    elif (m is None) and (k is None):
        message = 'no acceptable argument given for elliptic integral approximation'
        print_op.log_and_raise_error(message)

    elif (m is not None) and (k is not None):
        message = 'too many arguments given for elliptic integral approximation'
        print_op.log_and_raise_error(message)

    if (isinstance(m, float)) or (isinstance(m, int)) or (isinstance(m, cas.DM)):
        if not (m >= 0 and m <= 1):
            message = 'm argument of elliptic integral is outside of acceptable range.'
            print_op.log_and_raise_error(message)

    # else:
        # be advised: as the argument of elliptic integral is a casadi symbolic, cannot automatically check that m is within acceptable range of 0 <= m < 1

    return m

def elliptic_e(approximation_order_for_elliptic_integrals=3, k=None, m=None):

    m = get_elliptic_m_from_m_or_k(k=k, m=m)

    if approximation_order_for_elliptic_integrals == 0:
        aa = cas.DM([np.pi / 2.])
        bb = cas.DM([0.])

    elif approximation_order_for_elliptic_integrals == 1:
        aa = cas.DM([1.00826, 0.562537])
        bb = cas.DM([0., 0.175222])

    elif approximation_order_for_elliptic_integrals == 2:
        aa = cas.DM([1.00029, 0.477105, 0.0934031])
        bb = cas.DM([0., 0.239135, 0.0321714])

    elif approximation_order_for_elliptic_integrals == 3:
        aa = cas.DM([1.00001, 0.448231, 0.091466, 0.0310873])
        bb = cas.DM([0., 0.248772, 0.0733862, 0.00956869])

    elif approximation_order_for_elliptic_integrals == 4:
        aa = cas.DM([1., 0.443741, 0.0682298, 0.0471662, 0.0116587])
        bb = cas.DM([0., 0.249879, 0.0892654, 0.0332429, 0.00324792])

    elif approximation_order_for_elliptic_integrals == 5:
        aa = cas.DM([1., 0.443208, 0.0592761, 0.0367837, 0.0269668, 0.004562])
        bb = cas.DM([0., 0.249989, 0.0929817, 0.0498747, 0.0160718, 0.00117209])

    elif approximation_order_for_elliptic_integrals == 6:
        aa = cas.DM([1., 0.443153, 0.057224, 0.0270174, 0.0263564, 0.0152242, 0.00182195])
        bb = cas.DM([0., 0.249999, 0.0936388, 0.0563518, 0.0301448, 0.0078578, 0.000438041])

    else:
        message = 'elliptic_e approximation of order ' + str(approximation_order_for_elliptic_integrals) + ' is not yet available.'
        print_op.log_and_raise_error(message)

    # correct for rounding errors in the coefficients at initial condition (m=0)
    if aa.shape[0] > 1:
        aa[-1] = scipy.special.ellipe(0.) - sum(aa[:-1])

    one_minus_m = 1. - m

    part_1 = 0.
    part_2 = 0.
    for ndx in range(approximation_order_for_elliptic_integrals + 1):
        part_1 += aa[ndx] * one_minus_m**ndx
        part_2 += bb[ndx] * one_minus_m**ndx

    part_2 = part_2 * cas.log(1./one_minus_m)

    found = part_1 + part_2

    return found

def elliptic_e_approximation_max_abs_error(approximation_order_for_elliptic_integrals):
    if approximation_order_for_elliptic_integrals == 0:
        max_abs_error = 0.570796

    elif approximation_order_for_elliptic_integrals == 1:
        max_abs_error = 0.00825932

    elif approximation_order_for_elliptic_integrals == 2:
        max_abs_error = 0.000288165

    elif approximation_order_for_elliptic_integrals == 3:
        max_abs_error = 1.7184e-6

    elif approximation_order_for_elliptic_integrals == 4:
        max_abs_error = 5.96923e-7

    elif approximation_order_for_elliptic_integrals == 5:
        max_abs_error = 2.40275e-9

    elif approximation_order_for_elliptic_integrals == 6:
        max_abs_error = 2.59103e-7

    else:
        message = 'elliptic_e approximation of order ' + str(approximation_order_for_elliptic_integrals) + ' is not yet available.'
        print_op.log_and_raise_error(message)

    return max_abs_error

def test_elliptic_e_at_position(approximation_order_for_elliptic_integrals, elliptic_m, epsilon=1.e-5):

    found = elliptic_e(approximation_order_for_elliptic_integrals=approximation_order_for_elliptic_integrals, m=elliptic_m)
    expected = special.ellipe(elliptic_m)
    error_bound = elliptic_e_approximation_max_abs_error(approximation_order_for_elliptic_integrals) + epsilon

    diff = found - expected
    error = diff / expected
    if (abs(error) > error_bound):
        message = 'the error of the elliptic integral E(m) approximation (N=' + str(approximation_order_for_elliptic_integrals) + ') was not within the expected range, at m = ' + str(elliptic_m)
        print_op.log_and_raise_error(message)
    return None

def test_elliptic_e(epsilon=1.e-5):
    for approximation_order_for_elliptic_integrals in get_available_approximation_orders_for_elliptic_integrals():
        test_elliptic_e_at_position(approximation_order_for_elliptic_integrals, elliptic_m=0., epsilon=epsilon)
        test_elliptic_e_at_position(approximation_order_for_elliptic_integrals, elliptic_m=0.5, epsilon=epsilon)
        delta = 1.e-4
        test_elliptic_e_at_position(approximation_order_for_elliptic_integrals, elliptic_m=(1.0 - delta), epsilon=epsilon)
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
        print_op.log_and_raise_error(message)

    # obtuse triangle
    x_1 = 1. * xhat_np()
    x_2 = 2. * xhat_np()
    difference = get_altitude(x_1 - x_obs, x_2 - x_obs) - expected
    thresh = 1.e-6
    if thresh < difference**2.:
        message = 'biot-savart obtuse-triangle altitude test gives error of size: ' + str(difference)
        print_op.log_and_raise_error(message)

    return None

def is_numeric(val):
    return (isinstance(val, cas.DM) or isinstance(val, float) or isinstance(val, np.ndarray))

def is_numeric_scalar(val):
    if isinstance(val, float):
        return True
    elif is_numeric(val) and hasattr(val, 'shape') and val.shape == (1, 1):
        return True
    else:
        return False

def spline_interpolation(x_data, y_data, x_points):
    """ Interpolate solution values with b-splines
    """
    n_points = columnize(cas.DM(x_points)).shape[0]

    # can't use splines if all y_data values are zero
    if isinstance(y_data, cas.DM):
        all_zeros = y_data.is_zero()
    else:
        all_zeros = all([y_value == 0 for y_value in y_data])
    if all_zeros:
        return np.zeros(n_points)

    # create interpolating function
    name = 'spline_' + str(np.random.randint(10**5)) # hopefully unique name
    spline = cas.interpolant(name, 'bspline', [x_data], y_data, {})

    # function map to new discretization
    spline = spline.map(n_points)
    # interpolate
    y_points = spline(x_points).full()[0]

    return y_points

def test_spline_interpolation(epsilon=1.e-6):

    def cubic_function(xx):
        return 0.1235 * xx ** 3 - 2.3993 * xx ** 2. + 0.7344 * xx ** 1. - 3.231
    def parametrized(ss):
        x_max = 5.
        x_min = -1.
        return x_min + ss * (x_max - x_min)

    s_data = np.linspace(0., 1., 6)
    x_data = np.array([parametrized(ss) for ss in s_data])
    y_data = np.array([cubic_function(x_val) for x_val in x_data])

    x_test = np.array([parametrized(0.), parametrized(1./100.), parametrized(7./17.), parametrized(1.)])
    y_expected = columnize(cas.DM(np.array([cubic_function(x_val) for x_val in x_test])))
    y_found = columnize(spline_interpolation(x_data, y_data, x_points=x_test))

    diff = y_expected - y_found
    criteria = (cas.mtimes(diff.T, diff) < epsilon**2.)

    if not criteria:
        message = 'spline interpolation does not work as expected'
        print_op.log_and_raise_error(message)

    return None

def test():
    test_altitude()
    test_elliptic_e()
    test_elliptic_k()
    test_elliptic_pi()
    test_spline_interpolation()
    return None

# test()