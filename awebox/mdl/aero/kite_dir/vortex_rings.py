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
vortex rings code
- authors: Jochem De Schutter, Antonia Mühleck 
'''

import casadi as ca
from awebox.mdl.aero.kite_dir.elliptic_integrals_functions import *
import awebox.tools.cached_functions as cf

# Compensation model
def compensation_model(r, A, eps):
    return 1 + A / (r**2 + eps**2)

def vortex_ring_dipole_approx(q, q_v, n, Gamma, R_ring):
    """
    Approximate the induced velocity at point q from a vortex ring,
    using a far-field potential dipole aligned with the ring's normal vector.

    Parameters:
        q      : np.ndarray, shape (3,)
        q_v    : np.ndarray, shape (3,)
        n      : np.ndarray, shape (3,), unit vector normal to the ring
        Gamma  : float, vortex strength
        r_f    : float, ring radius

    Returns:
        u_ind : np.ndarray, shape (3,)
    """
    r = q - q_v
    r_norm = ca.sqrt(ca.mtimes(r.T, r))

    m = Gamma * n * 100**3 # Dipole moment (scalar * direction * scaling**3)
    dot = ca.mtimes(r.T, m)

    # compensation model
    rdot = ca.mtimes(r.T, n)
    # rdot = r
    rdot_norm = ca.sqrt(ca.mtimes(rdot.T, rdot))
    A = - 0.1508
    epsilon = 0.3827
    comp = compensation_model(rdot_norm / R_ring, A, epsilon)
    comp = 1
    u_ind = (1 / (4 * np.pi)) * ((3 * dot * r) / (r_norm**5) - m / r_norm**3)  * comp
    
    # u_ind = np.zeros((3,1))
    return u_ind

def far_wake_induction(p_k, p_r, n_ring, ec_ring, gamma_ring, R_ring, data):

    """ Model for the induced axial speed due to a vortex ring in the far wake.
    """

    if data['type'] == 'dipole':
        induction_fun = far_wake_dipole_induction_function(data)
        w_f = induction_fun(p_k, p_r, n_ring, gamma_ring, R_ring)

    elif data['type'] == 'rectangle':
        induction_fun = rectangle_induced_velocity_taylor_fun()
        w_f = induction_fun(p_k, p_r, 0.8 * R_ring, gamma_ring, n_ring, ec_ring)

    return w_f


def simplified_vortex_segment(q, q0, e_dir, l, Gamma):
    """
    Computes induced velocity at point q from a straight vortex filament.

    Parameters:
        q     : ndarray (3,) -- observation point
        q0    : ndarray (3,) -- start point of the filament
        e_dir : ndarray (3,) -- unit vector along the filament
        l     : float         -- length of the filament
        Gamma : float         -- circulation strength

    Returns:
        u     : ndarray (3,) -- induced velocity at q
    """
    r1 = q - q0
    r2 = r1 - l * e_dir

    cross = ca.cross(r1, e_dir)
    cross_norm_sq = ca.mtimes(cross.T, cross)

    r1_norm = ca.norm_2(r1)
    r2_norm = ca.norm_2(r2)

    # if cross_norm_sq < 1e-12 or r1_norm < 1e-12 or r2_norm < 1e-12:
    #     return np.zeros(3)

    proj1 = ca.mtimes(e_dir.T, r1)
    proj2 = proj1 - l

    factor = Gamma / (4 * np.pi * cross_norm_sq)
    return - factor * cross * (proj1 / r1_norm - proj2 / r2_norm)

def rectangle_induced_velocity(q, q_v, b, s, Gamma, n, e_c):
    """
    Computes induced velocity at point q from a vortex rectangle.

    Parameters:
        q     : ndarray (3,) -- observation point
        q_v   : ndarray (3,) -- center of the rectangle
        b     : float         -- spanwise length (along e_b)
        s     : float         -- chordwise length (along e_c)
        Gamma : float         -- circulation strength
        n     : ndarray (3,)  -- unit normal vector of the rectangle
        e_c   : ndarray (3,)  -- unit vector along chord direction

    Returns:
        u_total : ndarray (3,) -- total induced velocity at q
    """

    e_b = ca.cross(n, e_c)  # spanwise direction
    dy = s / 2
    dz = b / 2

    # Corner points of the rectangle (CCW order)
    p1 = q_v - dy * e_c - dz * e_b  # bottom left
    p2 = q_v - dy * e_c + dz * e_b  # bottom right
    p3 = q_v + dy * e_c + dz * e_b  # top right
    p4 = q_v + dy * e_c - dz * e_b  # top left

    u_total = np.zeros(3)

    # Side 1: bottom (p1 to p2), direction e_b, length 2*dz
    u_total += simplified_vortex_segment(q, p1, e_b, 2 * dz, Gamma)
    # Side 2: right (p2 to p3), direction e_c, length 2*dy
    u_total += simplified_vortex_segment(q, p2, e_c, 2 * dy, Gamma)
    # Side 3: top (p3 to p4), direction -e_b, length 2*dz
    u_total += simplified_vortex_segment(q, p3, -e_b, 2 * dz, Gamma)
    # Side 4: left (p4 to p1), direction -e_c, length 2*dy
    u_total += simplified_vortex_segment(q, p4, -e_c, 2 * dy, Gamma)

    return u_total * 100**3


def rectangle_induced_velocity_taylor_fun():

    # CasADi symbolic inputs
    q = ca.SX.sym("q", 3)
    q_v = ca.SX.sym("q_v", 3)
    b = ca.SX.sym("b")
    s = ca.SX.sym("s")
    Gamma = ca.SX.sym("Gamma")
    n = ca.SX.sym("n", 3)
    e_c = ca.SX.sym("e_c", 3)

    # Compute induced velocity
    u = rectangle_induced_velocity(q, q_v, b, s, Gamma, n, e_c)

    # Compute Jacobian w.r.t. s
    jac_u_s = ca.jacobian(u, s)

    # Linearization at s = 0
    jac_func = ca.Function(
        "jacobian_wrt_s",
        [q, q_v, b, s, Gamma, n, e_c],
        [jac_u_s]
    )

    jac_func_s_zero = ca.Function(
        "jac_func_s_zero",
        [q, q_v, b, Gamma, n, e_c],
        [ca.simplify(jac_func(q, q_v, b, 0, Gamma, n, e_c))]
    )

    return jac_func_s_zero


def far_wake_dipole_induction_function(data):

    p_k = ca.SX.sym('p_k', 3, 1)
    p_r = ca.SX.sym('p_r', 3, 1)
    n_ring = ca.SX.sym('n_ring', 3, 1)
    gamma_ring = ca.SX.sym('gamma_ring', 3, 1)
    R_ring = ca.SX.sym('R_ring', 1, 1)

    # FAR WAKE INDUCTION OF VORTEX RING
    # h = ca.mtimes((p_r - p_k).T, n_ring)
    # vec = (p_r - p_k) - h * n_ring
    # R_j = ca.sqrt(ca.mtimes(vec.T, vec))
    # w_f = - gamma_ring / (4 * np.pi) * (elliptic_integrand_series_axial(R_j, R_ring, h, data['N_elliptic_int'], method = data['elliptic_method']))

    # FAR WAKE INDUCTION OF VORTEX DIPOLE!!!!
    w_f = - vortex_ring_dipole_approx(p_k, p_r, n_ring, gamma_ring, R_ring)

    far_wake_axial_induction_fun = ca.Function('far_wake_axial_induction_fun', [p_k, p_r, n_ring, gamma_ring, R_ring], [w_f])
    # compilation_file_name = 'far_wake_axial_induction_fun_'
    # compilation_file_name += 'N_ell_int{}_ell_method_{}'.format(
    #             data['N_elliptic_int'],
    #             data['elliptic_method'],
    #         )
    # far_wake_axial_induction_fun_cf = cf.CachedFunction(compilation_file_name, far_wake_axial_induction_fun, do_compile=True)

    return far_wake_axial_induction_fun