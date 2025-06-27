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


def vortex_ring_dipole_approx(q, q_v, n, Gamma):
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
    
    m = Gamma * n * 100**3 # Dipole moment (scalar * direction)
    dot = ca.mtimes(r.T, m)
    
    u_ind = (1 / (4 * np.pi)) * (3 * dot * r - r_norm**2 * m) / r_norm**5
    
    # u_ind = np.zeros((3,1))
    return u_ind

def far_wake_ring_induction(p_k, p_r, n_ring, gamma_ring, R_ring, data):

    """ Model for the induced axial speed due to a vortex ring in the far wake.
    """

    induction_fun = far_wake_ring_induction_function(data)
    w_f = induction_fun(p_k, p_r, n_ring, gamma_ring, R_ring)

    return w_f

def far_wake_ring_induction_function(data):

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
    w_f = - vortex_ring_dipole_approx(p_k, p_r, n_ring, gamma_ring)

    far_wake_axial_induction_fun = ca.Function('far_wake_axial_induction_fun', [p_k, p_r, n_ring, gamma_ring, R_ring], [w_f])
    # compilation_file_name = 'far_wake_axial_induction_fun_'
    # compilation_file_name += 'N_ell_int{}_ell_method_{}'.format(
    #             data['N_elliptic_int'],
    #             data['elliptic_method'],
    #         )
    # far_wake_axial_induction_fun_cf = cf.CachedFunction(compilation_file_name, far_wake_axial_induction_fun, do_compile=True)

    return far_wake_axial_induction_fun