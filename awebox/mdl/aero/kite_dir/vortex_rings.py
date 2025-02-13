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

def far_wake_ring_induction(p_k, p_r, n_ring, gamma_ring, R_ring, data):

    """ Model for the induced axial speed due to a vortex ring in the far wake.
    """

    h = ca.mtimes((p_r - p_k).T, n_ring)
    vec = (p_r - p_k) - h * n_ring
    R_j = ca.sqrt(ca.mtimes(vec.T, vec))
    w_f = - gamma_ring / (4 * np.pi) * (elliptic_integrand_series_axial(R_j, R_ring, h, data['N_elliptic_int'], method = data['elliptic_method']))

    return w_f