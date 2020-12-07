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

import matplotlib.pylab as plt
import scipy
import scipy.io
import scipy.sparse as sps

import casadi.tools as cas
import numpy as np
from awebox.logger.logger import Logger as awelogger
import awebox.tools.vector_operations as vect_op

def get_loyd_power(power_density, CL, CD, s_ref, elevation_angle=0.):
    phf = get_loyd_phf(CL, CD, elevation_angle)
    p_loyd = power_density * s_ref * phf
    return p_loyd

def get_loyd_phf(CL, CD, elevation_angle=0.):
    epsilon = 1.e-4 #8
    CR = CL * vect_op.smooth_sqrt(1. + (CD / (CL + epsilon))**2.)

    phf = 4. / 27. * CR * (CR / CD) ** 2. * np.cos(elevation_angle) ** 3.
    return phf


def determine_if_periodic(options):

    enforce_periodicity = bool(True)
    if options['trajectory']['type'] in ['transition', 'compromised_landing', 'nominal_landing', 'launch','mpc']:
         enforce_periodicity = bool(False)

    return enforce_periodicity
