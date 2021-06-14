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
import numpy as np
from casadi.tools import vertcat

def data_dict():

    data_dict = {}
    data_dict['name'] = 'kitepower'

    data_dict['geometry'] = geometry() # kite geometry

    stab_derivs, aero_validity = aero()
    data_dict['stab_derivs'] = stab_derivs # stability derivatives
    data_dict['aero_validity'] = aero_validity

    return data_dict

def geometry():

    geometry = {}
    geometry['m_k'] = 36.8  # [kg]
    geometry['s_ref'] = 200.0 # [m^2]
    geometry['s_ref_side'] = 10.0 # [m^2]

    # tether attachment point
    geometry['r_tether'] = np.zeros((3,1)) # assumed at COM

    geometry['b_ref'] = 1.0  # [m]
    geometry['c_ref'] = geometry['s_ref'] / geometry['b_ref']  # [m]

    return geometry

def aero():
    # commented values are not currently supported, future implementation

    stab_derivs = {}

    stab_derivs['frame'] = {}
    stab_derivs['frame']['force'] = 'control'
    stab_derivs['frame']['moment'] = 'control'

    stab_derivs['CD'] = {}
    stab_derivs['CD']['0'] = [0.13]
    stab_derivs['CD']['alpha'] = [0.0, 0.0]

    stab_derivs['CS'] = {}
    stab_derivs['CS']['0'] = [-1.0]

    stab_derivs['CL'] = {}
    stab_derivs['CL']['0'] = [0.59]
    stab_derivs['CL']['alpha'] = [0.0, 0.0]

    aero_validity = {}
    # aero_validity['alpha_max_deg'] = 21.7724
    # aero_validity['alpha_min_deg'] = -7.4485
    # aero_validity['beta_max_deg'] = 15.
    # aero_validity['beta_min_deg'] = -15.0

    aero_validity['alpha_max_deg'] = 9.
    aero_validity['alpha_min_deg'] = -6.
    aero_validity['beta_max_deg'] = 20.
    aero_validity['beta_min_deg'] = -20.


    return stab_derivs, aero_validity
