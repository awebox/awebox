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
import numpy as np
from casadi.tools import *

def data_dict():

    data_dict = {}
    data_dict['name'] = 'flatplate'
    data_dict['geometry'] = geometry()
    data_dict['aero_deriv'] = aero_deriv()
    data_dict['battery'] = {}

    return data_dict

def geometry():

    geometry = {}
    # 'aerodynamic parameter identification for an airborne wind energy pumping system', licitra, williams, gillis, ghandchi, sierbling, ruiterkamp, diehl, 2017
    # 'numerical optimal trajectory for system in pumping mode described by differential algebraic equation (focus on ap2)' licitra, 2014
    geometry['m_k'] = 1.  # [kg]
    geometry['b_ref'] = 3.  # [m]
    geometry['c_ref'] = 1.  # [m]
    geometry['s_ref'] = geometry['b_ref'] * geometry['c_ref']
    geometry['ar'] = geometry['b_ref'] / geometry['c_ref']

    h = 0.01 * geometry['c_ref']
    d = geometry['c_ref']
    w = geometry['b_ref']

    geometry['j'] = np.array([[(h**2. + d**2.), 0., 0.],
                              [0., (w**2. + d**2.), 0.],
                              [0., 0., (w**2. + h**2.)]]) * geometry['m_k'] / 12.

    geometry['delta_max'] = vertcat(inf, inf, inf)
    geometry['ddelta_max'] = vertcat(inf, inf, inf)

    # plotting stuff
    geometry['length'] = d
    geometry['height'] = h

    geometry['c_root'] = geometry['c_ref']
    geometry['c_tip'] = geometry['c_ref']

    geometry['fuselage'] = False
    geometry['wing'] = True
    geometry['tail'] = False
    geometry['wing_profile'] = "0001"

    return geometry

def aero_deriv():

    aero_deriv = {}

    aero_deriv['CL0'] = 0.
    aero_deriv['CS0'] = 0.
    aero_deriv['CD0'] = 0.

    aero_deriv['CLalpha'] = 2. * np.pi
    aero_deriv['CSalpha'] = 0.
    aero_deriv['CDalpha'] = 0.4 / (20. * np.pi / 180.)

    aero_deriv['CLalpha2'] = 0
    aero_deriv['CSalpha2'] = 0.
    aero_deriv['CDalpha2'] = 0.

    aero_deriv['CLbeta'] = 0.
    aero_deriv['CSbeta'] = 0.
    aero_deriv['CDbeta'] = 0.

    aero_deriv['CLbeta2'] = 0.
    aero_deriv['CSbeta2'] = 0.
    aero_deriv['CDbeta2'] = 0.

    aero_deriv['CLdeltae'] = 0.
    aero_deriv['CLdeltaa'] = 0.
    aero_deriv['CLdeltar'] = 0.

    aero_deriv['CSdeltae'] = 0.
    aero_deriv['CSdeltaa'] = 0.
    aero_deriv['CSdeltar'] = 0.

    aero_deriv['CDdeltae'] = 0.
    aero_deriv['CDdeltaa'] = 0.
    aero_deriv['CDdeltar'] = 0.

    aero_deriv['CLdeltaa2'] = 0.
    aero_deriv['CSdeltaa2'] = 0.
    aero_deriv['CDdeltaa2'] = 0.

    aero_deriv['CLdeltae2'] = 0.
    aero_deriv['CSdeltae2'] = 0.
    aero_deriv['CDdeltae2'] = 0.

    aero_deriv['CLdeltar2'] = 0.
    aero_deriv['CSdeltar2'] = 0.
    aero_deriv['CDdeltar2'] = 0.

    aero_deriv['CLalpha_deltae'] = 0.
    aero_deriv['CSalpha_deltae'] = 0.
    aero_deriv['CDalpha_deltae'] = 0.

    aero_deriv['CLbeta_deltaa'] = 0.
    aero_deriv['CSbeta_deltaa'] = 0.
    aero_deriv['CDbeta_deltaa'] = 0.

    aero_deriv['CLbeta_deltar'] = 0.
    aero_deriv['CSbeta_deltar'] = 0.
    aero_deriv['CDbeta_deltar'] = 0.

    aero_deriv['CLp'] = 0.
    aero_deriv['CSp'] = 0.
    aero_deriv['CDp'] = 0.

    aero_deriv['CLq'] = 0.
    aero_deriv['CSq'] = 0.
    aero_deriv['CDq'] = 0.

    aero_deriv['CLr'] = 0.
    aero_deriv['CSr'] = 0.
    aero_deriv['CDr'] = 0.

    aero_deriv['Cl0'] = 0.
    aero_deriv['Cm0'] = 0.
    aero_deriv['Cn0'] = 0.

    aero_deriv['Cldeltae'] = 0.
    aero_deriv['Cldeltaa'] = 0.
    aero_deriv['Cldeltar'] = 0.

    aero_deriv['Cmdeltae'] = 0.
    aero_deriv['Cmdeltaa'] = 0.
    aero_deriv['Cmdeltar'] = 0.

    aero_deriv['Cndeltae'] = 0.
    aero_deriv['Cndeltaa'] = 0.
    aero_deriv['Cndeltar'] = 0.

    aero_deriv['Clalpha'] = 0.
    aero_deriv['Cmalpha'] = 0.
    aero_deriv['Cnalpha'] = 0.

    aero_deriv['Clbeta'] = 0.
    aero_deriv['Cmbeta'] = 0.
    aero_deriv['Cnbeta'] = 0.

    aero_deriv['Clp'] = 0.
    aero_deriv['Cmp'] = 0.
    aero_deriv['Cnp'] = 0.

    aero_deriv['Clq'] = 0.
    aero_deriv['Cmq'] = 0.
    aero_deriv['Cnq'] = 0.

    aero_deriv['Clr'] = 0.
    aero_deriv['Cmr'] = 0.
    aero_deriv['Cnr'] = 0.

    aero_deriv['alpha_max_deg'] = inf
    aero_deriv['alpha_min_deg'] = -inf
    aero_deriv['beta_max_deg'] = inf
    aero_deriv['beta_min_deg'] = -inf

    aero_deriv['alpha_max'] = aero_deriv['alpha_max_deg'] * np.pi / 180.
    aero_deriv['alpha_min'] = aero_deriv['alpha_min_deg'] * np.pi / 180.
    aero_deriv['beta_max'] = aero_deriv['beta_max_deg'] * np.pi / 180.
    aero_deriv['beta_min'] = aero_deriv['beta_min_deg'] * np.pi / 180.

    return aero_deriv
