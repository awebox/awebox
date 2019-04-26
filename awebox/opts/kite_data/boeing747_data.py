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
from casadi.tools import vertcat

def data_dict():

    data_dict = {}
    data_dict['name'] = 'boeing747'
    data_dict['geometry'] = geometry()
    data_dict['aero_deriv'] = aero_deriv()
    data_dict['battery'] = {}

    return data_dict

def geometry():
    # data from Heffley, Robert K. and Jewell, Wayne F.
    # Aircraft handling qualities data
    # NASA CR-2144
    # Hawthorne, CA.

    geometry = {}

    geometry['m_k'] = 288756.903  # [kg]
    geometry['s_ref'] = 510.9667  # [m^2]
    geometry['b_ref'] = 59.643264  # [m]
    geometry['c_ref'] = geometry['s_ref'] / geometry['b_ref']  # [m]

    geometry['ar'] = geometry['b_ref'] / geometry['c_ref']
    geometry['j'] = np.array([[24.67588669e6, 0., 1.315143e6],
                              [0.0, 44.87757e6, 0.0],
                              [1.315143e6, 0.0, 67.38415e6]])

    geometry['delta_max'] = np.array([20., 30., 30.]) * np.pi / 180.
    geometry['ddelta_max'] = np.array([2., 2., 2.])

    # only for plotting
    geometry['length'] = geometry['b_ref']  # only for plotting
    geometry['height'] = geometry['b_ref'] / 5.  # only for plotting
    geometry['c_root'] = 1.4 * geometry['c_ref']
    geometry['c_tip'] = 2. * geometry['c_ref'] - geometry['c_root']

    geometry['fuselage'] = True
    geometry['wing'] = True
    geometry['tail'] = True
    geometry['wing_profile'] = None

    return geometry

def aero_deriv():
    # data from Heffley, Robert K. and Jewell, Wayne F.
    # Aircraft handling qualities data
    # NASA CR-2144
    # Hawthorne, CA.

    aero_deriv = {}
    
    aero_deriv['CL0'] = 1.11
    aero_deriv['CS0'] = 0.
    aero_deriv['CD0'] = 0.102

    aero_deriv['CLalpha'] = 5.70
    aero_deriv['CSalpha'] = 0.
    aero_deriv['CDalpha'] = 0.66

    aero_deriv['CLalpha2'] = 0
    aero_deriv['CSalpha2'] = 0.
    aero_deriv['CDalpha2'] = 0.
    
    aero_deriv['CLbeta'] = 0.
    aero_deriv['CSbeta'] = -1.08
    aero_deriv['CDbeta'] = 0.

    aero_deriv['CLbeta2'] = 0.
    aero_deriv['CSbeta2'] = 0.
    aero_deriv['CDbeta2'] = 0.
    
    aero_deriv['CLdeltae'] = 0.338
    aero_deriv['CLdeltaa'] = 0.
    aero_deriv['CLdeltar'] = 0.

    aero_deriv['CSdeltae'] = 0.
    aero_deriv['CSdeltaa'] = 0.
    aero_deriv['CSdeltar'] = 0.179

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

    aero_deriv['CLp'] = 0.
    aero_deriv['CSp'] = 0.
    aero_deriv['CDp'] = 0.

    aero_deriv['CLq'] = 5.4
    aero_deriv['CSq'] = 0.
    aero_deriv['CDq'] = 0.

    aero_deriv['CLr'] = 0.
    aero_deriv['CSr'] = 0.
    aero_deriv['CDr'] = 0.

    aero_deriv['CLalpha_deltae'] = 0.
    aero_deriv['CSalpha_deltae'] = 0.
    aero_deriv['CDalpha_deltae'] = 0.

    aero_deriv['CLbeta_deltaa'] = 0.
    aero_deriv['CSbeta_deltaa'] = 0.
    aero_deriv['CDbeta_deltaa'] = 0.

    aero_deriv['CLbeta_deltar'] = 0.
    aero_deriv['CSbeta_deltar'] = 0.
    aero_deriv['CDbeta_deltar'] = 0.

    aero_deriv['Cl0'] = 0.
    aero_deriv['Cm0'] = 0.
    aero_deriv['Cn0'] = 0.

    aero_deriv['Cldeltae'] = 0.
    aero_deriv['Cldeltaa'] = 0.053
    aero_deriv['Cldeltar'] = 0.

    aero_deriv['Cmdeltae'] = 0.
    aero_deriv['Cmdeltaa'] = 0.
    aero_deriv['Cmdeltar'] = 0.

    aero_deriv['Cndeltae'] = 0.
    aero_deriv['Cndeltaa'] = 0.0083
    aero_deriv['Cndeltar'] = -0.112

    aero_deriv['Clalpha'] = 0.
    aero_deriv['Cmalpha'] = -1.45
    aero_deriv['Cnalpha'] = 0.
    
    aero_deriv['Clbeta'] = -0.281
    aero_deriv['Cmbeta'] = 0.
    aero_deriv['Cnbeta'] = 0.184

    aero_deriv['Clp'] = -.502
    aero_deriv['Cmp'] = 0.
    aero_deriv['Cnp'] = -0.222

    aero_deriv['Clq'] = 0.
    aero_deriv['Cmq'] = -21.4
    aero_deriv['Cnq'] = 0.
    
    aero_deriv['Clr'] = 0.195
    aero_deriv['Cmr'] = 0.
    aero_deriv['Cnr'] = -0.36

    aero_deriv['alpha_max_deg'] = 20.
    aero_deriv['alpha_min_deg'] = -20.
    aero_deriv['beta_max_deg'] = 15.
    aero_deriv['beta_min_deg'] = -15.0

    aero_deriv['alpha_max'] = aero_deriv['alpha_max_deg'] * np.pi / 180.
    aero_deriv['alpha_min'] = aero_deriv['alpha_min_deg'] * np.pi / 180.
    aero_deriv['beta_max'] = aero_deriv['beta_max_deg'] * np.pi / 180.
    aero_deriv['beta_min'] = aero_deriv['beta_min_deg'] * np.pi / 180.

    return aero_deriv
