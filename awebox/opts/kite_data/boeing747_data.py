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
    data_dict['name'] = 'boeing747'
    data_dict['geometry'] = geometry()

    stab_derivs, aero_validity = aero()
    data_dict['stab_derivs'] = stab_derivs # stability derivatives
    data_dict['aero_validity'] = aero_validity

    # (optional: on-board battery model)
    coeff_min = np.array([0, -80*np.pi/180.0])
    coeff_max = np.array([2, 80*np.pi/180.0])
    data_dict['battery'] = battery_model_parameters(coeff_max, coeff_min)

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

    # tether attachment point
    geometry['r_tether'] = np.zeros((3,1))

    return geometry

def battery_model_parameters(coeff_max, coeff_min):

    # copied from ampyx ap2

    battery_model = {}

    # guessed values for battery model
    battery_model['flap_length'] = 0.2
    battery_model['flap_width'] = 0.1
    battery_model['max_flap_defl'] = 20.*(np.pi/180.)
    battery_model['min_flap_defl'] = -20.*(np.pi/180.)
    battery_model['c_dl'] = (battery_model['max_flap_defl'] - battery_model['min_flap_defl'])/(coeff_min[0] - coeff_max[0])
    battery_model['c_dphi'] = (battery_model['max_flap_defl'] - battery_model['min_flap_defl'])/(coeff_min[1] - coeff_max[1])
    battery_model['defl_lift_0'] = battery_model['min_flap_defl'] - battery_model['c_dl']*coeff_max[0]
    battery_model['defl_roll_0'] = battery_model['min_flap_defl'] - battery_model['c_dphi']*coeff_max[1]
    battery_model['voltage'] = 3.7
    battery_model['mAh'] = 5000.
    battery_model['charge'] = battery_model['mAh']*3600.*1e-3
    battery_model['number_of_cells'] = 15.
    battery_model['conversion_efficiency'] = 0.7
    battery_model['power_controller'] = 50.
    battery_model['power_electronics'] = 10.
    battery_model['charge_fraction'] = 1.

    return battery_model




def aero():
    # data from Heffley, Robert K. and Jewell, Wayne F.
    # Aircraft handling qualities data
    # NASA CR-2144
    # Hawthorne, CA.
    # https://www.robertheffley.com/docs/Data/NASA%20CR-2144--Heffley--Aircraft%20Handling%20Qualities%20Data.pdf

    stab_derivs = {}

    stab_derivs['frame'] = {}
    stab_derivs['frame']['force'] = 'wind'
    stab_derivs['frame']['moment'] = 'control'

    stab_derivs['CL'] = {}
    stab_derivs['CL']['0'] = [1.11]
    stab_derivs['CL']['alpha'] = [5.70]
    stab_derivs['CL']['deltae'] = [0.338]
    stab_derivs['CL']['q'] = [5.4]

    stab_derivs['CS'] = {}
    stab_derivs['CS']['beta'] = [-1.08]
    stab_derivs['CS']['deltar'] = [0.179]

    stab_derivs['CD'] = {}
    stab_derivs['CD']['0'] = [0.102]
    stab_derivs['CD']['alpha'] = [0.66]

    stab_derivs['Cl'] = {}
    stab_derivs['Cl']['deltaa'] = [0.053]
    stab_derivs['Cl']['beta'] = [-0.281]
    stab_derivs['Cl']['p'] = [-0.502]
    stab_derivs['Cl']['r'] = [0.195]

    stab_derivs['Cm'] = {}
    stab_derivs['Cm']['alpha'] = [-1.45]
    stab_derivs['Cm']['q'] = [-21.4]

    stab_derivs['Cn'] = {}
    stab_derivs['Cn']['deltaa'] = [0.0083]
    stab_derivs['Cn']['deltar'] = [-0.112]
    stab_derivs['Cn']['beta'] = [0.184]
    stab_derivs['Cn']['p'] = [-0.222]
    stab_derivs['Cn']['r'] = [-0.36]

    aero_validity = {}
    aero_validity['alpha_max_deg'] = 20.
    aero_validity['alpha_min_deg'] = -20.
    aero_validity['beta_max_deg'] = 15.
    aero_validity['beta_min_deg'] = -15.0

    return stab_derivs, aero_validity
