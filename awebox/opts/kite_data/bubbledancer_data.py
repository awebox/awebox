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


def data_dict():

    data_dict = {}
    data_dict['name'] = 'bubbledancer'
    data_dict['geometry'] = geometry()

    stab_derivs, aero_validity = aero()
    data_dict['stab_derivs'] = stab_derivs # stability derivatives
    data_dict['aero_validity'] = aero_validity

    # (optional: on-board battery model)
    coeff_min = np.array([0, -80 * np.pi / 180.0])
    coeff_max = np.array([2, 80 * np.pi / 180.0])
    data_dict['battery'] = battery_model_parameters(coeff_max, coeff_min)

    return data_dict

def geometry():

    # values from AVL sample files
    # and, plan: http://www.charlesriverrc.org/articles/bubbledancer/PDFs/bd_V3.pdf

    geometry = {}

    geometry['s_ref'] = 0.6541922  # [m^2]
    geometry['b_ref'] = 2.9718  # [m]
    geometry['c_ref'] = geometry['s_ref'] / geometry['b_ref']  # [m]

    geometry['m_k'] = 0.9195  # [kg]

    geometry['ar'] = geometry['b_ref'] / geometry['c_ref']
    geometry['j'] = np.array([[0.2052, 0.0, 0.1702e-2],
                              [0.0, 0.7758e-1, 0.0],
                              [0.1702e-2, 0.0, 0.2790]])

    geometry['length'] = 1.534 #[m]  # only for plotting
    geometry['height'] = 0.26416  #[m] only for plotting
    geometry['delta_max'] = np.array([20., 30., 30.]) * np.pi / 180.
    geometry['ddelta_max'] = np.array([2., 2., 2.])

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

    # values copied from ampyx ap2

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

    # values from AVL run at zero-deg bank angle trim.

    stab_derivs = {}

    stab_derivs['frame'] = {}
    stab_derivs['frame']['force'] = 'wind'
    stab_derivs['frame']['moment'] = 'control'

    stab_derivs['CL'] = {}
    stab_derivs['CL']['0'] = [0.700]
    stab_derivs['CL']['alpha'] = [5.675616]
    stab_derivs['CL']['deltae'] = [0.008059]
    stab_derivs['CL']['p'] = [0.000008]
    stab_derivs['CL']['q'] = [7.286214]
    stab_derivs['CL']['r'] = [-0.000001]

    stab_derivs['CD'] = {}
    stab_derivs['CD']['0'] = [0.02862]
    stab_derivs['CD']['alpha'] = [0.1, 1.3] #arbitrary, based on ampyx ap2
    stab_derivs['CD']['deltae'] = [0.000284]

    stab_derivs['CS'] = {}
    stab_derivs['CS']['alpha'] = [-0.000003]
    stab_derivs['CS']['beta'] = [-0.404699]
    stab_derivs['CS']['deltar'] = [-0.003376]
    stab_derivs['CS']['p'] = [-0.380742]
    stab_derivs['CS']['q'] = [-0.000001]
    stab_derivs['CS']['r'] = [0.294666]

    stab_derivs['Cl'] = {}
    stab_derivs['Cl']['deltaa'] = [0.3] # arbitrary, assumed similar to ampyx ap2
    stab_derivs['Cl']['deltar'] = [-0.000076]
    stab_derivs['Cl']['p'] = [-0.634188]
    stab_derivs['Cl']['q'] = [-0.000002]
    stab_derivs['Cl']['r'] = [0.181038]
    stab_derivs['Cl']['alpha'] = [-0.000003]
    stab_derivs['Cl']['beta'] = [-0.257096]

    stab_derivs['Cm'] = {}
    stab_derivs['Cm']['alpha'] = [-0.895625]
    stab_derivs['Cm']['deltae'] = [-0.027418]
    stab_derivs['Cm']['q'] = [-12.180685]

    stab_derivs['Cn'] = {}
    stab_derivs['Cn']['deltar'] = [0.001245]
    stab_derivs['Cn']['alpha'] = [0.000001]
    stab_derivs['Cn']['beta'] = [0.057021]
    stab_derivs['Cn']['p'] = [-0.068262]
    stab_derivs['Cn']['r'] = [-0.066292]


    aero_validity = {}
    aero_validity['alpha_max_deg'] = 20.
    aero_validity['alpha_min_deg'] = -20.
    aero_validity['beta_max_deg'] = 15.
    aero_validity['beta_min_deg'] = -15.0

    return stab_derivs, aero_validity