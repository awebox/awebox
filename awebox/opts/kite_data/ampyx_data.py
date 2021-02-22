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
    data_dict['name'] = 'ampyx'

    data_dict['geometry'] = geometry() # kite geometry

    stab_derivs, aero_validity = aero()
    data_dict['stab_derivs'] = stab_derivs # stability derivatives
    data_dict['aero_validity'] = aero_validity

    # (optional: on-board battery model)
    coeff_min = np.array([0, -80*np.pi/180.0])
    coeff_max = np.array([2, 80*np.pi/180.0])
    data_dict['battery'] = battery_model_parameters(coeff_max, coeff_min)

    return data_dict

def geometry():

    geometry = {}
    # 'aerodynamic parameter identification for an airborne wind energy pumping system', licitra, williams, gillis, ghandchi, sierbling, ruiterkamp, diehl, 2017
    # 'numerical optimal trajectory for system in pumping mode described by differential algebraic equation (focus on ap2)' licitra, 2014
    geometry['b_ref'] = 5.5  # [m]
    geometry['s_ref'] = 3.  # [m^2]
    geometry['c_ref'] = geometry['s_ref'] / geometry['b_ref']  # [m]

    geometry['m_k'] = 36.8  # [kg]

    geometry['ar'] = geometry['b_ref'] / geometry['c_ref']
    geometry['j'] = np.array([[25., 0.0, 0.47],
                              [0.0, 32., 0.0],
                              [0.47, 0.0, 56.]])

    geometry['length'] = geometry['b_ref']  # only for plotting
    geometry['height'] = geometry['b_ref'] / 5.  # only for plotting
    # geometry['delta_max'] = np.array([20., 30., 30.]) * np.pi / 180.
    geometry['delta_max'] = np.array([5., 10., 5.]) * np.pi / 180.
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
    # commented values are not currently supported, future implementation

    stab_derivs = {}

    stab_derivs['frame'] = {}
    stab_derivs['frame']['force'] = 'wind'
    stab_derivs['frame']['moment'] = 'control'


    # A reference model for airborne wind energy systems for optimization and control
    # Article
    # March 2019 Renewable Energy
    # Elena Malz Jonas Koenemann S. Sieberling Sebastien Gros

    # commented values are not currently supported, future implementation

    stab_derivs = {}

    stab_derivs['frame'] = {}
    stab_derivs['frame']['force'] = 'control'
    stab_derivs['frame']['moment'] = 'control'

    stab_derivs['CX'] = {}
    stab_derivs['CX']['0'] = [-0.0293]
    stab_derivs['CX']['alpha'] = [0.4784, 2.5549]
    stab_derivs['CX']['q'] = [-0.6029, 4.4124]
    # stab_derivs['CX']['alpha_q'] = [4.4124]
    stab_derivs['CX']['deltae'] = [-0.0106, 0.1115]
    # stab_derivs['CX']['alpha_deltae'] = [0.1115]

    stab_derivs['CY'] = {}
    stab_derivs['CY']['beta'] = [-0.1855, -0.0299, 0.0936]
    # stab_derivs['CY']['alpha_beta'] = [-0.0299]
    # stab_derivs['CY']['alpha2_beta'] = [0.0936]
    stab_derivs['CY']['p'] = [-0.1022, -0.0140, 0.0496]
    # stab_derivs['CY']['alpha_p'] = [-0.0140]
    # stab_derivs['CY']['alpha2_p'] = [0.0496]
    stab_derivs['CY']['r'] = [0.1694, 0.1368]
    # stab_derivs['CY']['alpha_r'] = [0.1368]
    stab_derivs['CY']['deltaa'] = [-0.0514, -0.0024, 0.0579]
    # stab_derivs['CY']['alpha_deltaa'] = [-0.0024]
    # stab_derivs['CY']['alpha2_deltaa'] = [0.0579]
    stab_derivs['CY']['deltar'] = [0.10325, 0.0268, -0.1036]
    # stab_derivs['CY']['alpha_deltar'] = [0.0268]
    # stab_derivs['CY']['alpha2_deltar'] = [-0.1036]

    stab_derivs['CZ'] = {}
    stab_derivs['CZ']['0'] = [-0.5526]
    stab_derivs['CZ']['alpha'] = [-5.0676, 5.7736]
    stab_derivs['CZ']['q'] = [-7.5560, 0.1251, 6.1486]
    # stab_derivs['CZ']['alpha_q'] = [0.1251]
    # stab_derivs['CZ']['alpha2_q'] = [6.1486]
    stab_derivs['CZ']['deltae'] = [-0.315, -0.0013, 0.2923]
    # stab_derivs['CZ']['alpha_deltae'] = [-0.0013]
    # stab_derivs['CZ']['alpha2_deltae'] = [0.2923]

    #
    # # 'numerical optimal trajectory for system in pumping mode described by differential algebraic equation (focus on ap2)' licitra, 2014
    # stab_derivs['CL'] = {}
    # stab_derivs['CL']['0'] = [0.5284]
    # stab_derivs['CL']['alpha'] = [4.6306]
    # stab_derivs['CL']['q'] = [-0.6029]
    # stab_derivs['CL']['deltae'] = [0.1]
    #
    # stab_derivs['CS'] = {}
    # stab_derivs['CS']['0'] = [0.]
    # stab_derivs['CS']['beta'] = [-0.217]
    # stab_derivs['CS']['deltar'] = [0.113]
    #
    # stab_derivs['CD'] = {}
    # stab_derivs['CD']['0'] = [0.0273]
    # stab_derivs['CD']['alpha'] = [0.0965, 1.2697]
    # stab_derivs['CD']['beta'] = [0., -0.16247]
    # stab_derivs['CD']['deltae'] = [4.52856e-5, 4.19816e-5]
    # stab_derivs['CD']['deltaa'] = [0., 5.60583e-5]
    # stab_derivs['CD']['deltar'] = [0., 2.03105e-5]
    # # stab_derivs['CD']['alpha_deltae'] = [-9.79647e-5]
    # # stab_derivs['CD']['beta_deltaa'] = [-6.73139e-6]
    # # stab_derivs['CD']['beta_deltar'] = [5.55453e-5]


    # A reference model for airborne wind energy systems for optimization and control
    # Article
    # March 2019 Renewable Energy
    # Elena Malz Jonas Koenemann S. Sieberling Sebastien Gros


    stab_derivs['Cl'] = {}
    stab_derivs['Cl']['beta'] = [-0.0630, -0.0003, 0.0312]
    # stab_derivs['Cl']['alpha_beta'] = [-0.0003]
    # stab_derivs['Cl']['alpha2_beta'] = [0.0312]
    stab_derivs['Cl']['p'] = [-0.5632, -0.0247, 0.2813]
    # stab_derivs['Cl']['alpha_p'] = [-0.0247]
    # stab_derivs['Cl']['alpha2_p'] = [0.2813]
    stab_derivs['Cl']['r'] = [0.1811, 0.6448]
    # stab_derivs['Cl']['alpha_r'] = [0.6448]
    stab_derivs['Cl']['deltaa'] = [-0.2489, -0.0087, 0.2383]
    # stab_derivs['Cl']['alpha_deltaa'] = [-0.0087]
    # stab_derivs['Cl']['alpha2_deltaa'] = [0.2383]
    stab_derivs['Cl']['deltar'] = [0.00436, -0.0013]
    # stab_derivs['Cl']['alpha_deltar'] = [-0.0013]

    stab_derivs['Cm'] = {}
    stab_derivs['Cm']['0'] = [-0.0307]
    stab_derivs['Cm']['alpha'] = [-0.6027]
    stab_derivs['Cm']['q'] = [-11.3022, -0.0026, 5.2885]
    # stab_derivs['Cm']['alpha_q'] = [-0.0026]
    # stab_derivs['Cm']['alpha2_q'] = [5.2885]
    stab_derivs['Cm']['deltae'] = [-1.0427, -0.0061, 0.9974]
    # stab_derivs['Cm']['alpha_deltae'] = [-0.0061]
    # stab_derivs['Cm']['alpha2_deltae'] = [0.9974]

    stab_derivs['Cn'] = {}
    stab_derivs['Cn']['beta'] = [0.0577, -0.0849]
    # stab_derivs['Cn']['alpha_beta'] = [-0.0849]
    stab_derivs['Cn']['p'] = [-0.0565, -0.9137]
    # stab_derivs['Cn']['alpha_p'] = [-0.9137]
    stab_derivs['Cn']['r'] = [-0.0553, 0.0290, 0.0257]
    # stab_derivs['Cn']['alpha_r'] = [0.0290]
    # stab_derivs['Cn']['alpha2_r'] = [0.0257]
    stab_derivs['Cn']['deltaa'] = [0.01903, -0.1147]
    # stab_derivs['Cn']['alpha_deltaa'] = [-0.1147]
    stab_derivs['Cn']['deltar'] = [-0.0404, -0.0117, 0.04089]
    # stab_derivs['Cn']['alpha_deltar'] = [-0.0117]
    # stab_derivs['Cn']['alpha2_deltar'] = [0.04089]

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
