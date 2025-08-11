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

def data_dict(aero_model):

    data_dict = {}
    data_dict['name'] = 'megawes' + '_' + aero_model

    data_dict['geometry'] = geometry() # kite geometry

    stab_derivs, aero_validity = aero(aero_model)
    data_dict['stab_derivs'] = stab_derivs # stability derivatives
    data_dict['aero_validity'] = aero_validity

    # (optional: on-board battery model)
    coeff_min = np.array([0, -80*np.pi/180.0])
    coeff_max = np.array([2, 80*np.pi/180.0])
    data_dict['battery'] = battery_model_parameters(coeff_max, coeff_min)

    return data_dict

def geometry():

     # Values from: Eijkelhof, D.; Schmehl, R. Six-degrees-of-freedom simulation model for future multi-megawatt airborne wind energy systems. Renew. Energy 	   2022, 196, 137–150

    geometry = {}

    geometry['b_ref'] = 42.47  # [m]
    geometry['s_ref'] = 150.45  # [m^2]
    geometry['c_ref'] = geometry['s_ref'] / geometry['b_ref']  # [m] #todo:check with AVL model

    geometry['m_k'] = 6885.2  # [kg]

    geometry['ar'] = geometry['b_ref'] / geometry['c_ref'] #12.0
    geometry['j'] = np.array([[5.768e5, 0.0, 0.0],
                              [0.0, 0.8107e5, 0.0],
                              [0.47, 0.0, 6.5002e5]])

    geometry['length'] = geometry['b_ref']  # only for plotting
    geometry['height'] = geometry['b_ref'] / 5.  # only for plotting
    geometry['delta_max'] = np.array([20., 20., 20.]) * np.pi / 180.
    geometry['ddelta_max'] = np.array([2., 2., 2.])

    geometry['c_root'] = 4.46
    geometry['c_tip'] = 2.11

    geometry['fuselage'] = True
    geometry['wing'] = True
    geometry['tail'] = True
    geometry['wing_profile'] = None

    # tether attachment point
    geometry['r_tether'] = np.reshape([0, 0, 0], (3,1)) 

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

def aero(aero_model="VLM"):

    # commented values are not currently supported, future implementation

    # Stability derivatives of MegAWES aircraft from AVL analysis performed by Niels Pynaert (Ghent University, 2024)
    if aero_model == 'VLM':

        stab_derivs = {}
        stab_derivs['frame'] = {}
        stab_derivs['frame']['force'] = 'control'
        stab_derivs['frame']['moment'] = 'control'
        
        stab_derivs['CX'] = {}
        stab_derivs['CX']['0'] = [-0.046]
        stab_derivs['CX']['alpha'] = [0.5329, 3.6178]
        # stab_derivs['CX']['beta'] = [-0.0, 0.0, 0.0]
        # stab_derivs['CX']['p'] = [-0.0, 0.0001, 0.0]
        stab_derivs['CX']['q'] = [-0.1689, 3.1142, -0.3229]
        # stab_derivs['CX']['r'] = [0.0, 0.0, 0.0001]
        # stab_derivs['CX']['deltaa'] = [-0.0, 0.0, 0.0]
        stab_derivs['CX']['deltae'] = [-0.0203, 0.2281, 0.0541]
        # stab_derivs['CX']['deltar'] = [-0.0, -0.0, -0.0]
        
        stab_derivs['CY'] = {}
        # stab_derivs['CY']['0'] = [-0.0]
        # stab_derivs['CY']['alpha'] = [0.0, 0.0]
        stab_derivs['CY']['beta'] = [-0.2056, -0.1529, -0.3609]
        stab_derivs['CY']['p'] = [0.0588, 0.3069, -0.0109]
        # stab_derivs['CY']['q'] = [-0.0, -0.0, 0.0]
        stab_derivs['CY']['r'] = [0.0869, 0.0271, -0.0541]
        stab_derivs['CY']['deltaa'] = [0.0064, -0.0365, -0.0022]
        # stab_derivs['CY']['deltae'] = [-0.0, 0.0, 0.0]
        stab_derivs['CY']['deltar'] = [0.1801, 0.0196, -0.1724]
        
        stab_derivs['CZ'] = {}
        stab_derivs['CZ']['0'] = [-0.8781]
        stab_derivs['CZ']['alpha'] = [-4.7042, 0.0335]
        # stab_derivs['CZ']['beta'] = [-0.0, 0.0, 0.0]
        # stab_derivs['CZ']['p'] = [-0.0001, -0.0, -0.0]
        stab_derivs['CZ']['q'] = [-5.9365, -0.7263, 2.4422]
        # stab_derivs['CZ']['r'] = [0.0, 0.0001, -0.0]
        # stab_derivs['CZ']['deltaa'] = [-0.0, 0.0, 0.0]
        stab_derivs['CZ']['deltae'] = [-0.4867, -0.007, 0.4642]
        # stab_derivs['CZ']['deltar'] = [-0.0, 0.0, -0.0]
        
        stab_derivs['Cl'] = {}
        # stab_derivs['Cl']['0'] = [-0.0]
        # stab_derivs['Cl']['alpha'] = [-0.0001, -0.0001]
        stab_derivs['Cl']['beta'] = [-0.0101, -0.1834, 0.0023]
        stab_derivs['Cl']['p'] = [-0.4888, -0.027, 0.092]
        # stab_derivs['Cl']['q'] = [-0.0, -0.0, 0.0]
        stab_derivs['Cl']['r'] = [0.1966, 0.5629, -0.0498]
        stab_derivs['Cl']['deltaa'] = [-0.1972, 0.0574, 0.1674]
        # stab_derivs['Cl']['deltae'] = [-0.0, 0.0, 0.0]
        stab_derivs['Cl']['deltar'] = [0.0077, -0.0091, -0.0092]
        
        stab_derivs['Cm'] = {}
        stab_derivs['Cm']['0'] = [-0.065]
        stab_derivs['Cm']['alpha'] = [-0.3306, 0.2245]
        # stab_derivs['Cm']['beta'] = [-0.0, 0.0, 0.0]
        # stab_derivs['Cm']['p'] = [0.0, 0.0, 0.0]
        stab_derivs['Cm']['q'] = [-7.7531, -0.003, 3.8925]
        # stab_derivs['Cm']['r'] = [0.0, -0.0, -0.0]
        # stab_derivs['Cm']['deltaa'] = [-0.0, 0.0, 0.0]
        stab_derivs['Cm']['deltae'] = [-1.1885, -0.0007, 1.1612]
        # stab_derivs['Cm']['deltar'] = [-0.0, 0.0, -0.0]
        
        stab_derivs['Cn'] = {}
        # stab_derivs['Cn']['0'] = [-0.0]
        # stab_derivs['Cn']['alpha'] = [0.0, 0.0]
        stab_derivs['Cn']['beta'] = [0.0385, 0.0001, -0.0441]
        stab_derivs['Cn']['p'] = [-0.0597, -0.7602, 0.0691]
        # stab_derivs['Cn']['q'] = [0.0, 0.0, 0.0]
        stab_derivs['Cn']['r'] = [-0.0372, -0.0291, -0.2164]
        stab_derivs['Cn']['deltaa'] = [0.0054, -0.0425, 0.0354]
        # stab_derivs['Cn']['deltae'] = [-0.0, 0.0, -0.0]
        stab_derivs['Cn']['deltar'] = [-0.0404, -0.0031, 0.0385]
        
    # Stability derivatives of MegAWES aircraft from ALM analysis performed by Jean-Baptiste Crismer (UCLouvain, 2024)
    elif aero_model == 'ALM':

        stab_derivs = {}
        stab_derivs['frame'] = {}
        stab_derivs['frame']['force'] = 'control'
        stab_derivs['frame']['moment'] = 'control'
        
        stab_derivs['CX'] = {}
        stab_derivs['CX']['0'] = [-0.044491]
        stab_derivs['CX']['alpha'] = [0.643491, 4.312588]
        # stab_derivs['CX']['beta'] = [0.021606, 0.062355, 0.253736]
        # stab_derivs['CX']['p'] = [0.059195, -0.296454, -3.093771]
        stab_derivs['CX']['q'] = [-0.435813, 3.576224, 9.319524]
        # stab_derivs['CX']['r'] = [0.010688, 0.123894, 0.372218]
        # stab_derivs['CX']['deltaa'] = [0.002068, 0.034475, 0.14272]
        stab_derivs['CX']['deltae'] = [-0.032819, 0.382206, 0.336074]
        # stab_derivs['CX']['deltar'] = [0.002219, 0.035366, 0.143993]
        
        stab_derivs['CY'] = {}
        # stab_derivs['CY']['0'] = [-7e-06]
        # stab_derivs['CY']['alpha'] = [-4.7e-05, -7.9e-05]
        stab_derivs['CY']['beta'] = [-0.176968, -0.005307, 0.07225]
        stab_derivs['CY']['p'] = [-0.004113, -0.063994, -0.191728]
        # stab_derivs['CY']['q'] = [0.049085, -0.160077, -0.709422]
        stab_derivs['CY']['r'] = [0.075628, -0.001207, -0.002044]
        stab_derivs['CY']['deltaa'] = [-0.000378, -0.000256, 0.002923]
        # stab_derivs['CY']['deltae'] = [-4e-06, -5.1e-05, -0.000146]
        stab_derivs['CY']['deltar'] = [0.175971, 0.0086, -0.147732]
        
        stab_derivs['CZ'] = {}
        stab_derivs['CZ']['0'] = [-0.975917]
        stab_derivs['CZ']['alpha'] = [-4.482273, 2.927043]
        # stab_derivs['CZ']['beta'] = [-0.05144, 0.345141, 2.388404]
        # stab_derivs['CZ']['p'] = [0.009475, 2.499454, 8.584544]
        stab_derivs['CZ']['q'] = [-2.862521, 4.477878, 49.332075]
        # stab_derivs['CZ']['r'] = [-0.118169, 0.924362, 5.792087]
        # stab_derivs['CZ']['deltaa'] = [-0.028644, 0.204391, 1.378045]
        stab_derivs['CZ']['deltae'] = [-0.47211, 0.243975, 1.718238]
        # stab_derivs['CZ']['deltar'] = [-0.028764, 0.204725, 1.383893]
        
        stab_derivs['Cl'] = {}
        # stab_derivs['Cl']['0'] = [-7e-06]
        # stab_derivs['Cl']['alpha'] = [-7.9e-05, -0.000168]
        stab_derivs['Cl']['beta'] = [-0.007379, -6.2e-05, 0.00364]
        stab_derivs['Cl']['p'] = [-0.587414, 1.808228, 8.431355]
        # stab_derivs['Cl']['q'] = [-0.03176, -0.079475, 0.033337]
        stab_derivs['Cl']['r'] = [0.244036, 0.613158, -0.716677]
        stab_derivs['Cl']['deltaa'] = [-0.370799, 0.001332, 0.177379]
        # stab_derivs['Cl']['deltae'] = [-7e-06, 0.000143, 0.000583]
        stab_derivs['Cl']['deltar'] = [0.007332, 0.000383, -0.006009]
        
        stab_derivs['Cm'] = {}
        stab_derivs['Cm']['0'] = [0.057198]
        stab_derivs['Cm']['alpha'] = [-0.139756, -0.376294]
        # stab_derivs['Cm']['beta'] = [0.023998, -0.252359, -1.551892]
        # stab_derivs['Cm']['p'] = [0.037984, -0.751082, -1.519246]
        stab_derivs['Cm']['q'] = [-6.009464, -2.451568, -26.813118]
        # stab_derivs['Cm']['r'] = [0.072848, -0.55932, -3.307224]
        # stab_derivs['Cm']['deltaa'] = [0.018651, -0.144584, -0.878641]
        stab_derivs['Cm']['deltae'] = [-1.201313, -0.016661, 0.043642]
        # stab_derivs['Cm']['deltar'] = [0.018564, -0.146394, -0.883871]
        
        stab_derivs['Cn'] = {}
        # stab_derivs['Cn']['0'] = [-0.0]
        # stab_derivs['Cn']['alpha'] = [-6e-06, 8e-06]
        stab_derivs['Cn']['beta'] = [0.040269, 0.00141, -0.012721]
        stab_derivs['Cn']['p'] = [-0.067101, -0.999124, -1.066629]
        # stab_derivs['Cn']['q'] = [-0.011544, 0.014942, 0.125027]
        stab_derivs['Cn']['r'] = [-0.032551, 0.077463, 0.330239]
        stab_derivs['Cn']['deltaa'] = [0.004239, -0.338661, -0.009433]
        # stab_derivs['Cn']['deltae'] = [-0.0, -4e-06, 2.5e-05]
        stab_derivs['Cn']['deltar'] = [-0.040178, -0.00205, 0.031991]
        
    # Stability derivatives of MegAWES aircraft from CFD analysis performed by Niels Pynaert (Ghent University, 2024)
    if aero_model == 'CFD':

        stab_derivs = {}
        stab_derivs['frame'] = {}
        stab_derivs['frame']['force'] = 'control'
        stab_derivs['frame']['moment'] = 'control'
        
        stab_derivs['CX'] = {}
        stab_derivs['CX']['0'] = [-0.1164]
        stab_derivs['CX']['alpha'] = [0.4564, 2.3044]
        # stab_derivs['CX']['beta'] = [0.0279, 0.0414, 0.8307]
        # stab_derivs['CX']['p'] = [0.0342, 0.1529, -1.8588]
        stab_derivs['CX']['q'] = [-0.4645, 8.5417, -10.8181]
        # stab_derivs['CX']['r'] = [-0.0006, 0.0519, 0.4025]
        # stab_derivs['CX']['deltaa'] = [-0.0168, 0.0733, 1.3335]
        stab_derivs['CX']['deltae'] = [0.0002, -0.0182, 0.41]
        # stab_derivs['CX']['deltar'] = [-0.0173, -0.015, -0.2922]
        
        stab_derivs['CY'] = {}
        # stab_derivs['CY']['0'] = [-0.0]
        # stab_derivs['CY']['alpha'] = [0.0002, 0.0013]
        stab_derivs['CY']['beta'] = [-0.274, 0.1664, 0.8803]
        stab_derivs['CY']['p'] = [0.0198, -0.2312, -0.315]
        # stab_derivs['CY']['q'] = [0.0007, -0.001, 0.0799]
        stab_derivs['CY']['r'] = [0.0911, -0.0267, -0.4982]
        stab_derivs['CY']['deltaa'] = [0.0063, 0.0119, -0.0754]
        # stab_derivs['CY']['deltae'] = [0.0001, -0.0012, -0.0216]
        stab_derivs['CY']['deltar'] = [0.2259, -0.1198, 0.1955]
        
        stab_derivs['CZ'] = {}
        stab_derivs['CZ']['0'] = [-0.9245]
        stab_derivs['CZ']['alpha'] = [-3.7205, 4.7972]
        # stab_derivs['CZ']['beta'] = [0.1123, -0.125, -5.0971]
        # stab_derivs['CZ']['p'] = [0.1387, 0.1685, -27.9934]
        stab_derivs['CZ']['q'] = [-5.6405, 60.997, 240.6406]
        # stab_derivs['CZ']['r'] = [0.0067, 0.1349, -4.4412]
        # stab_derivs['CZ']['deltaa'] = [0.0638, -1.8662, -26.6776]
        stab_derivs['CZ']['deltae'] = [-0.4897, 0.2366, 3.4195]
        # stab_derivs['CZ']['deltar'] = [0.0044, 0.0123, -0.2717]
        
        stab_derivs['Cl'] = {}
        # stab_derivs['Cl']['0'] = [0.0]
        # stab_derivs['Cl']['alpha'] = [0.0002, 0.0001]
        stab_derivs['Cl']['beta'] = [0.0344, -0.1786, -2.6711]
        stab_derivs['Cl']['p'] = [-0.4052, 0.4109, -0.5721]
        # stab_derivs['Cl']['q'] = [0.018, 0.0258, -2.1828]
        stab_derivs['Cl']['r'] = [0.1802, 0.5792, -0.0129]
        stab_derivs['Cl']['deltaa'] = [-0.0941, -0.1921, -0.2034]
        # stab_derivs['Cl']['deltae'] = [0.0, -0.0063, -0.0912]
        stab_derivs['Cl']['deltar'] = [0.0106, -0.0214, -0.0874]
        
        stab_derivs['Cm'] = {}
        stab_derivs['Cm']['0'] = [0.0279]
        stab_derivs['Cm']['alpha'] = [-0.5307, -0.9786]
        # stab_derivs['Cm']['beta'] = [-0.0184, 0.7392, 8.2241]
        # stab_derivs['Cm']['p'] = [0.0008, -0.1007, -0.0845]
        stab_derivs['Cm']['q'] = [-8.0446, 1.1837, -20.8571]
        # stab_derivs['Cm']['r'] = [-0.0021, -0.2081, -2.4176]
        # stab_derivs['Cm']['deltaa'] = [0.0177, 0.9504, 4.4178]
        stab_derivs['Cm']['deltae'] = [-1.2524, -0.092, 11.6916]
        # stab_derivs['Cm']['deltar'] = [0.0165, 0.0416, 0.0795]
        
        stab_derivs['Cn'] = {}
        # stab_derivs['Cn']['0'] = [-0.0]
        # stab_derivs['Cn']['alpha'] = [-0.0, 0.0004]
        stab_derivs['Cn']['beta'] = [0.0682, 0.0048, -0.1193]
        stab_derivs['Cn']['p'] = [-0.0412, -0.4284, -1.0241]
        # stab_derivs['Cn']['q'] = [-0.0007, 0.0072, 0.0489]
        stab_derivs['Cn']['r'] = [-0.0555, 0.0316, 0.1057]
        stab_derivs['Cn']['deltaa'] = [0.0234, -0.0113, -0.6566]
        # stab_derivs['Cn']['deltae'] = [-0.0, -0.0001, 0.0014]
        stab_derivs['Cn']['deltar'] = [-0.0509, 0.0287, -0.0572]

    # Aero validity (MegAWES)
    aero_validity = {}
    aero_validity['alpha_max_deg'] = +5.  #4.2
    aero_validity['alpha_min_deg'] = -15. #-14.5
    aero_validity['beta_max_deg'] = 10.
    aero_validity['beta_min_deg'] = -10.


    return stab_derivs, aero_validity
