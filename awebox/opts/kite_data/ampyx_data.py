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
    data_dict['name'] = 'ampyx'

    data_dict['geometry'] = geometry() # kite geometry

    stab_derivs, aero_validity = aero()
    data_dict['aero_deriv'] = stab_derivs # stability derivatives
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
    # info from Licitra2019:
    # Licitra, G., Koenemann, J., Bürger, A., Williams, P., Ruiterkamp, R., & Diehl, M. (2019).
    # Performance assessment of a rigid wing Airborne Wind Energy pumping system.Energy, 173, 569–585.
    # https://doi.org/10.1016/j.energy.2019.02.064

    stab_derivs = {}

    stab_derivs['frame'] = {}
    stab_derivs['frame']['force'] = 'control'
    stab_derivs['frame']['moment'] = 'control'

    stab_derivs['CX'] = {}
    stab_derivs['CX']['alpha'] = [0., 8.320]
    stab_derivs['CX']['q'] = [-0.603, 4.412]
    stab_derivs['CX']['deltae'] = [-0.011, 0.112]
    stab_derivs['CX']['0'] = [0.456]

    stab_derivs['CY'] = {}
    stab_derivs['CY']['beta'] = [-0.186]
    stab_derivs['CY']['p'] = [-0.102]
    stab_derivs['CY']['r'] = [0.169, 0.137]
    stab_derivs['CY']['deltaa'] = [-0.05]
    stab_derivs['CY']['deltar'] = [0.103]

    stab_derivs['CZ'] = {}
    stab_derivs['CZ']['alpha'] = [0., 1.226, 10.203]
    stab_derivs['CZ']['q'] = [-7.556, 0.125, 6.149]
    stab_derivs['CZ']['deltae'] = [-0.315, -0.001, 0.292]
    stab_derivs['CZ']['0'] = [-5.4]

    stab_derivs['Cl'] = {}
    stab_derivs['Cl']['beta'] = [-0.062]
    stab_derivs['Cl']['p'] = [-0.559]
    stab_derivs['Cl']['r'] = [0.181, 0.645]
    stab_derivs['Cl']['deltaa'] = [-0.248, 0.041]
    stab_derivs['Cl']['deltar'] = [0.004]

    stab_derivs['Cm'] = {}
    stab_derivs['Cm']['alpha'] = [0., 0.205, 0.]
    stab_derivs['Cm']['q'] = [-11.302, -0.003, 5.289]
    stab_derivs['Cm']['deltae'] = [-1.019]
    stab_derivs['Cm']['0'] = [-0.315]

    stab_derivs['Cn'] = {}
    stab_derivs['Cn']['beta'] = [0.058, -0.085]
    stab_derivs['Cn']['p'] = [-0.057, -0.913]
    stab_derivs['Cn']['r'] = [-0.052]
    stab_derivs['Cn']['deltaa'] = [0.019, -0.115]
    stab_derivs['Cn']['deltar'] = [-0.041]

    aero_validity = {}
    aero_validity['alpha_max_deg'] = 21.7724
    aero_validity['alpha_min_deg'] = -7.4485
    aero_validity['beta_max_deg'] = 15.
    aero_validity['beta_min_deg'] = -15.0

    return stab_derivs, aero_validity
