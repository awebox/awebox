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


def data_dict():

    data_dict = {}
    data_dict['name'] = 'bubbledancer'
    data_dict['geometry'] = geometry()
    data_dict['aero_deriv'] = aero_deriv()

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




def aero_deriv():

    # values from AVL run at zero-deg bank angle trim.

    aero_deriv = {}
    
    aero_deriv['CL0'] = 0.700
    aero_deriv['CS0'] = 0.
    aero_deriv['CD0'] = 0.02862

    aero_deriv['CLalpha'] = 5.675616
    aero_deriv['CSalpha'] = -0.000003
    aero_deriv['CDalpha'] = 0.1 #arbitrary, based on ampyx ap2

    aero_deriv['CLalpha2'] = 0
    aero_deriv['CSalpha2'] = 0.
    aero_deriv['CDalpha2'] = 1.3 #arbitrary, based on ampyx ap2
    
    aero_deriv['CLbeta'] = 0.
    aero_deriv['CSbeta'] = -0.404699
    aero_deriv['CDbeta'] = 0.

    aero_deriv['CLbeta2'] = 0.
    aero_deriv['CSbeta2'] = 0.
    aero_deriv['CDbeta2'] = 0.
    
    aero_deriv['CLdeltae'] = 0.008059
    aero_deriv['CLdeltaa'] = 0.
    aero_deriv['CLdeltar'] = 0.

    aero_deriv['CSdeltae'] = 0.000000
    aero_deriv['CSdeltaa'] = 0.
    aero_deriv['CSdeltar'] = -0.003376

    aero_deriv['CDdeltae'] = 0.000284
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

    aero_deriv['CLp'] = 0.000008
    aero_deriv['CSp'] = -0.380742
    aero_deriv['CDp'] = 0.

    aero_deriv['CLq'] = 7.286214
    aero_deriv['CSq'] = -0.000001
    aero_deriv['CDq'] = 0.

    aero_deriv['CLr'] = -0.000001
    aero_deriv['CSr'] = 0.294666
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
    aero_deriv['Cldeltaa'] = 0.3 # arbitrary, assumed similar to ampyx ap2
    aero_deriv['Cldeltar'] = -0.000076

    aero_deriv['Cmdeltae'] = -0.027418
    aero_deriv['Cmdeltaa'] = 0.
    aero_deriv['Cmdeltar'] = 0.

    aero_deriv['Cndeltae'] = 0.
    aero_deriv['Cndeltaa'] = 0.
    aero_deriv['Cndeltar'] = 0.001245

    aero_deriv['Clalpha'] = -0.000003
    aero_deriv['Cmalpha'] = -0.895625
    aero_deriv['Cnalpha'] = 0.000001
    
    aero_deriv['Clbeta'] = -0.257096
    aero_deriv['Cmbeta'] = -0.000000
    aero_deriv['Cnbeta'] = 0.057021

    aero_deriv['Clp'] =  -0.634188
    aero_deriv['Cmp'] = 0.000000
    aero_deriv['Cnp'] = -0.068262

    aero_deriv['Clq'] = -0.000002
    aero_deriv['Cmq'] = -12.180685
    aero_deriv['Cnq'] = 0.
    
    aero_deriv['Clr'] = 0.181038
    aero_deriv['Cmr'] = 0.
    aero_deriv['Cnr'] = -0.066292

    aero_deriv['alpha_max_deg'] = 20.
    aero_deriv['alpha_min_deg'] = -20.
    aero_deriv['beta_max_deg'] = 15.
    aero_deriv['beta_min_deg'] = -15.0

    return aero_deriv
