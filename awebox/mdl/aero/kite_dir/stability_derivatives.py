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
'''
stability_derivatives aerodynamics modelling file
calculates stability derivatives based on orientation, angular velocity, and control surface deflection
_python-3.5 / casadi-3.4.5
- author: elena malz, chalmers 2016
- edited: rachel leuthold, alu-fr 2017-18
'''

import casadi.tools as cas

import awebox.mdl.aero.indicators as indicators

def stability_derivatives(options, alpha, beta, ua, omega, delta, parameters):
    # delta:
    # aileron left-right [right teu+, rad], ... positive delta a -> negative roll
    # elevator [ted+, rad],                 ... positive delta e -> negative pitch
    # rudder [tel+, rad])                   ... positive delta r -> positive yaw
    deltaa = delta[0]
    deltae = delta[1]
    deltar = delta[2]

    # CA -> axial force, along ehat1
    # CY -> side force, along ehat2
    # CN -> normal force, along ehat3

    # Cl -> roll moment, about -ehat1
    # Cm -> pitch moment, about -ehat2
    # Cn -> yaw moment, about ehat3

    # p -> roll rate, about -ehat1
    # q -> pitch rate, about -ehat2
    # r -> yaw rate, about ehat3

    ua_norm = cas.mtimes(ua.T, ua) ** 0.5

    # pqr - damping: in radians
    omega_hat = omega / (2. * ua_norm)

    b_ref = parameters['theta0','geometry','b_ref']
    c_ref = parameters['theta0','geometry','c_ref']

    omega_hat[0] *= b_ref  # pb/2|ua|
    omega_hat[1] *= c_ref  # qc/2|ua|
    omega_hat[2] *= b_ref  # rb/2|ua|

    # roll, pitch, yaw
    p = omega_hat[0]
    q = omega_hat[1]
    r = omega_hat[2]

    stab_deriv = consolidate_stability_derivatives(options, alpha, beta, parameters)

    CL0 = stab_deriv['CL0']
    CS0 = stab_deriv['CS0']
    CD0 = stab_deriv['CD0']

    Cl0 = stab_deriv['Cl0']
    Cm0 = stab_deriv['Cm0']
    Cn0 = stab_deriv['Cn0']

    # contribution from motion
    CD_wind = stab_deriv['CDalpha'] * alpha + stab_deriv['CDalpha2'] * alpha ** 2. + stab_deriv['CDbeta'] * beta + \
              stab_deriv['CDbeta2'] * beta ** 2.
    CL_wind = stab_deriv['CLalpha'] * alpha + stab_deriv['CLalpha2'] * alpha ** 2. + stab_deriv['CLbeta'] * beta + \
              stab_deriv['CLbeta2'] * beta ** 2.
    CS_wind = stab_deriv['CSalpha'] * alpha + stab_deriv['CSalpha2'] * alpha ** 2. + stab_deriv['CSbeta'] * beta + \
              stab_deriv['CSbeta2'] * beta ** 2.

    Cl_wind = stab_deriv['Clalpha'] * alpha + stab_deriv['Clbeta'] * beta
    Cm_wind = stab_deriv['Cmalpha'] * alpha + stab_deriv['Cmbeta'] * beta
    Cn_wind = stab_deriv['Cnalpha'] * alpha + stab_deriv['Cnbeta'] * beta

    CD_motion = stab_deriv['CDp'] * p + stab_deriv['CDq'] * q + stab_deriv['CDr'] * r
    CL_motion = stab_deriv['CLp'] * p + stab_deriv['CLq'] * q + stab_deriv['CLr'] * r
    CS_motion = stab_deriv['CSp'] * p + stab_deriv['CSq'] * q + stab_deriv['CSr'] * r

    Cl_motion = stab_deriv['Clp'] * p + stab_deriv['Clq'] * q + stab_deriv['Clr'] * r
    Cm_motion = stab_deriv['Cmp'] * p + stab_deriv['Cmq'] * q + stab_deriv['Cmr'] * r
    Cn_motion = stab_deriv['Cnp'] * p + stab_deriv['Cnq'] * q + stab_deriv['Cnr'] * r

    # contribution from control surfaces
    CD_surfs = stab_deriv['CDdeltaa'] * deltaa + stab_deriv['CDdeltae'] * deltae + stab_deriv['CDdeltar'] * deltar
    CL_surfs = stab_deriv['CLdeltaa'] * deltaa + stab_deriv['CLdeltae'] * deltae + stab_deriv['CLdeltar'] * deltar
    CS_surfs = stab_deriv['CSdeltaa'] * deltaa + stab_deriv['CSdeltae'] * deltae + stab_deriv['CSdeltar'] * deltar

    CD_surfs2 = stab_deriv['CDdeltaa2'] * deltaa ** 2. + stab_deriv['CDdeltae2'] * deltae ** 2. + stab_deriv[
                                                                                                      'CDdeltar2'] * deltar ** 2.
    CL_surfs2 = stab_deriv['CLdeltaa2'] * deltaa ** 2. + stab_deriv['CLdeltae2'] * deltae ** 2. + stab_deriv[
                                                                                                      'CLdeltar2'] * deltar ** 2.
    CS_surfs2 = stab_deriv['CSdeltaa2'] * deltaa ** 2. + stab_deriv['CSdeltae2'] * deltae ** 2. + stab_deriv[
                                                                                                      'CSdeltar2'] * deltar ** 2.

    Cl_surfs = parameters['theta0','aero','Cldeltaa'] * deltaa + parameters['theta0','aero','Cldeltar'] * deltar
    Cm_surfs = parameters['theta0','aero','Cmdeltae'] * deltae
    Cn_surfs = parameters['theta0','aero','Cndeltaa'] * deltaa + parameters['theta0','aero','Cndeltar'] * deltar

    CD_split = stab_deriv['CDalpha_deltae'] * alpha * deltae + stab_deriv['CDbeta_deltaa'] * beta * deltaa + stab_deriv[
        'CDbeta_deltar']
    CS_split = stab_deriv['CSalpha_deltae'] * alpha * deltae + stab_deriv['CSbeta_deltaa'] * beta * deltaa + stab_deriv[
        'CSbeta_deltar']
    CL_split = stab_deriv['CLalpha_deltae'] * alpha * deltae + stab_deriv['CLbeta_deltaa'] * beta * deltaa + stab_deriv[
        'CLbeta_deltar']

    # sum
    CD = CD0 + CD_wind + CD_surfs + CD_surfs2 + CD_split + CD_motion
    CL = CL0 + CL_wind + CL_surfs + CL_surfs2 + CL_split + CL_motion
    CS = CS0 + CS_wind + CS_surfs + CS_surfs2 + CS_split + CS_motion

    Cl = Cl0 + Cl_wind + Cl_motion + Cl_surfs
    Cm = Cm0 + Cm_wind + Cm_motion + Cm_surfs
    Cn = Cn0 + Cn_wind + Cn_motion + Cn_surfs

    # correct for alternate body reference frame
    drag_cross_lift = cas.vertcat(CD, CS, CL)
    axial_side_normal = indicators.convert_from_wind_to_body_axes(alpha, beta, drag_cross_lift)
    CA = axial_side_normal[0]
    CY = axial_side_normal[1]
    CN = axial_side_normal[2]

    # concatenate
    CF = cas.vertcat(CA, CY, CN)  # in body frame
    CM = cas.vertcat(Cl, Cm, Cn)  # in body frame

    return CF, CM

def consolidate_stability_derivatives(model_options, alpha, beta, parameters):

    stab_deriv = parameters.prefix['theta0','aero']
    keys = list(model_options['params']['aero'].keys())

    derivative = ['alpha', 'alpha2', 'beta', 'beta2', 'deltaa', 'deltae', 'deltar', 'deltaa2', 'deltae2', 'deltar2',
                  'alpha_deltae', 'beta_deltaa', 'beta_deltar']
    for indep in derivative:

        side_deriv_exists = ('CS' + indep) in keys
        lift_deriv_exists = ('CL' + indep) in keys
        drag_deriv_exists = ('CD' + indep) in keys

        span_deriv_exists = ('CY' + indep) in keys
        axial_deriv_exists = ('CA' + indep) in keys
        normal_deriv_exists = ('CN' + indep) in keys

        body_deriv_exists = side_deriv_exists and lift_deriv_exists and drag_deriv_exists
        wind_deriv_exists = span_deriv_exists and axial_deriv_exists and normal_deriv_exists

        if wind_deriv_exists and not body_deriv_exists:
            axial_side_normal = cas.vertcat(stab_deriv['CA' + indep], stab_deriv['CY' + indep], stab_deriv['CN' + indep])
            drag_cross_lift = indicators.convert_from_body_to_wind_axes(alpha, beta, axial_side_normal)

            stab_deriv['CD' + indep] = drag_cross_lift[0]
            stab_deriv['CS' + indep] = drag_cross_lift[1]
            stab_deriv['CL' + indep] = drag_cross_lift[2]

    return stab_deriv
