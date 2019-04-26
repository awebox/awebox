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
"""
actuator_disk model of awebox aerodynamics
sets up the axial-induction actuator disk equation
currently for untilted rotor with no tcf.
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2017-18
- edit: jochem de schutter, alu-fr 2019
"""

import casadi.tools as cas
import numpy as np
import logging
import pdb

import awebox.tools.vector_operations as vect_op
from . import geom as geom

## variables

def get_a_var(model_options, variables, parent):

    var_type = geom.get_var_type(model_options)
    a_var = variables[var_type]['a' + str(parent)]
    return a_var

def get_da_var(model_options, variables, parent):

    if model_options['aero']['actuator']['steadyness'] == 'unsteady':
        da_var = variables['xddot']['da' + str(parent)]
    else:
        da_var = 0.0
    return da_var

def get_qapp_var(atmos, wind, variables, parent):
    qapp_ref = get_qapp_ref(atmos, wind)
    qapp_var = qapp_ref * variables['xl']['qapp' + str(parent)]
    return qapp_var

def get_f_var(model_options, variables, parent):
    var_type = geom.get_var_type(model_options)
    f_var = variables[var_type]['f' + str(parent)]
    return f_var

def get_df_var(variables, parent):
    df_var = variables['xddot']['df' + str(parent)]
    return df_var

def get_cosgamma_var(variables, parent):
    cosgamma_var = variables['xl']['cosgamma' + str(parent)]
    return cosgamma_var


## residuals

def get_qapp_residual(model_options, parent, atmos, wind, variables, architecture):

    qapp_var = get_qapp_var(atmos, wind, variables, parent)
    qapp_val = get_actuator_dynamic_pressure(model_options, atmos, wind, variables, parent, architecture)
    resi_unscaled = qapp_var - qapp_val

    qapp_ref = get_qapp_ref(atmos, wind)
    resi_scaled = resi_unscaled / qapp_ref

    return resi_scaled

def get_f_residual(model_options, wind, parent, variables, architecture):
    f_var = get_f_var(model_options, variables, parent)
    f_val = get_f_val(model_options, wind, parent, variables, architecture)

    resi = f_var - f_val
    return resi

def get_cosgamma_residual(model_options, wind, parent, variables, architecture):

    cosgamma_var = get_cosgamma_var(variables, parent)
    u_app = get_rotor_apparent_velocity(model_options, wind, parent, variables, architecture)
    nhat_var = geom.get_nhat_var(variables, parent)

    num = cas.mtimes(u_app.T, nhat_var)**2.
    den = cas.mtimes(u_app.T, u_app)
    # den^(1/2) = norm(u_app) * norm(nhat)

    u_ref = vect_op.norm(get_uapp_ref(wind))

    resi_unscaled = cosgamma_var**2. * den - num
    resi = resi_unscaled / u_ref**2.

    return resi


## values

def get_f_val(model_options, wind, parent, variables, architecture):
    dl_t = variables['xd']['dl_t']
    u_infty = get_actuator_freestream_velocity(model_options, wind, parent, variables, architecture)
    f_val = dl_t / vect_op.smooth_norm(u_infty)

    return f_val



## references

def get_uapp_ref(wind):
    uinfty_ref = wind.get_velocity_ref()
    return uinfty_ref

def get_qapp_ref(atmos, wind):

    rho_ref = atmos.get_density_ref()
    uinfty_ref = wind.get_velocity_ref()
    qapp_ref = .5 * rho_ref * uinfty_ref**2.
    return qapp_ref

def get_a_ref(model_options):
    a_ref = model_options['aero']['a_ref']
    return a_ref


def get_local_induction_factor(model_options, variables, parent):

    a_local = get_a_var(model_options, variables, parent)
    return a_local


def get_rotor_apparent_velocity(model_options, wind, parent, variables, architecture):

    u_infty = get_actuator_freestream_velocity(model_options, wind, parent, variables, architecture)
    u_actuator = geom.get_center_velocity(model_options, parent, variables, architecture)

    u_apparent = u_infty - u_actuator

    return u_apparent

def get_actuator_freestream_velocity(model_options, wind, parent, variables, architecture):

    center = geom.get_center_point(model_options, parent, variables, architecture)
    u_infty = wind.get_velocity(center[2])

    return u_infty

def get_local_induced_velocity(model_options, variables, wind, parent, architecture):

    u_app = get_rotor_apparent_velocity(model_options, wind, parent, variables, architecture)
    nhat = geom.get_nhat_var(variables, parent)

    a_val = get_local_induction_factor(model_options, variables, parent)
    u_ind = -1. * a_val * vect_op.smooth_norm(u_app) * nhat

    return u_ind

def get_kite_effective_velocity(model_options, variables, wind, kite, parent, architecture):

    q_kite = variables['xd']['q' + str(kite) + str(parent)]
    u_infty = wind.get_velocity(q_kite[2])

    u_kite = variables['xd']['dq' + str(kite) + str(parent)]

    u_induced = get_local_induced_velocity(model_options, variables, wind, parent, architecture)

    u_app_kite = u_infty - u_kite + u_induced

    return u_app_kite

def get_actuator_dynamic_pressure(model_options, atmos, wind, variables, parent, architecture):

    center = geom.get_center_point(model_options, parent, variables, architecture)
    rho_infty = atmos.get_density(center[2])

    u_app = get_rotor_apparent_velocity(model_options, wind, parent, variables, architecture)

    q_infty = 0.5 * rho_infty * cas.mtimes(u_app.T, u_app)

    return q_infty

def get_actuator_yaw_angle(model_options, wind, parent, variables,architecture):

    u_app = get_rotor_apparent_velocity(model_options, wind, parent, variables, architecture)

    nhat = geom.get_nhat_var(variables, parent)

    gamma = vect_op.angle_between(u_app, nhat)

    return gamma

def get_glauert_a(model_options, wind, variables, parent, architecture):
    # this is NOT angle of attack.
    # this is the combi-function of the induction factor, as given by
    # see pg. 174 of Burton, et al. 2011

    a_var = get_a_var(model_options, variables, parent)

    cosgamma_var = get_cosgamma_var(variables, parent)

    a_fun_a = (1. - a_var * (2. * cosgamma_var - a_var))**(0.5)

    return a_fun_a



def get_nonlin_induction_correction(model_options, wind, variables, parent, architecture):
    a_var = get_a_var(model_options, variables, parent)

    if model_options['aero']['actuator']['correct_tilt']:
        nonlin_correction = get_glauert_a(model_options, wind, variables, parent, architecture)
    else:
        nonlin_correction = (1. - a_var)

    return nonlin_correction

def get_wake_angle_chi(model_options, wind, parent, variables, architecture):
    gamma = get_actuator_yaw_angle(model_options, wind, parent, variables, architecture)
    a = get_a_var(model_options, variables, parent)

    # use coleman approximation
    chi = (0.6 * a + 1.) * gamma

    return chi
