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
"""
actuator_disk model of awebox aerodynamics
sets up the axial-induction actuator disk equation
currently for untilted rotor with no tcf.
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2017-20
- edit: jochem de schutter, alu-fr 2019
"""

import casadi.tools as cas

import awebox.tools.vector_operations as vect_op
import awebox.mdl.aero.induction_dir.general_dir.geom as general_geom
import awebox.mdl.aero.induction_dir.actuator_dir.geom as actuator_geom
import awebox.mdl.aero.induction_dir.general_dir.flow as general_flow
import numpy as np
from awebox.logger.logger import Logger as awelogger
import awebox.tools.print_operations as print_op
import awebox.mdl.aero.induction_dir.actuator_dir.force as actuator_force


## variables

def get_local_a_var(variables, kite, parent):
    local_a = variables['xd']['local_a' + str(kite) + str(parent)]
    return local_a

def get_a_var(model_options, variables, parent, label):
    a_ref = get_a_ref(model_options)
    var_type = actuator_geom.get_var_type(model_options)
    a_var = a_ref * variables[var_type]['a_' + label + str(parent)]
    return a_var

def get_acos_var(model_options, variables, parent, label):
    acos_var = variables['xd']['acos_' + label + str(parent)]
    return acos_var

def get_asin_var(model_options, variables, parent, label):
    asin_var = variables['xd']['asin_' + label + str(parent)]
    return asin_var

def get_a_all_var(model_options, variables, parent, label):

    if 'asym' in label:
        a_var = get_a_var(model_options, variables, parent, label)
        acos_var = get_acos_var(model_options, variables, parent, label)
        asin_var = get_asin_var(model_options, variables, parent, label)
        a_all = cas.vertcat(a_var, acos_var, asin_var)
    else:
        a_all = get_a_var(model_options, variables, parent, label)
    return a_all

def get_da_var(model_options, variables, parent, label):
    a_ref = get_a_ref(model_options)
    da_var = a_ref * variables['xd']['da_' + label + str(parent)]
    return da_var

def get_dacos_var(model_options, variables, parent, label):
    dacos_var = variables['xd']['dacos_' + label + str(parent)]
    return dacos_var

def get_dasin_var(model_options, variables, parent, label):
    dasin_var = variables['xd']['dasin_' + label + str(parent)]
    return dasin_var

def get_da_all_var(model_options, variables, parent, label):
    if 'asym' in label:
        da_var = get_da_var(model_options, variables, parent, label)
        dacos_var = get_dacos_var(model_options, variables, parent, label)
        dasin_var = get_dasin_var(model_options, variables, parent, label)
        da_all = cas.vertcat(da_var, dacos_var, dasin_var)
    else:
        da_all = get_da_var(model_options, variables, parent, label)
    return da_all


def get_qzero_var(atmos, wind, variables, parent):
    qzero_ref = get_qzero_ref(atmos, wind)
    qzero_val = qzero_ref * variables['xl']['qzero' + str(parent)]
    return qzero_val

def get_uzero_matr_var(variables, parent):
    rot_cols = variables['xl']['uzero_matr' + str(parent)]
    rot_matr = cas.reshape(rot_cols, (3, 3))

    return rot_matr

def get_uzero_hat_var(variables, parent):
    rot_matr = get_uzero_matr_var(variables, parent)
    u_hat = rot_matr[:, 0]
    return u_hat

def get_vzero_hat_var(variables, parent):
    rot_matr = get_uzero_matr_var(variables, parent)
    v_hat = rot_matr[:, 1]
    return v_hat

def get_wzero_hat_var(variables, parent):
    rot_matr = get_uzero_matr_var(variables, parent)
    w_hat = rot_matr[:, 2]
    return w_hat

def get_gamma_var(variables, parent):
    gamma_var = variables['xl']['gamma' + str(parent)]
    return gamma_var

def get_cosgamma_var(variables, parent):
    cosgamma_var = variables['xl']['cosgamma' + str(parent)]
    return cosgamma_var

def get_singamma_var(variables, parent):
    singamma_var = variables['xl']['singamma' + str(parent)]
    return singamma_var

def get_uzero_vec_length_var(wind, variables, parent):
    scale = get_uzero_vec_length_ref(wind)
    len_var = scale * variables['xl']['u_vec_length' + str(parent)]
    return len_var

def get_g_vec_length_var(variables, parent):
    len_var = variables['xl']['g_vec_length' + str(parent)]
    return len_var


## residuals

def get_qzero_residual(model_options, parent, atmos, wind, variables, architecture):

    qzero_var = get_qzero_var(atmos, wind, variables, parent)
    qzero_val = get_actuator_dynamic_pressure(model_options, atmos, wind, variables, parent, architecture)
    resi_unscaled = qzero_var - qzero_val

    qzero_ref = get_qzero_ref(atmos, wind)
    resi_scaled = resi_unscaled / qzero_ref

    return resi_scaled


def get_gamma_residual(model_options, wind, parent, variables, parameters, architecture):

    uzero_hat_var = get_uzero_hat_var(variables, parent)
    vzero_hat_var = get_vzero_hat_var(variables, parent)

    n_hat_var = general_geom.get_n_hat_var(variables, parent)
    u_comp = cas.mtimes(n_hat_var.T, uzero_hat_var)
    v_comp = cas.mtimes(n_hat_var.T, vzero_hat_var)

    gamma_var = get_gamma_var(variables, parent)
    cosgamma_var = get_cosgamma_var(variables, parent)
    singamma_var = get_singamma_var(variables, parent)

    g_vec_length_var = get_g_vec_length_var(variables, parent)

    f_cosproj = g_vec_length_var * cosgamma_var - u_comp
    f_sinproj = g_vec_length_var * singamma_var - v_comp

    f_cos = np.cos(gamma_var) - cosgamma_var
    f_sin = np.sin(gamma_var) - singamma_var

    resi = cas.vertcat(f_cos, f_sin, f_cosproj, f_sinproj)

    return resi









def get_uzero_matr_ortho_residual(model_options, parent, variables, parameters, architecture):

    # rotation matrix is in SO3 = 6 constraints
    rot_matr_var = get_uzero_matr_var(variables, parent)
    ortho_matr = cas.mtimes(rot_matr_var.T, rot_matr_var) - np.eye(3)
    f_ortho = vect_op.upper_triangular_inclusive(ortho_matr)

    return f_ortho

def get_uzero_matr_u_along_uzero_residual(model_options, wind, parent, variables, parameters, architecture):

    u_vec_val = get_uzero_vec(model_options, wind, parent, variables, parameters, architecture)
    u_hat_var = get_uzero_hat_var(variables, parent)

    u_vec_length_var = get_uzero_vec_length_var(wind, variables, parent)

    u_diff = u_vec_val - u_hat_var * u_vec_length_var

    u_vec_length_ref = get_uzero_vec_length_ref(wind)
    f_u_vec = u_diff / u_vec_length_ref

    return f_u_vec

def get_wzero_hat_is_z_rotor_hat_residual(variables, parent):
    w_hat_var = get_wzero_hat_var(variables, parent)
    z_rot_length = general_geom.get_z_vec_length_var(variables, parent)
    z_rotor_hat = general_geom.get_z_rotor_hat_var(variables, parent)
    f_full = w_hat_var - z_rotor_hat * z_rot_length

    return f_full

def get_wzero_parallel_z_rotor_check(variables, parent):
    w_hat_var = get_wzero_hat_var(variables, parent)
    z_rotor_hat = general_geom.get_z_rotor_hat_var(variables, parent)
    check = cas.mtimes(w_hat_var.T, z_rotor_hat) - 1.
    return check

def get_uzero_matr_residual(model_options, wind, parent, variables, parameters, architecture):

    # total number of variables = 10 (9 from rot_matr, 1 lengths)
    f_ortho = get_uzero_matr_ortho_residual(model_options, parent, variables, parameters, architecture)
    f_n_vec = get_uzero_matr_u_along_uzero_residual(model_options, wind, parent, variables, parameters, architecture)
    f_w = get_wzero_hat_is_z_rotor_hat_residual(variables, parent)

    # join the constraints
    f_combi = cas.vertcat(f_ortho, f_n_vec, f_w)

    return f_combi



def get_local_a_residual(model_options, variables, kite, parent):
    a_var = get_local_a_var(variables, kite, parent)
    label = get_label(model_options)
    a_val = get_local_induction_factor(model_options, variables, kite, parent, label)
    resi = a_var - a_val
    return resi

## values

def get_f_val(model_options, wind, parent, variables, architecture):
    dl_t = variables['xd']['dl_t']
    u_infty = get_actuator_freestream_velocity(model_options, wind, parent, variables, architecture)
    f_val = dl_t / vect_op.smooth_norm(u_infty)

    return f_val

def get_df_val(model_options, wind, parent, variables, architecture):

    if 'ddl_t' in variables['xd'].keys():
        ddl_t = variables['xd']['ddl_t']
    else:
        ddl_t = variables['u']['ddl_t']

    u_infty = get_actuator_freestream_velocity(model_options, wind, parent, variables, architecture)
    df_val = ddl_t / vect_op.smooth_norm(u_infty)

    return df_val

def get_gamma_val(model_options, wind, parent, variables, parameters, architecture):

    uzero = get_uzero_vec(model_options, wind, parent, variables, parameters, architecture)
    n_vec = general_geom.get_n_vec_val(model_options, parent, variables, parameters, architecture)
    gamma = vect_op.angle_between(n_vec, uzero)
    return gamma

def get_gamma_check(model_options, wind, parent, variables, parameters, architecture):
    gamma_val = vect_op.abs(get_gamma_val(model_options, wind, parent, variables, parameters, architecture))
    gamma_var = vect_op.abs(get_gamma_var(variables, parent))
    check = gamma_val - gamma_var
    norm = cas.mtimes(check.T, check)
    return norm

## references

def get_uinfty_ref(wind):
    uinfty_ref = wind.get_velocity_ref()
    return uinfty_ref

def get_qzero_ref(atmos, wind):
    scale = 5.
    rho_ref = atmos.get_density_ref()
    uinfty_ref = wind.get_velocity_ref()
    qzero_ref = .5 * rho_ref * uinfty_ref**2. * scale
    return qzero_ref

def get_a_ref(model_options):
    a_ref = model_options['aero']['a_ref']
    return a_ref

def get_uzero_vec_length_ref(wind):
    return wind.get_velocity_ref()


def get_local_induction_factor(model_options, variables, kite, parent, label):

    cospsi = actuator_geom.get_cospsi_var(variables, kite, parent)
    sinpsi = actuator_geom.get_sinpsi_var(variables, kite, parent)
    mu = actuator_geom.get_mu_radial_ratio(model_options, variables, kite, parent)
    # mu = 1.
    # see Suzuki 2000 for motivation for evaluating at the edges of the "annulus"

    if 'asym' in label:
        a_uni = get_a_var(model_options, variables, parent, label)
        acos = get_acos_var(model_options, variables, parent, label)
        asin = get_asin_var(model_options, variables, parent, label)
        a_local = a_uni + acos * cospsi * mu + asin * sinpsi * mu
    elif 'axi' in label:
        a_local = get_a_var(model_options, variables, parent, label)
    else:
        awelogger.logger.error('induction code not yet implemented.')

    return a_local


def get_uzero_vec(model_options, wind, parent, variables, parameters, architecture):

    u_infty = get_actuator_freestream_velocity(model_options, wind, parent, variables, architecture)
    u_actuator = actuator_geom.get_center_velocity(model_options, parent, variables, parameters, architecture)

    u_apparent = u_infty - u_actuator

    return u_apparent

def get_actuator_freestream_velocity(model_options, wind, parent, variables, architecture):

    center = actuator_geom.get_center_point(model_options, parent, variables, architecture)
    u_infty = wind.get_velocity(center[2])

    return u_infty

def get_local_induced_velocity(model_options, variables, parameters, architecture, wind, kite, parent, label):

    uzero_vec_length = get_uzero_vec_length_var(wind, variables, parent)
    nhat = general_geom.get_n_hat_var(variables, parent)

    a_val = get_local_induction_factor(model_options, variables, kite, parent, label)
    u_ind = -1. * a_val * uzero_vec_length * nhat

    return u_ind

def get_kite_induced_velocity(model_options, variables, parameters, architecture, wind, kite, parent):
    label = get_label(model_options)
    u_ind_kite = get_local_induced_velocity(model_options, variables, parameters, architecture, wind, kite, parent, label)
    return u_ind_kite

def get_kite_effective_velocity(model_options, variables, parameters, architecture, wind, kite, parent):

    u_app_kite = general_flow.get_kite_apparent_velocity(variables, wind, kite, parent)
    u_ind_kite = get_kite_induced_velocity(model_options, variables, parameters, architecture, wind, kite, parent)

    u_eff_kite = u_app_kite + u_ind_kite

    return u_eff_kite

def get_actuator_dynamic_pressure(model_options, atmos, wind, variables, parent, architecture):

    center = actuator_geom.get_center_point(model_options, parent, variables, architecture)
    rho_infty = atmos.get_density(center[2])

    uzero_mag = get_uzero_vec_length_var(wind, variables, parent)

    qzero = 0.5 * rho_infty * uzero_mag**2.

    return qzero


def get_wake_angle_chi_equal(model_options, parent, variables, label):
    gamma = get_gamma_var(variables, parent)
    return gamma

def get_wake_angle_chi_coleman(model_options, parent, variables, label):
    gamma = get_gamma_var(variables, parent)
    a = get_a_var(model_options, variables, parent, label)

    chi = (0.6 * a + 1.) * gamma

    return chi

def get_wake_angle_chi_jimenez(model_options, atmos, wind, variables, outputs, parameters, parent, architecture):

    gamma = get_gamma_var(variables, parent)

    cosgamma = get_cosgamma_var(variables, parent)
    singamma = get_singamma_var(variables, parent)

    var_type = actuator_geom.get_var_type(model_options)

    thrust = actuator_force.get_actuator_thrust(model_options, variables, parameters, outputs, parent, architecture)
    area = actuator_geom.get_actuator_area(model_options, parent, variables, parameters)
    qzero = get_actuator_dynamic_pressure(model_options, atmos, wind, variables, parent, architecture)

    ct_val = thrust / area / qzero

    chi = gamma + 0.5 * ct_val * cosgamma**2. * singamma

    return chi

def get_wake_angle_chi(model_options, atmos, wind, variables, outputs, parameters, parent, architecture, label):

    wake_skew = model_options['aero']['actuator']['wake_skew']

    if wake_skew == 'equal':
        chi_val = get_wake_angle_chi_equal(model_options, parent, variables, label)
    elif wake_skew == 'coleman':
        chi_val = get_wake_angle_chi_coleman(model_options, parent, variables, label)
    elif wake_skew == 'jimenez':
        chi_val = get_wake_angle_chi_jimenez(model_options, atmos, wind, variables, outputs, parameters, parent, architecture)
    elif wake_skew == 'not_in_use':
        chi_val = 0.
    else:
        chi_val = 0.
        message = 'unknown wake skew angle (chi) model selected'
        raise Exception(message)
    return chi_val


def get_actuator_comparison_labels(model_options):
    comparison_labels = model_options['aero']['induction']['comparison_labels']

    actuator_comp_labels = []
    for label in comparison_labels:
        if label[:3] == 'act':
            actuator_comp_labels += [label[4:]]

    return actuator_comp_labels

def get_label(model_options):
    steadyness = model_options['aero']['actuator']['steadyness']
    symmetry = model_options['aero']['actuator']['symmetry']

    if steadyness == 'quasi-steady' or steadyness == 'steady':
        if symmetry == 'axisymmetric':
            label = 'qaxi'

        elif symmetry == 'asymmetric':
            label = 'qasym'

        else:
            awelogger.logger.error('steady model not yet implemented.')

    elif steadyness == 'unsteady':
        if symmetry == 'axisymmetric':
            label = 'uaxi'

        elif symmetry == 'asymmetric':
            label = 'uasym'

        else:
            awelogger.logger.error('unsteady model not yet implemented.')

    else:
        awelogger.logger.error('model not yet implemented.')

    return label


def get_corr_val_axisym(model_options, variables, parent, label):
    a_var = get_a_var(model_options, variables, parent, label)
    corr_val = (1. - a_var)
    return corr_val

def get_corr_val_glauert(model_options, variables, parent, label):
    a_var = get_a_var(model_options, variables, parent, label)
    cosgamma_var = get_cosgamma_var(variables, parent)

    corr_val = cas.sqrt( (1. - a_var * (2. * cosgamma_var - a_var)) )
    return corr_val

def get_corr_val_coleman(model_options, atmos, wind, variables, outputs, parameters, parent, architecture, label):
    a = get_a_var(model_options, variables, parent, label)
    singamma = get_singamma_var(variables, parent)
    cosgamma = get_cosgamma_var(variables, parent)
    chi = get_wake_angle_chi(model_options, atmos, wind, variables, outputs, parameters, parent, architecture, label)

    corr_val = cosgamma + np.tan(chi / 2.) * singamma - a / (np.cos(chi / 2.)**2.)
    return corr_val

def get_corr_val_simple(model_options, variables, parent, label):
    a_var = get_a_var(model_options, variables, parent, label)
    cosgamma_var = get_cosgamma_var(variables, parent)
    corr_val = (cosgamma_var - a_var)
    return corr_val

def get_corr_val(model_options, atmos, wind, variables, outputs, parameters, parent, architecture, label):

    actuator_skew = model_options['aero']['actuator']['actuator_skew']

    if actuator_skew == 'not_in_use':
        corr_val = get_corr_val_axisym(model_options, variables, parent, label)

    elif actuator_skew == 'coleman':
        corr_val = get_corr_val_coleman(model_options, atmos, wind, variables, outputs, parameters, parent, architecture, label)

    elif actuator_skew == 'glauert':
        corr_val = get_corr_val_glauert(model_options, variables, parent, label)

    elif actuator_skew == 'simple':
        corr_val = get_corr_val_simple(model_options, variables, parent, label)

    else:
        message = 'unknown actuator angle correction model selected'
        raise Exception(message)
        corr_val = get_corr_val_simple(model_options, variables, parent, label)

    return corr_val

