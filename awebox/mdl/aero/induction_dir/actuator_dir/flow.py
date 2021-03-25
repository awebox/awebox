#
#    This file is part of awebox.
#
#    awebox -- A modeling and optimization framework for multi-kite AWE systems.
#    Copyright (C) 2017-2021 Jochem De Schutter, Rachel Leuthold, Moritz Diehl,
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
- author: rachel leuthold, alu-fr 2017-21
- edit: jochem de schutter, alu-fr 2019
"""

import casadi.tools as cas
import numpy as np
from awebox.logger.logger import Logger as awelogger

import awebox.mdl.aero.induction_dir.actuator_dir.geom as actuator_geom
import awebox.mdl.aero.induction_dir.actuator_dir.force as actuator_force
import awebox.mdl.aero.induction_dir.tools_dir.flow as general_flow
import awebox.mdl.aero.induction_dir.tools_dir.geom as general_geom

import awebox.tools.vector_operations as vect_op
import awebox.tools.print_operations as print_op
import awebox.tools.constraint_operations as cstr_op

## variables

def get_a_var_type(label):
    """ Extract variable type of average induction factor.
        steady: algebraic variable
        unsteady: differential state"""

    if label[0] == 'q':
        var_type = 'xl'
    elif label[0] == 'u':
        var_type = 'xd'
    else:
        raise Exception('Invalid steadyness option for actuator disk model chosen')

    return var_type

def get_local_a_var(variables, kite, parent):
    local_a = variables['xl']['local_a' + str(kite) + str(parent)]
    return local_a

def get_a_var(variables, parent, label):
    var_type = get_a_var_type(label)
    a_var = variables[var_type]['a_' + label + str(parent)]
    return a_var

def get_acos_var(variables, parent, label):
    var_type = get_a_var_type(label)
    acos_var = variables[var_type]['acos_' + label + str(parent)]
    return acos_var

def get_asin_var(variables, parent, label):
    var_type = get_a_var_type(label)
    asin_var = variables[var_type]['asin_' + label + str(parent)]
    return asin_var

def get_a_all_var(variables, parent, label):

    if 'asym' in label:
        a_var = get_a_var(variables, parent, label)
        acos_var = get_acos_var(variables, parent, label)
        asin_var = get_asin_var(variables, parent, label)
        a_all = cas.vertcat(a_var, acos_var, asin_var)
    else:
        a_all = get_a_var(variables, parent, label)
    return a_all

def get_da_var(variables, parent, label):
    da_var = variables['xddot']['da_' + label + str(parent)]
    return da_var

def get_dacos_var(variables, parent, label):
    dacos_var = variables['xddot']['dacos_' + label + str(parent)]
    return dacos_var

def get_dasin_var(variables, parent, label):
    dasin_var = variables['xddot']['dasin_' + label + str(parent)]
    return dasin_var

def get_da_all_var(variables, parent, label):
    if 'asym' in label:
        da_var = get_da_var(variables, parent, label)
        dacos_var = get_dacos_var(variables, parent, label)
        dasin_var = get_dasin_var(variables, parent, label)
        da_all = cas.vertcat(da_var, dacos_var, dasin_var)
    else:
        da_all = get_da_var(variables, parent, label)
    return da_all

def get_wind_dcm_var(variables, parent):
    rot_cols = variables['xl']['wind_dcm' + str(parent)]
    wind_dcm = cas.reshape(rot_cols, (3, 3))
    return wind_dcm

def get_uzero_hat_var(variables, parent):
    wind_dcm = get_wind_dcm_var(variables, parent)
    u_hat = wind_dcm[:, 0]
    return u_hat

def get_vzero_hat_var(variables, parent):
    wind_dcm = get_wind_dcm_var(variables, parent)
    v_hat = wind_dcm[:, 1]
    return v_hat

def get_wzero_hat_var(variables, parent):
    wind_dcm = get_wind_dcm_var(variables, parent)
    w_hat = wind_dcm[:, 2]
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

def get_uzero_vec_length_var(variables, parent):
    len_var = variables['xl']['u_vec_length' + str(parent)]
    return len_var

def get_g_vec_length_var(variables, parent):
    len_var = variables['xl']['g_vec_length' + str(parent)]
    return len_var


## residuals

def get_gamma_cstr(parent, variables):

    # notice that the correction values are even functions of gamma

    uzero_hat_var = get_uzero_hat_var(variables, parent)
    vzero_hat_var = get_vzero_hat_var(variables, parent)

    n_hat_var = actuator_geom.get_n_hat_var(variables, parent)
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

    name = 'actuator_gamma_' + str(parent)
    cstr = cstr_op.Constraint(expr=resi,
                              name=name,
                              cstr_type='eq')

    return cstr


def get_wind_dcm_ortho_cstr(parent, variables):

    # rotation matrix is in SO3 = 6 constraints
    wind_dcm_var = get_wind_dcm_var(variables, parent)
    ortho_matr = cas.mtimes(wind_dcm_var.T, wind_dcm_var) - np.eye(3)
    f_ortho = vect_op.upper_triangular_inclusive(ortho_matr)

    name = 'actuator_wind_dcm_ortho_' + str(parent)
    cstr = cstr_op.Constraint(expr=f_ortho,
                              name=name,
                              cstr_type='eq')

    return cstr

def get_wind_dcm_u_along_uzero_cstr(model_options, wind, parent, variables, parameters, architecture):

    # 3 constraints

    u_vec_val = general_flow.get_uzero_vec(model_options, wind, parent, variables, architecture)
    u_hat_var = get_uzero_hat_var(variables, parent)

    u_vec_length_var = get_uzero_vec_length_var(variables, parent)

    u_diff = u_vec_val - u_hat_var * u_vec_length_var

    u_vec_length_ref = get_uzero_vec_length_ref(wind)
    f_u_vec = u_diff / u_vec_length_ref

    name = 'actuator_uhat_' + str(parent)
    cstr = cstr_op.Constraint(expr=f_u_vec,
                              name=name,
                              cstr_type='eq')

    return cstr

def get_orientation_z_along_wzero_cstr(variables, parent):

    # 3 constraints

    w_hat_var = get_wzero_hat_var(variables, parent)
    z_rot_length = actuator_geom.get_z_vec_length_var(variables, parent)
    z_rotor_hat = actuator_geom.get_z_rotor_hat_var(variables, parent)
    f_full = w_hat_var - z_rotor_hat * z_rot_length

    name = 'actuator_zhat_' + str(parent)
    cstr = cstr_op.Constraint(expr=f_full,
                              name=name,
                              cstr_type='eq')

    return cstr


    return f_full

def get_wzero_parallel_z_rotor_check(variables, parent):
    w_hat_var = get_wzero_hat_var(variables, parent)
    z_rotor_hat = actuator_geom.get_z_rotor_hat_var(variables, parent)
    check = cas.mtimes(w_hat_var.T, z_rotor_hat) - 1.
    return check




def get_induction_factor_assignment_cstr(model_options, variables, kite, parent):
    a_var = get_local_a_var(variables, kite, parent)

    label = get_label(model_options)
    a_val = get_local_induction_factor(model_options, variables, kite, parent, label)

    a_ref = model_options['aero']['actuator']['a_ref']
    resi = (a_var - a_val) / a_ref

    name = 'actuator_a_assignment_' + str(kite)
    cstr = cstr_op.Constraint(expr=resi,
                              name=name,
                              cstr_type='eq')

    return cstr

## values

def get_gamma_val(model_options, wind, parent, variables, parameters, architecture):

    uzero = general_flow.get_uzero_vec(model_options, wind, parent, variables, architecture)
    n_vec = actuator_geom.get_n_vec_val(model_options, parent, variables, parameters, architecture)
    gamma = vect_op.angle_between(n_vec, uzero)
    return gamma

def get_gamma_check(model_options, wind, parent, variables, parameters, architecture):
    gamma_val = vect_op.abs(get_gamma_val(model_options, wind, parent, variables, parameters, architecture))
    gamma_var = vect_op.abs(get_gamma_var(variables, parent))
    check = gamma_val - gamma_var
    norm = cas.mtimes(check.T, check)
    return norm

## references


def get_qzero_ref(atmos, wind):
    scale = 5.
    rho_ref = atmos.get_density_ref()
    uinfty_ref = wind.get_velocity_ref()
    qzero_ref = .5 * rho_ref * uinfty_ref**2. * scale
    return qzero_ref

def get_a_ref(model_options):
    a_ref = model_options['aero']['actuator']['a_ref']
    return a_ref

def get_uzero_vec_length_ref(wind):
    return wind.get_velocity_ref()


def get_local_induction_factor(model_options, variables, kite, parent, label):

    if 'asym' in label:
        cospsi = actuator_geom.get_cospsi_var(variables, kite, parent)
        sinpsi = actuator_geom.get_sinpsi_var(variables, kite, parent)

        if model_options['aero']['actuator']['asym_radial_linearity']:
            mu = actuator_geom.get_mu_radial_ratio(variables, kite, parent)
            # version given in original Kinner / Pitt&Peters / Burton et al.
        else:
            mu = 1.
            # see Suzuki2000 (phd thesis)
            # Application of Dynamic Inflow Theory to Wind Turbine Rotors
            # for motivation for evaluating at the edges of the "annulus"
            # also: seems to lead to less optimizer-trickery.

        a_uni = get_a_var(variables, parent, label)
        acos = get_acos_var(variables, parent, label)
        asin = get_asin_var(variables, parent, label)
        a_local = a_uni + acos * cospsi * mu + asin * sinpsi * mu

    elif 'axi' in label:
        a_local = get_a_var(variables, parent, label)

    else:
        message = 'an unfamiliar actuator model label was entered when computing the local induction factor'
        awelogger.logger.error(message)
        raise Exception(message)

    return a_local


def get_local_induced_velocity(model_options, variables, parameters, architecture, wind, kite, parent, label):

    uzero_vec_length = get_uzero_vec_length_var(variables, parent)
    nhat = actuator_geom.get_n_hat_var(variables, parent)

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

    center = general_geom.get_center_point(model_options, parent, variables, architecture)
    rho_infty = atmos.get_density(center[2])

    uzero_mag = get_uzero_vec_length_var(variables, parent)

    qzero = 0.5 * rho_infty * uzero_mag**2.

    return qzero


def get_wake_angle_chi_equal(model_options, parent, variables, label):
    gamma = get_gamma_var(variables, parent)
    return gamma

def get_wake_angle_chi_coleman(parent, variables, label):
    gamma = get_gamma_var(variables, parent)
    a = get_a_var(variables, parent, label)

    chi = (0.6 * a + 1.) * gamma

    return chi

def get_wake_angle_chi_jimenez(model_options, atmos, wind, variables, outputs, parameters, parent, architecture):

    gamma = get_gamma_var(variables, parent)
    cosgamma = get_cosgamma_var(variables, parent)
    singamma = get_singamma_var(variables, parent)

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
        chi_val = get_wake_angle_chi_coleman(parent, variables, label)
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
    steadyness = model_options['induction']['steadyness']
    symmetry = model_options['induction']['symmetry']

    accepted_steadyness_dict = {'quasi-steady':'q', 'steady':'q', 'unsteady':'u'}
    accepted_symmetry_dict = {'axisymmetric': 'axi', 'asymmetric': 'asym'}
    
    if (steadyness in accepted_steadyness_dict.keys()) and (symmetry in accepted_symmetry_dict.keys()):
        label = accepted_steadyness_dict[steadyness] + accepted_symmetry_dict[symmetry]

    else:
        message = 'unrecognized actuator option (' + steadyness + ', ' + symmetry + ') indicated. available ' \
                'options are: ' + repr(accepted_steadyness_dict.keys()) + ' and ' + repr(accepted_symmetry_dict.keys())
        awelogger.logger.error(message)
        raise Exception(message)

    return label


def get_corr_val_axisym(model_options, variables, parent, label):
    a_var = get_a_var(variables, parent, label)
    corr_val = (1. - a_var)
    return corr_val

def get_corr_val_glauert(model_options, variables, parent, label):
    a_var = get_a_var(variables, parent, label)
    cosgamma_var = get_cosgamma_var(variables, parent)

    corr_val = cas.sqrt( (1. - a_var * (2. * cosgamma_var - a_var)) )
    return corr_val

def get_corr_val_coleman(model_options, atmos, wind, variables, outputs, parameters, parent, architecture, label):
    a = get_a_var(variables, parent, label)
    singamma = get_singamma_var(variables, parent)
    cosgamma = get_cosgamma_var(variables, parent)
    chi = get_wake_angle_chi(model_options, atmos, wind, variables, outputs, parameters, parent, architecture, label)

    corr_val = cosgamma + np.tan(chi / 2.) * singamma - a / (np.cos(chi / 2.)**2.)
    return corr_val

def get_corr_val_simple(model_options, variables, parent, label):
    a_var = get_a_var(variables, parent, label)
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

