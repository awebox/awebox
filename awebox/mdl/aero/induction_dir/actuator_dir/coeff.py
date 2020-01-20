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
actuator_disk model of awebox aerodynamics
sets up the axial-induction actuator disk equation
currently for untilted rotor with no tcf.
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2017-19
- edit: jochem de schutter, alu-fr 2019
'''

import casadi.tools as cas
import numpy as np


import awebox.mdl.aero.induction_dir.general_dir.geom as general_geom

import awebox.mdl.aero.induction_dir.actuator_dir.geom as actuator_geom
import awebox.mdl.aero.induction_dir.actuator_dir.flow as actuator_flow

import awebox.tools.vector_operations as vect_op


def get_ct_var(model_options, variables, parent):

    var_type = actuator_geom.get_var_type(model_options)
    ct_var = variables[var_type]['ct' + str(parent)]
    return ct_var

def get_dct_var(variables, parent):

    dct_var = variables['xd']['dct' + str(parent)]
    return dct_var


def get_cmy_var(variables, parent):
    cm_ref = get_cm_ref()
    cmy_var = cm_ref * variables['xl']['cmy' + str(parent)]
    return cmy_var

def get_cmz_var(variables, parent):
    cm_ref = get_cm_ref()
    cmz_var = cm_ref * variables['xl']['cmz' + str(parent)]
    return cmz_var

def get_c_all_var(model_options, variables, parent, label):
    if 'asym' in label:
        ct_var = get_ct_var(model_options, variables, parent)
        cmy_var = get_cmy_var(variables, parent)
        cmz_var = get_cmz_var(variables, parent)
        c_all = cas.vertcat(ct_var, cmy_var, cmz_var)
    else:
        c_all = get_ct_var(model_options, variables, parent)
    return c_all

def get_t_star_var(variables, parent):
    t_star = variables['xl']['t_star' + str(parent)]
    return t_star

def get_c_tilde_var(variables, parent, label):
    c_tilde_var =variables['xl']['c_tilde_' + str(label) + str(parent)]
    return c_tilde_var

def get_LL_var(variables, parent, label):
    LL = variables['xl']['LL_' + label + str(parent)]
    return LL

def get_LL_matrix(variables, parent, label):
    LL_var = get_LL_var(variables, parent, label)
    LL_matr = cas.reshape(LL_var, (3, 3))

    return LL_matr

def get_MM_matrix():
    MM11 = 1.69765
    MM22 = 0.113177
    MM33 = 0.113177

    MM_col1 = MM11 * vect_op.xhat()
    MM_col2 = MM22 * vect_op.yhat()
    MM_col3 = MM33 * vect_op.zhat()

    MM = cas.horzcat(MM_col1, MM_col2, MM_col3)

    return MM

def get_actuator_force(outputs, parent, architecture):

    children = architecture.kites_map[parent]

    total_force_aero = np.zeros((3, 1))
    for kite in children:
        aero_force = outputs['aerodynamics']['f_aero' + str(kite)]

        total_force_aero = total_force_aero + aero_force

    return total_force_aero

def get_actuator_moment(model_options, variables, outputs, parent, architecture):

    children = architecture.kites_map[parent]

    total_moment_aero = np.zeros((3, 1))
    for kite in children:
        aero_force = outputs['aerodynamics']['f_aero' + str(kite)]
        kite_radius = actuator_geom.get_kite_radius_vector(model_options, kite, variables, architecture)
        aero_moment = vect_op.cross(kite_radius, aero_force)

        total_moment_aero = total_moment_aero + aero_moment

    return total_moment_aero

def get_actuator_thrust(model_options, variables, outputs, parent, architecture):

    total_force_aero = get_actuator_force(outputs, parent, architecture)
    normal = general_geom.get_n_hat_var(variables, parent)
    thrust = cas.mtimes(total_force_aero.T, normal)

    return thrust

def get_actuator_moment_y_rotor(model_options, variables, outputs, parent, architecture):

    total_moment_aero = get_actuator_moment(model_options, variables, outputs, parent, architecture)
    y_rotor = general_geom.get_y_rotor_hat_var(variables, parent)
    moment = cas.mtimes(total_moment_aero.T, y_rotor)

    return moment

def get_actuator_moment_z_rotor(model_options, variables, outputs, parent, architecture):

    total_moment_aero = get_actuator_moment(model_options, variables, outputs, parent, architecture)
    z_rotor = general_geom.get_z_rotor_hat_var(variables, parent)
    moment = cas.mtimes(total_moment_aero.T, z_rotor)

    return moment


def get_moment_denom(model_options, variables, parent, atmos, wind, parameters):

    qzero_var = actuator_flow.get_qzero_var(atmos, wind, variables, parent)
    area_var = actuator_geom.get_area_var(model_options, variables, parent, parameters)

    bar_varrho_var = actuator_geom.get_bar_varrho_var(model_options, variables, parent)
    b_ref = parameters['theta0', 'geometry', 'b_ref']
    radius_bar =  bar_varrho_var * b_ref

    moment = qzero_var * area_var * radius_bar

    return moment

def get_thrust_residual(model_options, atmos, wind, variables, parameters, outputs, parent, architecture):

    thrust_val = get_actuator_thrust(model_options, variables, outputs, parent, architecture)

    area_var = actuator_geom.get_area_var(model_options, variables, parent, parameters)
    qzero_var = actuator_flow.get_qzero_var(atmos, wind, variables, parent)

    ct_var = get_ct_var(model_options, variables, parent)

    resi_unscaled = thrust_val - ct_var * area_var * qzero_var

    thrust_ref = get_thrust_ref(model_options, atmos, wind, parameters)

    resi_scaled = resi_unscaled / thrust_ref

    return resi_scaled


def get_thrust_trivial(model_options, atmos, wind, variables, parameters, outputs, parent, architecture):
    thrust_ref = get_thrust_ref(model_options, atmos, wind, parameters)
    area_ref = actuator_geom.get_area_ref(model_options, parameters)
    qzero_ref = actuator_flow.get_qzero_ref(atmos, wind)

    ct_var = get_ct_var(model_options, variables, parent)

    resi_unscaled = thrust_ref - ct_var * area_ref * qzero_ref
    resi_scaled = resi_unscaled / thrust_ref

    return resi_scaled

def get_moments_residual(model_options, atmos, wind, variables, parameters, outputs, parent, architecture):

    cmy_var = get_cmy_var(variables, parent)
    cmz_var = get_cmz_var(variables, parent)

    # see page 122 of Burton
    # positive yaw = normal points towards + yhat
    # positive yaw = turning around + zhat

    moment_y_val = get_actuator_moment_y_rotor(model_options, variables, outputs, parent, architecture)
    moment_z_val = get_actuator_moment_z_rotor(model_options, variables, outputs, parent, architecture)

    moment_denom = get_moment_denom(model_options, variables, parent, atmos, wind, parameters)
    moment_ref = get_moment_ref(model_options, atmos, wind, parameters)

    resi_moment_y = ( cmy_var * moment_denom - moment_y_val ) / moment_ref
    resi_moment_z = ( cmz_var * moment_denom - moment_z_val ) / moment_ref

    resi_combi = cas.vertcat(resi_moment_y, resi_moment_z)

    return resi_combi

def get_moments_trivial(model_options, atmos, wind, variables, parameters, outputs, parent, architecture):

    cmy_var = get_cmy_var(variables, parent)
    cmz_var = get_cmz_var(variables, parent)

    resi_moment_y = cmy_var
    resi_moment_z = cmz_var

    resi_combi = cas.vertcat(resi_moment_y, resi_moment_z)

    return resi_combi




# references
def get_ct_ref(model_options):
    a_ref = actuator_flow.get_a_ref(model_options)
    ct_ref = 4. * a_ref * (1. - a_ref)

    return ct_ref

def get_t_star_ref(model_options, wind, parameters):

    # t_star = geom.get_tstar_ref(parameters, wind)
    # bar_varrho_var = geom.get_bar_varrho_var(model_options, variables, parent)
    # dt_dtimescale = t_star * (bar_varrho_var + 0.5)

    b_ref = parameters['theta0', 'geometry', 'b_ref']
    uzero_ref = wind.get_velocity_ref()
    bar_varrho_ref = actuator_geom.get_varrho_ref(model_options)

    ref = b_ref * (bar_varrho_ref + 0.5) / uzero_ref
    return ref

def get_thrust_ref(model_options, atmos, wind, parameters):

    qzero_ref = actuator_flow.get_qzero_ref(atmos, wind)
    area_ref = actuator_geom.get_area_ref(model_options, parameters)
    ct_ref = get_ct_ref(model_options)

    thrust_ref = ct_ref * qzero_ref * area_ref

    scaling = model_options['aero']['actuator']['scaling']
    reference = scaling * thrust_ref

    return reference

def get_cm_ref():
    return 1.e-2

def get_moment_ref(model_options, atmos, wind, parameters):

    qzero_ref = actuator_flow.get_qzero_ref(atmos, wind)
    area_ref = actuator_geom.get_area_ref(model_options, parameters)
    bar_varrho_ref = actuator_geom.get_varrho_ref(model_options)
    b_ref = parameters['theta0', 'geometry', 'b_ref']
    bar_radius = (bar_varrho_ref) * b_ref

    moment = qzero_ref * area_ref * bar_radius

    return moment

def get_c_tilde_residual(model_options, variables, parent, label):

    c_tilde = get_c_tilde_var(variables, parent, label)
    LL_matr = get_LL_matrix(variables, parent, label)
    a_all = actuator_flow.get_a_all_var(model_options, variables, parent, label)

    resi = a_all - cas.mtimes(LL_matr, c_tilde)

    return resi

def get_LL_residual(model_options, variables, parent, label):

    LL_matr = get_LL_matrix(variables, parent, label)
    corr = actuator_flow.get_corr_var(variables, parent, label)

    tanhalfchi = actuator_flow.get_tanhalfchi_var(variables, parent, label)
    sechalfchi = actuator_flow.get_sechalfchi_var(variables, parent, label)

    LL11 = 0.25 / corr
    LL12 = 0.
    LL13 = -0.368155 * tanhalfchi
    LL21 = 0.
    LL22 = -1. * sechalfchi**2.
    LL23 = 0.
    LL31 = (0.368155 * tanhalfchi ) / corr
    LL32 = 0.
    LL33 = -1. + tanhalfchi**2.

    LL_row1 = cas.horzcat(LL11, LL12, LL13)
    LL_row2 = cas.horzcat(LL21, LL22, LL23)
    LL_row3 = cas.horzcat(LL31, LL32, LL33)
    LL_comp = cas.vertcat(LL_row1, LL_row2, LL_row3)

    resi_unscaled = LL_matr - LL_comp
    resi_reshape = cas.reshape(resi_unscaled, (9, 1))

    a_ref = actuator_flow.get_a_ref(model_options)
    corr_ref = (1. - a_ref)
    LL11_ref = 0.25 / corr_ref

    resi = resi_reshape / LL11_ref

    return resi

def get_t_star_residual(model_options, atmos, wind, variables, parameters, outputs, parent, architecture):
    dt_var = get_t_star_var(variables, parent)

    # t_star = geom.get_tstar_ref(parameters, wind)
    # bar_varrho_var = geom.get_bar_varrho_var(model_options, variables, parent)
    # dt_dtimescale = t_star * (bar_varrho_var + 0.5)

    b_ref = parameters['theta0', 'geometry', 'b_ref']
    uzero_mag = actuator_flow.get_uzero_vec_length_var(wind, variables, parent)
    bar_varrho_var = actuator_geom.get_bar_varrho_var(model_options, variables, parent)

    t_star_num = b_ref * (bar_varrho_var + 0.5)
    t_star_den = uzero_mag

    uinfty_ref = wind.get_velocity_ref()
    t_star_ref = get_t_star_ref(model_options, wind, parameters)
    scale = uinfty_ref * t_star_ref

    resi = (dt_var * t_star_den - t_star_num) / scale

    return resi


def get_t_star_trivial(model_options, atmos, wind, variables, parameters, outputs, parent, architecture):
    dt_var = get_t_star_var(variables, parent)
    t_star_ref = get_t_star_ref(model_options, wind, parameters)

    resi = (dt_var / t_star_ref - 1.)

    return resi

