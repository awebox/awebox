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
'''
actuator_disk model of awebox aerodynamics
sets up the axial-induction actuator disk equation
currently for untilted rotor with no tcf.
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2017-21
- edit: jochem de schutter, alu-fr 2019
'''

import casadi.tools as cas
import numpy as np

import awebox.tools.vector_operations as vect_op
import awebox.tools.constraint_operations as cstr_op
import awebox.tools.print_operations as print_op

import awebox.mdl.aero.induction_dir.tools_dir.path_based_geom as path_based_geom
import awebox.mdl.aero.induction_dir.tools_dir.multi_kite_geom as multi_kite_geom
import awebox.mdl.aero.induction_dir.tools_dir.unit_normal as unit_normal

# switches

def get_kite_radius_vector(model_options, kite, variables, architecture):
    number_siblings = architecture.get_number_siblings(kite)

    if number_siblings > 1:
        r_vec = multi_kite_geom.approx_kite_radius_vector(variables, architecture, kite)
    else:
        parent = architecture.parent_map[kite]
        r_vec = path_based_geom.approx_kite_radius_vector(model_options, variables, kite, parent)
    return r_vec


def get_mu_radial_ratio(variables, kite, parent):
    varrho_var = get_varrho_var(variables, kite, parent)
    mu = varrho_var / (varrho_var + 0.5)

    return mu


# variables

def get_area_var(variables, parent):
    area_var = variables['xl']['area' + str(parent)]
    return area_var

def get_bar_varrho_var(variables, parent):
    varrho_var = variables['xl']['bar_varrho' + str(parent)]
    return varrho_var

def get_varrho_var(variables, kite, parent):
    varrho_var = variables['xl']['varrho' + str(kite) + str(parent)]
    return varrho_var

def get_psi_var(variables, kite, parent):
    psi_var = variables['xl']['psi' + str(kite) + str(parent)]
    return psi_var

def get_cospsi_var(variables, kite, parent):
    cospsi_var = variables['xl']['cospsi' + str(kite) + str(parent)]
    return cospsi_var

def get_sinpsi_var(variables, kite, parent):
    sinpsi_var = variables['xl']['sinpsi' + str(kite) + str(parent)]
    return sinpsi_var

def get_n_vec_length_var(variables, parent):
    len_var = variables['xl']['n_vec_length' + str(parent)]
    return len_var


# references

def get_tstar_ref(parameters, wind):
    b_ref = parameters['theta0', 'geometry', 'b_ref']
    uinfty_ref = wind.get_velocity_ref()
    tstar = b_ref / uinfty_ref
    return tstar


def get_n_vec_length_ref(model_options):
    return model_options['scaling']['xl']['n_vec_length']

def get_varrho_ref(model_options):
    varrho_ref = model_options['aero']['actuator']['varrho_ref']
    return varrho_ref

def get_area_ref(model_options, parameters):
    b_ref = parameters['theta0','geometry','b_ref']
    varrho_ref = get_varrho_ref(model_options)
    r_ref = varrho_ref * b_ref
    area_ref = 2. * np.pi * r_ref * b_ref
    return area_ref

# residuals

def get_bar_varrho_cstr(model_options, parent, variables, architecture):

    bar_varrho_val = get_bar_varrho_val(variables, parent, architecture)
    bar_varrho_var = get_bar_varrho_var(variables, parent)

    resi_unscaled = bar_varrho_var - bar_varrho_val

    varrho_ref = get_varrho_ref(model_options)
    resi = resi_unscaled / varrho_ref

    name = 'actuator_bar_varrho_' + str(parent)
    cstr = cstr_op.Constraint(expr=resi,
                              name=name,
                              cstr_type='eq')
    return cstr


def get_varrho_and_psi_cstr(model_options, kite, variables, parameters, architecture):

    # for positive yaw(turns around +zhat, normal towards +yhat):
    #     rhat = zhat * cos(psi) - yhat * sin(psi)
    #
    # rvec = radius((zhat') * cos(psi) + (-yhat') * sin(psi))
    # dot(rvec, zhat') = radius * cos(psi)
    # dot(rvec, yhat') = - radius * sin(psi)

    parent = architecture.parent_map[kite]
    b_ref = parameters['theta0', 'geometry', 'b_ref']

    radius_vec = get_kite_radius_vector(model_options, kite, variables, architecture)

    y_rotor_hat_var = get_y_rotor_hat_var(variables, parent)
    z_rotor_hat_var = get_z_rotor_hat_var(variables, parent)

    y_rotor_comp = cas.mtimes(radius_vec.T, y_rotor_hat_var)
    z_rotor_comp = cas.mtimes(radius_vec.T, z_rotor_hat_var)

    psi_var = get_psi_var(variables, kite, parent)
    cospsi_var = get_cospsi_var(variables, kite, parent)
    sinpsi_var = get_sinpsi_var(variables, kite, parent)

    f_sin = np.sin(psi_var) - sinpsi_var
    f_cos = np.cos(psi_var) - cospsi_var

    varrho_var = get_varrho_var(variables, kite, parent)
    radius = varrho_var * b_ref

    varrho_ref = get_varrho_ref(model_options)
    radius_ref = b_ref * varrho_ref

    f_cos_proj = (radius * cospsi_var - z_rotor_comp) / radius_ref
    f_sin_proj = (radius * sinpsi_var + y_rotor_comp) / radius_ref

    resi_combi = cas.vertcat(f_cos, f_sin, f_cos_proj, f_sin_proj)

    name = 'actuator_varrho_and_psi_' + str(kite)
    cstr = cstr_op.Constraint(expr=resi_combi,
                              name=name,
                              cstr_type='eq')
    return cstr


# processing

def get_actuator_area(model_options, parent, variables, parameters):

    b_ref = parameters['theta0','geometry','b_ref']
    bar_varrho_var = get_bar_varrho_var(variables, parent)

    radius = bar_varrho_var * b_ref
    annulus_area = 2. * np.pi * b_ref * radius

    area = annulus_area

    return area

def get_kite_radial_vector(model_options, kite, variables, architecture, parameters):

    parent = architecture.parent_map[kite]

    y_rotor_hat_var = get_y_rotor_hat_var(variables, parent)
    z_rotor_hat_var = get_z_rotor_hat_var(variables, parent)

    psi_var = get_psi_var(variables, kite, parent)
    cospsi_var = get_cospsi_var(variables, kite, parent)
    sinpsi_var = get_sinpsi_var(variables, kite, parent)

    # for positive yaw(turns around +zhat, normal towards +yhat):
    #     rhat = zhat * cos(psi) - yhat * sin(psi)
    rhat = z_rotor_hat_var * cospsi_var - y_rotor_hat_var * sinpsi_var

    return rhat

def get_kite_radius(kite, variables, architecture, parameters):

    b_ref = parameters['theta0', 'geometry', 'b_ref']
    parent = architecture.parent_map[kite]
    varrho_var = get_varrho_var(variables, kite, parent)

    radius = varrho_var * b_ref

    return radius

def get_average_radius(model_options, variables, parent, architecture, parameters):
    children = architecture.kites_map[parent]
    number_children = float(len(children))

    average_radius = 0.
    for kite in children:
        radius = get_kite_radius(kite, variables, architecture, parameters)

        average_radius = average_radius + radius / number_children

    return average_radius

def get_bar_varrho_val(variables, parent, architecture):
    children = architecture.kites_map[parent]
    number_children = float(len(children))

    sum_varrho = 0.
    for kite in children:
        varrho_kite = get_varrho_var(variables, kite, parent)
        sum_varrho = sum_varrho + varrho_kite

    bar_varrho_val = sum_varrho / number_children
    return bar_varrho_val


def approximate_tip_radius(model_options, variables, kite, architecture, tip, parameters):

    b_ref = parameters['theta0','geometry','b_ref']
    half_span_proj = b_ref / 2.
    parent = architecture.parent_map[kite]

    radial_vector = get_kite_radial_vector(model_options, kite, variables, architecture, parameters)

    if int(model_options['kite_dof']) == 6:

        r_column = variables['xd']['r' + str(kite) + str(parent)]
        r = cas.reshape(r_column, (3, 3))
        ehat2 = r[:, 1]  # spanwise, from pe to ne

        ehat2_proj_radial = vect_op.smooth_abs(cas.mtimes(radial_vector.T, ehat2))

        half_span_proj = b_ref * ehat2_proj_radial / 2.

    radius = get_kite_radius(kite, variables, architecture, parameters)

    tip_radius = radius
    if ('int' in tip) or (tip == 0):
        tip_radius = tip_radius - half_span_proj
    elif ('ext' in tip) or (tip == 1):
        tip_radius = tip_radius + half_span_proj
    else:
        raise Exception('invalid tip designated')

    return tip_radius

def get_average_exterior_radius(model_options, variables, parent, parameters, architecture):

    children = architecture.kites_map[parent]
    number_children = float(len(children))

    average_radius = 0.
    for kite in children:
        radius = approximate_tip_radius(model_options, variables, kite, architecture, 'ext', parameters)

        average_radius = average_radius + radius / number_children

    return average_radius



def get_act_dcm_var(variables, parent):

    name = 'act_dcm' + str(parent)
    rot_cols = variables['xl'][name]
    act_dcm = cas.reshape(rot_cols, (3, 3))

    return act_dcm

def get_n_hat_var(variables, parent):
    act_dcm = get_act_dcm_var(variables, parent)
    n_hat = act_dcm[:, 0]
    return n_hat

def get_y_rotor_hat_var(variables, parent):
    act_dcm = get_act_dcm_var(variables, parent)
    y_hat = act_dcm[:, 1]
    return y_hat

def get_z_rotor_hat_var(variables, parent):
    act_dcm = get_act_dcm_var(variables, parent)
    y_hat = act_dcm[:, 2]
    return y_hat

def get_z_vec_length_var(variables, parent):
    len_var = variables['xl']['z_vec_length' + str(parent)]
    return len_var



def get_act_dcm_ortho_cstr(parent, variables):
    # rotation matrix is in SO3 = 6 constraints
    act_dcm_var = get_act_dcm_var(variables, parent)
    ortho_matr = cas.mtimes(act_dcm_var.T, act_dcm_var) - np.eye(3)
    f_ortho = vect_op.upper_triangular_inclusive(ortho_matr)

    name = 'actuator_geom_dcm_ortho_' + str(parent)
    cstr = cstr_op.Constraint(expr=f_ortho,
                              name=name,
                              cstr_type='eq')

    return cstr

def get_act_dcm_n_along_normal_cstr(model_options, parent, variables, parameters, architecture):

    # n_hat * length equals normal direction = 3 constraints
    n_vec_val = unit_normal.get_n_vec(model_options, parent, variables, parameters, architecture)
    n_hat_var = get_n_hat_var(variables, parent)
    n_vec_length_var = get_n_vec_length_var(variables, parent)

    n_diff = n_vec_val - n_hat_var * n_vec_length_var

    n_vec_length_ref = get_n_vec_length_ref(model_options)
    f_n_vec = n_diff / n_vec_length_ref

    name = 'actuator_nhat_' + str(parent)
    cstr = cstr_op.Constraint(expr=f_n_vec,
                              name=name,
                              cstr_type='eq')

    return cstr

def get_n_vec_val(model_options, parent, variables, parameters, architecture):
    n_vec_val = unit_normal.get_n_vec(model_options, parent, variables, parameters, architecture)
    return n_vec_val

