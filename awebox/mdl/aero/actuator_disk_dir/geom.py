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

import awebox.tools.vector_operations as vect_op
import awebox.mdl.aero.actuator_disk_dir.geometry_dir.path_geom as path_based
import awebox.mdl.aero.actuator_disk_dir.geometry_dir.multi_kite_geom as multi_kite_geom
import awebox.mdl.aero.actuator_disk_dir.geometry_dir.n_hat_opt as n_hat_opt


# switches

def get_center_point(model_options, parent, variables, architecture):

    children = architecture.kites_map[parent]
    number_children = float(len(children))

    if number_children > 1:
        center = multi_kite_geom.approx_center_point(parent, variables, architecture)
    else:
        center = path_based.approx_center_point(model_options, children, variables, architecture)

    return center

def get_center_velocity(model_options, parent, variables, architecture):

    children = architecture.kites_map[parent]
    number_children = float(len(children))

    if number_children > 1:
        dcenter = multi_kite_geom.approx_center_velocity(parent, variables, architecture)
    else:
        # dcenter = path_based.approx_center_velocity(model_options, children, variables, architecture)

        n_hat_var = get_n_hat_var(variables, parent)
        dq = variables['xd']['dq' + str(children[0]) + str(parent)]
        dcenter = cas.mtimes(dq.T, n_hat_var) * n_hat_var

    return dcenter

def get_kite_radius_vector(model_options, kite, variables, architecture):
    number_siblings = architecture.get_number_siblings(kite)

    if number_siblings > 1:
        r_vec = multi_kite_geom.approx_kite_radius_vector(variables, architecture, kite)
    else:
        parent = architecture.parent_map[kite]
        r_vec = path_based.approx_kite_radius_vector(model_options, variables, kite, parent)
    return r_vec


def get_mu_radial_ratio(model_options, variables, kite, parent):
    varrho_var = get_varrho_var(model_options, variables, kite, parent)
    mu = varrho_var / (varrho_var + 0.5)

    return mu


def get_var_type(model_options):
    """ Extract variable type of average induction factor.
        steady: algebraic variable
        unsteady: differential state"""
    steadyness = model_options['aero']['actuator']['steadyness']

    # if steadyness == 'steady':
    #     var_type = 'xl'
    # elif steadyness == 'unsteady':
    #     var_type = 'xd'
    # else:
    #     raise ValueError('Invalid steadyness option for actuator disk model chosen')

    var_type = 'xd'
    return var_type


# variables

def get_area_var(model_options, variables, parent, parameters):
    area_ref = get_area_ref(model_options, parameters)
    area_var = area_ref * variables['xl']['area' + str(parent)]
    return area_var

def get_bar_varrho_var(model_options, variables, parent):
    type = get_var_type(model_options)
    varrho_ref = get_varrho_ref(model_options)
    varrho_var = varrho_ref * variables[type]['bar_varrho' + str(parent)]
    return varrho_var

def get_varrho_var(model_options, variables, kite, parent):
    varrho_ref = get_varrho_ref(model_options)
    varrho_var = varrho_ref * variables['xl']['varrho' + str(kite) + str(parent)]
    return varrho_var

def get_rot_matr_var(variables, parent):
    rot_cols = variables['xl']['rot_matr' + str(parent)]
    rot_matr = cas.reshape(rot_cols, (3, 3))

    return rot_matr

def get_n_hat_var(variables, parent):
    rot_matr = get_rot_matr_var(variables, parent)
    n_hat = rot_matr[:, 0]
    return n_hat

def get_n_hat_slack_lower(variables, parent):
    slack = variables['xl']['n_hat_slack' + str(parent)][:3]
    return slack

def get_n_hat_slack_upper(variables, parent):
    slack = variables['xl']['n_hat_slack' + str(parent)][3:]
    return slack

def get_y_rotor_hat_var(variables, parent):
    rot_matr = get_rot_matr_var(variables, parent)
    y_hat = rot_matr[:, 1]
    return y_hat

def get_z_rotor_hat_var(variables, parent):
    rot_matr = get_rot_matr_var(variables, parent)
    y_hat = rot_matr[:, 2]
    return y_hat

def get_z_vec_length_var(variables, parent):
    len_var = variables['xl']['z_vec_length' + str(parent)]
    return len_var

def get_psi_var(variables, kite, parent):
    psi_scale = 2. * np.pi
    psi_var = psi_scale * variables['xd']['psi' + str(kite) + str(parent)]
    return psi_var

def get_cospsi_var(variables, kite, parent):
    cospsi_var = variables['xl']['cospsi' + str(kite) + str(parent)]
    return cospsi_var

def get_sinpsi_var(variables, kite, parent):
    sinpsi_var = variables['xl']['sinpsi' + str(kite) + str(parent)]
    return sinpsi_var

# references

def get_tstar_ref(parameters, wind):
    b_ref = parameters['theta0', 'geometry', 'b_ref']
    uinfty_ref = wind.get_velocity_ref()
    tstar = b_ref / uinfty_ref
    return tstar

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

def get_area_residual(model_options, parent, variables, parameters):

    area_var = get_area_var(model_options, variables, parent, parameters)
    area_val = get_actuator_area(model_options, parent, variables, parameters)
    resi_unscaled = area_var - area_val

    area_ref = get_area_ref(model_options, parameters)
    resi_scaled = resi_unscaled / area_ref

    return resi_scaled

def get_area_trivial(model_options, parent, variables, parameters):

    area_var = get_area_var(model_options, variables, parent, parameters)
    area_ref = get_area_ref(model_options, parameters)
    resi_unscaled = area_var - area_ref
    resi_scaled = resi_unscaled / area_ref

    return resi_scaled

def get_bar_varrho_residual(model_options, parent, variables, architecture):

    bar_varrho_val = get_bar_varrho_val(model_options, variables, parent, architecture)
    bar_varrho_var = get_bar_varrho_var(model_options, variables, parent)

    resi_unscaled = bar_varrho_var - bar_varrho_val

    varrho_ref = get_varrho_ref(model_options)
    resi = resi_unscaled / varrho_ref

    # resi = bar_varrho_var - 7.

    return resi

def get_bar_varrho_trivial(model_options, parent, variables, architecture):
    varrho_ref = get_varrho_ref(model_options)
    bar_varrho_var = get_bar_varrho_var(model_options, variables, parent)

    resi_unscaled = bar_varrho_var - varrho_ref
    resi = resi_unscaled / varrho_ref

    return resi

def get_varrho_residual(model_options, kite, variables, parameters, architecture):

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

    varrho_var = get_varrho_var(model_options, variables, kite, parent)
    radius = varrho_var * b_ref

    varrho_ref = get_varrho_ref(model_options)
    radius_ref = b_ref * varrho_ref

    f_cos_proj = (radius * cospsi_var - z_rotor_comp) / radius_ref
    f_sin_proj = (radius * sinpsi_var + y_rotor_comp) / radius_ref

    resi_combi = cas.vertcat(f_cos, f_sin, f_cos_proj, f_sin_proj)

    return resi_combi

def get_rot_matr_ortho_residual(model_options, parent, variables, parameters, architecture):
    # rotation matrix is in SO3 = 6 constraints
    rot_matr_var = get_rot_matr_var(variables, parent)
    ortho_matr = cas.mtimes(rot_matr_var.T, rot_matr_var) - np.eye(3)
    f_ortho = vect_op.upper_triangular_inclusive(ortho_matr)

    return f_ortho

def get_rot_matr_n_along_normal_residual(model_options, parent, variables, parameters, architecture):
    # n_hat * length equals normal direction = 3 constraints
    n_vec_val = n_hat_opt.get_n_vec(model_options, parent, variables, parameters, architecture)
    n_hat_var = get_n_hat_var(variables, parent)
    n_vec_length_var = n_hat_opt.get_n_vec_length_var(variables, parent)

    slack_lower = get_n_hat_slack_lower(variables, parent)
    slack_upper = get_n_hat_slack_upper(variables, parent)

    n_diff = n_vec_val - (n_hat_var - slack_lower + slack_upper) * n_vec_length_var

    n_vec_length_ref = n_hat_opt.get_n_vec_length_ref(variables, parent)
    f_n_vec = n_diff / n_vec_length_ref

    return f_n_vec

def get_rot_matr_n_along_tether_residual(model_options, parent, variables, parameters, architecture):
    # n_hat * length equals normal direction = 3 constraints
    n_vec_val = n_hat_opt.get_n_vec_default(model_options, parent, variables, parameters, architecture)
    n_hat_var = get_n_hat_var(variables, parent)
    n_vec_length_var = n_hat_opt.get_n_vec_length_var(variables, parent)

    n_diff = n_vec_val - n_hat_var * n_vec_length_var

    n_vec_length_ref = n_hat_opt.get_n_vec_length_ref(variables, parent)
    f_n_vec = n_diff / n_vec_length_ref

    return f_n_vec


def get_rot_matr_residual(model_options, parent, variables, parameters, architecture):

    # total number of variables = 10 (9 from rot_matr, 1 lengths)
    f_ortho = get_rot_matr_ortho_residual(model_options, parent, variables, parameters, architecture)
    f_n_vec = get_rot_matr_n_along_normal_residual(model_options, parent, variables, parameters, architecture)
    #
    # join the constraints
    f_combi = cas.vertcat(f_ortho, f_n_vec)

    return f_combi

def get_rot_matr_trivial(model_options, parent, variables, parameters, architecture):

    # total number of variables = 10 (9 from rot_matr, 1 lengths)
    f_ortho = get_rot_matr_ortho_residual(model_options, parent, variables, parameters, architecture)
    f_n_vec = get_rot_matr_n_along_tether_residual(model_options, parent, variables, parameters, architecture)
    #
    # join the constraints
    f_combi = cas.vertcat(f_ortho, f_n_vec)

    return f_combi



# processing

def get_actuator_area(model_options, parent, variables, parameters):

    b_ref = parameters['theta0','geometry','b_ref']
    bar_varrho_var = get_bar_varrho_var(model_options, variables, parent)

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

def get_kite_radius(model_options, kite, variables, architecture, parameters):

    b_ref = parameters['theta0', 'geometry', 'b_ref']
    parent = architecture.parent_map[kite]
    varrho_var = get_varrho_var(model_options, variables, kite, parent)

    radius = varrho_var * b_ref

    return radius

def get_average_radius(model_options, variables, parent, architecture, parameters):
    children = architecture.kites_map[parent]
    number_children = float(len(children))

    average_radius = 0.
    for kite in children:
        radius = get_kite_radius(model_options, kite, variables, architecture, parameters)

        average_radius = average_radius + radius / number_children

    return average_radius

def get_bar_varrho_val(model_options, variables, parent, architecture):
    children = architecture.kites_map[parent]
    number_children = float(len(children))

    sum_varrho = 0.
    for kite in children:
        varrho_kite = get_varrho_var(model_options, variables, kite, parent)
        sum_varrho = sum_varrho + varrho_kite

    bar_varrho_val = sum_varrho / number_children
    return bar_varrho_val

def get_n_vec_val(model_options, parent, variables, parameters, architecture):
    n_vec_val = n_hat_opt.get_n_vec(model_options, parent, variables, parameters, architecture)
    return n_vec_val

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

    radius = get_kite_radius(model_options, kite, variables, architecture, parameters)

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
