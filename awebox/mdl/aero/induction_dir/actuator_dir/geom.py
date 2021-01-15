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
'''
actuator_disk model of awebox aerodynamics
sets up the axial-induction actuator disk equation
currently for untilted rotor with no tcf.
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2017-20
- edit: jochem de schutter, alu-fr 2019
'''

import casadi.tools as cas
import numpy as np

import awebox.tools.vector_operations as vect_op

import awebox.mdl.aero.induction_dir.tools_dir.path_based_geom as path_based_geom
import awebox.mdl.aero.induction_dir.tools_dir.multi_kite_geom as multi_kite_geom

import awebox.mdl.aero.induction_dir.general_dir.geom as general_geom

# switches

def get_center_point(model_options, parent, variables, architecture):

    children = architecture.kites_map[parent]
    number_children = float(len(children))

    if number_children > 1:
        center = multi_kite_geom.approx_center_point(parent, variables, architecture)
    else:
        center = path_based_geom.approx_center_point(model_options, children, variables, architecture)

    return center

def get_center_velocity(model_options, parent, variables, parameters, architecture):

    children = architecture.kites_map[parent]
    number_children = float(len(children))

    if (parent > 0) and (number_children > 1):
        dcenter = multi_kite_geom.approx_center_velocity(parent, variables, architecture)
    elif (parent > 0):
        # dcenter = path_based.approx_center_velocity(model_options, children, variables, architecture)
        n_hat_var = general_geom.get_n_hat_var(variables, parent)
        dq = variables['xd']['dq' + str(children[0]) + str(parent)]
        dcenter = cas.mtimes(dq.T, n_hat_var) * n_hat_var
    else:
        dq = variables['xd']['dq' + str(children[0]) + str(parent)]
        dcenter = dq

    return dcenter

def get_kite_radius_vector(model_options, kite, variables, architecture):
    number_siblings = architecture.get_number_siblings(kite)

    if number_siblings > 1:
        r_vec = multi_kite_geom.approx_kite_radius_vector(variables, architecture, kite)
    else:
        parent = architecture.parent_map[kite]
        r_vec = path_based_geom.approx_kite_radius_vector(model_options, variables, kite, parent)
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


def get_bar_varrho_residual(model_options, parent, variables, architecture):

    bar_varrho_val = get_bar_varrho_val(model_options, variables, parent, architecture)
    bar_varrho_var = get_bar_varrho_var(model_options, variables, parent)

    resi_unscaled = bar_varrho_var - bar_varrho_val

    varrho_ref = get_varrho_ref(model_options)
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

    y_rotor_hat_var = general_geom.get_y_rotor_hat_var(variables, parent)
    z_rotor_hat_var = general_geom.get_z_rotor_hat_var(variables, parent)

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

    y_rotor_hat_var = general_geom.get_y_rotor_hat_var(variables, parent)
    z_rotor_hat_var = general_geom.get_z_rotor_hat_var(variables, parent)

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
