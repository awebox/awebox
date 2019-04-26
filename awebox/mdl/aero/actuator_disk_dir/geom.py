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
- author: rachel leuthold, alu-fr 2017-18
- edit: jochem de schutter, alu-fr 2019
'''

import casadi.tools as cas
import numpy as np
import pdb

import awebox.tools.vector_operations as vect_op
import awebox.mdl.aero.actuator_disk_dir.geometry_dir.path_geom as path_based
import awebox.mdl.aero.actuator_disk_dir.geometry_dir.multi_kite_geom as multi_kite_geom
import awebox.mdl.aero.actuator_disk_dir.geometry_dir.nhat_opt as nhat_opt


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

        nhat_var = get_nhat_var(variables, parent)
        dq = variables['xd']['dq' + str(children[0]) + str(parent)]
        dcenter = cas.mtimes(dq.T, nhat_var) * nhat_var

    return dcenter

def get_kite_radius_vector(model_options, kite, variables, architecture):
    number_siblings = architecture.get_number_siblings(kite)

    if number_siblings > 1:
        r_vec = multi_kite_geom.approx_kite_radius_vector(variables, architecture, kite)
    else:
        parent = architecture.parent_map[kite]
        r_vec = path_based.approx_kite_radius_vector(model_options, variables, kite, parent)
    return r_vec

def get_normal_axis(model_options, parent, variables, parameters, architecture):

    nhat = nhat_opt.get_nhat(model_options, parent, variables, parameters, architecture)

    return nhat

def get_var_type(model_options):
    """ Extract variable type of average induction factor.
        steady: algebraic variable
        unsteady: differential state"""

    if model_options['aero']['actuator']['steadyness'] == 'steady':
        var_type = 'xl'
    elif model_options['aero']['actuator']['steadyness'] == 'unsteady':
        var_type = 'xd'
    else:
        raise ValueError('Invalid steadyness option for actuator disk model chosen')

    return var_type



# variables

def get_area_var(model_options, variables, parent, parameters):
    area_ref = get_area_ref(model_options, parameters)
    area_var = area_ref * variables['xl']['area' + str(parent)]
    return area_var

def get_bar_varrho_var(model_options, variables, parent):
    type = get_var_type(model_options)
    varrho_var = variables[type]['bar_varrho' + str(parent)]
    return varrho_var

def get_dbar_varrho_var(variables, parent):
    dvarrho_var = variables['xddot']['dbar_varrho' + str(parent)]
    return dvarrho_var

def get_varrho_var(variables, kite, architecture):
    parent = architecture.parent_map[kite]
    varrho_var = variables['xl']['varrho' + str(kite) + str(parent)]
    return varrho_var

def get_nhat_var(variables, parent):
    nhat_var = variables['xl']['nhat' + str(parent)]
    return nhat_var


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

    bar_varrho_val = get_bar_varrho_val(variables, parent, architecture)
    bar_varrho_var = get_bar_varrho_var(model_options, variables, parent)

    resi_unscaled = bar_varrho_var - bar_varrho_val

    varrho_ref = get_varrho_ref(model_options)
    resi = resi_unscaled / varrho_ref

    return resi

def get_varrho_residual(model_options, kite, variables, parameters, architecture):
    varrho_var = get_varrho_var(variables, kite, architecture)
    varrho_ref = get_varrho_ref(model_options)

    b_ref = parameters['theta0', 'geometry', 'b_ref']
    radius_sqared_val = get_kite_radius_proj_squared(model_options, kite, variables, architecture)
    varrho_val_squared = radius_sqared_val / b_ref**2.

    resi_unscaled = varrho_var**2. - varrho_val_squared
    resi_scaled = resi_unscaled / varrho_ref**2.

    return resi_scaled

def get_nhat_residual(model_options, parent, variables, parameters, architecture):

    nhat_val = get_normal_axis(model_options, parent, variables, parameters, architecture)
    nhat_var = get_nhat_var(variables, parent)

    nvec_resi = nhat_var - nhat_val

    factor_resi = vect_op.smooth_norm(nhat_var) - 1.

    resi = cas.vertcat(nvec_resi, factor_resi)

    return resi


# processing

def get_actuator_area(model_options, parent, variables, parameters):

    b_ref = parameters['theta0','geometry','b_ref']
    bar_varrho_var = get_bar_varrho_var(model_options, variables, parent)

    radius = bar_varrho_var * b_ref
    annulus_area = 2. * np.pi * b_ref * radius

    area = annulus_area

    return area

def get_kite_radial_vector(model_options, kite, variables, architecture):
    radius_vec = get_kite_radius_vector(model_options, kite, variables, architecture)
    rhat = vect_op.smooth_normalize(radius_vec)
    return rhat

def get_kite_radius_proj_squared(model_options, kite, variables, architecture):
    radius_vec = get_kite_radius_vector(model_options, kite, variables, architecture)

    parent = architecture.parent_map[kite]
    nhat_vec = get_nhat_var(variables, parent)

    hypotenuse_squared = cas.mtimes(radius_vec.T, radius_vec)
    normal_squared = cas.mtimes(radius_vec.T, nhat_vec)**2.

    radius_proj_squared = hypotenuse_squared - normal_squared

    return radius_proj_squared

def get_kite_radius(model_options, kite, variables, architecture):

    # radius_vec = get_kite_radius_vector(model_options, kite, variables, architecture)
    # radius = vect_op.smooth_norm(radius_vec)

    radius_squared = get_kite_radius_proj_squared(model_options, kite, variables, architecture)
    radius = vect_op.smooth_sqrt(radius_squared)

    return radius

def get_average_radius(model_options, variables, parent, architecture):
    children = architecture.kites_map[parent]
    number_children = float(len(children))

    average_radius = 0.
    for kite in children:
        radius = get_kite_radius(model_options, kite, variables, architecture)

        average_radius = average_radius + radius / number_children

    return average_radius

def get_bar_varrho_val(variables, parent, architecture):
    children = architecture.kites_map[parent]
    number_children = float(len(children))

    sum_varrho = 0.
    for kite in children:
        varrho_kite = get_varrho_var(variables, kite, architecture)
        sum_varrho = sum_varrho + varrho_kite

    bar_varrho_val = sum_varrho / number_children
    return bar_varrho_val

def approximate_tip_radius(model_options, variables, kite, architecture, tip, parameters):

    b_ref = parameters['theta0','geometry','b_ref']
    half_span_proj = b_ref / 2.
    parent = architecture.parent_map[kite]

    radial_vector = get_kite_radial_vector(model_options, kite, variables, architecture)

    if int(model_options['kite_dof']) == 6:

        r_column = variables['xd']['r' + str(kite) + str(parent)]
        r = cas.reshape(r_column, (3, 3))
        ehat2 = r[:, 1]  # spanwise, from pe to ne

        ehat2_proj_radial = vect_op.smooth_abs(cas.mtimes(radial_vector.T, ehat2))

        half_span_proj = b_ref * ehat2_proj_radial / 2.

    radius = get_kite_radius(model_options, kite, variables, architecture)

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