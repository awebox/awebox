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


import awebox.mdl.aero.induction_dir.actuator_dir.geom as actuator_geom
import awebox.mdl.aero.induction_dir.actuator_dir.flow as actuator_flow
import awebox.mdl.aero.induction_dir.actuator_dir.force as actuator_force

import awebox.tools.vector_operations as vect_op
import awebox.tools.print_operations as print_op

def get_LL_matrix_val(model_options, atmos, wind, variables, outputs, parameters, parent, architecture, label):
    corr = actuator_flow.get_corr_val(model_options, atmos, wind, variables, outputs, parameters, parent, architecture, label)
    chi = actuator_flow.get_wake_angle_chi(model_options, atmos, wind, variables, outputs, parameters, parent, architecture, label)
    tanhalfchi = cas.tan(chi / 2.)
    sechalfchi = 1. / cas.cos(chi / 2.)

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
    LL_matr = cas.vertcat(LL_row1, LL_row2, LL_row3)

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

def get_ct_val(model_options, atmos, wind, variables, outputs, parameters, parent, architecture):
    thrust = actuator_force.get_actuator_thrust(model_options, variables, parameters, outputs, parent, architecture)
    area = actuator_geom.get_actuator_area(model_options, parent, variables, parameters)
    qzero = actuator_flow.get_actuator_dynamic_pressure(model_options, atmos, wind, variables, parent, architecture)

    ct = thrust / area / qzero

    return ct


def get_actuator_moment_y_rotor(model_options, variables, outputs, parent, architecture):

    total_moment_aero = actuator_force.get_actuator_moment(model_options, variables, outputs, parent, architecture)
    y_rotor = actuator_geom.get_y_rotor_hat_var(variables, parent)
    moment = cas.mtimes(total_moment_aero.T, y_rotor)

    return moment

def get_actuator_moment_z_rotor(model_options, variables, outputs, parent, architecture):

    total_moment_aero = actuator_force.get_actuator_moment(model_options, variables, outputs, parent, architecture)
    z_rotor = actuator_geom.get_z_rotor_hat_var(variables, parent)
    moment = cas.mtimes(total_moment_aero.T, z_rotor)

    return moment




# references
def get_ct_ref(model_options):
    a_ref = actuator_flow.get_a_ref(model_options)
    ct_ref = 4. * a_ref * (1. - a_ref)

    return ct_ref


def get_thrust_ref(model_options, atmos, wind, parameters):

    qzero_ref = actuator_flow.get_qzero_ref(atmos, wind)
    area_ref = actuator_geom.get_area_ref(model_options, parameters)
    ct_ref = get_ct_ref(model_options)

    thrust_ref = ct_ref * qzero_ref * area_ref

    scaling = model_options['aero']['actuator']['scaling']
    reference = scaling * thrust_ref

    return reference


def get_moment_ref(model_options, atmos, wind, parameters):

    qzero_ref = actuator_flow.get_qzero_ref(atmos, wind)
    area_ref = actuator_geom.get_area_ref(model_options, parameters)
    bar_varrho_ref = actuator_geom.get_varrho_ref(model_options)
    b_ref = parameters['theta0', 'geometry', 'b_ref']
    bar_radius = (bar_varrho_ref) * b_ref

    moment = qzero_ref * area_ref * bar_radius

    return moment


def get_t_star(variables, parameters, parent):
    # radius / u_0 = [m] / [m/s]
    t_star_num = get_t_star_numerator_val(variables, parameters, parent)
    t_star_den = get_t_star_denominator_val(variables, parent)
    return t_star_num / t_star_den

def get_t_star_numerator_val(variables, parameters, parent):
    b_ref = parameters['theta0', 'geometry', 'b_ref']
    bar_varrho_var = actuator_geom.get_bar_varrho_var(variables, parent)
    t_star_num = b_ref * (bar_varrho_var + 0.5)
    return t_star_num

def get_t_star_numerator_ref(model_options, parameters):
    varrho_ref = actuator_geom.get_varrho_ref(model_options)
    b_ref = parameters['theta0', 'geometry', 'b_ref']
    t_star_num = b_ref * (varrho_ref + 0.5)
    return t_star_num

def get_t_star_denominator_val(variables, parent):
    uzero_mag = actuator_flow.get_uzero_vec_length_var(variables, parent)
    t_star_den = uzero_mag
    return t_star_den

def get_t_star_denominator_ref(wind):
    t_star_den_ref = actuator_flow.get_uzero_vec_length_ref(wind)
    return t_star_den_ref


def get_c_all_components(model_options, atmos, wind, variables, parameters, outputs, parent, architecture):
    thrust = actuator_force.get_actuator_thrust(model_options, variables, parameters, outputs, parent, architecture)
    moment_y_val = get_actuator_moment_y_rotor(model_options, variables, outputs, parent, architecture)
    moment_z_val = get_actuator_moment_z_rotor(model_options, variables, outputs, parent, architecture)

    area = actuator_geom.get_actuator_area(model_options, parent, variables, parameters)
    qzero = actuator_flow.get_actuator_dynamic_pressure(model_options, atmos, wind, variables, parent, architecture)

    bar_varrho_var = actuator_geom.get_bar_varrho_var(variables, parent)
    b_ref = parameters['theta0', 'geometry', 'b_ref']
    radius_bar =  bar_varrho_var * b_ref

    thrust_denom = area * qzero
    moment_denom = thrust_denom * radius_bar

    thrust_radius = thrust * radius_bar
    c_all = cas.vertcat(thrust_radius, moment_y_val, moment_z_val)

    return c_all, moment_denom
