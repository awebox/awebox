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
general flow functions for the induction model
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2017-21
- edit: jochem de schutter, alu-fr 2019
"""
import pdb

import awebox.mdl.aero.induction_dir.general_dir.geom as general_geom

import numpy as np
import casadi.tools as cas
from awebox.logger.logger import Logger as awelogger
import awebox.tools.print_operations as print_op
import awebox.tools.vector_operations as vect_op

def get_kite_apparent_velocity(variables, wind, kite, parent):
    u_infty = get_kite_uinfy_vec(variables, wind, kite, parent)
    u_kite = variables['xd']['dq' + str(kite) + str(parent)]
    u_app_kite = u_infty - u_kite

    return u_app_kite

def get_kite_uinfy_vec(variables, wind, kite, parent):
    q_kite = variables['xd']['q' + str(kite) + str(parent)]
    u_infty = wind.get_velocity(q_kite[2])
    return u_infty

def get_uzero_vec(model_options, wind, parent, variables, architecture):

    u_infty = get_actuator_freestream_velocity(model_options, wind, parent, variables, architecture)
    u_actuator = general_geom.get_center_velocity(parent, variables, architecture)

    u_apparent = u_infty - u_actuator

    return u_apparent


def compute_induction_factor(vec_u_ind, n_hat, u_normalizing):
    u_projected = cas.mtimes(vec_u_ind.T, n_hat)
    a_calc = -1. * u_projected / u_normalizing

    return a_calc


def get_f_val(model_options, wind, parent, variables, architecture):
    dl_t = variables['xd']['dl_t']
    u_infty = get_actuator_freestream_velocity(model_options, wind, parent, variables, architecture)
    f_val = dl_t / vect_op.smooth_norm(u_infty)

    return f_val

def get_actuator_freestream_velocity(model_options, wind, parent, variables, architecture):

    center = general_geom.get_center_point(model_options, parent, variables, architecture)
    u_infty = wind.get_velocity(center[2])

    return u_infty


def get_far_wake_cylinder_pitch(l_hat, vec_u_zero, total_circulation, average_period_of_rotation):
    sum_of_kite_circulations_in_layer = total_circulation
    period_of_rotation = average_period_of_rotation

    # pitch = distance traveled of (one layer's) far-wake during one rotation
    # pitch = period_of_rotation * (u_zero_in_longitudinal_direction + contribution_of_all_vortex_cylinders)
    # far downstream, a tangential vortex cylinder of strength gamma_tan induces a velocity of gamma_tan within the cylinder
    # we consider only within the streamtube, so where only the postive-edge/exterior cylinders are located. they have strength gamma_tan = kite_circulation / pitch
    # pitch = period_of_rotation * (u_zero_in_longitudinal_direction + Sum[gamma_tan, all kites in layer])
    # pitch = period_of_rotation * (u_zero_in_longitudinal_direction + Sum[kite_circulation / pitch, all kites in layer])
    # pitch^2 = pitch * period_of_rotation * u_zero_in_longitudinal_direction - period_of_rotation * sum_of_kite_circulations_in_layer
    # (1) * pitch^2 - (period_of_rotation * u_zero_in_longitudinal_direction) * pitch - (period_of_rotation * sum_of_kite_circulations_in_layer)
    # therefore, solve with quadratic formula
    u_zero_in_longitudinal_direction = cas.mtimes(vec_u_zero.T, l_hat)

    quad_a = 1.
    neg_quad_b = period_of_rotation * u_zero_in_longitudinal_direction
    quad_c = period_of_rotation * sum_of_kite_circulations_in_layer

    # we choose the 'plus' variant of the quadratic formula, so that when circulation = 0, pitch = t * (vec_u_zero \cdot lhat)
    # notice that this corresponds to the 1D case, mentioned in Branlard et al. 2014
    # pitch = (neg_quad_b + sqrt(neg_quad_b ** 2. - 4. * quad_a * quad_c) ) / 2.

    # b^2 - 4 a c = period^2 * u_zero_in_longitudinal_direction^2 - 4 period^2 * frequency * sum_of_kite_circulations_in_layer
    # = period^2 (u_zero_in_longitudinal_direction^2 - 4 * frequency_of_rotation * sum_of_kite_circulations_in_layer)
    pitch = (neg_quad_b + period_of_rotation * vect_op.smooth_abs(neg_quad_b ** 2. - 4. * quad_a * quad_c) ** 0.5) / 2.

    return pitch

def get_far_wake_cylinder_residual(pitch, l_hat, vec_u_zero, total_circulation, average_period_of_rotation):
    sum_of_kite_circulations_in_layer = total_circulation
    period_of_rotation = average_period_of_rotation

    # pitch = distance traveled of (one layer's) far-wake during one rotation
    # pitch = period_of_rotation * (u_zero_in_longitudinal_direction + contribution_of_all_vortex_cylinders)
    # far downstream, a tangential vortex cylinder of strength gamma_tan induces a velocity of gamma_tan within the cylinder
    # we consider only within the streamtube, so where only the postive-edge/exterior cylinders are located. they have strength gamma_tan = kite_circulation / pitch
    # pitch = period_of_rotation * (u_zero_in_longitudinal_direction + Sum[gamma_tan, all kites in layer])
    # pitch = period_of_rotation * (u_zero_in_longitudinal_direction + Sum[kite_circulation / pitch, all kites in layer])
    # pitch^2 = pitch * period_of_rotation * (vec_u_zero \cdot lhat) - period_of_rotation * sum_of_kite_circulations_in_layer
    # therefore, solve with quadratic formula
    u_zero_in_longitudinal_direction = cas.mtimes(vec_u_zero.T, l_hat)

    resi = pitch**2. - (pitch * period_of_rotation * u_zero_in_longitudinal_direction) + (period_of_rotation * sum_of_kite_circulations_in_layer)
    return resi

