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
import pdb

import numpy as np
import casadi.tools as cas
import awebox.tools.vector_operations as vect_op
import awebox.tools.print_operations as print_op
import awebox.tools.struct_operations as struct_op
from awebox.logger.logger import Logger as awelogger


def print_warning_if_relevant(variables, architecture):
    kite = architecture.kite_nodes[0]
    parent = architecture.parent_map[kite]
    warning_is_relevant = not variables_have_third_derivative_information(variables, kite, parent)

    if warning_is_relevant:
        message = 'frenet geometry does not have enough information (the third derivative d^3 gamma(t) / dt^3) to correctly compute the trajectory torsion. this may lead to distorted calculations, for example of the center velocity, if the trajectory torsion is non-negligible.'
        awelogger.logger.warning(message)
    return None

def get_center_position(parent, variables, architecture):

    x_center = cas.DM.zeros((3, 1))
    for kite in architecture.children_map[parent]:
        q_kite = get_gamma(variables, kite, parent)

        local_radius = get_radius_of_curvature(variables, kite, parent)
        local_ehat_2 = get_trajectory_normal_unit_vector(variables, kite, parent)
        vec_from_kite_to_center = local_radius * local_ehat_2

        local_center = q_kite + vec_from_kite_to_center
        x_center += local_center / float(architecture.get_number_children(parent))

    return x_center

def get_center_velocity(parent, variables, architecture):

    dx_center = cas.DM.zeros((3, 1))
    for kite in architecture.children_map[parent]:
        q_kite = get_gamma(variables, kite, parent)
        dq_kite = get_dgamma_dt(variables, kite, parent)
        # gamma_dot = dq_kite

        local_radius = get_radius_of_curvature(variables, kite, parent)
        d_local_radius_dt = get_d_radius_of_curvature_dt(variables, kite, parent)

        curvature = 1./local_radius
        torsion = get_torsion(variables, kite, parent)

        ehat_1 = get_trajectory_tangent_unit_vector(variables, kite, parent)
        ehat_2 = get_trajectory_normal_unit_vector(variables, kite, parent)
        ehat_3 = get_trajectory_binormal_unit_vector(variables, kite, parent)

        # frenet-serret
        d_ehat_2_ds = -1. * curvature * ehat_1 + torsion * ehat_3
        d_ehat_2_dt = vect_op.smooth_norm(dq_kite) * d_ehat_2_ds
        d_vec_from_kite_to_center_dt = d_local_radius_dt * ehat_2 + local_radius * d_ehat_2_dt

        d_local_center_dt = dq_kite + d_vec_from_kite_to_center_dt
        dx_center += d_local_center_dt / float(architecture.get_number_children(parent))

    return dx_center

def get_d_radius_of_curvature_dt(variables_si, kite, parent):

    gamma_dot = get_dgamma_dt(variables_si, kite, parent)
    gamma_ddot = get_ddgamma_ddt(variables_si, kite, parent)
    gamma_dddot = get_dddgamma_dddt(variables_si, kite, parent)

    # from frenet vectors, see wiki page 'curvature'
    # kappa = Norm[gamma' cross gamma''] / Norm[gamma']^3
    # r = 1/kappa

    num_kappa = vect_op.smooth_norm(vect_op.cross(gamma_dot, gamma_ddot))
    den_kappa = vect_op.smooth_norm(gamma_dot)**3.

    cross_vec = vect_op.cross(gamma_dot, gamma_ddot)
    # note that (gamma_ddot x gamma_ddot) = vec 0
    d_cross_vec_dt = vect_op.cross(gamma_dot, gamma_dddot)
    cross_dot_itself = cas.mtimes(cross_vec.T, cross_vec)
    d_cross_dot_itself_dt = 2. * cas.mtimes(cross_vec.T, d_cross_vec_dt)
    d_num_kappa_dt = 0.5 * d_cross_dot_itself_dt / vect_op.smooth_sqrt(cross_dot_itself)

    gamma_dot_dot_itself = cas.mtimes(gamma_dot.T, gamma_dot)
    d_norm_gamma_dot_dt = 2. * cas.mtimes(gamma_dot.T, gamma_ddot)
    d_den_kappa_dt = (3./2.) * vect_op.smooth_sqrt(gamma_dot_dot_itself) * d_norm_gamma_dot_dt

    num = den_kappa
    den = num_kappa
    d_num_dt = d_den_kappa_dt
    d_den_dt = d_num_kappa_dt

    d_radius_dt = (den * d_num_dt - num * d_den_dt) / den**2.

    return d_radius_dt

def get_torsion(variables_si, kite, parent):

    gamma_dot = get_dgamma_dt(variables_si, kite, parent)
    gamma_ddot = get_ddgamma_ddt(variables_si, kite, parent)
    gamma_dddot = get_dddgamma_dddt(variables_si, kite, parent)

    # see wiki page 'frenet-serret formulas'
    # tau = gamma''' \cdot (gamma' \cross gamma'') / Norm[gamma' cross gamma'']^2

    epsilon = 1.e-8

    cross = vect_op.cross(gamma_dot, gamma_ddot)
    num_tau = cas.mtimes(gamma_dddot.T, cross)
    den_tau = cas.mtimes(cross.T, cross) + epsilon

    tau = num_tau / den_tau

    return tau

def get_radius_of_curvature(variables_si, kite, parent):

    gamma_dot = get_dgamma_dt(variables_si, kite, parent)
    gamma_ddot = get_ddgamma_ddt(variables_si, kite, parent)

    # see wiki page 'frenet-serret formulas'
    # kappa = Norm[gamma' cross gamma''] / Norm[gamma']^3
    # r = 1/kappa

    num_kappa = vect_op.smooth_norm(vect_op.cross(gamma_dot, gamma_ddot))
    den_kappa = vect_op.smooth_norm(gamma_dot)**3.

    num = den_kappa
    den = num_kappa

    radius = num / den
    return radius

def get_gamma(variables, kite, parent):
    gamma = struct_op.get_variable_from_model_or_reconstruction(variables, 'xd', 'q' + str(kite) + str(parent))
    return gamma

def get_dgamma_dt(variables, kite, parent):
    dgamma_dt = struct_op.get_variable_from_model_or_reconstruction(variables, 'xd', 'dq' + str(kite) + str(parent))
    return dgamma_dt

def get_ddgamma_ddt(variables, kite, parent):
    ddgamma_ddt = struct_op.get_variable_from_model_or_reconstruction(variables, 'xddot', 'ddq' + str(kite) + str(parent))
    return ddgamma_ddt

def variables_have_third_derivative_information(variables, kite, parent):
    label = 'dddq' + str(kite) + str(parent) + ',0'
    if (isinstance(variables, dict)) and ('xddot' in variables.keys()):
        return ('[' + label + ']' in variables['xddot'].labels())
    if (isinstance(variables, dict)) and not ('xddot' in variables.keys()):
        message = 'cannot determine if the variables have third degree-of-freedom information'
        awelogger.logger.error(message)
        raise Exception(message)
    else:
        return ('[xddot,' + label + ']' in variables.labels())

def get_dddgamma_dddt(variables_si, kite, parent):
    if variables_have_third_derivative_information(variables_si, kite, parent):
        dddgamma_dddt = struct_op.get_variable_from_model_or_reconstruction(variables_si, 'xddot',
                                                                          'dddq' + str(kite) + str(parent))
    else:
        dddgamma_dddt = cas.DM.zeros((3, 1))
    return dddgamma_dddt

def get_trajectory_tangent_unit_vector(variables, kite, parent):
    # e1 = gamma'
    # ehat_1 = e1 / || e1 ||

    dgamma_dt = get_dgamma_dt(variables, kite, parent)
    vec_e1 = dgamma_dt
    ehat_1 = vect_op.smooth_normalize(vec_e1)
    return ehat_1

def get_trajectory_normal_unit_vector(variables, kite, parent):
    # ehat_2 = (gamma' cross (gamma'' cross gamma')) / ( Norm[gamma'] Norm[gamma'' cross gamma'] )

    dgamma_dt = get_dgamma_dt(variables, kite, parent)
    ddgamma_ddt = get_ddgamma_ddt(variables, kite, parent)

    interior_cross = vect_op.cross(ddgamma_ddt, dgamma_dt)

    num = vect_op.cross(dgamma_dt, interior_cross)
    den = vect_op.smooth_norm(dgamma_dt) * vect_op.smooth_norm(interior_cross)

    ehat_2 = num / den

    return ehat_2

def get_trajectory_binormal_unit_vector(variables, kite, parent):

    ehat_1 = get_trajectory_tangent_unit_vector(variables, kite, parent)
    ehat_2 = get_trajectory_normal_unit_vector(variables, kite, parent)
    ehat_3 = vect_op.smooth_normed_cross(ehat_1, ehat_2)

    return ehat_3