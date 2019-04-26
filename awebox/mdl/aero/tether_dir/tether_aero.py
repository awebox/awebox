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
tether aerodynamics model of an awe system
takes states, finds approximate total force and moment for a tether element
finds equivalent forces corresponding to the total force and moment.
_python-3.5 / casadi-3.4.5
- author: elena malz, chalmers 2016
- edited: rachel leuthold, jochem de schutter alu-fr 2017
'''

import casadi.tools as cas
import numpy as np

import awebox.tools.vector_operations as vect_op

def get_forces(model_options, variables, atmos, wind, n, cd_tether_fun, outputs, parameters, architecture):

    if 'tether_aero' not in list(outputs.keys()):
        outputs['tether_aero'] = {}

    [trivial_lower, trivial_upper] = get_trivial_forces(model_options, variables, atmos, wind, n, parameters,architecture)
    [physical_lower, physical_upper] = get_physical_forces(model_options, variables, atmos, wind, n, cd_tether_fun, architecture)
    [simple_lower, simple_upper] = get_simple_forces(model_options, variables, atmos, wind, n, cd_tether_fun, architecture)

    [diam, q_upper, q_lower, dq_upper, dq_lower, ua_upper, ua_lower] = get_upper_lower_q_and_dq(model_options,
                                                                                                  variables, wind, n, architecture)

    ua = (ua_upper + ua_lower)/2.
    reynolds = get_reynolds_number(atmos, ua, diam, q_upper, q_lower)


    outputs['tether_aero']['physical_upper' + str(n)] = physical_upper
    outputs['tether_aero']['physical_lower' + str(n)] = physical_lower
    outputs['tether_aero']['simple_upper' + str(n)] = simple_upper
    outputs['tether_aero']['simple_lower' + str(n)] = simple_lower
    outputs['tether_aero']['trivial_upper' + str(n)] = trivial_upper
    outputs['tether_aero']['trivial_lower' + str(n)] = trivial_lower
    outputs['tether_aero']['reynolds' + str(n)] = reynolds

    return outputs

def get_trivial_forces(model_options, variables, atmos, wind, n, parameters, architecture):

    [diam, q_upper, q_lower, dq_upper, dq_lower, ua_upper, ua_lower] = get_upper_lower_q_and_dq(model_options,
                                                                                                  variables, wind, n,architecture)

    length = vect_op.norm(q_upper - q_lower)
    q_average = 0.5 * (q_upper + q_lower)
    dq_average = 0.5 * (dq_upper + dq_lower)
    rho = atmos.get_density(q_average[2])

    u_a = wind.get_velocity(q_average[2]) - dq_average

    cd = parameters['theta0','tether','cd']
    drag_force = cd * 0.5 * rho * vect_op.smooth_norm(u_a, 1e-6) * u_a * diam * length

    force_upper = drag_force / 2.
    force_lower = drag_force / 2.

    return [force_lower, force_upper]

def get_simple_forces(model_options, variables, atmos, wind, n, cd_tether_fun, architecture):

    [diam, q_upper, q_lower, dq_upper, dq_lower, ua_upper, ua_lower] = get_upper_lower_q_and_dq(model_options,
                                                                                                  variables, wind, n,architecture)

    drag = get_segment_force(diam, q_upper, q_lower, dq_upper, dq_lower, atmos, wind, cd_tether_fun)

    force_upper = drag/2.
    force_lower = drag/2.

    return [force_lower, force_upper]

def get_physical_forces(model_options, variables, atmos, wind, n, cd_tether_fun, architecture):

    [diam, q_upper, q_lower, dq_upper, dq_lower, ua_upper, ua_lower] = get_upper_lower_q_and_dq(model_options, variables, wind, n,architecture)

    [force_upper, force_lower] = get_equivalent_tether_drag_forces(model_options, diam, q_upper, q_lower, dq_upper, dq_lower, atmos, wind, cd_tether_fun)

    return [force_lower, force_upper]

def get_upper_lower_q_and_dq(model_options, variables, wind, n, architecture):

    parent_map = architecture.parent_map
    parent = parent_map[n]

    xd = variables['xd']
    theta = variables['theta']

    q_upper = xd['q' + str(n) + str(parent)]
    dq_upper = xd['dq' + str(n) + str(parent)]
    u_infty_upper = wind.get_velocity(q_upper[2])

    if n > 1:
        grandparent = parent_map[parent]
        q_lower = xd['q' + str(parent) + str(grandparent)]
        dq_lower = xd['dq' + str(parent) + str(grandparent)]

        diam = theta['diam_s']

        u_infty_lower = wind.get_velocity(q_lower[2])

    else:
        q_lower = np.zeros((3, 1))
        dq_lower = np.zeros((3, 1))

        diam = theta['diam_t']

        u_infty_lower = np.zeros((3, 1))

    ua_upper = u_infty_upper - dq_upper
    ua_lower = u_infty_lower - dq_lower

    return diam, q_upper, q_lower, dq_upper, dq_lower, ua_upper, ua_lower

def get_reynolds_number(atmos, ua, diam, q_upper, q_lower):

    q_average = (q_upper + q_lower) / 2.
    zz = q_average[2]
    rho_infty = atmos.get_density(zz)
    mu_infty = atmos.get_viscosity(zz)

    norm_ua = cas.mtimes(ua.T, ua) ** 0.5

    reynolds = rho_infty * norm_ua * diam / mu_infty

    return reynolds

def get_segment_force(diam, q_upper, q_lower, dq_upper, dq_lower, atmos, wind, cd_tether_fun):

    q_average = (q_upper + q_lower) / 2.
    zz = q_average[2]

    uw_average = wind.get_velocity(zz)
    density = atmos.get_density(zz)

    dq_average = (dq_upper + dq_lower) / 2.
    ua = uw_average - dq_average

    ua_norm = vect_op.smooth_norm(ua, 1e-6)
    ehat_ua = vect_op.smooth_normalize(ua, 1e-6)

    tether = q_upper - q_lower

    length = vect_op.norm(tether)
    length_parallel_to_wind = cas.mtimes(tether.T, ehat_ua)
    length_perp_to_wind = (length**2. - length_parallel_to_wind**2.)**0.5

    reynolds = get_reynolds_number(atmos, ua, diam, q_upper, q_lower)
    cd = cd_tether_fun(reynolds)

    drag = cd * 0.5 * density * ua_norm * diam * length_perp_to_wind * ua

    return drag

def get_body_axes(q_upper, q_lower):

    tether = q_upper - q_lower

    xhat = vect_op.xhat()
    yhat = vect_op.yhat()

    ehat_z = vect_op.normalize(tether)
    ehat_x = vect_op.normed_cross(yhat, tether)
    ehat_y = vect_op.normed_cross(ehat_z, ehat_x)

    return ehat_x, ehat_y, ehat_z

def from_earthfixed_to_body(earthfixed_vector, q_upper, q_lower):

    [ehat_x, ehat_y, ehat_z] = get_body_axes(q_upper, q_lower)

    body_x = cas.mtimes(earthfixed_vector.T, ehat_x)
    body_y = cas.mtimes(earthfixed_vector.T, ehat_y)
    body_z = cas.mtimes(earthfixed_vector.T, ehat_z)

    body_vector = cas.vertcat(body_x, body_y, body_z)

    return body_vector

def from_body_to_earthfixed(body_vector, q_upper, q_lower):

    [ehat_x, ehat_y, ehat_z] = get_body_axes(q_upper, q_lower)

    earthfixed_x = body_vector[0] * ehat_x
    earthfixed_y = body_vector[1] * ehat_y
    earthfixed_z = body_vector[2] * ehat_z

    earthfixed_vector = earthfixed_x + earthfixed_y + earthfixed_z

    return earthfixed_vector

def get_total_drag(model_options, diam, q_upper, q_lower, dq_upper, dq_lower, atmos, wind, cd_tether_fun):

    elem = model_options['tether']['aero_elements']
    q_average = (q_upper + q_lower) / 2.

    total_force = np.zeros((3, 1))
    total_moment = np.zeros((3, 1))

    for idx in range(elem):

        loc_s_upper = float(idx + 1) / float(elem)
        loc_s_lower = float(idx) / float(elem)

        q_loc_upper = q_lower + loc_s_upper * (q_upper - q_lower)
        q_loc_lower = q_lower + loc_s_lower * (q_upper - q_lower)

        q_loc_average = (q_loc_lower + q_loc_upper)/2.
        moment_arm = q_average - q_loc_average

        dq_loc_upper = dq_lower + loc_s_upper * (dq_upper - dq_lower)
        dq_loc_lower = dq_lower + loc_s_lower * (dq_upper - dq_lower)

        loc_force = get_segment_force(diam, q_loc_upper, q_loc_lower, dq_loc_upper, dq_loc_lower, atmos, wind, cd_tether_fun)

        loc_moment = vect_op.cross(moment_arm, loc_force)

        total_force = total_force + loc_force
        total_moment = total_moment + loc_moment

    return [total_force, total_moment]

def get_inverse_equivalence_matrix(tether_length):
    # equivalent forces at upper node = [a, b, c]
    # equivalent forces at lower node = [d, e, f]
    # total forces = [Fx, Fy, Fz]
    # total moment = [Mx, My, 0]

    # a + d = Fx
    # b + e = Fy
    # c + f = Fz
    # L (a - d) = My
    # L (a - e) = Mx
    # c - f = 0

    # A [a, b, c, d, e, f].T = [Fx, Fy, Fz, Mx, My, 0].T
    # [a, b, c, d, e, f].T = Ainv [Fx, Fy, Fz, Mx, My, 0].T

    L = tether_length / 2.

    Ainv = np.matrix([[1., 0., 0., 0., 1./L, 0.],
                      [0., 1., 0., 1./L, 0., 0.],
                      [0., 0., 1., 0., 0., 1./L],
                      [1., 0., 0., 0., -1. / L, 0.],
                      [0., 1., 0., -1. / L, 0., 0.],
                      [0., 0., 1., 0., 0., -1. / L]]) / 2.

    return Ainv


def get_equivalent_tether_drag_forces(model_options, diam, q_upper, q_lower, dq_upper, dq_lower, atmos, wind, cd_tether_fun):

    tether = q_upper - q_lower

    [total_force_earthfixed, total_moment_earthfixed] = get_total_drag(model_options, diam, q_upper, q_lower, dq_upper, dq_lower, atmos, wind, cd_tether_fun)

    total_force_body = from_earthfixed_to_body(total_force_earthfixed, q_upper, q_lower)
    total_moment_body = from_earthfixed_to_body(total_moment_earthfixed, q_upper, q_lower)

    total_moment_body[2] = 0.

    total_vect = cas.vertcat(total_force_body, total_moment_body)

    Ainv = get_inverse_equivalence_matrix(vect_op.norm(tether))

    equiv_vect = cas.mtimes(Ainv, total_vect)

    equiv_force_upper_body = equiv_vect[0:3]
    equiv_force_lower_body = equiv_vect[3:6]

    equiv_force_upper_earthfixed = from_body_to_earthfixed(equiv_force_upper_body, q_upper, q_lower)
    equiv_force_lower_earthfixed = from_body_to_earthfixed(equiv_force_lower_body, q_upper, q_lower)

    return [equiv_force_upper_earthfixed, equiv_force_lower_earthfixed]
