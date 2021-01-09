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
specific aerodynamics for a 6dof kite
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2017-20
'''

import casadi.tools as cas

import awebox.tools.vector_operations as vect_op
import awebox.tools.constraint_operations as cstr_op
import awebox.tools.print_operations as print_op


import awebox.mdl.mdl_constraint as mdl_constraint

import awebox.mdl.aero.indicators as indicators
import awebox.mdl.aero.kite_dir.stability_derivatives as stability_derivatives
import awebox.mdl.aero.kite_dir.frames as frames
import awebox.mdl.aero.kite_dir.tools as tools

from awebox.logger.logger import Logger as awelogger





def get_kite_dcm(kite, variables, architecture):
    parent = architecture.parent_map[kite]
    kite_dcm = cas.reshape(variables['xd']['r' + str(kite) + str(parent)], (3, 3))
    return kite_dcm


def get_force_and_moment_vector(options, variables, atmos, wind, architecture, parameters, kite, outputs):

    surface_control = options['surface_control']

    parent = architecture.parent_map[kite]

    kite_dcm = cas.reshape(variables['xd']['r' + str(kite) + str(parent)], (3, 3))

    vec_u_earth = tools.get_local_air_velocity_in_earth_frame(options, variables, wind, kite, kite_dcm, architecture,
                                                              parameters, outputs)

    if int(surface_control) == 0:
        delta = variables['u']['delta' + str(kite) + str(parent)]
    elif int(surface_control) == 1:
        delta = variables['xd']['delta' + str(kite) + str(parent)]

    omega = variables['xd']['omega' + str(kite) + str(parent)]
    q = variables['xd']['q' + str(kite) + str(parent)]
    rho = atmos.get_density(q[2])
    force_info, moment_info = get_force_and_moment(options, parameters, vec_u_earth, kite_dcm, omega, delta, rho)

    f_found_frame = force_info['frame']
    f_found_vector = force_info['vector']

    m_found_frame = moment_info['frame']
    m_found_vector = moment_info['vector']

    return f_found_vector, f_found_frame, m_found_vector, m_found_frame, vec_u_earth, kite_dcm


def get_force_cstr(options, variables, atmos, wind, architecture, parameters, outputs):

    f_scale = tools.get_f_scale(parameters, options)
    m_scale = tools.get_m_scale(parameters, options)

    cstr_list = mdl_constraint.MdlConstraintList()

    for kite in architecture.kite_nodes:

        parent = architecture.parent_map[kite]

        f_found_vector, f_found_frame, m_found_vector, m_found_frame, vec_u_earth, kite_dcm = get_force_and_moment_vector(
            options, variables, atmos, wind, architecture, parameters, kite, outputs)

        forces_dict = tools.get_framed_forces(vec_u_earth, kite_dcm, variables, kite, architecture)
        f_var_frame = tools.force_variable_frame()
        f_var = forces_dict[f_var_frame]
        f_val = frames.from_named_frame_to_named_frame(from_name=f_found_frame,
                                                       to_name=f_var_frame,
                                                       vec_u=vec_u_earth,
                                                       kite_dcm=kite_dcm,
                                                       vector=f_found_vector)

        moments_dict = tools.get_framed_moments(vec_u_earth, kite_dcm, variables, kite, architecture)
        m_var_frame = tools.moment_variable_frame()
        m_var = moments_dict[m_var_frame]
        m_val = frames.from_named_frame_to_named_frame(from_name=m_found_frame,
                                                       to_name=m_var_frame,
                                                       vec_u=vec_u_earth,
                                                       kite_dcm=kite_dcm,
                                                       vector=m_found_vector)

        resi_f_kite = (f_var - f_val) / f_scale
        resi_m_kite = (m_var - m_val) / m_scale

        f_kite_cstr = cstr_op.Constraint(expr=resi_f_kite,
                                       name='f_aero' + str(kite) + str(parent),
                                       cstr_type='eq')
        cstr_list.append(f_kite_cstr)

        m_kite_cstr = cstr_op.Constraint(expr=resi_m_kite,
                                       name='m_aero' + str(kite) + str(parent),
                                       cstr_type='eq')
        cstr_list.append(m_kite_cstr)

    return cstr_list


def get_force_and_moment(options, parameters, vec_u_earth, kite_dcm, omega, delta, rho):

    # we use the vec_u_earth and the kite_dcm to give the relative orientation.
    # this means, that they must be in the same frame. otherwise, the frame of
    # the wind vector is not used in this function.

    alpha = indicators.get_alpha(vec_u_earth, kite_dcm)
    beta = indicators.get_beta(vec_u_earth, kite_dcm)

    airspeed = vect_op.norm(vec_u_earth)
    force_coeff_info, moment_coeff_info = stability_derivatives.stability_derivatives(options, alpha, beta,
                                                                                      airspeed, omega,
                                                                                      delta, parameters)

    force_info = {}
    moment_info = {}

    force_info['frame'] = force_coeff_info['frame']
    moment_info['frame'] = moment_coeff_info['frame']

    CF = force_coeff_info['coeffs']
    CM = moment_coeff_info['coeffs']

    # notice that magnitudes don't change under rotation
    dynamic_pressure = 1. / 2. * rho * cas.mtimes(vec_u_earth.T, vec_u_earth)
    planform_area = parameters['theta0', 'geometry', 's_ref']

    force = CF * dynamic_pressure * planform_area
    force_info['vector'] = force

    b_ref = parameters['theta0', 'geometry', 'b_ref']
    c_ref = parameters['theta0', 'geometry', 'c_ref']
    reference_lengths = cas.diag(cas.vertcat(b_ref, c_ref, b_ref))

    moment = dynamic_pressure * planform_area * cas.mtimes(reference_lengths, CM)
    moment_info['vector'] = moment

    return force_info, moment_info





def get_wingtip_position(kite, model, variables, parameters, ext_int):
    parent_map = model.architecture.parent_map

    xd = model.variables_dict['xd'](variables['xd'])

    if ext_int == 'ext':
        span_sign = 1.
    elif ext_int == 'int':
        span_sign = -1.
    else:
        awelogger.logger.error('wing side not recognized for 6dof kite.')

    parent = parent_map[kite]

    name = 'q' + str(kite) + str(parent)
    q_unscaled = xd[name]
    scale = model.scaling['xd'][name]
    q = q_unscaled * scale

    kite_dcm = cas.reshape(xd['kite_dcm' + str(kite) + str(parent)], (3, 3))
    ehat_span = kite_dcm[:, 1]

    b_ref = parameters['theta0','geometry','b_ref']

    wingtip_position = q + ehat_span * span_sign * b_ref / 2.

    return wingtip_position
