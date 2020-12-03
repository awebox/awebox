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
import awebox.mdl.aero.indicators as indicators
import numpy as np

import awebox.mdl.aero.kite_dir.stability_derivatives as stability_derivatives
import awebox.mdl.aero.kite_dir.frames as frames
import awebox.mdl.aero.kite_dir.tools as tools

from awebox.logger.logger import Logger as awelogger
import awebox.tools.print_operations as print_op

import copy


def arbitrarily_desired_force_frame():
    return 'earth'

def get_kite_dcm(kite, variables, architecture):
    parent = architecture.parent_map[kite]
    kite_dcm = cas.reshape(variables['xd']['r' + str(kite) + str(parent)], (3, 3))
    return kite_dcm

def get_framed_forces(vec_u, options, variables, kite, architecture, parameters):

    kite_dcm = get_kite_dcm(kite, variables, architecture)

    parent = architecture.parent_map[kite]

    # frame_name = options['aero']['stab_derivs']['force_frame']
    frame_name = arbitrarily_desired_force_frame()

    f_aero_var = tools.get_f_aero_var(variables, kite, parent, parameters, options)

    f_aero_body = frames.from_named_frame_to_body(frame_name, vec_u, kite_dcm, f_aero_var)
    f_aero_wind = frames.from_named_frame_to_wind(frame_name, vec_u, kite_dcm, f_aero_var)
    f_aero_control = frames.from_named_frame_to_control(frame_name, vec_u, kite_dcm, f_aero_var)
    f_aero_earth = frames.from_named_frame_to_earth(frame_name, vec_u, kite_dcm, f_aero_var)

    dict = {'body':f_aero_body, 'control': f_aero_control, 'wind': f_aero_wind, 'earth': f_aero_earth}

    return dict

def get_framed_moments(vec_u, options, variables, kite, architecture, parameters):

    kite_dcm = get_kite_dcm(kite, variables, architecture)

    parent = architecture.parent_map[kite]

    frame_name = options['aero']['stab_derivs']['moment_frame']
    m_aero_var = tools.get_m_aero_var(variables, kite, parent, parameters, options)

    m_aero_body = frames.from_named_frame_to_body(frame_name, vec_u, kite_dcm, m_aero_var)

    dict = {'body':m_aero_body}

    return dict


def get_force_resi(options, variables, atmos, wind, architecture, parameters):

    surface_control = options['surface_control']

    f_scale = tools.get_f_scale(parameters, options)
    m_scale = tools.get_m_scale(parameters, options)

    resi = []
    for kite in architecture.kite_nodes:

        parent = architecture.parent_map[kite]
        f_aero_var = tools.get_f_aero_var(variables, kite, parent, parameters, options)
        m_aero_var = tools.get_m_aero_var(variables, kite, parent, parameters, options)

        if int(surface_control) == 0:
            delta = variables['u']['delta' + str(kite) + str(parent)]
        elif int(surface_control) == 1:
            delta = variables['xd']['delta' + str(kite) + str(parent)]

        omega = variables['xd']['omega' + str(kite) + str(parent)]
        kite_dcm = cas.reshape(variables['xd']['r' + str(kite) + str(parent)], (3, 3))

        q = variables['xd']['q' + str(kite) + str(parent)]
        rho = atmos.get_density(q[2])

        vec_u_earth = tools.get_local_air_velocity_in_earth_frame(options, variables, atmos, wind, kite, kite_dcm,
                                                             architecture, parameters)

        force_info, moment_info = get_force_and_moment(options, parameters, vec_u_earth, kite_dcm, omega, delta, rho)

        arb_force_frame = arbitrarily_desired_force_frame()
        force_arb_info = copy.deepcopy(force_info)
        force_arb_info['vector'] = frames.from_named_frame_to_named_frame(force_info['frame'], arb_force_frame, vec_u_earth, kite_dcm, force_info['vector'])
        force_arb_info['frame'] = arb_force_frame

        f_aero_val = force_arb_info['vector']
        m_aero_val = moment_info['vector']

        resi_f_kite = (f_aero_var - f_aero_val) / f_scale
        resi_m_kite = (m_aero_var - m_aero_val) / m_scale

        resi = cas.vertcat(resi, resi_f_kite, resi_m_kite)

    return resi


def get_force_and_moment(options, parameters, vec_u, kite_dcm, omega, delta, rho):

    alpha = indicators.get_alpha(vec_u, kite_dcm)
    beta = indicators.get_beta(vec_u, kite_dcm)

    airspeed = vect_op.norm(vec_u)
    force_coeff_info, moment_coeff_info = stability_derivatives.stability_derivatives(options, alpha, beta,
                                                                                      airspeed, omega,
                                                                                      delta, parameters)

    force_info = {}
    moment_info = {}

    force_info['frame'] = force_coeff_info['frame']
    moment_info['frame'] = moment_coeff_info['frame']

    CF = force_coeff_info['coeffs']
    CM = moment_coeff_info['coeffs']

    dynamic_pressure = 1. / 2. * rho * cas.mtimes(vec_u.T, vec_u)
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
