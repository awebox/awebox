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
common tools for all kite models
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2017-20
'''

import casadi.tools as cas
import awebox.mdl.aero.induction_dir.induction as induction
import awebox.mdl.aero.kite_dir.frames as frames
import awebox.mdl.aero.indicators as indicators
import awebox.tools.vector_operations as vect_op
import numpy as np
import awebox.tools.print_operations as print_op
from awebox.logger.logger import Logger as awelogger


##### the force and moment lifted variables

def get_f_aero_var(variables, kite, parent):

    var_type = 'xl'
    var_name = 'f_aero' + str(kite) + str(parent)

    if var_name in variables[var_type].keys():
        f_aero_si = variables[var_type][var_name]
    else:
        awelogger.logger.error('lifted aero forces not found.')

    return f_aero_si

def get_m_aero_var(variables, kite, parent):

    var_type = 'xl'
    var_name = 'm_aero' + str(kite) + str(parent)

    if var_name in variables[var_type].keys():
        m_aero_si = variables[var_type][var_name]
    else:
        awelogger.logger.error('lifted aero moments not found.')

    return m_aero_si

def force_variable_frame():
    return 'earth'

def moment_variable_frame():
    return 'body'

def get_f_scale(parameters, options):
    scale = options['scaling']['xl']['f_aero']
    return scale

def get_m_scale(parameters, options):
    m_scale = options['scaling']['xl']['m_aero']
    return m_scale


def get_framed_forces(vec_u, kite_dcm, variables, kite, architecture, f_vector=None, f_frame=None):

    parent = architecture.parent_map[kite]

    if f_vector is None:
        f_vector = get_f_aero_var(variables, kite, parent)

    if f_frame is None:
        f_frame = force_variable_frame()

    f_aero_body = frames.from_named_frame_to_body(f_frame, vec_u, kite_dcm, f_vector)
    f_aero_wind = frames.from_named_frame_to_wind(f_frame, vec_u, kite_dcm, f_vector)
    f_aero_control = frames.from_named_frame_to_control(f_frame, vec_u, kite_dcm, f_vector)
    f_aero_earth = frames.from_named_frame_to_earth(f_frame, vec_u, kite_dcm, f_vector)

    dict = {'body':f_aero_body, 'control': f_aero_control, 'wind': f_aero_wind, 'earth': f_aero_earth}

    return dict

def get_framed_moments(vec_u, kite_dcm, variables, kite, architecture, m_vector=None, m_frame=None):

    parent = architecture.parent_map[kite]

    if m_vector is None:
        m_vector = get_m_aero_var(variables, kite, parent)

    if m_frame is None:
        m_frame = moment_variable_frame()

    m_aero_body = frames.from_named_frame_to_body(m_frame, vec_u, kite_dcm, m_vector)
    m_aero_wind = frames.from_named_frame_to_wind(m_frame, vec_u, kite_dcm, m_vector)
    m_aero_control = frames.from_named_frame_to_control(m_frame, vec_u, kite_dcm, m_vector)
    m_aero_earth = frames.from_named_frame_to_earth(m_frame, vec_u, kite_dcm, m_vector)

    dict = {'body':m_aero_body, 'control': m_aero_control, 'wind': m_aero_wind, 'earth': m_aero_earth}

    return dict




##### the velocities

def get_local_air_velocity_in_earth_frame(options, variables, wind, kite, kite_dcm, architecture, parameters, outputs):

    aero_coeff_ref_velocity = options['aero']['aero_coeff_ref_velocity']

    if aero_coeff_ref_velocity == 'app':
        vec_u_app_body = get_u_app_alone_in_body_frame(options, variables, wind, kite, kite_dcm, architecture, parameters, outputs)
        vec_u_earth = frames.from_body_to_earth(kite_dcm, vec_u_app_body)

    elif aero_coeff_ref_velocity == 'eff':
        vec_u_eff = get_u_eff_in_earth_frame(options, variables, wind, kite, architecture)
        vec_u_earth = vec_u_eff

    return vec_u_earth


def get_u_eff_in_body_frame(options, variables, wind, kite, kite_dcm, architecture):
    u_eff_in_earth_frame = get_u_eff_in_earth_frame(options, variables, wind, kite, architecture)
    u_eff_in_body_frame = frames.from_earth_to_body(kite_dcm, u_eff_in_earth_frame)
    return u_eff_in_body_frame

def get_u_eff_in_earth_frame(options, variables, wind, kite, architecture):
    if (options['induction_model'] == 'not_in_use'):
        u_eff = get_u_eff_in_earth_frame_without_induction(variables, wind, kite, architecture)

    else:
        u_eff = get_u_eff_in_earth_frame_with_induction(options, variables, wind, kite, architecture)

    return u_eff

def get_u_eff_in_earth_frame_without_induction(variables, wind, kite, architecture):
    vec_u_app_alone_in_earth_frame = get_u_app_alone_in_earth_frame_without_induction(variables, wind, kite, architecture)

    # approximation!
    vec_u_eff_in_earth_frame = vec_u_app_alone_in_earth_frame

    return vec_u_eff_in_earth_frame



def get_u_eff_in_earth_frame_with_induction(options, variables, wind, kite, architecture):
    vec_u_eff_mawes_in_earth_frame = induction.get_kite_effective_velocity(options, variables, wind, kite, architecture)
    return vec_u_eff_mawes_in_earth_frame



def get_u_app_alone_in_body_frame(options, variables, wind, kite, kite_dcm, architecture, parameters, outputs):
    if (options['induction_model'] == 'not_in_use'):
        u_app = get_u_app_alone_in_body_frame_without_induction(variables, wind, kite, kite_dcm, architecture)

    else:
        u_app = get_u_app_alone_in_body_frame_with_induction(options, variables, wind, kite, kite_dcm, architecture, parameters, outputs)

    return u_app

def get_u_app_alone_in_earth_frame_without_induction(variables, wind, kite, architecture):

    parent = architecture.parent_map[kite]

    q = variables['xd']['q' + str(kite) + str(parent)]
    dq = variables['xd']['dq' + str(kite) + str(parent)]

    uw_infty = wind.get_velocity(q[2])

    vec_u_app_alone_in_earth_frame = uw_infty - dq

    return vec_u_app_alone_in_earth_frame

def get_u_app_alone_in_body_frame_without_induction(variables, wind, kite, kite_dcm, architecture):
    vec_u_app_alone_in_earth_frame = get_u_app_alone_in_earth_frame_without_induction(variables, wind, kite, architecture)
    vec_u_app_alone_in_body_frame = frames.from_earth_to_body(kite_dcm, vec_u_app_alone_in_earth_frame)

    return vec_u_app_alone_in_body_frame



def get_u_app_alone_in_body_frame_with_induction(options, variables, wind, kite, kite_dcm, architecture, parameters, outputs):

    vec_u_eff_mawes_in_earth_frame = get_u_eff_in_earth_frame_with_induction(options, variables, wind, kite, architecture)
    vec_u_eff_mawes_in_body_frame = frames.from_earth_to_body(kite_dcm, vec_u_eff_mawes_in_earth_frame)

    vec_u_eff_alone_in_body_frame = vec_u_eff_mawes_in_body_frame

    vec_u_ind_alone_in_body_frame = get_u_ind_alone_in_body_frame(vec_u_eff_alone_in_body_frame, kite, parameters, outputs)

    vec_u_app_alone_in_body_frame = vec_u_eff_alone_in_body_frame - vec_u_ind_alone_in_body_frame

    return vec_u_app_alone_in_body_frame


def get_u_ind_alone_in_body_frame(vec_u_eff_in_body_frame, kite, parameters, outputs):

    b_ref = parameters['theta0', 'geometry', 'b_ref']
    dcm_body_frame = cas.DM.eye(3)
    alpha_eff = indicators.get_alpha(vec_u_eff_in_body_frame, dcm_body_frame)

    gamma = outputs['aerodynamics']['circulation' + str(kite)]

    u_ind_x = (gamma / np.pi / b_ref) * alpha_eff
    u_ind_y = 0
    u_ind_z = (gamma / np.pi / b_ref) * (-1.)

    u_ind = cas.vertcat(u_ind_x, u_ind_y, u_ind_z)

    return u_ind
