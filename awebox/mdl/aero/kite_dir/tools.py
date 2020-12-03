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

##### the force and moment lifted variables

def get_f_aero_var(variables, kite, parent, parameters, options):
    unscaled = variables['xl']['f_aero' + str(kite) + str(parent)]
    f_scale = get_f_scale(parameters, options)
    rescaled = unscaled * f_scale

    return rescaled

def get_m_aero_var(variables, kite, parent, parameters, options):
    unscaled = variables['xl']['m_aero' + str(kite) + str(parent)]
    m_scale = get_m_scale(parameters, options)
    rescaled = unscaled * m_scale
    return rescaled

def get_f_scale(parameters, options):

    g = options['scaling']['other']['g']
    m_k = parameters['theta0', 'geometry', 'm_k']
    scale = g * m_k * 10.
    return scale

def get_m_scale(parameters, options):
    b_ref = parameters['theta0', 'geometry', 'b_ref']
    g = options['scaling']['other']['g']
    m_k = parameters['theta0', 'geometry', 'm_k']
    scale = b_ref * g * m_k / 2.
    return scale


##### the velocities

def get_local_air_velocity_in_earth_frame(options, variables, atmos, wind, kite, kite_dcm,
                                                             architecture, parameters):

    aero_coeff_ref_velocity = options['aero']['aero_coeff_ref_velocity']

    vec_u_eff = get_u_eff_in_earth_frame(options, variables, wind, kite, architecture)
    if aero_coeff_ref_velocity == 'app':
        vec_u_app_body = get_u_app_alone_in_body_frame(options, variables, atmos, wind, kite, kite_dcm,
                                                             architecture, parameters)
        vec_u_earth = frames.from_body_to_earth(kite_dcm, vec_u_app_body)
    elif aero_coeff_ref_velocity == 'eff':
        vec_u_earth = vec_u_eff

    return vec_u_earth

def get_local_air_velocity_in_body_frame(options, variables, atmos, wind, kite, kite_dcm, architecture, parameters):
    aero_coeff_ref_velocity = options['aero']['aero_coeff_ref_velocity']

    if aero_coeff_ref_velocity == 'app':
        vec_u = get_u_app_alone_in_body_frame(options, variables, atmos, wind, kite, kite_dcm, architecture,
                                                    parameters)
    elif aero_coeff_ref_velocity == 'eff':
        vec_u = get_u_eff_in_body_frame(options, variables, wind, kite, kite_dcm, architecture)

    return vec_u


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



def get_u_app_alone_in_body_frame(options, variables, atmos, wind, kite, kite_dcm, architecture, parameters):
    if (options['induction_model'] == 'not_in_use'):
        u_app = get_u_app_alone_in_body_frame_without_induction(options, variables, atmos, wind,
                                                                                     kite, kite_dcm, architecture,
                                                                                     parameters)

    else:
        u_app = get_u_app_alone_in_body_frame_with_induction(options, variables, atmos, wind,
                                                                                     kite, kite_dcm, architecture,
                                                                                     parameters)

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



def get_u_app_alone_in_body_frame_with_induction(options, variables, atmos, wind, kite, kite_dcm, architecture, parameters):
    parent = architecture.parent_map[kite]
    q = variables['xd']['q' + str(kite) + str(parent)]
    rho = atmos.get_density(q[2])

    vec_u_eff_mawes_in_earth_frame = get_u_eff_in_earth_frame_with_induction(options, variables, wind, kite, architecture)
    vec_u_eff_mawes_in_body_frame = frames.from_earth_to_body(kite_dcm, vec_u_eff_mawes_in_earth_frame)

    vec_u_eff_alone_in_body_frame = vec_u_eff_mawes_in_body_frame

    vec_u_ind_alone_in_body_frame = get_u_ind_alone_in_body_frame(vec_u_eff_alone_in_body_frame, rho, variables, kite, parent, parameters, options)

    vec_u_app_alone_in_body_frame = vec_u_eff_alone_in_body_frame - vec_u_ind_alone_in_body_frame

    return vec_u_app_alone_in_body_frame


def get_u_ind_alone_in_body_frame(vec_u_eff_in_body_frame, rho, variables, kite, parent, parameters, options):

    b_ref = parameters['theta0', 'geometry', 'b_ref']

    f_aero_var = get_f_aero_var(variables, kite, parent, parameters, options)
    dcm_body_frame = cas.DM.eye(3)

    ehat2 = dcm_body_frame[:, 1]

    alpha_eff = indicators.get_alpha(vec_u_eff_in_body_frame, dcm_body_frame)
    beta_eff = indicators.get_beta(vec_u_eff_in_body_frame, dcm_body_frame)

    ue_cross_e2 = vect_op.cross(vec_u_eff_in_body_frame, ehat2)
    gamma = cas.mtimes(f_aero_var.T, ue_cross_e2) / b_ref / rho / cas.mtimes(ue_cross_e2.T, ue_cross_e2)

    u_ind_x = (gamma / np.pi / b_ref) * alpha_eff
    u_ind_y = 0
    u_ind_z = (gamma / np.pi / b_ref) * (-1.)

    u_ind = cas.vertcat(u_ind_x, u_ind_y, u_ind_z)

    return u_ind
