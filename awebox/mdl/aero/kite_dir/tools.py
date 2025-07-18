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
import awebox.mdl.aero.kite_dir.vortex_rings as vortex_rings

def construct_wingtip_position(q_kite, dcm_kite, parameters, tip):

    if tip == 'ext':
        span_sign = 1.
    elif tip == 'int':
        span_sign = -1.
    else:
        message = 'wingtip ' + tip + ' is not recognized'
        print_op.log_and_raise_error(message)

    ehat_span = dcm_kite[:, 1]

    b_ref = parameters['theta0', 'geometry', 'b_ref']

    wingtip_position = q_kite + ehat_span * span_sign * b_ref / 2.

    return wingtip_position

##### the force and moment lifted variables

def get_f_aero_var(variables, kite, parent):

    var_type = 'z'
    var_name = 'f_aero' + str(kite) + str(parent)

    if var_name in variables[var_type].keys():
        f_aero_si = variables[var_type][var_name]
    else:
        print_op.log_and_raise_error('lifted aero forces not found.')

    return f_aero_si

def get_m_aero_var(variables, kite, parent):

    var_type = 'z'
    var_name = 'm_aero' + str(kite) + str(parent)

    if var_name in variables[var_type].keys():
        m_aero_si = variables[var_type][var_name]
    else:
        print_op.log_and_raise_error('lifted aero moments not found.')

    return m_aero_si

def force_variable_frame():
    return 'earth'

def moment_variable_frame():
    return 'body'

def get_f_scale(parameters, options):
    scale = options['scaling']['z']['f_aero']
    return scale

def get_m_scale(parameters, options):
    m_scale = options['scaling']['z']['m_aero']
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
        vec_u_eff = get_u_eff_in_earth_frame(options, variables, parameters, wind, kite, architecture)
        vec_u_earth = vec_u_eff

    return vec_u_earth


def get_u_eff_in_body_frame(options, variables, parameters, wind, kite, kite_dcm, architecture):
    u_eff_in_earth_frame = get_u_eff_in_earth_frame(options, variables, parameters, wind, kite, architecture)
    u_eff_in_body_frame = frames.from_earth_to_body(kite_dcm, u_eff_in_earth_frame)
    return u_eff_in_body_frame

def get_u_eff_in_earth_frame(options, variables, parameters, wind, kite, architecture):
    if (options['induction_model'] == 'not_in_use'):
        u_eff = get_u_eff_in_earth_frame_without_induction(variables, parameters, wind, kite, architecture, options)
    else:
        u_eff = get_u_eff_in_earth_frame_with_induction(options, variables, wind, kite, architecture)

    return u_eff

def get_u_eff_in_earth_frame_without_induction(variables, parameters, wind, kite, architecture, options):
    vec_u_app_alone_in_earth_frame = get_u_app_alone_in_earth_frame_without_induction(variables, parameters,  wind, kite, architecture, options)

    # approximation!
    vec_u_eff_in_earth_frame = vec_u_app_alone_in_earth_frame

    return vec_u_eff_in_earth_frame



def get_u_eff_in_earth_frame_with_induction(options, variables, wind, kite, architecture):

    if options['induction_model'] == 'averaged':

        q = variables['x']['q{}'.format(architecture.node_label(kite))]
        dq = variables['x']['dq{}'.format(architecture.node_label(kite))]
        uw_infty = (1-variables['theta']['a']) * wind.get_velocity(q[2])

        vec_u_eff_mawes_in_earth_frame = uw_infty - dq

    else:
        vec_u_eff_mawes_in_earth_frame = induction.get_kite_effective_velocity(variables, wind, kite, architecture)
    return vec_u_eff_mawes_in_earth_frame



def get_u_app_alone_in_body_frame(options, variables, wind, kite, kite_dcm, architecture, parameters, outputs):
    if (options['induction_model'] == 'not_in_use'):
        u_app = get_u_app_alone_in_body_frame_without_induction(variables, wind, kite, kite_dcm, architecture)

    else:
        u_app = get_u_app_alone_in_body_frame_with_induction(options, variables, wind, kite, kite_dcm, architecture, parameters, outputs)

    return u_app

def get_u_app_alone_in_earth_frame_without_induction(variables, parameters, wind, kite, architecture, options):

    parent = architecture.parent_map[kite]

    q = variables['x']['q' + str(kite) + str(parent)]
    dq = variables['x']['dq' + str(kite) + str(parent)]

    uw_infty = wind.get_velocity(q[2])

    if 'dp_ring_2_0_0' in variables['x'].keys() and kite in [2,3]:
        u_induced = u_induced_vortex_rings(variables, parameters, kite, architecture, options)
    else:
        u_induced = np.array([0,0,0])

    vec_u_app_alone_in_earth_frame = uw_infty - dq + u_induced

    return vec_u_app_alone_in_earth_frame

def u_induced_vortex_rings(variables, parameters, kite, architecture, options):

    parent = architecture.parent_map[kite]
    q = variables['x']['q' + str(kite) + str(parent)]
    t_f = variables['theta']['t_f']
    u_induced = np.zeros((3,1))
    initial_guess =  np.array([[-1],[0],[0]])
    params = 'p_near_{}'.format(kite)
    h = t_f / options['aero']['vortex_rings']['N'] / options['aero']['vortex_rings']['N_rings']

    vortex_type = options['aero']['vortex_rings']['type']

    for k in range(options['aero']['vortex_rings']['N']):
        for i in range(options['aero']['vortex_rings']['N_rings']):
            for j in [2, 3]:
                w_ind_f = 0
                p_r = variables['x']['p_ring_{}_{}_{}'.format(j, k, i)]
                dp_r = variables['x']['dp_ring_{}_{}_{}'.format(j, k, i)]
                gamma_r = variables['x']['gamma_ring_{}_{}_{}'.format(j, k, i)]
                n_r = variables['x']['n_ring_{}_{}_{}'.format(j, k, i)]
                if vortex_type == 'dipole':
                    e_c = None
                elif vortex_type == 'rectangle':
                    e_c = variables['x']['ec_ring_{}_{}_{}'.format(j, k, i)]
                R_ring = parameters['theta0', 'aero', 'vortex_rings', 'R_ring']
                param = parameters['p_far_{}'.format(kite), 'p_far_{}_{}'.format(j, k)] * parameters[params, 'p_near_{}_{}'.format(j, k)]
                w_ind_f += - h * param * vortex_rings.far_wake_induction(q, p_r, n_r, e_c, gamma_r, R_ring, options['aero']['vortex_rings'])
                for d in range(options['aero']['vortex_rings']['N_duplicates']):
                    param = parameters['p_far_{}'.format(kite), 'p_far_{}_{}'.format(j, k)]
                    p_r_dup = p_r + cas.vertcat(dp_r*(d+1)*t_f, 0, 0)
                    w_ind_f += - h * param * vortex_rings.far_wake_induction(q, p_r_dup, n_r, e_c, gamma_r, R_ring, options['aero']['vortex_rings'])

                u_induced = u_induced  + w_ind_f

    u_induced_homotopy = parameters['phi', 'iota'] * initial_guess + (1 - parameters['phi', 'iota']) * u_induced

    return u_induced_homotopy

def get_u_app_alone_in_body_frame_without_induction(variables, wind, kite, kite_dcm, architecture, options):
    vec_u_app_alone_in_earth_frame = get_u_app_alone_in_earth_frame_without_induction(variables, wind, kite, architecture, options)
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
