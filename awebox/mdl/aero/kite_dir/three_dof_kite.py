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
specific aerodynamics for a 3dof kite with roll_control
_python-3.5 / casadi-3.4.5
- author: elena malz, chalmers 2016
- edited: jochem de schutter, rachel leuthold, alu-fr 2017-20
'''

import casadi.tools as cas

import awebox.tools.vector_operations as vect_op
import awebox.mdl.aero.indicators as indicators
import numpy as np

import awebox.mdl.aero.kite_dir.frames as frames
import awebox.mdl.aero.kite_dir.tools as tools

import pdb
from awebox.logger.logger import Logger as awelogger

def get_outputs(options, atmos, wind, variables, outputs, parameters, architecture):

    xd = variables['xd']
    elevation_angle = indicators.get_elevation_angle(xd)

    s_ref = parameters['theta0', 'geometry', 's_ref']

    kite_nodes = architecture.kite_nodes
    for kite in kite_nodes:
        parent = architecture.parent_map[kite]

        q = xd['q' + str(kite) + str(parent)]
        kite_dcm = get_kite_dcm(options, variables, wind, kite, architecture)
        ehat1 = kite_dcm[:, 0]
        ehat2 = kite_dcm[:, 1]

        vec_u_eff = tools.get_u_eff_in_earth_frame(options, variables, wind, kite, architecture)
        u_eff = vect_op.smooth_norm(vec_u_eff)

        rho = atmos.get_density(q[2])
        q_eff = 0.5 * rho * cas.mtimes(vec_u_eff.T, vec_u_eff)

        f_aero_body = tools.get_f_aero_var(variables, kite, parent, parameters)
        coeff_body = f_aero_body / q_eff / s_ref
        CA = coeff_body[0]
        CY = coeff_body[1]
        CN = coeff_body[2]

        f_aero_wind = frames.from_body_to_wind(vec_u_eff, kite_dcm, f_aero_body)
        wind_dcm = frames.get_wind_dcm(vec_u_eff, kite_dcm)
        f_drag = f_aero_wind[0] * wind_dcm[:, 0]
        f_side = f_aero_wind[1] * wind_dcm[:, 1]
        f_lift = f_aero_wind[2] * wind_dcm[:, 2]

        coeff_wind = f_aero_wind / q_eff / s_ref
        CD = coeff_wind[0]
        CS = coeff_wind[1]
        CL = coeff_wind[2]

        f_aero = frames.from_body_to_earth(kite_dcm, f_aero_body)
        m_aero = cas.DM(np.zeros((3, 1)))

        aero_coefficients = {}
        aero_coefficients['CD'] = CD
        aero_coefficients['CS'] = CS
        aero_coefficients['CL'] = CL
        aero_coefficients['CA'] = CA
        aero_coefficients['CY'] = CY
        aero_coefficients['CN'] = CN

        outputs = indicators.collect_kite_aerodynamics_outputs(options, atmos, vec_u_eff, u_eff, aero_coefficients,
                                                               f_aero, f_lift, f_drag, f_side, m_aero,
                                                               ehat1, ehat2, kite_dcm, q, kite,
                                                               outputs, parameters)
        outputs = indicators.collect_environmental_outputs(atmos, wind, q, kite, outputs)
        outputs = indicators.collect_aero_validity_outputs(options, xd, vec_u_eff, kite, parent, outputs, parameters)
        outputs = indicators.collect_local_performance_outputs(options, atmos, wind, variables, CL, CD, elevation_angle,
                                                               vec_u_eff, kite, parent, outputs, parameters)
        outputs = indicators.collect_power_balance_outputs(variables, kite, outputs, architecture)

    return outputs



def get_force_resi(options, variables, atmos, wind, architecture, parameters):

    aero_coeff_ref_velocity = options['aero']['aero_coeff_ref_velocity']
    if aero_coeff_ref_velocity == 'app':
        force_fun = get_force_fun_from_u_app_alone_in_kite_frame(parameters)
    elif aero_coeff_ref_velocity == 'eff':
        force_fun = get_force_fun_from_u_eff_in_kite_frame(parameters)
    else:
        awelogger.logger.error('unrecognized velocity field associated with stability derivative computation')

    resi = []
    for kite in architecture.kite_nodes:

        parent = architecture.parent_map[kite]
        f_aero_var = tools.get_f_aero_var(variables, kite, parent, parameters)

        q = variables['xd']['q' + str(kite) + str(parent)]
        rho = atmos.get_density(q[2])

        kite_dcm = get_kite_dcm(options, variables, wind, kite, architecture)

        if aero_coeff_ref_velocity == 'app':
            vec_u = tools.get_u_app_alone_in_body_frame(options, variables, atmos, wind, kite, kite_dcm, architecture, parameters)
        elif aero_coeff_ref_velocity == 'eff':
            vec_u = tools.get_u_eff_in_body_frame(options, variables, wind, kite, kite_dcm, architecture)

        coeff = variables['xd']['coeff' + str(kite) + str(parent)]
        f_found = force_fun(vec_u, coeff, rho)

        f_scale = tools.get_f_scale(parameters)

        resi_f_kite = (f_aero_var - f_found) / f_scale

        resi = cas.vertcat(resi, resi_f_kite)

    return resi





def get_force_fun_from_u_eff_in_kite_frame(parameters):

    # creates a casadi function that finds the force, all calculations in kite-body reference frame.

    coeff_sym = cas.SX.sym('coeff_sym', 2)
    vec_u_eff_sym = cas.SX.sym('vec_u_eff_sym', 3)
    rho_sym = cas.SX.sym('rho_sym')

    force = get_force_and_moment_from_u_eff_in_kite_frame(parameters, vec_u_eff_sym, coeff_sym, rho_sym)

    force_fun = cas.Function('force_fun', [vec_u_eff_sym, coeff_sym, rho_sym], [force])

    return force_fun


def get_force_and_moment_from_u_eff_in_kite_frame(parameters, vec_u_eff_sym, coeff_sym, rho_sym):

    kite_dcm = cas.DM.eye(3)
    ehat2 = kite_dcm[:, 1]

    # lift and drag coefficients
    CL = coeff_sym[0]
    CD = parameters['theta0', 'aero', 'CD0'] + CL ** 2 / (np.pi * parameters['theta0', 'geometry', 'ar'])

    q_app = 0.5 * rho_sym * cas.mtimes(vec_u_eff_sym.T, vec_u_eff_sym)
    s_ref = parameters['theta0', 'geometry', 's_ref']

    ehat_drag = vect_op.smooth_normalize(vec_u_eff_sym)
    ehat_lift = vect_op.smooth_normed_cross(vec_u_eff_sym, ehat2)

    # lift and drag force
    f_lift = CL * q_app * s_ref * ehat_lift
    f_drag = CD * q_app * s_ref * ehat_drag
    f_side = cas.DM(np.zeros((3, 1)))

    force = f_drag + f_lift + f_side

    return force







def get_force_fun_from_u_app_alone_in_kite_frame(parameters):

    # creates a casadi function that finds the force, all calculations in kite-body reference frame.

    coeff_sym = cas.SX.sym('coeff_sym', 2)
    vec_u_app_alone_sym = cas.SX.sym('vec_u_app_alone_sym', 3)
    rho_sym = cas.SX.sym('rho_sym')

    force = get_force_and_moment_from_u_app_alone_in_kite_frame(parameters, vec_u_app_alone_sym, coeff_sym, rho_sym)

    force_fun = cas.Function('force_fun', [vec_u_app_alone_sym, coeff_sym, rho_sym], [force])

    return force_fun


def get_force_and_moment_from_u_app_alone_in_kite_frame(parameters, vec_u_app_alone_sym, coeff_sym, rho_sym):

    kite_dcm = cas.DM.eye(3)
    ehat2 = kite_dcm[:, 1]

    # lift and drag coefficients
    CL = coeff_sym[0]
    CD = parameters['theta0', 'aero', 'CD0'] + CL ** 2 / (np.pi * parameters['theta0', 'geometry', 'ar'])

    q_app = 0.5 * rho_sym * cas.mtimes(vec_u_app_alone_sym.T, vec_u_app_alone_sym)
    s_ref = parameters['theta0', 'geometry', 's_ref']

    ehat_drag = vect_op.smooth_normalize(vec_u_app_alone_sym)
    ehat_lift = vect_op.smooth_normed_cross(vec_u_app_alone_sym, ehat2)

    # lift and drag force
    f_lift = CL * q_app * s_ref * ehat_lift
    f_drag = CD * q_app * s_ref * ehat_drag
    f_side = cas.DM(np.zeros((3, 1)))

    force = f_drag + f_lift + f_side

    return force












def get_kite_dcm(options, variables, wind, kite, architecture):
    vec_u_eff = tools.get_u_eff_in_earth_frame(options, variables, wind, kite, architecture)

    ehat3, ehat2 = get_ehat_up_and_span(kite, options, wind, variables, architecture)
    ehat1 = vect_op.smooth_normalize(vec_u_eff)
    kite_dcm = cas.horzcat(ehat1, ehat2, ehat3)

    return kite_dcm

def get_ehat_up_and_span(kite, options, wind, variables, architecture):
    parent_map = architecture.parent_map
    xd = variables['xd']

    parent = parent_map[kite]

    # get relevant variables for kite
    q = xd['q' + str(kite) + str(parent)]
    coeff = xd['coeff' + str(kite) + str(parent)]

    vec_u_eff = tools.get_u_eff_in_earth_frame(options, variables, wind, kite, architecture)

    # in kite body:
    if parent > 0:
        grandparent = parent_map[parent]
        qparent = xd['q' + str(parent) + str(grandparent)]
    else:
        qparent = np.array([0., 0., 0.])

    ehat_r = (q - qparent) / vect_op.norm(q - qparent)
    ehat_t = vect_op.normed_cross(vec_u_eff, ehat_r)
    ehat_s = vect_op.normed_cross(ehat_t, vec_u_eff)

    # roll angle
    psi = coeff[1]

    ehat_up = cas.cos(psi) * ehat_s + cas.sin(psi) * ehat_t
    ehat_span = cas.cos(psi) * ehat_t - cas.sin(psi) * ehat_s

    return ehat_up, ehat_span


def get_wingtip_position(kite, options, model, variables, parameters, ext_int):

    parent_map = model.architecture.parent_map
    xd = model.variables_dict['xd'](variables['xd'])

    if ext_int == 'ext':
        span_sign = 1.
    elif ext_int == 'int':
        span_sign = -1.
    else:
        pdb.set_trace()

    parent = parent_map[kite]

    q = xd['q' + str(kite) + str(parent)]

    _, ehat_span = get_ehat_up_and_span(kite, options, model.wind, variables, model.architecture)

    b_ref = parameters['theta0', 'geometry', 'b_ref']

    wingtip_position = q + ehat_span * span_sign * b_ref / 2.

    return wingtip_position