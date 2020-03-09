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

        vec_u_eff = tools.get_u_eff_in_earth_frame(options, variables, wind, kite, architecture)
        u_eff = vect_op.smooth_norm(vec_u_eff)

        kite_dcm = get_kite_dcm(vec_u_eff, kite, variables, architecture)
        ehat1 = kite_dcm[:, 0]
        ehat2 = kite_dcm[:, 1]

        rho = atmos.get_density(q[2])
        q_eff = 0.5 * rho * cas.mtimes(vec_u_eff.T, vec_u_eff)

        f_aero_earth = tools.get_f_aero_var(variables, kite, parent, parameters)

        f_aero_body = frames.from_earth_to_body(kite_dcm, f_aero_earth)
        coeff_body = f_aero_body / q_eff / s_ref
        CA = coeff_body[0]
        CY = coeff_body[1]
        CN = coeff_body[2]

        f_aero_wind = frames.from_earth_to_wind(vec_u_eff, kite_dcm, f_aero_earth)
        wind_dcm = frames.get_wind_dcm(vec_u_eff, kite_dcm)
        f_drag = f_aero_wind[0] * wind_dcm[:, 0]
        f_side = f_aero_wind[1] * wind_dcm[:, 1]
        f_lift = f_aero_wind[2] * wind_dcm[:, 2]

        coeff_wind = f_aero_wind / q_eff / s_ref
        CD = coeff_wind[0]
        CS = coeff_wind[1]
        CL = coeff_wind[2]

        f_aero = f_aero_earth
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

    resi = []
    for kite in architecture.kite_nodes:

        parent = architecture.parent_map[kite]
        f_aero_var = tools.get_f_aero_var(variables, kite, parent, parameters)

        vec_u_eff = tools.get_u_eff_in_earth_frame(options, variables, wind, kite, architecture)
        kite_dcm = get_kite_dcm(vec_u_eff, kite, variables, architecture)

        if aero_coeff_ref_velocity == 'app':
            vec_u_app_body = tools.get_u_app_alone_in_body_frame(options, variables, atmos, wind, kite, kite_dcm, architecture, parameters)
            vec_u = frames.from_body_to_earth(kite_dcm, vec_u_app_body)
        elif aero_coeff_ref_velocity == 'eff':
            vec_u = vec_u_eff

        f_earth_frame = get_force_from_u_sym_in_earth_frame(vec_u, options, variables, kite, atmos, wind, architecture, parameters)

        f_scale = tools.get_f_scale(parameters)

        resi_f_kite = (f_aero_var - f_earth_frame) / f_scale

        resi = cas.vertcat(resi, resi_f_kite)

    return resi




def get_force_from_u_sym_in_earth_frame(vec_u, options, variables, kite, atmos, wind, architecture, parameters):

    parent = architecture.parent_map[kite]

    # get relevant variables for kite n
    q = variables['xd']['q' + str(kite) + str(parent)]
    coeff = variables['xd']['coeff' + str(kite) + str(parent)]

    # wind parameters
    rho_infty = atmos.get_density(q[2])

    vec_u_eff = tools.get_u_eff_in_earth_frame(options, variables, wind, kite, architecture)
    kite_dcm = get_kite_dcm(vec_u_eff, kite, variables, architecture)
    Lhat = kite_dcm[:,2]

    # lift and drag coefficients
    CL = coeff[0]
    CD = parameters['theta0', 'aero', 'CD0'] + CL ** 2 / (np.pi * parameters['theta0', 'geometry', 'ar'])

    s_ref = parameters['theta0', 'geometry', 's_ref']

    # lift and drag force
    f_lift = CL * 1. / 2. * rho_infty * cas.mtimes(vec_u.T, vec_u) * s_ref * Lhat
    f_drag = CD * 1. / 2. * rho_infty * vect_op.norm(vec_u) * s_ref * vec_u

    f_aero = f_lift + f_drag

    return f_aero











def get_planar_dmc(vec_u_eff, variables, kite, architecture):

    parent = architecture.parent_map[kite]

    # get relevant variables for kite n
    q = variables['xd']['q' + str(kite) + str(parent)]

    # in kite body:
    if parent > 0:
        grandparent = architecture.parent_map[parent]
        q_parent = variables['xd']['q' + str(parent) + str(grandparent)]
    else:
        q_parent = np.array([0., 0., 0.])

    vec_t = q - q_parent # should be roughly "up-wards", ie, act like vec_w

    vec_v = vect_op.cross(vec_t, vec_u_eff)
    vec_w = vect_op.cross(vec_u_eff, vec_v)

    uhat = vect_op.smooth_normalize(vec_u_eff)
    vhat = vect_op.smooth_normalize(vec_v)
    what = vect_op.smooth_normalize(vec_w)

    planar_dcm = cas.horzcat(uhat, vhat, what)

    return planar_dcm


def get_kite_dcm(vec_u_eff, kite, variables, architecture):

    parent = architecture.parent_map[kite]

    # roll angle
    coeff = variables['xd']['coeff' + str(kite) + str(parent)]
    psi = coeff[1]

    planar_dcm = get_planar_dmc(vec_u_eff, variables, kite, architecture)
    uhat = planar_dcm[:, 0]
    vhat = planar_dcm[:, 1]
    what = planar_dcm[:, 2]

    ehat1 = uhat
    ehat2 = np.cos(psi) * vhat + np.sin(psi) * what
    ehat3 = np.cos(psi) * what - np.sin(psi) * vhat

    kite_dcm = cas.horzcat(ehat1, ehat2, ehat3)

    return kite_dcm


def get_wingtip_position(kite, options, model, variables, parameters, ext_int):

    aero_coeff_ref_velocity = options['aero']['aero_coeff_ref_velocity']

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

    vec_u_eff = tools.get_u_eff_in_earth_frame(options, variables, model.wind, kite, model.architecture)

    kite_dcm = get_kite_dcm(vec_u_eff, kite, variables, model.architecture)
    ehat_span = kite_dcm[:, 1]

    b_ref = parameters['theta0', 'geometry', 'b_ref']

    wingtip_position = q + ehat_span * span_sign * b_ref / 2.

    return wingtip_position