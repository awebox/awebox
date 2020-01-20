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
aerodynamics indicators helper file
calculates indicators based on states and environment
_python-3.5 / casadi-3.4.5
- author: elena malz, chalmers 2016
- edited: rachel leuthold, jochem de schutter alu-fr 2017-18
'''

import casadi.tools as cas
import numpy as np

import awebox.mdl.aero.induction_dir.tools_dir.path_based_geom as path_based_geom
import awebox.tools.vector_operations as vect_op
import pdb

def get_mach(options, atmos, ua, q):
    norm_ua = vect_op.smooth_norm(ua)
    a = atmos.get_speed_of_sound( q[2])
    mach = norm_ua / a
    return mach

def get_reynolds(options, atmos, ua, q, parameters):
    norm_ua = cas.mtimes(ua.T, ua) ** 0.5
    rho_infty = atmos.get_density( q[2])
    mu_infty = atmos.get_viscosity( q[2])
    c_ref = parameters['theta0','geometry','c_ref']
    reynolds = rho_infty * norm_ua * c_ref / mu_infty
    return reynolds

def get_performance_outputs(options, atmos, wind, variables, outputs, parameters,architecture):

    if 'performance' not in list(outputs.keys()):
        outputs['performance'] = {}

    kite_nodes = architecture.kite_nodes
    xd = variables['xd']

    outputs['performance']['freelout'] = xd['dl_t'] / vect_op.norm(
        wind.get_velocity(xd['q10'][2]))

    outputs['performance']['elevation'] = get_elevation_angle(variables['xd'])

    outputs['performance']['p_loyd_total'] = 0.

    for n in kite_nodes:
        outputs['performance']['p_loyd_total'] += outputs['local_performance']['p_loyd' + str(n)]

    [current_power, phf, phf_hubheight, hubheight_power_availability] = get_power_harvesting_factor(options, atmos,
                                                                                                    wind, variables, parameters,architecture)
    outputs['performance']['phf'] = phf
    outputs['performance']['phf_hubheight'] = phf_hubheight
    outputs['performance']['hubheight_power_availability'] = hubheight_power_availability

    outputs['performance']['phf_loyd_total'] = outputs['performance']['p_loyd_total'] / hubheight_power_availability

    outputs['performance']['p_current'] = current_power

    epsilon = 1.0e-8

    p_loyd_total = outputs['performance']['p_loyd_total']
    outputs['performance']['loyd_factor'] = current_power / (p_loyd_total + epsilon)

    outputs['performance']['power_density'] = current_power / len(kite_nodes) / parameters['theta0','geometry','s_ref']

    return outputs

def collect_kite_aerodynamics_outputs(options, atmos, ua, ua_norm, aero_coefficients, f_aero, f_lift, f_drag, f_side, m_aero, ehat_chord, ehat_span, r, q, n, outputs, parameters):

    if 'aerodynamics' not in list(outputs.keys()):
        outputs['aerodynamics'] = {}

    for name in set(aero_coefficients.keys()):
        outputs['aerodynamics'][name + str(n)] = aero_coefficients[name]

    outputs['aerodynamics']['v_app' + str(n)] = ua
    outputs['aerodynamics']['speed' + str(n)] = ua_norm

    outputs['aerodynamics']['f_aero' + str(n)] = f_aero
    outputs['aerodynamics']['f_lift' + str(n)] = f_lift
    outputs['aerodynamics']['f_drag' + str(n)] = f_drag
    outputs['aerodynamics']['f_side' + str(n)] = f_side

    outputs['aerodynamics']['ehat_chord' + str(n)] = ehat_chord
    outputs['aerodynamics']['ehat_span' + str(n)] = ehat_span
    outputs['aerodynamics']['ehat_up' + str(n)] = vect_op.normed_cross(ehat_chord, ehat_span)

    b_ref = parameters['theta0', 'geometry', 'b_ref']
    c_ref = parameters['theta0', 'geometry', 'c_ref']
    rho = atmos.get_density(q[2])
    gamma_cross = vect_op.norm(f_lift) / b_ref / rho / vect_op.norm(vect_op.cross(ua, ehat_span))
    gamma_cl = 0.5 * ua_norm**2. * aero_coefficients['CL'] * c_ref / vect_op.norm(vect_op.cross(ua, ehat_span))
    gamma_unity = cas.DM(1.)
    outputs['aerodynamics']['gamma_cross' + str(n)] = gamma_cross
    outputs['aerodynamics']['gamma_cl' + str(n)] = gamma_cl
    outputs['aerodynamics']['gamma_unity' + str(n)] = gamma_unity
    outputs['aerodynamics']['gamma' + str(n)] = gamma_cl

    outputs['aerodynamics']['wingtip_ext' + str(n)] = q + ehat_span * b_ref / 2.
    outputs['aerodynamics']['wingtip_int' + str(n)] = q - ehat_span * b_ref / 2.

    outputs['aerodynamics']['fstar_aero' + str(n)] = cas.mtimes(ua.T, ehat_chord) / c_ref

    outputs['aerodynamics']['r'+str(n)] = r.reshape((9,1))

    if int(options['kite_dof']) == 6:
        outputs['aerodynamics']['m_aero' + str(n)] = m_aero

    outputs['aerodynamics']['mach' + str(n)] = get_mach(options, atmos, ua, q)
    outputs['aerodynamics']['reynolds' + str(n)] = get_reynolds(options, atmos, ua, q, parameters)

    return outputs

def collect_power_balance_outputs(variables, n, outputs, architecture):

    if 'power_balance' not in list(outputs.keys()):
        # initialize
        outputs['power_balance'] = {}

    # kite velocity
    dq_n = variables['xd']['dq'+str(n)+str(architecture.parent_map[n])]

    # get lift, drag and aero-moment power
    outputs['power_balance']['P_lift'+str(n)] = cas.mtimes(outputs['aerodynamics']['f_lift'+str(n)].T, dq_n)
    outputs['power_balance']['P_drag'+str(n)] = cas.mtimes(outputs['aerodynamics']['f_drag'+str(n)].T, dq_n)
    if 'r'+str(n)+str(architecture.parent_map[n]) in list(variables['xd'].keys()):
        outputs['power_balance']['P_side'+str(n)] = cas.mtimes(outputs['aerodynamics']['f_side'+str(n)].T, dq_n)

    if 'm_aero'+str(n) in list(outputs['aerodynamics'].keys()):
        omega_n = variables['xd']['omega'+str(n)+str(architecture.parent_map[n])]
        outputs['power_balance']['P_moment'+str(n)] = cas.mtimes(outputs['aerodynamics']['m_aero'+str(n)].T, omega_n)

    return outputs

def collect_tether_drag_losses(variables, tether_drag_forces, outputs, architecture):

    if 'power_balance' not in list(outputs.keys()):
        # initialize
        outputs['power_balance'] = {}

    # get dissipation power from tether drag
    for n in range(1, architecture.number_of_nodes):
        parent = architecture.parent_map[n]
        dq_n   = variables['xd']['dq'+str(n)+str(parent)] # node velocity
        outputs['power_balance']['P_tetherdrag'+str(n)] = cas.mtimes(tether_drag_forces['f'+str(n)+(str(parent))].T,dq_n)

    return outputs

def collect_aero_validity_outputs(options, xd, ua, n, parent, outputs, parameters):

    if 'aero_validity' not in list(outputs.keys()):
        outputs['aero_validity'] = {}
    tightness = options['model_bounds']['aero_validity']['scaling']
    num_ref = options['model_bounds']['aero_validity']['num_ref']

    if 'aerodynamics' not in list(outputs.keys()):
        outputs['aerodynamics'] = {}

    alpha = cas.DM(0.)
    beta = cas.DM(0.)
    if int(options['kite_dof']) == 6:

        r = cas.reshape(xd['r' + str(n) + str(parent)], (3, 3))
        ehat1 = r[:, 0]  # chordwise, froml_e tot_e
        ehat2 = r[:, 1]  # spanwise, fromp_e ton_e
        ehat3 = r[:, 2]  # up

        alpha = get_alpha(ua, r)
        beta = get_beta(ua, r)

        alpha_min = options['aero']['alpha_min_deg']*np.pi/180.0
        alpha_max = options['aero']['alpha_max_deg']*np.pi/180.0
        beta_min = options['aero']['beta_min_deg']*np.pi/180.0
        beta_max = options['aero']['beta_max_deg']*np.pi/180.0

        alpha_ub = (cas.mtimes(ua.T, ehat3) - cas.mtimes(ua.T, ehat1) * alpha_max) * tightness / num_ref
        alpha_lb = (- cas.mtimes(ua.T, ehat3) + cas.mtimes(ua.T, ehat1) * alpha_min) * tightness / num_ref
        beta_ub = (cas.mtimes(ua.T, ehat2) - cas.mtimes(ua.T, ehat1) * beta_max) * tightness / num_ref
        beta_lb = (- cas.mtimes(ua.T, ehat2) + cas.mtimes(ua.T, ehat1) * beta_min) * tightness / num_ref

        outputs['aero_validity']['alpha_ub' + str(n)] = alpha_ub
        outputs['aero_validity']['alpha_lb' + str(n)] = alpha_lb
        outputs['aero_validity']['beta_ub' + str(n)] = beta_ub
        outputs['aero_validity']['beta_lb' + str(n)] = beta_lb

    outputs['aerodynamics']['alpha' + str(n)] = alpha
    outputs['aerodynamics']['beta' + str(n)] = beta
    outputs['aerodynamics']['alpha_deg' + str(n)] = alpha * 180. / np.pi
    outputs['aerodynamics']['beta_deg' + str(n)] = beta * 180. / np.pi

    return outputs

def collect_local_performance_outputs(options, atmos, wind, variables, CL, CD, elevation_angle, ua, n, parent, outputs,parameters):

    xd = variables['xd']
    q = xd['q' + str(n) + str(parent)]

    if 'local_performance' not in list(outputs.keys()):
        outputs['local_performance'] = {}

    [CR, f_crosswind, p_loyd, loyd_speed, loyd_phf] = get_loyd_comparison(options, atmos, wind, xd, n, parent, CL, CD, parameters, elevation_angle)

    norm_ua = cas.mtimes(ua.T, ua)**0.5

    outputs['local_performance']['CR' + str(n)] = CR
    outputs['local_performance']['f_crosswind' + str(n)] = f_crosswind
    outputs['local_performance']['p_loyd' + str(n)] = p_loyd
    outputs['local_performance']['loyd_speed' + str(n)] = loyd_speed
    outputs['local_performance']['loyd_phf' + str(n)] = loyd_phf
    outputs['local_performance']['radius' + str(n)] = path_based_geom.get_radius_of_curvature(variables, n, parent)

    outputs['local_performance']['speed_ratio' + str(n)] = norm_ua / vect_op.norm(wind.get_velocity(q[2]))
    outputs['local_performance']['speed_ratio_loyd' + str(n)] = loyd_speed / vect_op.norm(wind.get_velocity(q[2]))

    outputs['local_performance']['radius_of_curvature' + str(n)] = path_based_geom.get_radius_of_curvature(variables, n, parent)


    return outputs

def collect_environmental_outputs(atmos, wind, q, n, outputs):

    if 'environment' not in list(outputs.keys()):
        outputs['environment'] = {}

    outputs['environment']['windspeed' + str(n)] = vect_op.norm(wind.get_velocity(q[2]))
    outputs['environment']['pressure' + str(n)] = atmos.get_pressure(q[2])
    outputs['environment']['temperature' + str(n)] = atmos.get_temperature(q[2])
    outputs['environment']['density' + str(n)] = atmos.get_density(q[2])

    return outputs


def get_loyd_comparison(options, atmos, wind, xd, n, parent, CL, CD, parameters, elevation_angle=0.):
    # for elevation angle cosine losses see Van der Lind, p. 477, AWE book

    q = xd['q' + str(n) + str(parent)]

    epsilon = 1.e-8
    CR = CL * (1. + (CD / (CL + epsilon))**2.)**0.5

    windspeed = vect_op.norm(wind.get_velocity(q[2]))
    power_density = get_power_density(atmos, wind, q[2])

    f_crosswind = 4. / 27. * CR * (CR / CD) ** 2. * np.cos(elevation_angle) ** 3.

    s_ref = parameters['theta0','geometry','s_ref']
    p_loyd = power_density * s_ref * f_crosswind

    loyd_speed = 2. * CR / 3. / CD * windspeed * np.cos(elevation_angle)

    loyd_phf = f_crosswind

    return [CR, f_crosswind, p_loyd, loyd_speed, loyd_phf]

def get_power_harvesting_factor(options, atmos, wind, variables, parameters,architecture):

    number_of_kites = architecture.number_of_kites

    xd = variables['xd']
    xa = variables['xa']

    s_ref = parameters['theta0', 'geometry', 's_ref']

    power_availability = 0.
    for n in architecture.kite_nodes:
        parent = architecture.parent_map[n]
        height = xd['q' + str(n) + str(parent)][2]

        power_availability += get_power_density(atmos, wind, height) * s_ref

    current_power = xa['lambda10'] * xd['l_t'] * xd['dl_t']

    hubheight = xd['q10'][2]
    hubheight_power_availability = get_power_density(atmos, wind, hubheight) * s_ref * number_of_kites

    phf = current_power / power_availability
    phf_hubheight = current_power / hubheight_power_availability

    return [current_power, phf, phf_hubheight, hubheight_power_availability]

def get_elevation_angle(xd):
    length_along_ground = (xd['q10'][0] ** 2. + xd['q10'][1] ** 2.) ** 0.5
    elevation_angle = np.arctan2(xd['q10'][2], length_along_ground)

    return elevation_angle

def convert_from_body_to_wind_axes(alpha, beta, axial_side_normal):
    rotation1 = cas.horzcat(np.cos(alpha) * np.cos(beta),      np.sin(beta),   np.sin(alpha) * np.cos(beta))
    rotation2 = cas.horzcat(-np.cos(alpha) * np.sin(beta),     np.cos(beta),   - np.sin(alpha) * np.sin(beta))
    rotation3 = cas.horzcat(-np.sin(alpha),                    0.          ,   np.cos(alpha))
    rotation = cas.vertcat(rotation1, rotation2, rotation3)

    drag_cross_lift = cas.mtimes(rotation, axial_side_normal)
    return drag_cross_lift

def convert_from_wind_to_body_axes(alpha, beta, drag_cross_lift):
    rotation1 = cas.horzcat(np.cos(alpha) * np.cos(beta),  -np.cos(alpha) * np.sin(beta),  -np.sin(alpha))
    rotation2 = cas.horzcat(np.sin(beta),                   np.cos(beta),                   0.)
    rotation3 = cas.horzcat(np.cos(beta) * np.sin(alpha),   -np.sin(alpha) * np.sin(beta),  np.cos(alpha))
    rotation = cas.vertcat(rotation1, rotation2, rotation3)

    axial_side_normal = cas.mtimes(rotation, drag_cross_lift)
    return axial_side_normal

def test_conversions():
    # must return zeros...


    #  -------------
    print('test 1')

    alpha = 0.
    beta = 0.
    # then CA = CD, CY = CS, CN = CL

    test = vect_op.xhat_np
    check = vect_op.xhat_np
    calc = convert_from_body_to_wind_axes(alpha, beta, test)
    resultant = calc - check
    print(resultant)

    test = vect_op.yhat_np
    check = vect_op.yhat_np
    calc = convert_from_body_to_wind_axes(alpha, beta, test)
    resultant = calc - check
    print(resultant)

    test = vect_op.zhat_np
    check = vect_op.zhat_np
    calc = convert_from_body_to_wind_axes(alpha, beta, test)
    resultant = calc - check
    print(resultant)

    #  -------------
    print('test 2')

    alpha = np.pi / 2
    beta = 0.
    # then CA = -CL, CY = CS, CN = CD

    test = vect_op.xhat_np
    check = -1. * vect_op.zhat_np
    calc = convert_from_body_to_wind_axes(alpha, beta, test)
    resultant = calc - check
    print(resultant)

    test = vect_op.yhat_np
    check = vect_op.yhat_np
    calc = convert_from_body_to_wind_axes(alpha, beta, test)
    resultant = calc - check
    print(resultant)

    test = vect_op.zhat_np
    check = vect_op.xhat_np
    calc = convert_from_body_to_wind_axes(alpha, beta, test)
    resultant = calc - check
    print(resultant)

    # -----
    print('test 3')

    # then CD = CN, CS = CY, CL = -CA

    test = vect_op.xhat_np
    check = vect_op.zhat_np
    calc = convert_from_wind_to_body_axes(alpha, beta, test)
    resultant = calc - check
    print(resultant)

    test = vect_op.yhat_np
    check = vect_op.yhat_np
    calc = convert_from_wind_to_body_axes(alpha, beta, test)
    resultant = calc - check
    print(resultant)

    test = vect_op.zhat_np
    check = -1. * vect_op.xhat_np
    calc = convert_from_wind_to_body_axes(alpha, beta, test)
    resultant = calc - check
    print(resultant)

    print('')

def get_alpha(ua, r):
    ehat1 = r[:, 0]  # chordwise, from le to te
    ehat2 = r[:, 1]  # spanwise, from pe to ne
    ehat3 = r[:, 2]  # up

    z_component = cas.mtimes(ua.T, ehat3)
    x_component = vect_op.smooth_abs(cas.mtimes(ua.T, ehat1))
    # x component had better be positive

    alpha = z_component / x_component

    return alpha

def get_beta(ua, r):
    ehat1 = r[:, 0]  # chordwise, from le to te
    ehat2 = r[:, 1]  # spanwise, from pe to ne
    ehat3 = r[:, 2]  # up

    y_component = cas.mtimes(ua.T, ehat2)
    x_component = vect_op.smooth_abs(cas.mtimes(ua.T, ehat1))
    # x component had better be positive

    beta = y_component / x_component

    return beta

def get_dynamic_pressure(atmos, wind, zz):
    u = wind.get_velocity(zz)
    rho = atmos.get_density(zz)
    q = 0.5 * rho * cas.mtimes(u.T, u)

    return q

def get_power_density(atmos, wind, zz):

    power_density = .5 * atmos.get_density( zz) * vect_op.norm(wind.get_velocity(zz)) ** 3.

    return power_density


def get_radius_inequality(model_options, variables, kite, parent, parameters):
    inequality = path_based_geom.get_radius_inequality(model_options, variables, kite, parent, parameters)
    return inequality
