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
aerodynamics indicators helper file
calculates indicators based on states and environment
_python-3.5 / casadi-3.4.5
- author: elena malz, chalmers 2016
- edited: rachel leuthold, jochem de schutter alu-fr 2017-21
'''

import casadi.tools as cas
import numpy as np
from awebox.logger.logger import Logger as awelogger

import awebox.mdl.aero.induction_dir.tools_dir.path_based_geom as path_based_geom
import awebox.mdl.aero.induction_dir.tools_dir.geom as general_geom
import awebox.mdl.aero.induction_dir.tools_dir.flow as general_flow

import awebox.tools.vector_operations as vect_op
import awebox.tools.performance_operations as perf_op
import awebox.tools.print_operations as print_op

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

    outputs['performance']['freelout'] = xd['dl_t'] / vect_op.norm(wind.get_velocity(xd['q10'][2]))
    outputs['performance']['elevation'] = get_elevation_angle(variables['xd'])

    layer_nodes = architecture.layer_nodes
    for parent in layer_nodes:
        outputs['performance']['actuator_center' + str(parent)] = general_geom.get_center_point(options, parent,
                                                                                                variables, architecture)
        outputs['performance']['f' + str(parent)] = general_flow.get_f_val(options, wind, parent, variables,
                                                                           architecture)

        average_radius = 0.

        number_children = len(architecture.children_map[parent])
        for kite in architecture.children_map[parent]:

            rad_curv_name = 'radius_of_curvature' + str(kite)
            if rad_curv_name in outputs['local_performance'].keys():
                local_radius = outputs['local_performance'][rad_curv_name]
            else:
                qkite = variables['xd']['q' + str(kite) + str(parent)]
                local_radius = vect_op.norm(qkite - outputs['performance']['actuator_center' + str(parent)])

            average_radius += local_radius / float(number_children)

        outputs['performance']['average_radius' + str(parent)] = average_radius


    outputs['performance']['p_loyd_total'] = 0.
    for kite in kite_nodes:
        outputs['performance']['p_loyd_total'] += outputs['local_performance']['p_loyd' + str(kite)]

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


def collect_kite_aerodynamics_outputs(options, architecture, atmos, wind, variables, parameters, base_aerodynamic_quantities, outputs):

    if 'aerodynamics' not in list(outputs.keys()):
        outputs['aerodynamics'] = {}

    # unpack
    kite = base_aerodynamic_quantities['kite']
    air_velocity = base_aerodynamic_quantities['air_velocity']
    aero_coefficients = base_aerodynamic_quantities['aero_coefficients']
    f_aero_earth = base_aerodynamic_quantities['f_aero_earth']
    f_aero_body = base_aerodynamic_quantities['f_aero_body']


    f_aero_control = base_aerodynamic_quantities['f_aero_control']
    f_aero_wind = base_aerodynamic_quantities['f_aero_wind']
    f_lift_earth = base_aerodynamic_quantities['f_lift_earth']
    f_drag_earth = base_aerodynamic_quantities['f_drag_earth']
    f_side_earth = base_aerodynamic_quantities['f_side_earth']
    m_aero_body = base_aerodynamic_quantities['m_aero_body']
    kite_dcm = base_aerodynamic_quantities['kite_dcm']
    q = base_aerodynamic_quantities['q']

    f_lift_earth_overwrite = options['aero']['overwrite']['f_lift_earth']
    if f_lift_earth_overwrite is not None:
        f_lift_earth = f_lift_earth_overwrite

    for name in set(base_aerodynamic_quantities['aero_coefficients'].keys()):
        outputs['aerodynamics'][name + str(kite)] = base_aerodynamic_quantities['aero_coefficients'][name]

    outputs['aerodynamics']['air_velocity' + str(kite)] = air_velocity
    airspeed = vect_op.norm(air_velocity)
    outputs['aerodynamics']['airspeed' + str(kite)] = airspeed
    outputs['aerodynamics']['u_infty' + str(kite)] = wind.get_velocity(q[2])

    rho = atmos.get_density(q[2])
    outputs['aerodynamics']['air_density' + str(kite)] = rho
    outputs['aerodynamics']['dyn_pressure' + str(kite)] = 0.5 * rho * cas.mtimes(air_velocity.T, air_velocity)

    ehat_chord = kite_dcm[:, 0]
    ehat_span = kite_dcm[:, 1]
    ehat_up = kite_dcm[:, 2]

    outputs['aerodynamics']['ehat_chord' + str(kite)] = ehat_chord
    outputs['aerodynamics']['ehat_span' + str(kite)] = ehat_span
    outputs['aerodynamics']['ehat_up' + str(kite)] = ehat_up

    outputs['aerodynamics']['f_aero_body' + str(kite)] = f_aero_body
    outputs['aerodynamics']['f_aero_control' + str(kite)] = f_aero_control
    outputs['aerodynamics']['f_aero_earth' + str(kite)] = f_aero_earth
    outputs['aerodynamics']['f_aero_wind' + str(kite)] = f_aero_wind

    ortho = cas.reshape(cas.mtimes(kite_dcm.T, kite_dcm) - np.eye(3), (9, 1))
    ortho_resi = cas.mtimes(ortho.T, ortho)
    outputs['aerodynamics']['ortho_resi' + str(kite)] = ortho_resi

    outputs['aerodynamics']['f_lift_earth' + str(kite)] = f_lift_earth
    outputs['aerodynamics']['f_drag_earth' + str(kite)] = f_drag_earth
    outputs['aerodynamics']['f_side_earth' + str(kite)] = f_side_earth

    b_ref = parameters['theta0', 'geometry', 'b_ref']
    c_ref = parameters['theta0', 'geometry', 'c_ref']

    circulation_cross = vect_op.smooth_norm(f_lift_earth) / b_ref / rho / vect_op.smooth_norm(vect_op.cross(air_velocity, ehat_span))
    circulation_cl = 0.5 * airspeed**2. * aero_coefficients['CL'] * c_ref / vect_op.smooth_norm(vect_op.cross(air_velocity, ehat_span))

    outputs['aerodynamics']['circulation_cross' + str(kite)] = circulation_cross
    outputs['aerodynamics']['circulation_cl' + str(kite)] = circulation_cl
    outputs['aerodynamics']['circulation' + str(kite)] = circulation_cross

    outputs['aerodynamics']['wingtip_ext' + str(kite)] = q + ehat_span * b_ref / 2.
    outputs['aerodynamics']['wingtip_int' + str(kite)] = q - ehat_span * b_ref / 2.

    outputs['aerodynamics']['fstar_aero' + str(kite)] = cas.mtimes(air_velocity.T, ehat_chord) / c_ref

    outputs['aerodynamics']['r' + str(kite)] = kite_dcm.reshape((9, 1))

    if int(options['kite_dof']) == 6:
        outputs['aerodynamics']['m_aero_body' + str(kite)] = m_aero_body

    outputs['aerodynamics']['mach' + str(kite)] = get_mach(options, atmos, air_velocity, q)
    outputs['aerodynamics']['reynolds' + str(kite)] = get_reynolds(options, atmos, air_velocity, q, parameters)

    return outputs


def collect_power_balance_outputs(options, architecture, variables, base_aerodynamic_quantities, outputs):

    kite = base_aerodynamic_quantities['kite']

    if 'power_balance' not in list(outputs.keys()):
        # initialize
        outputs['power_balance'] = {}

    # kite velocity
    parent = architecture.parent_map[kite]
    dq = variables['xd']['dq'+str(kite)+str(parent)]

    f_lift_earth = outputs['aerodynamics']['f_lift_earth' + str(kite)]
    f_drag_earth = outputs['aerodynamics']['f_drag_earth' + str(kite)]
    f_side_earth = outputs['aerodynamics']['f_side_earth' + str(kite)]

    # get lift, drag and aero-moment power
    outputs['power_balance']['P_lift' + str(kite)] = cas.mtimes(f_lift_earth.T, dq)
    outputs['power_balance']['P_drag' + str(kite)] = cas.mtimes(f_drag_earth.T, dq)
    outputs['power_balance']['P_side' + str(kite)] = cas.mtimes(f_side_earth.T, dq)

    if int(options['kite_dof']) == 6:
        omega = variables['xd']['omega'+str(kite)+str(parent)]
        m_aero_body = outputs['aerodynamics']['m_aero_body'+str(kite)]
        outputs['power_balance']['P_moment'+str(kite)] = cas.mtimes(m_aero_body.T, omega)

    return outputs

def collect_tether_drag_losses(variables, tether_drag_forces, outputs, architecture):

    if 'power_balance' not in list(outputs.keys()):
        # initialize
        outputs['power_balance'] = {}

    # get dissipation power from tether drag
    for n in range(1, architecture.number_of_nodes):
        parent = architecture.parent_map[n]
        dq_n = variables['xd']['dq' + str(n) + str(parent)]  # node velocity
        force = tether_drag_forces['f' + str(n) + str(parent)]
        outputs['power_balance']['P_tetherdrag' + str(n)] = cas.mtimes(force.T, dq_n)

    return outputs

def collect_aero_validity_outputs(options, base_aerodynamic_quantities, outputs):

    kite = base_aerodynamic_quantities['kite']
    ua = base_aerodynamic_quantities['air_velocity']
    kite_dcm = base_aerodynamic_quantities['kite_dcm']

    if 'aero_validity' not in list(outputs.keys()):
        outputs['aero_validity'] = {}
    tightness = options['model_bounds']['aero_validity']['scaling']
    airspeed_ref = options['model_bounds']['aero_validity']['airspeed_ref']

    if 'aerodynamics' not in list(outputs.keys()):
        outputs['aerodynamics'] = {}

    ehat1 = kite_dcm[:, 0]  # chordwise, from leading edge to trailing edge
    ehat2 = kite_dcm[:, 1]  # spanwise, from positive edge to negative edge
    ehat3 = kite_dcm[:, 2]  # up

    alpha = get_alpha(ua, kite_dcm)
    beta = get_beta(ua, kite_dcm)

    alpha_min = options['aero']['alpha_min_deg'] * np.pi / 180.0
    alpha_max = options['aero']['alpha_max_deg'] * np.pi / 180.0
    beta_min = options['aero']['beta_min_deg'] * np.pi / 180.0
    beta_max = options['aero']['beta_max_deg'] * np.pi / 180.0

    alpha_ub_unscaled = (cas.mtimes(ua.T, ehat3) - cas.mtimes(ua.T, ehat1) * alpha_max)
    alpha_lb_unscaled = (- cas.mtimes(ua.T, ehat3) + cas.mtimes(ua.T, ehat1) * alpha_min)
    beta_ub_unscaled = (cas.mtimes(ua.T, ehat2) - cas.mtimes(ua.T, ehat1) * beta_max)
    beta_lb_unscaled = (- cas.mtimes(ua.T, ehat2) + cas.mtimes(ua.T, ehat1) * beta_min)

    alpha_ub = alpha_ub_unscaled * tightness / airspeed_ref / vect_op.smooth_abs(alpha_max)
    alpha_lb = alpha_lb_unscaled * tightness / airspeed_ref / vect_op.smooth_abs(alpha_min)
    beta_ub = beta_ub_unscaled * tightness / airspeed_ref / vect_op.smooth_abs(beta_max)
    beta_lb = beta_lb_unscaled * tightness / airspeed_ref / vect_op.smooth_abs(beta_min)

    outputs['aero_validity']['alpha_ub' + str(kite)] = alpha_ub
    outputs['aero_validity']['alpha_lb' + str(kite)] = alpha_lb
    outputs['aero_validity']['beta_ub' + str(kite)] = beta_ub
    outputs['aero_validity']['beta_lb' + str(kite)] = beta_lb

    outputs['aerodynamics']['alpha' + str(kite)] = alpha
    outputs['aerodynamics']['beta' + str(kite)] = beta
    outputs['aerodynamics']['alpha_deg' + str(kite)] = alpha * 180. / np.pi
    outputs['aerodynamics']['beta_deg' + str(kite)] = beta * 180. / np.pi


    CD = base_aerodynamic_quantities['aero_coefficients']['CD_var']
    CD_min = options['model_bounds']['aero_validity']['CD_min']
    drag_lb = CD_min - CD
    outputs['aero_validity']['drag_lb' + str(kite)] = drag_lb


    return outputs

def collect_local_performance_outputs(architecture, atmos, wind, variables, parameters, base_aerodynamic_quantities, outputs):

    kite = base_aerodynamic_quantities['kite']
    q = base_aerodynamic_quantities['q']
    airspeed = base_aerodynamic_quantities['airspeed']
    CL = base_aerodynamic_quantities['aero_coefficients']['CL']
    CD = base_aerodynamic_quantities['aero_coefficients']['CD']

    xd = variables['xd']
    elevation_angle = get_elevation_angle(xd)

    parent = architecture.parent_map[kite]

    if 'local_performance' not in list(outputs.keys()):
        outputs['local_performance'] = {}

    [CR, phf_loyd, p_loyd, speed_loyd] = get_loyd_comparison(atmos, wind, xd, kite, parent, CL, CD, parameters, elevation_angle)

    outputs['local_performance']['CR' + str(kite)] = CR
    outputs['local_performance']['p_loyd' + str(kite)] = p_loyd
    outputs['local_performance']['speed_loyd' + str(kite)] = speed_loyd
    outputs['local_performance']['phf_loyd' + str(kite)] = phf_loyd

    outputs['local_performance']['speed_ratio' + str(kite)] = airspeed / vect_op.norm(wind.get_velocity(q[2]))
    outputs['local_performance']['speed_ratio_loyd' + str(kite)] = speed_loyd / vect_op.norm(wind.get_velocity(q[2]))

    outputs['local_performance']['radius_of_curvature' + str(kite)] = path_based_geom.get_radius_of_curvature(variables, kite, parent)


    return outputs

def collect_environmental_outputs(atmos, wind, base_aerodynamic_quantities, outputs):

    kite = base_aerodynamic_quantities['kite']
    q = base_aerodynamic_quantities['q']

    if 'environment' not in list(outputs.keys()):
        outputs['environment'] = {}

    outputs['environment']['windspeed' + str(kite)] = vect_op.norm(wind.get_velocity(q[2]))
    outputs['environment']['pressure' + str(kite)] = atmos.get_pressure(q[2])
    outputs['environment']['temperature' + str(kite)] = atmos.get_temperature(q[2])
    outputs['environment']['density' + str(kite)] = atmos.get_density(q[2])

    return outputs


def get_loyd_comparison(atmos, wind, xd, n, parent, CL, CD, parameters, elevation_angle=0.):
    # for elevation angle cosine losses see Van der Lind, p. 477, AWE book

    q = xd['q' + str(n) + str(parent)]

    epsilon = 1.e-8
    CR = CL * (1. + (CD / (CL + epsilon))**2.)**0.5

    windspeed = vect_op.norm(wind.get_velocity(q[2]))
    power_density = get_power_density(atmos, wind, q[2])

    phf_loyd = perf_op.get_loyd_phf(CL, CD, elevation_angle)

    s_ref = parameters['theta0','geometry','s_ref']
    p_loyd = perf_op.get_loyd_power(power_density, CL, CD, s_ref, elevation_angle)

    speed_loyd = 2. * CR / 3. / CD * windspeed * np.cos(elevation_angle)

    return [CR, phf_loyd, p_loyd, speed_loyd]

def get_power_harvesting_factor(options, atmos, wind, variables, parameters,architecture):

    number_of_kites = architecture.number_of_kites

    xd = variables['xd']
    xa = variables['xa']

    s_ref = parameters['theta0', 'geometry', 's_ref']

    available_power_at_kites = 0.
    for n in architecture.kite_nodes:
        parent = architecture.parent_map[n]
        height = xd['q' + str(n) + str(parent)][2]

        available_power_at_kites += get_power_density(atmos, wind, height) * s_ref

    current_power = xa['lambda10'] * xd['l_t'] * xd['dl_t']

    node_1_height = xd['q10'][2]
    available_power_at_node_1_height = get_power_density(atmos, wind, node_1_height) * s_ref * number_of_kites

    phf = current_power / available_power_at_kites
    phf_hubheight = current_power / available_power_at_node_1_height

    return [current_power, phf, phf_hubheight, available_power_at_node_1_height]

def get_elevation_angle(xd):
    length_along_ground = (xd['q10'][0] ** 2. + xd['q10'][1] ** 2.) ** 0.5
    elevation_angle = np.arctan2(xd['q10'][2], length_along_ground)

    return elevation_angle




def get_alpha(ua, r):
    ehat1 = r[:, 0]  # chordwise, from le to te
    ehat2 = r[:, 1]  # spanwise, from pe to ne
    ehat3 = r[:, 2]  # up

    z_component = cas.mtimes(ua.T, ehat3)
    x_component = vect_op.smooth_abs(cas.mtimes(ua.T, ehat1))
    # x component had better be positive

    # the small angle approximation of:
    # alpha = cas.arctan(z_component / x_component)
    alpha = z_component / x_component

    return alpha

def get_beta(ua, r):
    ehat1 = r[:, 0]  # chordwise, from le to te
    ehat2 = r[:, 1]  # spanwise, from pe to ne
    ehat3 = r[:, 2]  # up

    y_component = cas.mtimes(ua.T, ehat2)
    x_component = vect_op.smooth_abs(cas.mtimes(ua.T, ehat1))
    # x component had better be positive

    # the small angle approximation of:
    # beta = cas.arctan(y_component / x_component)
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
