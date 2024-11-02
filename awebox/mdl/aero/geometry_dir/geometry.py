#
#    This file is part of awebox.
#
#    awebox -- A modeling and optimization framework for multi-kite AWE systems.
#    Copyright (C) 2017-2021 Jochem De Schutter, Rachel Leuthold, Moritz Diehl,
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
geometry values needed for general induction modelling
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2017-21
- edit: jochem de schutter, alu-fr 2019
'''
import pdb

import casadi.tools as cas
import numpy as np
from awebox.logger.logger import Logger as awelogger
import awebox.mdl.architecture as archi
import awebox.mdl.wind as wind_module

import awebox.mdl.aero.geometry_dir.frenet_geometry as frenet_geom
import awebox.mdl.aero.geometry_dir.averaged_geometry as averaged_geom
import awebox.mdl.aero.geometry_dir.parent_geometry as parent_geom
import awebox.mdl.aero.geometry_dir.unit_normal as unit_normal

import awebox.tools.vector_operations as vect_op
import awebox.tools.struct_operations as struct_op
import awebox.tools.print_operations as print_op

def get_available_geometry_types():
    return ['averaged', 'parent', 'frenet']

def get_geometry_type(model_options):
    geometry_type = model_options['aero']['geometry_type']
    return geometry_type

def raise_and_log_unavailable_geometry_type_error(model_options):
    geometry_type = get_geometry_type(model_options)
    if geometry_type not in get_available_geometry_types():
        message = geometry_type + ' aerodynamic geometry type is not yet available. perhaps check your spelling?'
        print_op.log_and_raise_error(message)

    return None

def print_warning_if_relevant(model_options, variables, architecture):
    geometry_type = get_geometry_type(model_options)
    if geometry_type == 'averaged':
        return averaged_geom.print_warning_if_relevant(architecture)
    elif geometry_type == 'frenet':
        return frenet_geom.print_warning_if_relevant(variables, architecture)
    elif geometry_type == 'parent':
        return parent_geom.print_warning_if_relevant(architecture)

    raise_and_log_unavailable_geometry_type_error(model_options)
    return None

def get_center_position(model_options, parent, variables, architecture):
    geometry_type = get_geometry_type(model_options)
    if geometry_type == 'averaged':
        return averaged_geom.get_center_position(parent, variables, architecture)
    elif geometry_type == 'frenet':
        return frenet_geom.get_center_position(parent, variables, architecture)
    elif geometry_type == 'parent':
        return parent_geom.get_center_position(parent, variables, architecture)

    raise_and_log_unavailable_geometry_type_error(model_options)
    return None

def get_center_velocity(model_options, parent, variables, architecture):
    geometry_type = get_geometry_type(model_options)
    if geometry_type == 'averaged':
        return averaged_geom.get_center_velocity(parent, variables, architecture)
    elif geometry_type == 'frenet':
        return frenet_geom.get_center_velocity(parent, variables, architecture)
    elif geometry_type == 'parent':
        return parent_geom.get_center_velocity(parent, variables, architecture)

    raise_and_log_unavailable_geometry_type_error(model_options)
    return None


def get_local_period_of_rotation(model_options, variables_si, kite, architecture, scaling):

    # the velocity we care about, is the one associated with rotation. so:
    # vec_v = vec_omega x vec_r
    # vec_v_comp_perp_to_r = vec_omega x vec_r
    # v_comp_perp_to_r = omega r sin(90deg)
    # omega = v_comp_perp_to_r / r

    # notice that if you use anti-clockwise initialization, this routine will give a radius pointing away from the kite.

    parent = architecture.parent_map[kite]

    dq_kite = struct_op.get_variable_from_model_or_reconstruction(variables_si, 'x', 'dq' + str(kite) + str(parent))
    dx_center = get_center_velocity(model_options, parent, variables_si, architecture)
    vec_v = dq_kite - dx_center

    vec_to_kite = get_vector_from_center_to_kite(model_options, variables_si, architecture, kite)
    rotation_outputs = unit_normal.get_rotation_axes_outputs(model_options, variables_si, {}, architecture)['rotation']
    ehat_radial = rotation_outputs['ehat_radial' + str(kite)]
    radius = vect_op.abs(cas.mtimes(ehat_radial.T, vec_to_kite))

    vec_v_perpendicular_to_r = vec_v - cas.mtimes(ehat_radial.T, vec_v) * ehat_radial
    v_component_perpendicular_to_r = vect_op.norm(vec_v_perpendicular_to_r)

    omega = v_component_perpendicular_to_r / radius

    period = 2. * np.pi / omega

    return period

def get_vector_from_center_to_kite(model_options, variables, architecture, kite):

    parent = architecture.parent_map[kite]

    q_kite = struct_op.get_variable_from_model_or_reconstruction(variables, 'x', 'q' + str(kite) + str(parent))
    center = get_center_position(model_options, parent, variables, architecture)

    radius_vec = q_kite - center

    return radius_vec

def kite_motion_is_right_hand_rule_positive_around_wind_direction(model_options, variables_si, kite, architecture, wind):
    parent = architecture.parent_map[kite]

    q_kite = struct_op.get_variable_from_model_or_reconstruction(variables_si, 'x', 'q' + str(kite) + str(parent))
    dq_kite = struct_op.get_variable_from_model_or_reconstruction(variables_si, 'x', 'dq' + str(kite) + str(parent))

    x_center = get_center_position(model_options, parent, variables_si, architecture)
    dx_center = get_center_velocity(model_options, parent, variables_si, architecture)

    u_hat = wind.get_wind_direction()

    # the direction of the tangential vorticity must be opposite to the direction of the kite rotation
    pseudo_r_vec = q_kite - x_center
    theta_vec = vect_op.cross(u_hat, pseudo_r_vec)
    pseudo_dr_dt_vec = dq_kite - dx_center
    component_in_theta_vec_direction = cas.mtimes(pseudo_dr_dt_vec.T, theta_vec)
    kite_motion_is_right_hand_rule_positive_around_wind_direction = vect_op.unitstep(component_in_theta_vec_direction)

    return kite_motion_is_right_hand_rule_positive_around_wind_direction


def collect_geometry_outputs(model_options, wind, variables_si, outputs, parameters, architecture, scaling):
    if 'geometry' not in outputs.keys():
        outputs['geometry'] = {}

    for kite in architecture.kite_nodes:
        local_period_of_rotation = get_local_period_of_rotation(model_options, variables_si, kite, architecture, scaling)
        vector_from_center_to_kite = get_vector_from_center_to_kite(model_options, variables_si, architecture, kite)
        clockwise_rotation = kite_motion_is_right_hand_rule_positive_around_wind_direction(model_options, variables_si, kite, architecture, wind)
        radius_of_curvature = frenet_geom.get_radius_of_curvature(variables_si, kite, architecture.parent_map[kite])

        outputs['geometry']['local_period_of_rotation' + str(kite)] = local_period_of_rotation
        outputs['geometry']['vector_from_center_to_kite' + str(kite)] = vector_from_center_to_kite
        outputs['geometry']['kite_motion_is_right_hand_rule_positive_around_wind_direction' + str(kite)] = clockwise_rotation
        outputs['geometry']['radius_of_curvature' + str(kite)] = radius_of_curvature

    for parent in architecture.layer_nodes:
        x_center = get_center_position(model_options, parent, variables_si, architecture)
        dx_center = get_center_velocity(model_options, parent, variables_si, architecture)
        wind_velocity_at_center = wind.get_velocity(x_center[2])
        vec_u_zero = wind_velocity_at_center - dx_center

        outputs['geometry']['x_center' + str(parent)] = x_center
        outputs['geometry']['dx_center' + str(parent)] = dx_center
        outputs['geometry']['wind_velocity_at_center' + str(parent)] = wind_velocity_at_center
        outputs['geometry']['vec_u_zero' + str(parent)] = vec_u_zero

        average_radius = cas.DM.zeros((1, 1))
        average_period_of_rotation = cas.DM.zeros((1, 1))
        for kite in architecture.kites_map[parent]:
            average_radius += outputs['geometry']['radius_of_curvature' + str(kite)] / float(architecture.get_number_siblings(kite))
            average_period_of_rotation += outputs['geometry']['local_period_of_rotation' + str(kite)] / float(
                architecture.get_number_siblings(kite))

        b_ref = parameters['theta0', 'geometry', 'b_ref']

        outputs['geometry']['average_radius' + str(parent)] = average_radius
        outputs['geometry']['average_relative_radius' + str(parent)] = average_radius / b_ref
        outputs['geometry']['average_curvature' + str(parent)] = 1./average_radius
        outputs['geometry']['average_period_of_rotation' + str(parent)] = average_period_of_rotation

    outputs = unit_normal.get_rotation_axes_outputs(model_options, variables_si, outputs, architecture)

    return outputs

def construct_geometry_test_object(geometry_type='averaged'):

    model_options = {}
    model_options['aero'] = {}
    model_options['aero']['geometry_type'] = geometry_type

    model_options['wind'] = {}
    model_options['wind']['u_ref'] = 1.
    model_options['wind']['model'] = 'uniform'
    model_options['wind']['z_ref'] = -999.
    model_options['wind']['log_wind'] = {'z0_air': -999}
    model_options['wind']['power_wind'] = {'exp_ref': -999}

    model_options['induction'] = {'normal_vector_model': 'xhat'}

    wind_struct = cas.struct([
        cas.entry('u_ref', shape=(1, 1)),
        cas.entry('z_ref', shape=(1, 1))
    ])
    theta_struct = cas.struct_symSX([
        cas.entry('wind', struct=wind_struct)
    ])
    param_struct = cas.struct_symSX([
        cas.entry('theta0', struct=theta_struct),
    ])

    wind_params = param_struct(0.)
    wind_params['theta0', 'wind', 'u_ref'] = model_options['wind']['u_ref']
    wind_params['theta0', 'wind', 'z_ref'] = model_options['wind']['z_ref']

    wind = wind_module.Wind(model_options['wind'], wind_params, suppress_type_incompatibility_warning=True)
    architecture = archi.Architecture({1: 0, 2: 1, 3: 1})

    system_states = []
    system_derivatives = []
    for node in range(1, architecture.number_of_nodes):
        parent = architecture.parent_map[node]
        var_name = 'q' + str(node) + str(parent)

        system_states.extend([(var_name, (3, 1))])
        system_states.extend([('d' + var_name, (3, 1))])
        system_derivatives.extend([('dd' + var_name, (3, 1))])

    system_variable_list = {}
    system_variable_list['xdot'] = system_derivatives
    system_variable_list['x'] = system_states

    var_struct, variables_dict = struct_op.generate_variable_struct(system_variable_list)

    variables_si = var_struct(0.)

    xhat = vect_op.xhat_dm()
    ahat = vect_op.yhat_dm()
    bhat = vect_op.zhat_dm()
    clockwise = 1.

    # note, if this value is increased, then the missing frenet torsion information will lead to test failure.
    velocity = 0. * xhat

    x_center = 2. * xhat + 3. * ahat + 4. * bhat
    dx_center = velocity
    ddx_center = cas.DM.zeros((3, 1))
    radius = 5.

    omega = 0.5 * 2. * np.pi
    time = 0.5
    psi_of_t = omega * time

    period = 2. * np.pi / omega

    variables_si['x', 'q10'] = x_center
    variables_si['x', 'dq10'] = dx_center
    variables_si['xdot', 'ddq10'] = ddx_center

    for kdx in range(len(architecture.kite_nodes)):

        psi_0 = 2. * np.pi * float(kdx) / float(architecture.number_of_kites)
        psi = psi_0 + psi_of_t

        q_kite = x_center + radius * (ahat * np.cos(psi) + bhat * np.sin(psi) )
        dq_kite = dx_center + radius * omega * (-1. * ahat * np.sin(psi) + bhat * np.cos(psi) )
        ddq_kite = ddx_center + radius * omega**2. * (-1. * ahat * np.cos(psi) - bhat * np.sin(psi) )
        # dddq_kite = radius * omega**3. * (ahat * np.sin(psi) - bhat * np.cos(psi))

        kite = architecture.kite_nodes[kdx]
        parent = architecture.parent_map[kite]
        variables_si['x', 'q' + str(kite) + str(parent)] = q_kite
        variables_si['x', 'dq' + str(kite) + str(parent)] = dq_kite
        variables_si['xdot', 'ddq' + str(kite) + str(parent)] = ddq_kite

    return model_options, variables_si, architecture, wind, x_center, dx_center, period, clockwise


def test_specific_geometry_type(geometry_type='averaged', epsilon=1.e-6):
    model_options, variables_si, architecture, wind, expected_position, expected_velocity, expected_period, expected_clockwise = construct_geometry_test_object(geometry_type=geometry_type)
    scaling = cas.DM.ones(variables_si.shape)
    parent = architecture.layer_nodes[0]

    found_position = get_center_position(model_options, parent, variables_si, architecture)
    found_velocity = get_center_velocity(model_options, parent, variables_si, architecture)
    found_period2 = get_local_period_of_rotation(model_options, variables_si, 2, architecture, scaling)
    found_period3 = get_local_period_of_rotation(model_options, variables_si, 3, architecture, scaling)
    found_clockwise2 = kite_motion_is_right_hand_rule_positive_around_wind_direction(model_options, variables_si, 2, architecture, wind)
    found_clockwise3 = kite_motion_is_right_hand_rule_positive_around_wind_direction(model_options, variables_si, 3, architecture, wind)

    error_position = vect_op.norm(expected_position - found_position) / vect_op.norm(expected_position)
    diff_velocity = vect_op.norm(expected_velocity - found_velocity)
    error_period2 = (expected_period - found_period2) / expected_period
    error_period3 = (expected_period - found_period3) / expected_period
    diff_clockwise2 = (expected_clockwise - found_clockwise2)
    diff_clockwise3 = (expected_clockwise - found_clockwise3)

    cond_position = (error_position < epsilon)
    cond_velocity = (diff_velocity < epsilon)
    cond_period2 = (error_period2**2. < epsilon**2)
    cond_period3 = (error_period3**2. < epsilon**2.)
    cond_clockwise2 = (diff_clockwise2**2. < epsilon**2.)
    cond_clockwise3 = (diff_clockwise3**2. < epsilon**2.)
    criteria = cond_position and cond_velocity and cond_period2 and cond_period3 and cond_clockwise2 and cond_clockwise3

    if not criteria:
        message = 'geometry type (' + geometry_type + ') does not behave as expected in the highly-simplified test-case'
        print_op.log_and_raise_error(message)

    return None


def test(epsilon=1.e-6):
    available_types = get_available_geometry_types()
    for geometry_type in available_types:
        test_specific_geometry_type(geometry_type=geometry_type, epsilon=epsilon)
    return None

if __name__ == "__main__":
    test()