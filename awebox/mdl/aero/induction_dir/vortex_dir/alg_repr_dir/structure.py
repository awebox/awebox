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
define the structure of an algebtraic-representation vortex-modelled wake
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2022
'''
import copy
import pdb

import casadi.tools as cas
import numpy as np
import matplotlib.pyplot as plt

import awebox.mdl.aero.induction_dir.vortex_dir.tools as vortex_tools

import awebox.mdl.architecture as archi
import awebox.mdl.wind as wind_module
import awebox.tools.struct_operations as struct_op
import awebox.tools.vector_operations as vect_op
import awebox.tools.print_operations as print_op

from awebox.logger.logger import Logger as awelogger


# get the names of the specific variables, for use in model_system and in local wake objects
# remember, that the presence of the wake-node-variables in model_variables is not itself the wake objects,
# the wake objects are constructed *with* those variables.
# which means that each of the vortex elements induced velocities depends on the wake structure,
# but not the positions and strengths; those belong to the wake nodes






def construct_test_model_variable_structures(element_type='finite_filament'):

    options = {}

    options['wind'] = {}
    options['wind']['u_ref'] = 1.
    options['wind']['model'] = 'uniform'
    options['wind']['z_ref'] = -999.
    options['wind']['log_wind'] = {'z0_air': -999}
    options['wind']['power_wind'] = {'exp_ref': -999}

    wake_nodes = 2
    rings = wake_nodes
    options['aero'] = {}
    options['aero']['geometry_type'] = 'frenet'
    options['aero']['vortex'] = {}
    options['aero']['vortex']['wake_nodes'] = wake_nodes
    options['aero']['vortex']['rings'] = rings
    options['aero']['vortex']['core_to_chord_ratio'] = 0.1
    options['aero']['vortex']['far_wake_element_type'] = element_type
    options['aero']['vortex']['approximation_order_for_elliptic_integrals'] = 3

    options['induction'] = {}
    options['induction']['vortex_wake_nodes'] = wake_nodes
    options['induction']['vortex_rings'] = rings
    options['induction']['vortex_far_wake_convection_time'] = 1.
    options['induction']['vortex_far_wake_element_type'] = element_type
    options['induction']['vortex_representation'] = 'alg'
    options['induction']['vortex_epsilon_m'] = 1.0e-8
    options['induction']['vortex_epsilon_r'] = 1.0e-8

    options['n_k'] = 10
    options['collocation'] = {'d':4, 'scheme':'radau'}
    options['discretization'] = 'direct_collocation'
    options['phase_fix'] = 'single_reelout'
    options['phase_fix_reelout'] = 0.5

    architecture = archi.Architecture({1: 0})

    system_lifted, system_states = vortex_tools.extend_system_variables(options, [], [], architecture)
    system_derivatives = []
    for kite in architecture.kite_nodes:
        parent = architecture.parent_map[kite]
        system_states.extend([('q' + str(kite) + str(parent), (3, 1))])
        system_states.extend([('dq' + str(kite) + str(parent), (3, 1))])
        system_derivatives.extend([('ddq' + str(kite) + str(parent), (3, 1))])

    system_variable_list = {'xl': system_lifted,
                            'xd': system_states,
                            'xddot': system_derivatives
                            }

    var_struct, variables_dict = struct_op.generate_variable_struct(system_variable_list)

    geometry_struct = cas.struct([
        cas.entry("c_ref", shape=(1, 1)),
        cas.entry("b_ref", shape=(1, 1))
    ])
    wind_struct = cas.struct([
        cas.entry('u_ref', shape=(1, 1)),
        cas.entry('z_ref', shape=(1, 1))
    ])
    theta_struct = cas.struct_symSX([
        cas.entry('geometry', struct=geometry_struct),
        cas.entry('wind', struct=wind_struct)
    ])
    param_struct = cas.struct_symSX([
        cas.entry('theta0', struct=theta_struct),

    ])

    wind_params = param_struct(0.)
    wind_params['theta0', 'wind', 'u_ref'] = options['wind']['u_ref']
    wind_params['theta0', 'wind', 'z_ref'] = options['wind']['z_ref']

    wind = wind_module.Wind(options['wind'], wind_params, suppress_type_incompatibility_warning=True)

    return options, architecture, wind, var_struct, param_struct, variables_dict

def construct_straight_flight_test_object(element_type='finite_filament'):
    options, architecture, wind, var_struct, param_struct, variables_dict = construct_test_model_variable_structures(
        element_type)
    kite = architecture.kite_nodes[0]

    half_b_span_vec = 0.5 * vect_op.yhat_dm()
    x_kite = cas.DM.zeros((3, 1))

    x_NE = x_kite - half_b_span_vec
    x_PE = x_kite + half_b_span_vec

    variables_si = var_struct(0.)

    variables_si['xl', vortex_tools.get_wake_node_position_name(kite, 'ext', 0)] = x_PE
    variables_si['xl', vortex_tools.get_wake_node_position_name(kite, 'int', 0)] = x_NE
    variables_si['xl', vortex_tools.get_wake_node_position_name(kite, 'ext', 1)] = x_PE + vect_op.xhat_dm()
    variables_si['xl', vortex_tools.get_wake_node_position_name(kite, 'int', 1)] = x_NE + vect_op.xhat_dm()
    variables_si['xl', vortex_tools.get_vortex_ring_strength_name(kite, 0)] = cas.DM(4.)
    variables_si['xl', vortex_tools.get_vortex_ring_strength_name(kite, 1)] = cas.DM(1.)

    parameters = param_struct(0.)
    parameters['theta0', 'geometry', 'c_ref'] = 0.1

    return options, architecture, wind, var_struct, param_struct, variables_dict, variables_si, parameters

def construct_vortex_ring_test_object(element_type='semi_infinite_filament'):
    options, architecture, wind, var_struct, param_struct, variables_dict = construct_test_model_variable_structures(
        element_type)
    kite = architecture.kite_nodes[0]

    half_b_span_vec = 0.5 * vect_op.yhat_dm()
    x_kite = cas.DM.zeros((3, 1))

    x_NE = x_kite - half_b_span_vec
    x_PE = x_kite + half_b_span_vec

    variables_si = var_struct(0.)

    variables_si['xl', vortex_tools.get_wake_node_position_name(kite, 'ext', 0)] = x_PE
    variables_si['xl', vortex_tools.get_wake_node_position_name(kite, 'int', 0)] = x_NE
    variables_si['xl', vortex_tools.get_wake_node_position_name(kite, 'ext', 1)] = x_PE + vect_op.xhat_dm()
    variables_si['xl', vortex_tools.get_wake_node_position_name(kite, 'int', 1)] = x_NE + vect_op.xhat_dm()
    variables_si['xl', vortex_tools.get_vortex_ring_strength_name(kite, 0)] = cas.DM(4.)
    variables_si['xl', vortex_tools.get_vortex_ring_strength_name(kite, 1)] = cas.DM(0.)

    parameters = param_struct(0.)
    parameters['theta0', 'geometry', 'c_ref'] = 0.1

    return options, architecture, wind, var_struct, param_struct, variables_dict, variables_si, parameters


def construct_circular_flight_test_object(element_type='semi_infinite_cylinder'):
    options, architecture, wind, var_struct, param_struct, variables_dict = construct_test_model_variable_structures(
        element_type)
    kite = architecture.kite_nodes[0]
    parent = architecture.parent_map[kite]

    x_center = cas.DM.zeros((3, 1))
    a_hat = vect_op.yhat()
    b_hat = vect_op.zhat()

    vec_u_wind = wind.get_velocity(1.)

    omega = 2. * np.pi * 1.e8
    delta_t = 1.

    pitch = cas.DM(2. * np.pi * vect_op.norm(vec_u_wind) / omega)  # -> 1.e-8

    theta0 = 0.
    theta1 = -1. * omega * delta_t

    b_span = 1.
    radius_kite = 2.
    r_ext = radius_kite + b_span / 2.
    r_int = radius_kite - b_span / 2.

    # notice, that this describes clockwise (right-hand-rule) rotation about positive xhat
    x_kite = x_center + radius_kite * (np.cos(theta0) * a_hat + np.sin(theta0) * b_hat)
    dx_kite = radius_kite * omega * (-1. * np.sin(theta0) * a_hat + np.cos(theta0) * b_hat)
    ddx_kite = radius_kite * omega**2. * (-1. * np.cos(theta0) * a_hat - np.sin(theta0) * b_hat )

    x_center0 = x_center
    x_center1 = x_center0 + delta_t * vec_u_wind

    x_PE_0 = x_center0 + r_ext * (np.cos(theta0) * a_hat + np.sin(theta0) * b_hat)
    x_PE_1 = x_center1 + r_ext * (np.cos(theta1) * a_hat + np.sin(theta1) * b_hat)

    x_NE_0 = x_center0 + r_int * (np.cos(theta0) * a_hat + np.sin(theta0) * b_hat)
    x_NE_1 = x_center1 + r_int * (np.cos(theta1) * a_hat + np.sin(theta1) * b_hat)

    variables_si = var_struct(0.)

    variables_si['xd', 'q10'] = x_kite
    variables_si['xd', 'dq10'] = dx_kite
    variables_si['xddot', 'ddq10'] = ddx_kite

    variables_si['xl', vortex_tools.get_wake_node_position_name(kite, 'ext', 0)] = x_PE_0
    variables_si['xl', vortex_tools.get_wake_node_position_name(kite, 'int', 0)] = x_NE_0
    variables_si['xl', vortex_tools.get_wake_node_position_name(kite, 'ext', 1)] = x_PE_1
    variables_si['xl', vortex_tools.get_wake_node_position_name(kite, 'int', 1)] = x_NE_1
    variables_si['xl', vortex_tools.get_vortex_ring_strength_name(kite, 0)] = cas.DM(4.)
    variables_si['xl', vortex_tools.get_vortex_ring_strength_name(kite, 1)] = cas.DM(1.)

    variables_si['xl', vortex_tools.get_far_wake_cylinder_center_position_name(parent)] = x_center
    variables_si['xl', vortex_tools.get_far_wake_cylinder_pitch_name(parent)] = pitch

    parameters = param_struct(0.)
    parameters['theta0', 'geometry', 'c_ref'] = 0.1

    return options, architecture, wind, var_struct, param_struct, variables_dict, variables_si, parameters