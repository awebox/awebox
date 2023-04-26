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
energy terms
_python-3.5 / casadi-3.4.5
- edited: rachel leuthold, jochem de schutter alu-fr 2017-20
'''
import pdb

import casadi.tools as cas
import numpy as np

import awebox.tools.vector_operations as vect_op
import awebox.tools.struct_operations as struct_op
import awebox.tools.print_operations as print_op
import awebox.mdl.aero.tether_dir.tether_aero as tether_aero

from awebox.logger.logger import Logger as awelogger


def energy_outputs(options, parameters, outputs, variables_si, architecture, scaling):

    # kinetic and potential energy in the system
    energy_types = ['e_kinetic', 'e_potential']
    for etype in energy_types:
        if etype not in list(outputs.keys()):
            outputs[etype] = {}

    number_of_nodes = architecture.number_of_nodes
    for node in range(1, number_of_nodes):
        outputs = add_node_kinetic(node, options, variables_si, parameters, outputs, architecture, scaling)
        outputs = add_node_potential(node, options, variables_si, parameters, outputs, architecture, scaling)

    outputs = add_ground_station_kinetic(options, variables_si, parameters, outputs, architecture, scaling)
    outputs = add_ground_station_potential(outputs)

    return outputs


def add_node_kinetic(node, options, variables_si, parameters, outputs, architecture, scaling):

    label = architecture.node_label(node)
    parent_label = architecture.parent_label(node)

    node_has_a_kite = node in architecture.kite_nodes
    kites_have_6dof = int(options['kite_dof']) == 6

    q_node = variables_si['x']['q' + label]
    dq_node = variables_si['x']['dq' + label]
    if node == 1:
        q_parent = cas.DM.zeros((3, 1))

        if 'dl_t' in variables_si['x'].keys():
            segment_vector = q_node - q_parent
            ehat_tether = vect_op.normalize(segment_vector)
            dq_parent = variables_si['x']['dl_t'] * ehat_tether
        else:
            dq_parent = cas.DM.zeros((3, 1))

    else:
        q_parent = variables_si['x']['q' + parent_label]
        dq_parent = variables_si['x']['dq' + parent_label]

    segment_properties = tether_aero.get_tether_segment_properties(options, architecture, scaling, variables_si, parameters, node)
    segment_vector = q_node - q_parent
    segment_length = segment_properties['seg_length']
    mass_segment = segment_properties['seg_mass']

    dq_average = (dq_node + dq_parent) / 2.
    e_kin_trans = 0.5 * mass_segment * cas.mtimes(dq_average.T, dq_average)

    if node_has_a_kite:
        mass_kite = parameters['theta0', 'geometry', 'm_k']
        e_kin_trans += 0.5 * mass_kite * cas.mtimes(dq_node.T, dq_node)

    v_difference = dq_node - dq_average
    ehat_tether = vect_op.normalize(segment_vector)
    v_rotation = v_difference - cas.mtimes(v_difference.T, ehat_tether) * ehat_tether
    radius_of_rod_rotation = (segment_length / 2.)
    omega = v_rotation / radius_of_rod_rotation
    moment_of_inertia = (1. / 12.) * mass_segment * segment_length**2.
    e_kin_rot = 0.5 * moment_of_inertia * cas.mtimes(omega.T, omega)

    if node_has_a_kite and kites_have_6dof:
        omega = variables_si['x']['omega' + label]
        j_kite = parameters['theta0', 'geometry', 'j']
        e_kin_rot += 0.5 * cas.mtimes(cas.mtimes(omega.T, j_kite), omega)

    e_kinetic = e_kin_trans + e_kin_rot

    outputs['e_kinetic']['q' + label] = e_kinetic

    return outputs


def add_node_potential(node, options, variables_si, parameters, outputs, architecture, scaling):

    label = architecture.node_label(node)
    parent_label = architecture.parent_label(node)

    node_has_a_kite = node in architecture.kite_nodes

    gravity = parameters['theta0', 'atmosphere', 'g']

    q_node = variables_si['x']['q' + label]
    if node == 1:
        q_parent = cas.DM.zeros((3, 1))
    else:
        q_parent = variables_si['x']['q' + parent_label]
    q_mean = (q_node + q_parent) / 2.

    segment_properties = tether_aero.get_tether_segment_properties(options, architecture, scaling, variables_si, parameters, node)
    mass_segment = segment_properties['seg_mass']

    e_potential = gravity * mass_segment * q_mean[2]

    if node_has_a_kite:
        mass = parameters['theta0', 'geometry', 'm_k']
        e_potential += gravity * mass * q_node[2]

    outputs['e_potential']['q' + label] = e_potential

    return outputs


def add_ground_station_potential(outputs):
    # the winch is at ground level
    e_potential = cas.DM(0.)
    outputs['e_potential']['ground_station'] = e_potential
    return outputs


def add_ground_station_kinetic(options, variables_si, parameters, outputs, architecture, scaling):

    # assume that ground station is two cylinders:
    # - the inner one a solid, homogenous cylinder, made of metal (the drum),
    #           with I_drum = 1/2 (m_drum) (r_drum)^2
    # - and the outer one a thin, homogenous cylindrical shell, made of wound tether,
    #           with I_shell = (m_tether) (r_drum + r_tether)^2
    # so that the total kinetic energy = 1/2 I_total omega^2
    # and the rotation of both is determined by a no-slip condition on the winding tether
    # omega radius_of_tether_winding = dl_t

    if options['tether']['use_wound_tether']:

        if not (('l_t_full' in variables_si['theta'].keys()) and ('dl_t' in variables_si['x'].keys())):
            message = 'awebox does not have necessary variables to compute the kinetic energy of the ground station, ' \
                      'according to the wound-tether model'
            print_op.log_and_raise_error(message)

        radius_drum = parameters['theta0', 'ground_station', 'r_gen']

        segment_properties = tether_aero.get_tether_segment_properties(options, architecture, scaling, variables_si, parameters, 1)
        tether_density = segment_properties['density']
        unwound_length = segment_properties['seg_length']
        length_full = variables_si['theta']['l_t_full']
        tether_diameter = segment_properties['seg_diam']
        wound_length = length_full - unwound_length
        cross_sectional_area = segment_properties['cross_section_area']
        mass_wound_tether = tether_density * cross_sectional_area * wound_length
        radius_shell = radius_drum + tether_diameter / 2.
        moment_of_inertia_of_wound_tether = mass_wound_tether * radius_shell**2.

        mass_drum = parameters['theta0', 'ground_station', 'm_gen']
        moment_of_inertia_of_drum = 0.5 * mass_drum * radius_drum**2.

        moment_of_inertia = moment_of_inertia_of_wound_tether + moment_of_inertia_of_drum

        tangential_speed = variables_si['x']['dl_t']
        omega = tangential_speed / radius_shell

        e_kinetic = 0.5 * moment_of_inertia * omega**2.

    else:
        e_kinetic = cas.DM(0.)

    outputs['e_kinetic']['ground_station'] = e_kinetic

    return outputs