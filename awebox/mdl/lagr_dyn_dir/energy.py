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

    outputs = add_ground_station_kinetic(options, variables_si, parameters, outputs)
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

    v_rotation = dq_node - dq_average
    ehat_tether = vect_op.normalize(segment_vector)
    v_ortho = v_rotation - cas.mtimes(v_rotation.T, ehat_tether) * ehat_tether
    radius_of_rod_rotation = (segment_length / 2.)
    omega = v_ortho / radius_of_rod_rotation
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


def add_ground_station_kinetic(options, variables_si, parameters, outputs):

    # E_kinetic_ground_station
    # = 1/2 J omega_gen^2, with no-slip condition
    # = 1/2 (1/2 m r^2) omega^2
    # = 1/4 m dl_t^2
    # add mass of first half of main tether, and the mass of wound tether.

    total_ground_station_mass = outputs['masses']['ground_station']

    dq10 = variables_si['x']['dq10']
    q10 = variables_si['x']['q10']
    if 'l_t' in variables_si['x'].keys():
        l_t = variables_si['x']['l_t']
    else:
        l_t = variables_si['theta']['l_t']

    speed_ground_station = cas.mtimes(dq10.T, q10) / l_t

    e_kinetic = 0.5 * total_ground_station_mass * speed_ground_station ** 2.

    outputs['e_kinetic']['ground_station'] = e_kinetic

    return outputs