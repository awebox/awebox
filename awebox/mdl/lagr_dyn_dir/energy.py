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

import awebox.tools.vector_operations as vect_op
import awebox.tools.struct_operations as struct_op
import awebox.tools.print_operations as print_op

from awebox.logger.logger import Logger as awelogger

def energy_outputs(options, parameters, outputs, variables_si, architecture):

    # kinetic and potential energy in the system
    energy_types = ['e_kinetic', 'e_potential']
    for etype in energy_types:
        if etype not in list(outputs.keys()):
            outputs[etype] = {}

    number_of_nodes = architecture.number_of_nodes
    for node in range(1, number_of_nodes):
        outputs = add_node_kinetic(node, options, variables_si, parameters, outputs, architecture)
        outputs = add_node_potential(node, options, variables_si, parameters, outputs, architecture)

    outputs = add_ground_station_kinetic(options, variables_si, parameters, outputs)
    outputs = add_ground_station_potential(outputs)

    return outputs


def add_node_kinetic(node, options, variables_si, parameters, outputs, architecture):

    label = architecture.node_label(node)
    parent_label = architecture.parent_label(node)

    node_has_a_kite = node in architecture.kite_nodes
    kites_have_6dof = int(options['kite_dof']) == 6
    node_has_rotational_energy = node_has_a_kite and kites_have_6dof

    # add tether translational kinetic energy
    m_t = outputs['masses']['m_tether{}'.format(node)]
    dq_n = variables_si['xd']['dq' + label]
    if node == 1:
        dq_parent = cas.DM.zeros((3, 1))
    else:
        dq_parent = variables_si['xd']['dq' + parent_label]
    e_kin_trans = 0.5 * m_t/3 * (cas.mtimes(dq_n.T, dq_n) + cas.mtimes(dq_parent.T, dq_parent) + cas.mtimes(dq_n.T, dq_parent))
    
    # add kite translational kinetic energy
    if node_has_a_kite:
        mass = parameters['theta0', 'geometry', 'm_k']
        e_kin_trans += 0.5 * mass * cas.mtimes(dq_n.T, dq_n)

    # add kite rotational energy
    if node_has_rotational_energy:
        omega = variables_si['xd']['omega' + label]
        j_kite = parameters['theta0', 'geometry', 'j']
        e_kin_rot = 0.5 * cas.mtimes(cas.mtimes(omega.T, j_kite), omega)

    else:
        e_kin_rot = cas.DM.zeros((1, 1))

    e_kinetic = e_kin_trans + e_kin_rot

    outputs['e_kinetic']['q' + label] = e_kinetic

    return outputs


def add_node_potential(node, options, variables_si, parameters, outputs, architecture):

    label = architecture.node_label(node)
    parent_label = architecture.parent_label(node)

    node_has_a_kite = node in architecture.kite_nodes

    gravity = parameters['theta0', 'atmosphere', 'g']

    m_t = outputs['masses']['m_tether{}'.format(node)]
    q_n = variables_si['xd']['q' + label]
    if node == 1:
        q_parent = cas.DM.zeros((3,1))
    else:
        q_parent = variables_si['xd']['q'+label]
    e_potential = gravity * m_t/2 *(q_n[2] + q_parent[2])

    if node_has_a_kite:
        mass = parameters['theta0', 'geometry', 'm_k']
        e_potential = gravity * mass * q_n[2]

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

    dq10 = variables_si['xd']['dq10']
    q10 = variables_si['xd']['q10']
    l_t = variables_si['xd']['l_t']

    speed_ground_station = cas.mtimes(dq10.T, q10) / l_t

    e_kinetic = 0.25 * total_ground_station_mass * speed_ground_station ** 2.

    outputs['e_kinetic']['ground_station'] = e_kinetic

    return outputs