#
#    This file is part of awebox.
#
#    awebox -- A modeling and optimization framework for multi-kite AWE systems.
#    Copyright (C) 2017-2020 Jochem De Schutter, Rachel Leuthold, Moritz Diehl,
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
energy terms
_python-3.5 / casadi-3.4.5
- edited: rachel leuthold, jochem de schutter alu-fr 2017-20
'''

import casadi.tools as cas

import awebox.tools.vector_operations as vect_op
import awebox.tools.struct_operations as struct_op
import awebox.tools.print_operations as print_op


from awebox.logger.logger import Logger as awelogger

def energy_outputs(options, parameters, outputs, node_masses_si, variables_si, architecture):

    # kinetic and potential energy in the system
    energy_types = ['e_kinetic', 'e_potential']
    for type in energy_types:
        if type not in list(outputs.keys()):
            outputs[type] = {}

    number_of_nodes = architecture.number_of_nodes
    for node in range(1, number_of_nodes):
        outputs = add_node_kinetic(node, options, node_masses_si, variables_si, parameters, outputs, architecture)
        outputs = add_node_potential(node, node_masses_si, variables_si, parameters, outputs, architecture)

    outputs = add_groundstation_kinetic(options, node_masses_si, variables_si, outputs)
    outputs = add_groundstation_potential(outputs)

    return outputs


def add_node_kinetic(node, options, node_masses_si, variables_si, parameters, outputs, architecture):

    parent = architecture.parent_map[node]
    label = str(node) + str(parent)

    node_has_a_kite = node in architecture.kite_nodes
    kites_have_6dof = int(options['kite_dof']) == 6
    node_has_rotational_energy = node_has_a_kite and kites_have_6dof

    mass = node_masses_si['m' + label]
    dq = variables_si['xd']['dq' + label]
    e_kin_trans = 0.5 * mass * cas.mtimes(dq.T, dq)

    if node_has_rotational_energy:
        omega = variables_si['xd']['omega' + label]
        j_kite = parameters['theta0', 'geometry', 'j']
        e_kin_rot = 0.5 * cas.mtimes(cas.mtimes(omega.T, j_kite), omega)

    else:
        e_kin_rot = cas.DM.zeros((1, 1))

    e_kinetic = e_kin_trans + e_kin_rot

    outputs['e_kinetic']['q' + label] = e_kinetic

    return outputs


def add_node_potential(node, node_masses_si, variables_si, parameters, outputs, architecture):

    parent = architecture.parent_map[node]
    label = str(node) + str(parent)

    gravity = parameters['theta0', 'atmosphere', 'g']
    mass = node_masses_si['m' + label]
    q = variables_si['xd']['q' + label]

    e_potential = gravity * mass * q[2]
    outputs['e_potential']['q' + label] = e_potential

    return outputs


def add_groundstation_potential(outputs):
    # the winch is at ground level
    e_potential = cas.DM(0.)
    outputs['e_potential']['groundstation'] = e_potential
    return outputs


def add_groundstation_kinetic(options, node_masses_si, variables_si, outputs):

    # = 1/2 i omega_gen^2, with no-slip condition
    # add mass of first half of main tether, and the mass of wound tether.

    total_groundstation_mass = node_masses_si['groundstation']

    if options['tether']['use_wound_tether']:
        total_groundstation_mass += node_masses_si['m00']

    dq10 = variables_si['xd']['dq10']
    q10 = variables_si['xd']['q10']
    l_t = variables_si['xd']['l_t']

    speed_groundstation = cas.mtimes(dq10.T, q10) / l_t

    e_kinetic = 0.25 * total_groundstation_mass * speed_groundstation ** 2.

    outputs['e_kinetic']['groundstation'] = e_kinetic

    return outputs