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
import awebox.mdl.lagr_dyn_dir.tools as lagr_tools
import awebox.mdl.arm as arm

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

    return outputs


def get_reelout_speed(variables_si):

    # Rocking mode : never called in rocking mode, no need to define q_parent
    q_node = variables_si['x']['q10']

    # q_parent = cas.DM.zeros((3, 1))
    # segment_vector = q_node - q_parent
    segment_vector = q_node
    ehat_tether = vect_op.normalize(segment_vector)

    reelout_speed = cas.mtimes(variables_si['x']['dq10'].T, ehat_tether)

    return reelout_speed


def add_node_kinetic(node, options, variables_si, parameters, outputs, architecture, scaling):

    label = architecture.node_label(node)
    parent_label = architecture.parent_label(node)

    rocking_mode = options['trajectory']['system_type'] == 'rocking_mode'
    node_has_a_kite = node in architecture.kite_nodes
    kites_have_6dof = int(options['kite_dof']) == 6

    segment_properties = tether_aero.get_tether_segment_properties(options, architecture, scaling, variables_si, parameters, node)
    mass_segment = segment_properties['seg_mass']

    q_node = variables_si['x']['q' + label]
    dq_node = variables_si['x']['dq' + label]

    if node == 1:
        if rocking_mode:
            dq_parent = arm.get_dq_arm_tip(variables_si['x']['arm_angle'], variables_si['x']['darm_angle'], variables_si['theta']['arm_length'])
        else:
            q_parent = cas.DM.zeros((3, 1))
            segment_vector = q_node - q_parent
            ehat_tether = vect_op.normalize(segment_vector)
            reelout_speed = get_reelout_speed(variables_si)
            dq_parent = reelout_speed * ehat_tether
    else:
        dq_parent = variables_si['x']['dq' + parent_label]

    e_kin_trans = 0.5 * mass_segment / 3 * (
                cas.mtimes(dq_node.T, dq_node) + cas.mtimes(dq_parent.T, dq_parent) + cas.mtimes(dq_node.T, dq_parent))
    outputs['e_kinetic']['tether' + label] = e_kin_trans

    e_kin_kite_trans = cas.DM(0.)
    if node_has_a_kite:
        mass_kite = parameters['theta0', 'geometry', 'm_k']
        e_kin_kite_trans = 0.5 * mass_kite * cas.mtimes(dq_node.T, dq_node)
    outputs['e_kinetic']['kite_trans' + label] = e_kin_kite_trans

    e_kinetic_kite_rot = cas.DM(0.)
    if node_has_a_kite and kites_have_6dof:
        omega = variables_si['x']['omega' + label]
        j_kite = parameters['theta0', 'geometry', 'j']
        e_kinetic_kite_rot = 0.5 * cas.mtimes(cas.mtimes(omega.T, j_kite), omega)

    outputs['e_kinetic']['kite_rot' + label] = e_kinetic_kite_rot

    e_kinetic_arm_rot = cas.DM(0.)
    if rocking_mode:
        arm_inertia = variables_si['theta']['arm_inertia']
        darm_angle = variables_si['x']['darm_angle']
        e_kinetic_arm_rot = 0.5 * arm_inertia * darm_angle**2
    outputs['e_kinetic']['arm_rot'] = e_kinetic_arm_rot

    return outputs


def add_node_potential(node, options, variables_si, parameters, outputs, architecture, scaling):

    label = architecture.node_label(node)
    parent_label = architecture.parent_label(node)

    node_has_a_kite = node in architecture.kite_nodes

    gravity = parameters['theta0', 'atmosphere', 'g']

    q_node = variables_si['x']['q' + label]
    if node == 1:
        # For rocking mode: even if wrong, q_parent = origin is fine because only q_mean[2] (z component) is used
        q_parent = cas.DM.zeros((3, 1))
    else:
        q_parent = variables_si['x']['q' + parent_label]
    q_mean = (q_node + q_parent) / 2.

    segment_properties = tether_aero.get_tether_segment_properties(options, architecture, scaling, variables_si, parameters, node)
    mass_segment = segment_properties['seg_mass']

    e_potential_tether = gravity * mass_segment * q_mean[2]
    outputs['e_potential']['tether' + label] = e_potential_tether

    e_potential_kite = cas.DM(0.)
    if node_has_a_kite:
        mass_kite = parameters['theta0', 'geometry', 'm_k']
        e_potential_kite += gravity * mass_kite * q_node[2]

    outputs['e_potential']['kite' + label] = e_potential_kite

    return outputs