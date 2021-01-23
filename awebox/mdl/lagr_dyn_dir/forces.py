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
the forces on the rhs of the lagrangian dynamics
python-3.5 / casadi-3.4.5
- authors: jochem de schutter, rachel leuthold alu-fr 2017-20
'''

import casadi.tools as cas
import numpy as np

import awebox.mdl.aero.kite_dir.kite_aero as kite_aero
import awebox.mdl.aero.indicators as indicators
import awebox.mdl.aero.tether_dir.tether_aero as tether_aero
import awebox.mdl.aero.tether_dir.coefficients as tether_drag_coeff

import awebox.tools.vector_operations as vect_op
import awebox.tools.struct_operations as struct_op
import awebox.tools.print_operations as print_op

from awebox.logger.logger import Logger as awelogger


def generate_f_nodes(options, atmos, wind, variables_si, parameters, outputs, architecture):
    # initialize dictionary
    node_forces = {}
    for node in range(1, architecture.number_of_nodes):
        parent = architecture.parent_map[node]
        node_forces['f' + str(node) + str(parent)] = cas.SX.zeros((3, 1))
        if int(options['kite_dof']) == 6:
            node_forces['m' + str(node) + str(parent)] = cas.SX.zeros((3, 1))

    aero_forces, outputs = generate_aerodynamic_forces(options, variables_si, parameters, atmos, wind, outputs,
                                                       architecture)

    # # this must be after the kite aerodynamics, because the tether model "kite_only" depends on the kite outputs.
    tether_drag_forces, outputs = generate_tether_drag_forces(options, variables_si, parameters, atmos, wind, outputs,
                                                              architecture)

    if options['trajectory']['system_type'] == 'drag_mode':
        generator_forces, outputs = generate_drag_mode_forces(variables_si, outputs, architecture)

    for force in list(node_forces.keys()):
        if force[0] == 'f':
            node_forces[force] += tether_drag_forces[force]
            if force in list(aero_forces.keys()):
                node_forces[force] += aero_forces[force]
            if options['trajectory']['system_type'] == 'drag_mode':
                if force in list(generator_forces.keys()):
                    node_forces[force] += generator_forces[force]
        if (force[0] == 'm') and force in list(aero_forces.keys()):
            node_forces[force] += aero_forces[force]

    return node_forces, outputs


def generate_drag_mode_forces(variables_si, outputs, architecture):
    # create generator forces
    generator_forces = {}
    for n in architecture.kite_nodes:
        parent = architecture.parent_map[n]

        # compute generator force
        kappa = variables_si['xd']['kappa{}{}'.format(n, parent)]
        speed = outputs['aerodynamics']['airspeed{}'.format(n)]
        v_app = outputs['aerodynamics']['air_velocity{}'.format(n)]
        gen_force = kappa * speed * v_app

        # store generator force
        generator_forces['f{}{}'.format(n, parent)] = gen_force
        outputs['aerodynamics']['f_gen{}'.format(n)] = gen_force

    return generator_forces, outputs


def generate_tether_drag_forces(options, variables_si, parameters, atmos, wind, outputs, architecture):

    # tether_drag_coeff.plot_cd_vs_reynolds(100, options)
    tether_cd_fun = tether_drag_coeff.get_tether_cd_fun(options, parameters)

    # mass vector, containing the mass of all nodes
    for node in range(1, architecture.number_of_nodes):
        outputs = tether_aero.get_force_outputs(options, variables_si, parameters, atmos, wind, node, tether_cd_fun, outputs,
                                                architecture)

    if options['tether']['lift_tether_force']:
        tether_drag_forces = {}
        for node in range(1, architecture.number_of_nodes):
            parent = architecture.parent_map[node]
            tether_drag_forces['f' + str(node) + str(parent)] = tether_aero.get_force_var(variables_si, node, architecture)
    else:
        tether_drag_forces = tether_aero.distribute_tether_drag_forces(options, variables_si, architecture, outputs)

    # collect tether drag losses
    outputs = indicators.collect_tether_drag_losses(variables_si, tether_drag_forces, outputs, architecture)

    return tether_drag_forces, outputs


def generate_aerodynamic_forces(options, variables_si, parameters, atmos, wind, outputs, architecture):
    # homotopy parameters
    p_dec = parameters.prefix['phi']

    # get aerodynamic forces and moments
    outputs = kite_aero.get_forces_and_moments(options, atmos, wind, variables_si, outputs, parameters, architecture)

    # attribute aerodynamic forces to kites
    aero_forces = {}
    for kite in architecture.kite_nodes:

        parent = architecture.parent_map[kite]
        [homotopy_force, homotopy_moment] = fictitious_embedding(options, p_dec, variables_si, kite, parent, outputs)
        aero_forces['f' + str(kite) + str(parent)] = homotopy_force

        if int(options['kite_dof']) == 6:
            aero_forces['m' + str(kite) + str(parent)] = homotopy_moment

    return aero_forces, outputs


def fictitious_embedding(options, p_dec, variables_si, kite, parent, outputs):

    kite_has_6_dof = (int(options['kite_dof']) == 6)

    # remember: generalized coordinates are in earth-fixed cartesian coordinates for translational dynamics

    if options['aero']['lift_aero_force']:
        f_aero_var, m_aero_var = kite_aero.get_force_and_moment_vars(variables_si, kite, parent, options)
        true_force = f_aero_var
        true_moment = m_aero_var
    else:
        true_force = outputs['aerodynamics']['f_aero_earth' + str(kite)]
        if kite_has_6_dof:
            true_moment = outputs['aerodynamics']['m_aero_body' + str(kite)]

    fict_force = variables_si['u']['f_fict' + str(kite) + str(parent)]
    homotopy_force = p_dec['gamma'] * fict_force + true_force

    if int(options['kite_dof']) == 6:
        fict_moment = variables_si['u']['m_fict' + str(kite) + str(parent)]
        homotopy_moment = p_dec['gamma'] * fict_moment + true_moment
    else:
        homotopy_moment = []

    return homotopy_force, homotopy_moment

def generate_tether_moments(options, variables_si, variables_scaled, holonomic_constraints, outputs, architecture):
    kite_nodes = architecture.kite_nodes
    parent_map = architecture.parent_map

    xd_si = variables_si['xd']

    if int(options['kite_dof']) == 6:
        outputs['tether_moments'] = {}
        for kite in kite_nodes:
            parent = parent_map[kite]

            # tether constraint contribution
            tether_moment = 2 * vect_op.jacobian_dcm(holonomic_constraints, xd_si, variables_scaled, kite, parent).T
            outputs['tether_moments']['n{}{}'.format(kite, parent)] = tether_moment

    return outputs

