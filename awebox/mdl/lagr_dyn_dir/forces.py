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

import pdb

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
    # tether_drag_forces, outputs = generate_tether_drag_forces(options, variables_si, parameters, atmos, wind, outputs,
    #                                                           architecture)

    if options['trajectory']['system_type'] == 'drag_mode':
        generator_forces, outputs = generate_drag_mode_forces(variables_si, outputs, architecture)

    for force in list(node_forces.keys()):
        if force[0] == 'f':
            # node_forces[force] += tether_drag_forces[force]
            if force in list(aero_forces.keys()):
                node_forces[force] += aero_forces[force]
            if options['trajectory']['system_type'] == 'drag_mode':
                if force in list(generator_forces.keys()):
                    node_forces[force] += generator_forces[force]
        if (force[0] == 'm') and force in list(aero_forces.keys()):
            node_forces[force] += aero_forces[force]

    print_op.warn_about_temporary_funcationality_removal(location='lagr_dyn.forces')

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
    # homotopy parameters
    p_dec = parameters.prefix['phi']

    # tether_drag_coeff.plot_cd_vs_reynolds(100, options)
    tether_cd_fun = tether_drag_coeff.get_tether_cd_fun(options, parameters)

    # initialize dictionary
    tether_drag_forces = {}
    for n in range(1, architecture.number_of_nodes):
        parent = architecture.parent_map[n]
        tether_drag_forces['f' + str(n) + str(parent)] = cas.SX.zeros((3, 1))

    # mass vector, containing the mass of all nodes
    for n in range(1, architecture.number_of_nodes):

        parent = architecture.parent_map[n]

        outputs = tether_aero.get_force_outputs(options, variables_si, atmos, wind, n, tether_cd_fun, outputs,
                                                architecture)

        multi_upper = outputs['tether_aero']['multi_upper' + str(n)]
        multi_lower = outputs['tether_aero']['multi_lower' + str(n)]
        single_upper = outputs['tether_aero']['single_upper' + str(n)]
        single_lower = outputs['tether_aero']['single_lower' + str(n)]
        split_upper = outputs['tether_aero']['split_upper' + str(n)]
        split_lower = outputs['tether_aero']['split_lower' + str(n)]
        kite_only_upper = outputs['tether_aero']['kite_only_upper' + str(n)]
        kite_only_lower = outputs['tether_aero']['kite_only_lower' + str(n)]

        tether_model = options['tether']['tether_drag']['model_type']

        if tether_model == 'multi':
            drag_node = p_dec['tau'] * split_upper + (1. - p_dec['tau']) * multi_upper
            drag_parent = p_dec['tau'] * split_lower + (1. - p_dec['tau']) * multi_lower

        elif tether_model == 'single':
            drag_node = p_dec['tau'] * split_upper + (1. - p_dec['tau']) * single_upper
            drag_parent = p_dec['tau'] * split_lower + (1. - p_dec['tau']) * single_lower

        elif tether_model == 'split':
            drag_node = split_upper
            drag_parent = split_lower

        elif tether_model == 'kite_only':
            drag_node = kite_only_upper
            drag_parent = kite_only_lower

        elif tether_model == 'not_in_use':
            drag_parent = cas.DM.zeros((3, 1))
            drag_node = cas.DM.zeros((3, 1))

        else:
            raise ValueError('tether drag model not supported.')

        # attribute portion of segment drag to parent
        if n > 1:
            grandparent = architecture.parent_map[parent]
            tether_drag_forces['f' + str(parent) + str(grandparent)] += drag_parent

        # attribute portion of segment drag to node
        tether_drag_forces['f' + str(n) + str(parent)] += drag_node

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
        [homotopy_force, homotopy_moment] = fictitious_embedding(options, p_dec, variables_si['u'], outputs, kite, parent)
        aero_forces['f' + str(kite) + str(parent)] = homotopy_force

        if int(options['kite_dof']) == 6:
            aero_forces['m' + str(kite) + str(parent)] = homotopy_moment

    return aero_forces, outputs


def fictitious_embedding(options, p_dec, u_si, outputs, kite, parent):

    # remember: generalized coordinates are in earth-fixed cartesian coordinates for translational dynamics

    fict_force = u_si['f_fict' + str(kite) + str(parent)]
    true_force = outputs['aerodynamics']['f_aero_earth' + str(kite)]

    # homotopy_force = p_dec['gamma'] * fict_force + true_force
    print_op.warn_about_temporary_funcationality_removal(location='lagr_dyn.forces')
    homotopy_force = p_dec['gamma'] * fict_force

    if int(options['kite_dof']) == 6:
        fict_moment = u_si['m_fict' + str(kite) + str(parent)]
        true_moment = outputs['aerodynamics']['m_aero_body' + str(kite)]

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

