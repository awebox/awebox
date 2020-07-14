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
lagrangian dynamics auto-generation module
that generates the dynamics residual
python-3.5 / casadi-3.4.5
- authors: jochem de schutter, rachel leuthold alu-fr 2017-20
'''

import casadi.tools as cas
import numpy as np

import itertools

from collections import OrderedDict

from . import system

import awebox.mdl.aero.kite_dir.kite_aero as kite_aero

import awebox.mdl.aero.induction_dir.induction as induction

import awebox.mdl.aero.indicators as indicators

import awebox.mdl.aero.tether_dir.tether_aero as tether_aero

import awebox.mdl.aero.tether_dir.coefficients as tether_drag_coeff

import awebox.tools.vector_operations as vect_op
import awebox.tools.struct_operations as struct_op
import awebox.tools.print_operations as print_op

import pdb

from awebox.logger.logger import Logger as awelogger


def make_dynamics(options, atmos, wind, parameters, architecture):
    # system architecture (see zanon2013a)
    number_of_nodes = architecture.number_of_nodes
    parent_map = architecture.parent_map

    # --------------------------------------------------------------------------------------
    # generate system states, controls, algebraic vars, lifted vars, generalized coordinates
    # --------------------------------------------------------------------------------------
    [system_variable_list, system_gc] = system.generate_structure(options, architecture)

    # -----------------------------------
    # generate structured SX.sym objects
    # -----------------------------------
    system_variables = {}
    system_variables['scaled'], variables_dict = struct_op.generate_variable_struct(system_variable_list)
    system_variables['SI'], scaling = generate_scaled_variables(options['scaling'], system_variables['scaled'])

    # --------------------------------------------
    # generate system constraints and derivatives
    # --------------------------------------------

    # generalized coordinates, velocities and accelerations
    generalized_coordinates = {}
    generalized_coordinates['scaled'] = generate_generalized_coordinates(system_variables['scaled'], system_gc)
    generalized_coordinates['SI'] = generate_generalized_coordinates(system_variables['SI'], system_gc)

    # define outputs to monitor system constraints etc.
    outputs = {}

    holonomic_constraints, outputs, g, gdot, gddot, holonomic_fun = generate_holonomic_constraints(
        architecture,
        outputs,
        system_variables,
        generalized_coordinates,
        parameters,
        options)
    holonomic_scaling = generate_holonomic_scaling(options, architecture, system_variables['SI'], parameters)

    # -------------------------------
    # mass distribution in the system
    # -------------------------------
    node_masses, outputs = generate_m_nodes(options, system_variables['SI'], outputs, parameters, architecture)
    node_masses_scaling = generate_m_nodes_scaling(options, system_variables['SI'], outputs, parameters, architecture)

    # --------------------------------
    # generalized forces in the system
    # --------------------------------

    f_nodes, outputs = generate_f_nodes(
        options, atmos, wind, system_variables['SI'], parameters, outputs, architecture)

    # add outputs for constraints
    outputs = tether_stress_inequality(options, system_variables['SI'], outputs, parameters, architecture)
    outputs = wound_tether_length_inequality(options, system_variables['SI'], outputs, parameters, architecture)
    outputs = airspeed_inequality(options, system_variables['SI'], outputs, parameters, architecture)
    outputs = xddot_outputs(options, system_variables['SI'], outputs)
    outputs = anticollision_inequality(options, system_variables['SI'], outputs, parameters, architecture)
    outputs = anticollision_radius_inequality(options, system_variables['SI'], outputs, parameters, architecture)
    outputs = acceleration_inequality(options, system_variables['SI'], outputs, parameters)
    outputs = angular_velocity_inequality(options, system_variables['SI'], outputs, parameters, architecture)
    outputs = dcoeff_actuation_inequality(options, system_variables['SI'], parameters, outputs)
    outputs = coeff_actuation_inequality(options, system_variables['SI'], parameters, outputs)
    outputs = tether_power_outputs(system_variables['SI'], outputs, architecture)

    if options['trajectory']['system_type'] == 'drag_mode':
        outputs = drag_mode_outputs(system_variables['SI'], outputs, architecture)

    if options['kite_dof'] == 6:
        outputs = rotation_inequality(options, system_variables['SI'], parameters, architecture, outputs)

    if options['induction_model'] != 'not_in_use':
        outputs = induction_equations(options, atmos, wind, system_variables['SI'], outputs, parameters, architecture)

    # ---------------------------------
    # rotation second law
    # ---------------------------------
    rotation_dynamics, outputs = generate_rotational_dynamics(options, system_variables, f_nodes, holonomic_constraints,
                                                              parameters, outputs, architecture)

    # ---------------------------------
    # lagrangian function of the system
    # ---------------------------------
    outputs = energy_outputs(options, parameters, outputs, node_masses, system_variables, generalized_coordinates,
                             architecture)
    outputs = conservative_power_outputs(options, parameters, outputs, node_masses, system_variables,
                                         generalized_coordinates, architecture)

    e_kinetic = sum(outputs['e_kinetic'][nodes] for nodes in list(outputs['e_kinetic'].keys()))
    e_potential = sum(outputs['e_potential'][nodes] for nodes in list(outputs['e_potential'].keys()))

    # lagrangian function
    lag = e_kinetic - e_potential - holonomic_constraints

    # system output function
    [out, out_fun, out_dict] = make_output_structure(outputs, system_variables, parameters)
    [constraint_out, constraint_out_fun] = make_output_constraint_structure(options, outputs, system_variables,
                                                                            parameters)

    # ----------------------------------------
    #  dynamics of the system in implicit form
    # ----------------------------------------
    dynamics_list = []

    # lhs of lagrange equations
    q_translation = cas.vertcat(generalized_coordinates['scaled']['xgc'].cat,
                                system_variables['scaled']['xd', 'l_t'],
                                generalized_coordinates['scaled']['xgcdot'].cat,
                                system_variables['scaled']['xd', 'dl_t'])
    qdot_translation = cas.vertcat(generalized_coordinates['scaled']['xgcdot'].cat,
                                   system_variables['scaled']['xd', 'dl_t'],
                                   generalized_coordinates['scaled']['xgcddot'].cat,
                                   system_variables['scaled'][
                                       struct_op.get_variable_type(variables_dict, 'ddl_t'), 'ddl_t'])

    lagrangian_lhs_translation = cas.mtimes(cas.jacobian(cas.jacobian(
        lag, generalized_coordinates['scaled']['xgcdot'].cat).T, q_translation), qdot_translation) - cas.jacobian(lag,
                                                                                                                  generalized_coordinates[
                                                                                                                      'scaled'][
                                                                                                                      'xgc'].cat).T
    lagrangian_lhs_constraints = gddot + 2. * \
                                 parameters['theta0', 'tether', 'kappa'] * gdot + parameters[
                                     'theta0', 'tether', 'kappa'] ** 2. * g
    # lagrangian_lhs_constraints = gddot

    # lagrangian momentum correction
    lagrangian_momentum_correction = momentum_correction(generalized_coordinates, system_variables, node_masses,
                                                         outputs, architecture)

    # rhs of lagrange equations
    lagrangian_rhs_translation = cas.vertcat(
        *[f_nodes['f' + str(n) + str(parent_map[n])] for n in range(1, number_of_nodes)]) + \
                                 lagrangian_momentum_correction
    lagrangian_rhs_constraints = np.zeros(g.shape)
    # trivial kinematics
    trivial_dynamics_states = cas.vertcat(
        *[system_variables['scaled']['xddot', name] - system_variables['scaled']['xd', name] for name in
          list(system_variables['SI']['xddot'].keys()) if name in list(system_variables['SI']['xd'].keys())])
    trivial_dynamics_controls = cas.vertcat(
        *[system_variables['scaled']['xddot', name] - system_variables['scaled']['u', name] for name in
          list(system_variables['SI']['xddot'].keys()) if name in list(system_variables['SI']['u'].keys())])

    # lagrangian dynamics
    forces_scaling = node_masses_scaling * options['scaling']['other']['g']
    dynamics_translation = (lagrangian_lhs_translation - lagrangian_rhs_translation) / forces_scaling
    dynamics_constraints = (lagrangian_lhs_constraints - lagrangian_rhs_constraints) / holonomic_scaling

    # rotation dynamics
    # above

    dynamics_list += [
        trivial_dynamics_states,
        dynamics_translation,
        rotation_dynamics,
        trivial_dynamics_controls,
        dynamics_constraints]

    # generate empty integral_outputs struct
    integral_outputs = cas.struct_SX([])
    integral_outputs_struct = cas.struct_symSX([])

    integral_scaling = {}

    # energy
    if options['trajectory']['system_type'] == 'drag_mode':
        power = cas.SX.zeros(1, 1)
        for n in architecture.kite_nodes:
            power += - outputs['power_balance']['P_gen{}'.format(n)]

    else:
        power = system_variables['SI']['xa']['lambda10'] * system_variables['SI']['xd']['l_t'] * \
                system_variables['SI']['xd']['dl_t']

    if options['integral_outputs']:
        integral_outputs = cas.struct_SX([cas.entry('e', expr=power / options['scaling']['xd']['e'])])
        integral_outputs_struct = cas.struct_symSX([cas.entry('e')])

        integral_scaling['e'] = options['scaling']['xd']['e']
    else:
        energy_dynamics = (system_variables['SI']['xddot']['de'] - power) / options['scaling']['xd']['e']
        dynamics_list += [energy_dynamics]

    aero_force_resi = kite_aero.get_force_resi(options, system_variables['SI'], atmos, wind, architecture, parameters)
    dynamics_list += [aero_force_resi]

    # induction constraint
    if options['induction_model'] != 'not_in_use':
        induction_init = outputs['induction']['induction_init']
        induction_final = outputs['induction']['induction_final']

        induction_constraint = parameters['phi', 'iota'] * induction_init \
                               + (1. - parameters['phi', 'iota']) * induction_final

        dynamics_list += [induction_constraint]

    # system dynamics function (implicit form)

    # order of V is: q, dq, omega, R, delta, lt, dlt, e
    res = cas.vertcat(*dynamics_list)

    # dynamics function options
    if options['jit_code_gen']['include']:
        opts = {'jit': True, 'compiler': options['jit_code_gen']['compiler']}
    else:
        opts = {}

    # generate dynamics function
    dynamics = cas.Function(
        'dynamics', [system_variables['scaled'], parameters], [res], opts)

    # generate integral outputs function
    integral_outputs_fun = cas.Function('integral_outputs', [system_variables['scaled'], parameters],
                                        [integral_outputs], opts)

    return [
        system_variables['scaled'],
        variables_dict,
        scaling,
        dynamics,
        out,
        out_fun,
        out_dict,
        constraint_out,
        constraint_out_fun,
        holonomic_fun,
        integral_outputs_struct,
        integral_outputs_fun,
        integral_scaling]
    # bounds_xdot]


def make_output_structure(outputs, system_variables, parameters):
    outputs_vec = []
    full_list = []

    outputs_dict = {}

    for output_type in list(outputs.keys()):

        local_list = []
        for name in list(outputs[output_type].keys()):
            # prepare empty entry list to generate substruct
            local_list += [cas.entry(name, shape=outputs[output_type][name].shape)]

            # generate vector with outputs - SX expressions
            outputs_vec = cas.vertcat(outputs_vec, outputs[output_type][name])

        # generate dict with sub-structs
        outputs_dict[output_type] = cas.struct_symMX(local_list)
        # prepare list with sub-structs to generate struct
        full_list += [cas.entry(output_type, struct=outputs_dict[output_type])]

    # generate "empty" structure
    out_struct = cas.struct_symMX(full_list)
    # generate structure with SX expressions
    outputs_struct = out_struct(outputs_vec)
    # generate outputs function
    outputs_fun = cas.Function('outputs', [system_variables['scaled'], parameters], [outputs_struct.cat])

    return [out_struct, outputs_fun, outputs_dict]


def make_output_constraint_structure(options, outputs, system_variables, parameters):
    constraint_vec = []

    represented_constraints = list(options['model_bounds'].keys())

    full_list = []
    for output_type in list(outputs.keys()):

        if output_type in set(represented_constraints):

            local_list = []
            for name in list(outputs[output_type].keys()):
                local_list += [cas.entry(name, shape=outputs[output_type][name].shape)]

                constraint_vec = cas.vertcat(constraint_vec, outputs[output_type][name])

            local_output = cas.struct_symMX(local_list)

            full_list += [cas.entry(output_type, struct=local_output)]

    constraint_struct = cas.struct_symMX(full_list)
    constraints = constraint_struct(constraint_vec)
    constraint_fun = cas.Function('constraint_out_fun', [system_variables['scaled'], parameters], [constraints])

    return [constraint_struct, constraint_fun]


def generate_m_nodes(options, variables, outputs, parameters, architecture):
    # system architecture (see zanon2013a)
    number_of_nodes = architecture.number_of_nodes
    kite_nodes = architecture.kite_nodes
    parent_map = architecture.parent_map

    node_masses = generate_mass_dictionary_for_all_nodes(options, variables, parameters, architecture, 'vals')

    # save some space in the outputs for the node masses
    if 'masses' not in list(outputs.keys()):
        outputs['masses'] = {}

    # save the each node's responsible mass into the outputs
    for node in range(1, number_of_nodes):
        parent = parent_map[node]
        outputs['masses']['m' + str(node) + str(parent)] = node_masses['m' + str(node) + str(parent)]

    if options['tether']['use_wound_tether']:
        outputs['masses']['m00'] = node_masses['m00']

    return node_masses, outputs


def generate_mass_dictionary_for_all_nodes(options, variables, parameters, architecture, vals_or_scaling):
    number_of_nodes = architecture.number_of_nodes
    kite_nodes = architecture.kite_nodes
    parent_map = architecture.parent_map

    use_wound_tether = options['tether']['use_wound_tether']

    # initialize dictionary
    node_masses = {}
    for n in range(1, number_of_nodes):
        parent = parent_map[n]
        node_masses['m' + str(n) + str(parent)] = 0.

    if use_wound_tether:
        node_masses['m00'] = 0.  # mass added to the ground-station

    # sum the masses of each segment, to determine the mass for which each node is responsible
    for n in range(1, number_of_nodes):

        parent = parent_map[n]

        seg_props = get_tether_segment_properties(options, architecture, variables, parameters, upper_node=n)
        if vals_or_scaling == 'vals':
            seg_mass = seg_props['seg_mass']
        elif vals_or_scaling == 'scaling':
            seg_mass = seg_props['scaling_mass']
        else:
            awelogger.logger.error('unknown option in mass dictionary generation')

        # attribute half of segment mass to node (at top of segment)
        node_masses['m' + str(n) + str(parent)] += seg_mass / 2.

        # attribute half of segment mass to parent (node at bottom of segment)
        # for non-ground-station parents
        if n > 1:
            grandparent = parent_map[parent]
            node_masses['m' + str(parent) + str(grandparent)] += seg_mass / 2.

        if n == 1 and use_wound_tether:
            # this adds to the ground-station entry 'm00'
            node_masses['m00'] += seg_mass / 2.

        # attribute kite mass to kite node
        if n in kite_nodes:
            node_masses['m' + str(n) + str(parent)] += parameters['theta0', 'geometry', 'm_k']

    # attribute wound-up tether mass to the ground-station
    if use_wound_tether:
        main_props = get_tether_segment_properties(options, architecture, variables, parameters, upper_node=1)
        if vals_or_scaling == 'vals':
            wound_length = variables['theta']['l_t_full'] - main_props['seg_length']
            wound_cross_section = main_props['cross_section_area']
        elif vals_or_scaling == 'scaling':
            wound_length = options['scaling']['theta']['l_t_full'] - main_props['scaling_length']
            wound_cross_section = main_props['scaling_area']
        else:
            awelogger.logger.error('unknown option in mass dictionary generation')

        wound_mass = wound_cross_section * parameters['theta0', 'tether', 'rho'] * wound_length
        node_masses['m00'] += wound_mass

    return node_masses


def generate_m_nodes_scaling(options, variables, outputs, parameters, architecture):
    # system architecture (see zanon2013a)
    number_of_nodes = architecture.number_of_nodes
    parent_map = architecture.parent_map

    node_masses_scaling = generate_mass_dictionary_for_all_nodes(options, variables, parameters, architecture,
                                                                 'scaling')

    # this will be used to scale the forces,
    # so we need to repeat the scaling forces 3x, once per dimension for force.
    # and, only on those nodes for which we will construct dynamics equations via. Lagrangian mechanics

    node_masses_scaling_stacked = []
    for node in range(1, number_of_nodes):
        parent = parent_map[node]
        mass = node_masses_scaling['m' + str(node) + str(parent)]
        node_masses_scaling_stacked = cas.vertcat(node_masses_scaling_stacked, mass, mass, mass)

    return node_masses_scaling_stacked


def generate_f_nodes(options, atmos, wind, variables, parameters, outputs, architecture):
    # initialize dictionary
    node_forces = {}
    for n in range(1, architecture.number_of_nodes):
        parent = architecture.parent_map[n]
        node_forces['f' + str(n) + str(parent)] = cas.SX.zeros((3, 1))
        if int(options['kite_dof']) == 6:
            node_forces['m' + str(n) + str(parent)] = cas.SX.zeros((3, 1))

    aero_forces, outputs = generate_aerodynamic_forces(options, variables, parameters, atmos, wind, outputs,
                                                       architecture)

    # this must be after the kite aerodynamics, because the tether model "kite_only" depends on the kite outputs.
    tether_drag_forces, outputs = generate_tether_drag_forces(options, variables, parameters, atmos, wind, outputs,
                                                              architecture)

    if options['trajectory']['system_type'] == 'drag_mode':
        generator_forces, outputs = generate_drag_mode_forces(options, variables, parameters, outputs, architecture)

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


def generate_drag_mode_forces(options, variables, parameters, outputs, architecture):
    # create generator forces
    generator_forces = {}
    for n in architecture.kite_nodes:
        parent = architecture.parent_map[n]

        # compute generator force
        kappa = variables['xd']['kappa{}{}'.format(n, parent)]
        speed = outputs['aerodynamics']['airspeed{}'.format(n)]
        v_app = outputs['aerodynamics']['v_app{}'.format(n)]
        gen_force = kappa * speed * v_app

        # store generator force
        generator_forces['f{}{}'.format(n, parent)] = gen_force
        outputs['aerodynamics']['f_gen{}'.format(n)] = gen_force

    return generator_forces, outputs


def generate_tether_drag_forces(options, variables, parameters, atmos, wind, outputs, architecture):
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

        outputs = tether_aero.get_force_outputs(options, variables, atmos, wind, n, tether_cd_fun, outputs,
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
            drag_parent = np.zeros((3, 1))
            drag_node = np.zeros((3, 1))

        else:
            raise ValueError('tether drag model not supported.')

        # attribute portion of segment drag to parent
        if n > 1:
            grandparent = architecture.parent_map[parent]
            tether_drag_forces['f' + str(parent) + str(grandparent)] += drag_parent

        # attribute portion of segment drag to node
        tether_drag_forces['f' + str(n) + str(parent)] += drag_node

    # collect tether drag losses
    outputs = indicators.collect_tether_drag_losses(variables, tether_drag_forces, outputs, architecture)

    return tether_drag_forces, outputs


def generate_aerodynamic_forces(options, variables, parameters, atmos, wind, outputs, architecture):
    # homotopy parameters
    p_dec = parameters.prefix['phi']
    # get aerodynamic forces and moments
    outputs = kite_aero.get_forces_and_moments(options, atmos, wind, variables, outputs, parameters, architecture)

    # attribute aerodynamic forces to kites
    aero_forces = {}
    for n in architecture.kite_nodes:

        parent = architecture.parent_map[n]
        [homotopy_force, homotopy_moment] = fictitious_embedding(options, p_dec, variables['u'], outputs, n, parent)
        aero_forces['f' + str(n) + str(parent)] = homotopy_force

        if int(options['kite_dof']) == 6:
            aero_forces['m' + str(n) + str(parent)] = homotopy_moment

    return aero_forces, outputs


def fictitious_embedding(options, p_dec, u, outputs, n, parent):
    fict_force = u['f_fict' + str(n) + str(parent)]
    true_force = outputs['aerodynamics']['f_aero' + str(n)]

    homotopy_force = p_dec['gamma'] * fict_force + true_force

    if int(options['kite_dof']) == 6:
        fict_moment = u['m_fict' + str(n) + str(parent)]
        true_moment = outputs['aerodynamics']['m_aero' + str(n)]

        homotopy_moment = p_dec['gamma'] * fict_moment + true_moment
    else:
        homotopy_moment = []

    return homotopy_force, homotopy_moment


def energy_outputs(options, parameters, outputs, node_masses, system_variables, generalized_coordinates, architecture):
    number_of_nodes = architecture.number_of_nodes
    parent_map = architecture.parent_map

    energy_types = ['e_kinetic', 'e_potential']
    for type in energy_types:
        if type not in list(outputs.keys()):
            outputs[type] = {}

    # kinetic and potential energy in the system
    for n in range(1, number_of_nodes):
        label = str(n) + str(parent_map[n])

        # translational kinetic energy
        e_kinetic = 0.5 * node_masses['m' + label] * \
                    cas.mtimes(generalized_coordinates['SI']['xgcdot']['dq' + label].T,
                               generalized_coordinates['SI']['xgcdot']['dq' + label])

        # add rotational kinetic energy
        if (n in architecture.kite_nodes) and (int(options['kite_dof']) == 6):
            e_kinetic += 0.5 * cas.mtimes(cas.mtimes(system_variables['SI']['xd']['omega' + label].T,
                                                     parameters['theta0', 'geometry', 'j']),
                                          system_variables['SI']['xd']['omega' + label])

        outputs['e_kinetic']['q' + label] = e_kinetic

        e_potential = parameters['theta0', 'atmosphere', 'g'] * \
                      node_masses['m' + label] * \
                      generalized_coordinates['SI']['xgc']['q' + label][2]
        outputs['e_potential']['q' + label] = e_potential

    # = 1/2 i omega_gen^2, with no-slip condition
    # add mass of first half of main tether, and the mass of wound tether.
    m_groundstation = parameters['theta0', 'ground_station', 'm_gen']
    if options['tether']['use_wound_tether']:
        m_groundstation += node_masses['m00']
    e_kinetic_groundstation = 1. / 4. * m_groundstation * system_variables['SI']['xd']['dl_t'] ** 2.0
    outputs['e_kinetic']['groundstation'] = e_kinetic_groundstation

    # the winch is at ground level
    outputs['e_potential']['groundstation'] = cas.DM(0.)

    return outputs


def drag_mode_outputs(variables, outputs, architecture):
    for n in architecture.kite_nodes:
        parent = architecture.parent_map[n]
        outputs['power_balance']['P_gen{}'.format(n)] = cas.mtimes(
            variables['xd']['dq{}{}'.format(n, parent)].T,
            outputs['aerodynamics']['f_gen{}'.format(n)]
        )

    return outputs


def tether_power_outputs(variables, outputs, architecture):
    # compute instantaneous power transferred by each tether
    for n in range(1, architecture.number_of_nodes):
        parent = architecture.parent_map[n]

        # node positions
        q_n = variables['xd']['q' + str(n) + str(parent)]
        if n > 1:
            grandparent = architecture.parent_map[parent]
            q_p = variables['xd']['q' + str(parent) + str(grandparent)]
        else:
            q_p = cas.SX.zeros((3, 1))

        # node velocity
        dq_n = variables['xd']['dq' + str(n) + str(parent)]
        # force and direction
        tether_force = outputs['local_performance']['tether_force' + str(n) + str(parent)]
        tether_direction = vect_op.normalize(q_n - q_p)
        # power transferred (convention: negative when power is transferred down the tree)
        outputs['power_balance']['P_tether' + str(n)] = -cas.mtimes(tether_force * tether_direction.T, dq_n)

    return outputs


def conservative_power_outputs(options, parameters, outputs, node_masses, system_variables, generalized_coordinates,
                               architecture):
    """Compute rate of change of kinetic and potential energy for all system nodes

    @return: outputs updated outputs dict
    """

    # extract variables
    xd = system_variables['scaled'].prefix['xd']
    xddot = system_variables['scaled'].prefix['xddot']
    xdSI = system_variables['SI']['xd']
    xddotSI = system_variables['SI']['xddot']

    # kinetic and potential energy in the system
    for n in range(1, architecture.number_of_nodes):
        label = str(n) + str(architecture.parent_map[n])

        # input variables for kinetic energy
        x_kin = cas.vertcat(xd['dq' + label])
        xdot_kin = cas.vertcat(xddot['ddq' + label])

        if (n in architecture.kite_nodes) and (int(options['kite_dof']) == 6):
            x_kin = cas.vertcat(x_kin, xd['omega' + label])
            xdot_kin = cas.vertcat(xdot_kin, xddot['domega' + label])

        # rate of change in kinetic energy (ignore in-outflow of kinetic energy)
        P_kinetic = cas.mtimes(cas.jacobian(outputs['e_kinetic']['q' + label], x_kin), xdot_kin)

        # convention: negative when kinetic energy is added to the system
        outputs['power_balance']['P_kin' + str(n)] = -P_kinetic

        # input variables for potential energy
        x_pot = cas.vertcat(xd['q' + label])
        xdot_pot = cas.vertcat(xd['dq' + label])

        # rate of change in potential energy (ignore in-outflow of potential energy)
        P_potential = cas.mtimes(cas.jacobian(outputs['e_potential']['q' + label], x_pot), xdot_pot)

        # convention: negative when potential energy is added to the system
        outputs['power_balance']['P_pot' + str(n)] = -P_potential

    return outputs


def momentum_correction(generalized_coordinates, system_variables, node_masses, outputs, architecture):
    """Compute momentum correction for translational lagrangian dynamics of an open system.
    Here the system is "open" because the main tether mass is changing in time. During reel-out,
    momentum is injected in the system, and during reel-in, momentum is extracted.
    It is assumed that the tether mass is concentrated at the main tether node.

    See "Lagrangian Dynamics for Open Systems", R. L. Greene and J. J. Matese 1981, Eur. J. Phys. 2, 103.

    @return: lagrangian_momentum_correction - correction term that can directly be added to rhs of transl_dyn
    """

    # initialize
    xgcdot = generalized_coordinates['scaled']['xgcdot'].cat
    lagrangian_momentum_correction = cas.DM(np.zeros(xgcdot.shape))

    for n in range(1, architecture.number_of_nodes):
        label = str(n) + str(architecture.parent_map[n])
        mass = node_masses['m' + label]
        velocity = system_variables['SI']['xd']['dq' + label]  # velocity of the mass particles leaving the system
        mass_flow = cas.mtimes(cas.jacobian(mass, system_variables['scaled']['xd', 'l_t']),
                               system_variables['scaled']['xd', 'dl_t'])
        lagrangian_momentum_correction += mass_flow * cas.mtimes(velocity.T, cas.jacobian(velocity,
                                                                                          xgcdot)).T  # see formula in reference

    return lagrangian_momentum_correction


def xddot_outputs(options, variables, outputs):
    outputs['xddot_from_var'] = variables['xddot']
    return outputs


def anticollision_radius_inequality(options, variables, outputs, parameters, architecture):
    kite_nodes = architecture.kite_nodes
    parent_map = architecture.parent_map

    tightness = options['model_bounds']['anticollision_radius']['scaling']

    if 'anticollision_radius' not in list(outputs.keys()):
        outputs['anticollision_radius'] = {}

    for kite in kite_nodes:
        parent = parent_map[kite]

        local_ineq = indicators.get_radius_inequality(options, variables, kite, parent, parameters)
        outputs['anticollision_radius']['n' + str(kite)] = tightness * local_ineq

    return outputs


def anticollision_inequality(options, variables, outputs, parameters, architecture):
    kite_nodes = architecture.kite_nodes
    parent_map = architecture.parent_map

    if 'anticollision' not in list(outputs.keys()):
        outputs['anticollision'] = {}

    if len(kite_nodes) == 1:
        outputs['anticollision']['n10'] = cas.DM(0.0)
        return outputs

    safety_factor = options['model_bounds']['anticollision']['safety_factor']
    dist_min = safety_factor * parameters['theta0', 'geometry', 'b_ref']

    kite_combinations = itertools.combinations(kite_nodes, 2)
    for kite_pair in kite_combinations:
        kite_a = kite_pair[0]
        kite_b = kite_pair[1]
        parent_a = parent_map[kite_a]
        parent_b = parent_map[kite_b]
        dist = variables['xd']['q' + str(kite_a) + str(parent_a)] - variables['xd']['q' + str(kite_b) + str(parent_b)]
        dist_inequality = 1 - (cas.mtimes(dist.T, dist) / dist_min ** 2)

        outputs['anticollision']['n' + str(kite_a) + str(kite_b)] = dist_inequality

    return outputs


def dcoeff_actuation_inequality(options, variables, parameters, outputs):
    # nu*u_max + (1 - nu)*u_compromised_max > u
    # u - nu*u_max + (1 - nu)*u_compromised_max < 0
    if int(options['kite_dof']) != 3:
        return outputs
    if 'dcoeff_actuation' not in list(outputs.keys()):
        outputs['dcoeff_actuation'] = OrderedDict()
    nu = parameters['phi', 'nu']
    dcoeff_max = options['model_bounds']['dcoeff_max']
    dcoeff_compromised_max = parameters['theta0', 'model_bounds', 'dcoeff_compromised_max']
    dcoeff_min = options['model_bounds']['dcoeff_min']
    dcoeff_compromised_min = parameters['theta0', 'model_bounds', 'dcoeff_compromised_min']
    traj_type = options['trajectory']['type']
    scenario = None
    broken_kite = None

    for variable in list(variables['u'].keys()):
        if variable[:6] == 'dcoeff':

            dcoeff = variables['u'][variable]
            if traj_type == 'compromised_landing':
                scenario, broken_kite = options['compromised_landing']['emergency_scenario']
            if (variable[-2] == str(broken_kite) and scenario == 'broken_roll'):
                outputs['dcoeff_actuation']['max_n' + variable[6:]] = dcoeff - (
                            nu * dcoeff_max + (1 - nu) * dcoeff_compromised_max)
                outputs['dcoeff_actuation']['min_n' + variable[6:]] = - dcoeff + (
                            nu * dcoeff_min + (1 - nu) * dcoeff_compromised_min)
            else:
                outputs['dcoeff_actuation']['max_n' + variable[6:]] = dcoeff - dcoeff_max
                outputs['dcoeff_actuation']['min_n' + variable[6:]] = - dcoeff + dcoeff_min

    return outputs


def coeff_actuation_inequality(options, variables, parameters, outputs):
    # nu*xd_max + (1 - nu)*u_compromised_max > xd
    # xd - nu*xd_max + (1 - nu)*xd_compromised_max < 0
    if int(options['kite_dof']) != 3:
        return outputs
    if 'coeff_actuation' not in list(outputs.keys()):
        outputs['coeff_actuation'] = OrderedDict()
    nu = parameters['phi', 'nu']
    coeff_max = options['aero']['three_dof']['coeff_max']
    coeff_compromised_max = parameters['theta0', 'model_bounds', 'coeff_compromised_max']
    coeff_min = options['aero']['three_dof']['coeff_min']
    coeff_compromised_min = parameters['theta0', 'model_bounds', 'coeff_compromised_min']
    traj_type = options['trajectory']['type']
    scenario = None
    broken_kite = None

    for variable in list(variables['xd'].keys()):
        if variable[:5] == 'coeff':

            coeff = variables['xd'][variable]
            if traj_type == 'compromised_landing':
                scenario, broken_kite = options['compromised_landing']['emergency_scenario']

            if (variable[-2] == str(broken_kite) and scenario == 'structural_damages'):
                outputs['coeff_actuation']['max_n' + variable[5:]] = coeff - (
                            nu * coeff_max + (1 - nu) * coeff_compromised_max)
                outputs['coeff_actuation']['min_n' + variable[5:]] = - coeff + (
                            nu * coeff_min + (1 - nu) * coeff_compromised_min)
            else:
                outputs['coeff_actuation']['max_n' + variable[5:]] = coeff - coeff_max
                outputs['coeff_actuation']['min_n' + variable[5:]] = - coeff + coeff_min

    return outputs


def acceleration_inequality(options, variables, outputs, parameters):
    acc_max = options['model_bounds']['acceleration']['acc_max'] * options['scaling']['other']['g']

    if 'acceleration' not in list(outputs.keys()):
        outputs['acceleration'] = {}

    for name in list(variables['xddot'].keys()):

        var_name = name[0:3]
        var_label = name[3:]

        if var_name == 'ddq':
            acc = variables['xddot'][name]
            acc_sq = cas.mtimes(acc.T, acc)
            acc_sq_norm = acc_sq / acc_max ** 2.

            # acc^2 < acc_max^2 -> acc^2 / acc_max^2 - 1 < 0
            local_ineq = acc_sq_norm - 1.

            outputs['acceleration']['n' + var_label] = local_ineq

    return outputs


def airspeed_inequality(options, variables, outputs, parameters, architecture):
    # system architecture
    kite_nodes = architecture.kite_nodes
    parent_map = architecture.parent_map

    # constraint bounds
    airspeed_max = parameters['theta0', 'model_bounds', 'airspeed_limits'][1]
    airspeed_min = parameters['theta0', 'model_bounds', 'airspeed_limits'][0]

    if 'airspeed_max' not in list(outputs.keys()):
        outputs['airspeed_max'] = {}
    if 'airspeed_min' not in list(outputs.keys()):
        outputs['airspeed_min'] = {}

    for kite in kite_nodes:
        airspeed = outputs['aerodynamics']['airspeed' + str(kite)]
        parent = parent_map[kite]
        outputs['airspeed_max']['n' + str(kite) + str(parent)] = airspeed / airspeed_max - 1.
        outputs['airspeed_min']['n' + str(kite) + str(parent)] = - airspeed / airspeed_min + 1.

    return outputs


def angular_velocity_inequality(options, variables, outputs, parameters, architecture):
    kite_nodes = architecture.kite_nodes
    parent_map = architecture.parent_map

    kite_dof = options['kite_dof']

    omega_norm_max_deg = parameters['theta0', 'model_bounds', 'angular_velocity_max']
    omega_norm_max = omega_norm_max_deg * np.pi / 180.

    if int(kite_dof) == 6:

        if 'angular_velocity' not in list(outputs.keys()):
            outputs['angular_velocity'] = {}

        for kite in kite_nodes:
            parent = parent_map[kite]
            omega = variables['xd']['omega' + str(kite) + str(parent)]

            # |omega| <= omega_norm_max
            # omega^\top omega <= omega_norm_max^2
            # omega^\top omega - omega_norm_max^2 <= 0
            # (omega^\top omega)/omega_norm_max^2 - 1 <= 0

            ineq = cas.mtimes(omega.T, omega) / omega_norm_max ** 2. - 1.
            outputs['angular_velocity']['n' + str(kite) + str(parent)] = ineq

    return outputs


def tether_stress_inequality(options, variables, outputs, parameters, architecture):
    # system architecture (see zanon2013a)
    number_of_nodes = architecture.number_of_nodes
    kite_nodes = architecture.kite_nodes
    parent_map = architecture.parent_map

    # system (scaled) variables
    xd = variables['xd']
    xa = variables['xa']
    theta = variables['theta']

    tightness = options['model_bounds']['tether_stress']['scaling']

    if 'tether_stress' not in list(outputs.keys()):
        outputs['tether_stress'] = {}

    if 'tether_force_max' not in list(outputs.keys()):
        outputs['tether_force_max'] = {}

    if 'tether_force_min' not in list(outputs.keys()):
        outputs['tether_force_min'] = {}

    if 'tether_tension' not in list(outputs.keys()):
        outputs['tether_tension'] = {}

    # mass vector, containing the mass of all nodes
    for n in range(1, number_of_nodes):

        parent = parent_map[n]

        seg_props = get_tether_segment_properties(options, architecture, variables, parameters, upper_node=n)
        seg_length = seg_props['seg_length']
        cross_section_area = seg_props['cross_section_area']
        max_area = seg_props['max_area']

        tension = xa['lambda' + str(n) + str(parent)] * seg_length

        min_tension = parameters['theta0', 'model_bounds', 'tether_force_limits'][0]
        max_tension = parameters['theta0', 'model_bounds', 'tether_force_limits'][1]

        maximum_allowed_stress = parameters['theta0', 'tether', 'max_stress'] / parameters['theta0', 'tether', 'stress_safety_factor']
        max_tension_from_stress = maximum_allowed_stress * max_area

        # stress_max = max_tension_from_stress / A_max
        # (tension / A) < stress_max
        # tension / A < max_tension_from_stress / Amax
        # tension / max_tension_from_stress < A / Amax
        # tension / max_tension_from_stress - A / Amax < 0
        stress_inequality_untightened = tension / max_tension_from_stress - cross_section_area / max_area
        stress_inequality = stress_inequality_untightened * tightness

        # outputs related to the constraints themselves
        tether_constraint_includes = options['model_bounds']['tether']['tether_constraint_includes']
        if n in tether_constraint_includes['stress']:
            outputs['tether_stress']['n' + str(n) + str(parent)] = stress_inequality
        if n in tether_constraint_includes['force']:
            outputs['tether_force_max']['n' + str(n) + str(parent)] = (tension - max_tension) / max_tension
            outputs['tether_force_min']['n' + str(n) + str(parent)] = -(tension - min_tension) / min_tension

        # outputs so that the user can find the stress and tension
        outputs['local_performance']['tether_stress' + str(n) + str(parent)] = tension / cross_section_area
        outputs['local_performance']['tether_force' + str(n) + str(parent)] = tension

    if options['cross_tether'] and len(architecture.kite_nodes) > 1:
        for l in architecture.layer_nodes:
            kites = architecture.kites_map[l]
            seg_length = theta['l_c{}'.format(l)]
            seg_diam = theta['diam_c{}'.format(l)]
            cross_section = np.pi * seg_diam ** 2. / 4.
            cross_section_max = np.pi * options['system_bounds']['theta']['diam_c'][1] ** 2.0 / 4.
            max_tension_from_stress = maximum_allowed_stress * cross_section_max

            if len(kites) == 2:
                tension = xa['lambda{}{}'.format(kites[0], kites[1])] * seg_length
                stress_inequality_untightened = tension / max_tension_from_stress - cross_section / cross_section_max
                outputs['tether_stress']['n{}{}'.format(kites[0], kites[1])] = stress_inequality_untightened * tightness
                outputs['local_performance']['tether_stress{}{}'.format(kites[0], kites[1])] = tension / cross_section
            else:
                for k in range(len(kites)):
                    tension = xa['lambda{}{}'.format(kites[k], kites[(k + 1) % len(kites)])] * seg_length
                    stress_inequality_untightened = tension / max_tension_from_stress - cross_section / cross_section_max
                    outputs['tether_stress']['n{}{}'.format(kites[k], kites[
                        (k + 1) % len(kites)])] = stress_inequality_untightened * tightness
                    outputs['local_performance'][
                        'tether_stress{}{}'.format(kites[k], kites[(k + 1) % len(kites)])] = tension / cross_section

    return outputs


def wound_tether_length_inequality(options, variables, outputs, parameters, architecture):
    outputs['wound_tether_length'] = {}
    outputs['wound_tether_length']['wound_tether_length'] = cas.DM((0,0))

    if options['tether']['use_wound_tether']:
        l_t_full = variables['theta']['l_t_full']
        l_t = variables['xd']['l_t']

        length_scaling = options['scaling']['xd']['l_t']

        outputs['wound_tether_length']['wound_tether_length'] = (l_t - l_t_full) / length_scaling

    return outputs


def induction_equations(options, atmos, wind, variables, outputs, parameters, architecture):
    if 'induction' not in list(outputs.keys()):
        outputs['induction'] = {}

    outputs['induction']['induction_init'] = induction.get_trivial_residual(options, atmos, wind, variables, parameters,
                                                                            outputs, architecture)
    outputs['induction']['induction_final'] = induction.get_final_residual(options, atmos, wind, variables, parameters,
                                                                           outputs, architecture)

    return outputs


def generate_scaled_variables(scaling_options, variables):
    # set scaling for xddot equal to that of xd
    scaling_options['xddot'] = {}
    for name in list(scaling_options['xd'].keys()):
        scaling_options['xddot']['d' + name] = scaling_options['xd'][name]

    # generate scaling for all variables
    scaling = {}
    for variable_type in list(variables.keys()):  # iterate over variable type (e.g. states, controls,...)
        scaling[variable_type] = {}
        for name in struct_op.subkeys(variables, variable_type):
            scaling_name = struct_op.get_scaling_name(scaling_options, variable_type,
                                                      name)  # seek highest integral variable name in scaling options
            # check if node-specific scaling option is provided (e.g. 'q10')
            if len(scaling_name) > 0:
                scaling[variable_type][name] = scaling_options[variable_type][scaling_name]
            # otherwise: check if global node variable scaling option is provided (e.g. 'q')
            else:
                var_name = struct_op.get_node_variable_name(name)  # omit node numbers
                scaling_name = struct_op.get_scaling_name(scaling_options, variable_type,
                                                          var_name)  # seek highest integral variable name in scaling options
                if len(scaling_name) > 0:  # check if scaling option provided
                    scaling[variable_type][name] = scaling_options[variable_type][scaling_name]
                else:  # non-provided scaling factor is equal to one
                    scaling[variable_type][name] = 1.0

    # set scaling for u-elements equal to that of its integrals in xd-vars (e.g. u.ddlt <=> xd.lt <=> xddot.ddlt)
    for name in struct_op.subkeys(variables, 'u'):
        if name in list(scaling['xddot'].keys()):
            scaling['u'][name] = scaling['xddot'][name]

    # scale variables
    scaled_variables = {}
    for variable_type in list(scaling.keys()):
        scaled_variables[variable_type] = cas.struct_SX(
            [cas.entry(name, expr=variables[variable_type, name] * scaling[variable_type][name]) for name in
             struct_op.subkeys(variables, variable_type)])

    return scaled_variables, scaling


def generate_generalized_coordinates(system_variables, system_gc):
    try:
        test = system_variables['xd', 'l_t']
        struct_flag = 1
    except:
        struct_flag = 0

    if struct_flag == 1:
        generalized_coordinates = {}
        generalized_coordinates['xgc'] = cas.struct_SX(
            [cas.entry(name, expr=system_variables['xd', name]) for name in system_gc])
        generalized_coordinates['xgcdot'] = cas.struct_SX(
            [cas.entry('d' + name, expr=system_variables['xd', 'd' + name])
             for name in system_gc])
        generalized_coordinates['xgcddot'] = cas.struct_SX(
            [cas.entry('dd' + name, expr=system_variables['xddot', 'dd' + name])
             for name in system_gc])
    else:
        generalized_coordinates = {}
        generalized_coordinates['xgc'] = cas.struct_SX(
            [cas.entry(name, expr=system_variables['xd'][name]) for name in system_gc])
        generalized_coordinates['xgcdot'] = cas.struct_SX(
            [cas.entry('d' + name, expr=system_variables['xd']['d' + name])
             for name in system_gc])
        generalized_coordinates['xgcddot'] = cas.struct_SX(
            [cas.entry('dd' + name, expr=system_variables['xddot']['dd' + name])
             for name in system_gc])

    return generalized_coordinates


def generate_holonomic_constraints(architecture, outputs, variables, generalized_coordinates, parameters, options):
    number_of_nodes = architecture.number_of_nodes
    parent_map = architecture.parent_map
    kite_nodes = architecture.kite_nodes

    # extract necessary SI variables
    xd_si = variables['SI']['xd']
    theta_si = variables['SI']['theta']
    xgc_si = generalized_coordinates['SI']['xgc']
    xa_si = variables['SI']['xa']
    xddot_si = variables['SI']['xddot']

    # extract necessary scaled variables
    xgc = generalized_coordinates['scaled']['xgc']
    xgcdot = generalized_coordinates['scaled']['xgcdot']
    xgcddot = generalized_coordinates['scaled']['xgcddot']

    # scaled variables struct
    var = variables['scaled']
    ddl_t_scaled = var[struct_op.get_variable_type(variables['SI'], 'ddl_t'), 'ddl_t']

    if 'tether_length' not in list(outputs.keys()):
        outputs['tether_length'] = {}

    # build constraints with si variables and obtain derivatives w.r.t scaled variables
    g = []
    gdot = []
    gddot = []
    holonomic_constraints = 0.0
    for n in range(1, number_of_nodes):
        parent = parent_map[n]

        if n not in kite_nodes or options['tether']['attachment'] == 'com':
            current_node = xgc_si['q' + str(n) + str(parent)]
        elif n in kite_nodes and options['tether']['attachment'] == 'stick':
            if int(options['kite_dof']) == 6:
                dcm = cas.reshape(xd_si['r{}{}'.format(n, parent)], (3, 3))
            elif int(options['kite_dof']) == 3:
                raise ValueError('Stick tether attachment option not implemented for 3DOF kites')
            current_node = xgc_si['q{}{}'.format(n, parent)] + cas.mtimes(dcm,
                                                                          parameters['theta0', 'geometry', 'r_tether'])
        else:
            raise ValueError('Unknown tether attachment option: {}'.format(options['tether']['attachment']))

        if n == 1:
            previous_node = cas.vertcat(0., 0., 0.)
            segment_length = xd_si['l_t']
        elif n in kite_nodes:
            grandparent = parent_map[parent]
            previous_node = xgc_si['q' + str(parent) + str(grandparent)]
            segment_length = theta_si['l_s']
        else:
            grandparent = parent_map[parent]
            previous_node = xgc_si['q' + str(parent) + str(grandparent)]
            segment_length = theta_si['l_i']

        # holonomic constraint
        length_constraint = 0.5 * (
                cas.mtimes(
                    (current_node - previous_node).T,
                    (current_node - previous_node)) - segment_length ** 2.0)
        g.append(length_constraint)

        # first-order derivative
        gdot.append(
            cas.mtimes(
                cas.jacobian(
                    g[-1],
                    cas.vertcat(xgc.cat, var['xd', 'l_t'])
                ),
                cas.vertcat(xgcdot.cat, var['xd', 'dl_t'])
            )
        )

        if int(options['kite_dof']) == 6:
            for k in kite_nodes:
                kparent = parent_map[k]
                gdot[-1] += 2 * cas.mtimes(
                    vect_op.jacobian_dcm(g[-1], xd_si, var, k, kparent),
                    var['xd', 'omega{}{}'.format(k, kparent)]
                )

        # second-order derivative
        gddot.append(
            cas.mtimes(
                cas.jacobian(
                    gdot[-1],
                    cas.vertcat(xgc.cat, var['xd', 'l_t'], xgcdot.cat, var['xd', 'dl_t'])
                ),
                cas.vertcat(xgcdot.cat, var['xd', 'dl_t'], xgcddot.cat, ddl_t_scaled)
            )
        )

        if int(options['kite_dof']) == 6:
            for k in kite_nodes:
                kparent = parent_map[k]

                # add time derivative due to angular velocity
                gddot[-1] += 2 * cas.mtimes(
                    vect_op.jacobian_dcm(gdot[-1], xd_si, var, k, kparent),
                    var['xd', 'omega{}{}'.format(k, kparent)]
                )

                # add time derivative due to angular acceleration
                gddot[-1] += 2 * cas.mtimes(
                    vect_op.jacobian_dcm(g[-1], xd_si, var, k, kparent),
                    var['xddot', 'domega{}{}'.format(k, kparent)]
                )

        outputs['tether_length']['c' + str(n) + str(parent)] = g[-1]
        outputs['tether_length']['dc' + str(n) + str(parent)] = gdot[-1]
        outputs['tether_length']['ddc' + str(n) + str(parent)] = gddot[-1]
        holonomic_constraints += xa_si['lambda{}{}'.format(n, parent)] * g[-1]

    # add cross-tethers
    if options['cross_tether'] and len(kite_nodes) > 1:
        for l in architecture.layer_nodes:
            kite_children = architecture.kites_map[l]

            # dual kite system (per layer) only has one tether
            if len(kite_children) == 2:
                no_tethers = 1
            else:
                no_tethers = len(kite_children)

            # add cross-tether constraints
            for k in range(no_tethers):

                # set-up relevant node numbers
                n0 = '{}{}'.format(kite_children[k], parent_map[kite_children[k]])
                n1 = '{}{}'.format(kite_children[(k + 1) % len(kite_children)],
                                   parent_map[kite_children[(k + 1) % len(kite_children)]])
                n01 = '{}{}'.format(kite_children[k], kite_children[(k + 1) % len(kite_children)])

                # center-of-mass attachment
                if options['tether']['cross_tether']['attachment'] == 'com':
                    first_node = xgc_si['q{}'.format(n0)]
                    second_node = xgc_si['q{}'.format(n1)]

                # stick or wing-tip attachment
                else:

                    # only implemented for 6DOF
                    if int(options['kite_dof']) == 6:

                        # rotation matrices of relevant kites
                        dcm_first = cas.reshape(xd_si['r{}'.format(n0)], (3, 3))
                        dcm_second = cas.reshape(xd_si['r{}'.format(n1)], (3, 3))

                        # stick: same attachment point as secondary tether
                        if options['tether']['cross_tether']['attachment'] == 'stick':
                            r_tether = parameters['theta0', 'geometry', 'r_tether']

                        # wing_tip: attachment half a wing span in negative span direction
                        elif options['tether']['cross_tether']['attachment'] == 'wing_tip':
                            r_tether = cas.vertcat(0.0, -parameters['theta0', 'geometry', 'b_ref'] / 2.0, 0.0)

                        # unknown option notifier
                        else:
                            raise ValueError('Unknown cross-tether attachment option: {}'.format(
                                options['tether']['cross_tether']['attachment']))

                        # create attachment nodes
                        first_node = xgc_si['q{}'.format(n0)] + cas.mtimes(dcm_first, r_tether)
                        second_node = xgc_si['q{}'.format(n1)] + cas.mtimes(dcm_second, r_tether)

                    # not implemented for 3DOF
                    elif int(options['kite_dof']) == 3:
                        raise ValueError('Stick cross-tether attachment options not implemented for 3DOF kites')

                # cross-tether length
                segment_length = theta_si['l_c{}'.format(l)]

                # create constraint
                length_constraint = 0.5 * (
                        cas.mtimes(
                            (first_node - second_node).T,
                            (first_node - second_node)) - segment_length ** 2.0)

                # append constraint
                g.append(length_constraint)

                # first-order derivative
                gdot.append(
                    cas.mtimes(
                        cas.jacobian(g[-1], cas.vertcat(xgc.cat, var['xd', 'l_t'])),
                        cas.vertcat(xgcdot.cat, var['xd', 'dl_t'])
                    )
                )
                if int(options['kite_dof']) == 6:
                    for kite in kite_children:
                        kparent = parent_map[kite]
                        gdot[-1] += 2 * cas.mtimes(
                            vect_op.jacobian_dcm(g[-1], xd_si, var, kite, kparent),
                            var['xd', 'omega{}{}'.format(kite, kparent)]
                        )

                # second-order derivative
                gddot.append(
                    cas.mtimes(
                        cas.jacobian(gdot[-1], cas.vertcat(xgc.cat, var['xd', 'l_t'], xgcdot.cat, var['xd', 'dl_t'])),
                        cas.vertcat(xgcdot.cat, var['xd', 'dl_t'], xgcddot.cat, ddl_t_scaled)
                    )
                )
                if int(options['kite_dof']) == 6:
                    for kite in kite_children:
                        kparent = parent_map[kite]
                        gddot[-1] += 2 * cas.mtimes(
                            vect_op.jacobian_dcm(gdot[-1], xd_si, var, kite, kparent),
                            var['xd', 'omega{}{}'.format(kite, kparent)]
                        )
                        gddot[-1] += 2 * cas.mtimes(
                            vect_op.jacobian_dcm(g[-1], xd_si, var, kite, kparent),
                            var['xddot', 'domega{}{}'.format(kite, kparent)]
                        )

                # save invariants to outputs
                outputs['tether_length']['c{}'.format(n01)] = g[-1]
                outputs['tether_length']['dc{}'.format(n01)] = gdot[-1]
                outputs['tether_length']['ddc{}'.format(n01)] = gddot[-1]

                # add to holonomic constraints
                holonomic_constraints += xa_si['lambda{}'.format(n01)] * g[-1]

        if n in kite_nodes:
            if 'r' + str(n) + str(parent) in list(xd_si.keys()):
                r = cas.reshape(var['xd', 'r' + str(n) + str(parent)], (3, 3))
                orthonormality = cas.mtimes(r.T, r) - np.eye(3)
                orthonormality = cas.reshape(orthonormality, (9, 1))

                outputs['tether_length']['orthonormality' + str(n) + str(parent)] = orthonormality

                dr_dt = variables['SI']['xddot']['dr' + str(n) + str(parent)]
                dr_dt = cas.reshape(dr_dt, (3, 3))
                omega = variables['SI']['xd']['omega' + str(n) + str(parent)]
                omega_skew = vect_op.skew(omega)
                dr = cas.mtimes(r, omega_skew)
                rot_kinematics = dr_dt - dr
                rot_kinematics = cas.reshape(rot_kinematics, (9, 1))

                outputs['tether_length']['rot_kinematics10'] = rot_kinematics

    g = cas.vertcat(*g)
    gdot = cas.vertcat(*gdot)
    gddot = cas.vertcat(*gddot)
    # holonomic_fun = cas.Function('holonomic_fun', [xgc,xgcdot,xgcddot,var['xd','l_t'],var['xd','dl_t'],ddl_t_scaled],[g,gdot,gddot])
    holonomic_fun = None  # todo: still used?

    return holonomic_constraints, outputs, g, gdot, gddot, holonomic_fun


def generate_holonomic_scaling(options, architecture, variables, parameters):
    scaling = options['scaling']
    holonomic_scaling = []

    # mass vector, containing the mass of all nodes
    for n in range(1, architecture.number_of_nodes):
        seg_props = get_tether_segment_properties(options, architecture, variables, parameters, upper_node=n)
        loc_scaling = seg_props['scaling_length'] ** 2.
        holonomic_scaling = cas.vertcat(holonomic_scaling, loc_scaling)

    number_of_kites = len(architecture.kite_nodes)
    if number_of_kites > 1 and options['cross_tether']:
        for l in architecture.layer_nodes:
            layer_kites = architecture.kites_map[l]
            number_of_layer_kites = len(layer_kites)

            if number_of_layer_kites == 2:
                holonomic_scaling = cas.vertcat(holonomic_scaling, scaling['theta']['l_c'] ** 2)
            else:
                for kdx in layer_kites:
                    holonomic_scaling = cas.vertcat(holonomic_scaling, scaling['theta']['l_c'] ** 2)

    return holonomic_scaling


def generate_rotational_dynamics(options, variables, f_nodes, holonomic_constraints, parameters, outputs, architecture):
    kite_nodes = architecture.kite_nodes
    parent_map = architecture.parent_map

    j_inertia = parameters['theta0', 'geometry', 'j']

    xd = variables['SI']['xd']
    xddot = variables['SI']['xddot']

    rotation_dynamics = []
    if int(options['kite_dof']) == 6:
        outputs['tether_moments'] = {}
        for n in kite_nodes:
            parent = parent_map[n]
            moment = f_nodes['m' + str(n) + str(parent)]

            rlocal = cas.reshape(xd['r' + str(n) + str(parent)], (3, 3))
            drlocal = cas.reshape(xddot['dr' + str(n) + str(parent)], (3, 3))

            omega = xd['omega' + str(n) + str(parent)]
            omega_skew = vect_op.skew(omega)

            domega = xddot['domega' + str(n) + str(parent)]

            # moment = J dot(omega) + omega x (J omega)
            omega_derivative = cas.mtimes(j_inertia, domega) + vect_op.cross(omega,
                                                                             cas.mtimes(j_inertia, omega)) - moment

            # tether constraint contribution
            tether_moment = 2 * vect_op.rot_op(
                rlocal,
                cas.reshape(cas.jacobian(holonomic_constraints, variables['scaled']['xd', 'r{}{}'.format(n, parent)]),
                            (3, 3))
            )
            omega_derivative += tether_moment
            outputs['tether_moments']['n{}{}'.format(n, parent)] = tether_moment

            # concatenate
            rotation_dynamics = cas.vertcat(rotation_dynamics, omega_derivative / vect_op.norm(cas.diag(j_inertia)))

            # Rdot = R omega_skew -> R ( kappa/2 (I - R.T R) + omega_skew )
            orthonormality = parameters['theta0', 'kappa_r'] / 2. * (np.eye(3) - cas.mtimes(rlocal.T, rlocal))
            ref_frame_deriv_matrix = drlocal - cas.mtimes(rlocal, orthonormality + omega_skew)
            ref_frame_derivative = cas.reshape(ref_frame_deriv_matrix, (9, 1))
            rotation_dynamics = cas.vertcat(rotation_dynamics, ref_frame_derivative)

    return rotation_dynamics, outputs


def get_roll_expr(xd, n0, n1, parent_map):
    """ Return the expression that allows to compute the bridle roll angle via roll = atan(expr),
    :param xd: system variables
    :param n0: node number of kite node
    :param n1: node number of tether attachment node 
    :param parent_map: architecture parent map
    :return: tan(roll)
    """

    # node + parent position
    q0 = xd['q{}{}'.format(n0, parent_map[n0])]
    if n1 == 0:
        q1 = np.zeros((3, 1))
    else:
        q1 = xd['q{}{}'.format(n1, parent_map[n1])]

    q_hat = q0 - q1  # tether direction
    r = cas.reshape(xd['r{}{}'.format(n0, parent_map[n0])], (3, 3))  # rotation matrix

    return cas.mtimes(q_hat.T, r[:, 1] / cas.mtimes(q_hat.T, r[:, 2]))


def get_pitch_expr(xd, n0, n1, parent_map):
    """ Return the expression that allows to compute the bridle pitch angle via pitch = asin(expr),
    :param xd: system variables
    :param n0: node number of kite node
    :param n1: node number of tether attachment node 
    :param parent_map: architecture parent map
    :return: sin(pitch)
    """

    # node + parent position
    q0 = xd['q{}{}'.format(n0, parent_map[n0])]
    if n1 == 0:
        q1 = np.zeros((3, 1))
    else:
        q1 = xd['q{}{}'.format(n1, parent_map[n1])]

    q_hat = q0 - q1  # tether direction
    r = cas.reshape(xd['r{}{}'.format(n0, parent_map[n0])], (3, 3))  # rotation matrix

    return cas.mtimes(q_hat.T, r[:, 0] / vect_op.norm(q_hat))


def get_span_angle_expr(options, xd, n0, n1, parent_map, parameters):
    """ Return the expression that allows to compute the cross-tether vs. body span-vector angle and related inequality,
    :param xd: system variables
    :param n0: node number of kite node
    :param n1: node number of tether attachment node
    :param parent_map: architecture parent map
    :return: span_inequality, span_angle
    """

    # node + parent position
    q0 = xd['q{}{}'.format(n0, parent_map[n0])]
    r0 = cas.reshape(xd['r{}{}'.format(n0, parent_map[n0])], (3, 3))  # rotation matrix
    r_wtip = cas.vertcat(0.0, -parameters['theta0', 'geometry', 'b_ref'] / 2, 0.0)

    if n1 == 0:
        q1 = np.zeros((3, 1))
    else:
        q1 = xd['q{}{}'.format(n1, parent_map[n1])]
        r1 = cas.reshape(xd['r{}{}'.format(n1, parent_map[n1])], (3, 3))

    # first node
    q_first = q0 + cas.mtimes(r0, r_wtip)
    q_second = q1 + cas.mtimes(r1, r_wtip)

    # tether direction
    q_hat = q_first - q_second

    # span inequality
    span_ineq = cas.cos(parameters['theta0', 'model_bounds', 'span_angle']) * vect_op.norm(q_hat) - cas.mtimes(
        r0[:, 1].T, q_hat)

    # scale span inequality
    span_ineq = span_ineq / options['scaling']['theta']['l_s']

    # angle between aircraft span vector and cross-tether
    span_angle = cas.acos(cas.mtimes(r0[:, 1].T, q_hat) / vect_op.norm(q_hat))

    return span_ineq, span_angle


def get_yaw_expr(options, xd, n0, n1, parent_map, gamma_max):
    """ Compute angle between kite yaw vector and tether, including corresponding inequality.
    :param xd: system variables
    :param n0: node number of kite node
    :param n1: node number of tether attachment node
    :param parent_map: architecture parent map
    :return: yaw expression, yaw angle
    """
    # node + parent position
    q0 = xd['q{}{}'.format(n0, parent_map[n0])]

    if n1 == 0:
        q1 = np.zeros((3, 1))
    else:
        q1 = xd['q{}{}'.format(n1, parent_map[n1])]

    q_hat = q0 - q1  # tether direction
    r = cas.reshape(xd['r{}{}'.format(n0, parent_map[n0])], (3, 3))  # rotation matrix

    yaw_angle = cas.arccos(cas.mtimes(q_hat.T, r[:, 2]) / vect_op.norm(q_hat))
    yaw_expr = (cas.mtimes(q_hat.T, r[:, 2]) - cas.cos(gamma_max) * vect_op.norm(q_hat))

    # scale yaw_expression
    if n0 == 1:
        scale = options['scaling']['xd']['l_t']
    else:
        scale = options['scaling']['theta']['l_s']
    yaw_expr = yaw_expr / scale

    return yaw_expr, yaw_angle


def rotation_inequality(options, variables, parameters, architecture, outputs):
    # system architecture
    number_of_nodes = architecture.number_of_nodes
    kite_nodes = architecture.kite_nodes
    parent_map = architecture.parent_map

    xd = variables['xd']

    outputs['rotation'] = {}

    # create bound expressions from angle bounds
    if options['model_bounds']['rotation']['type'] == 'roll_pitch':
        expr = cas.vertcat(
            cas.tan(parameters['theta0', 'model_bounds', 'rot_angles', 0]),
            cas.sin(parameters['theta0', 'model_bounds', 'rot_angles', 1])
        )

    for n in kite_nodes:
        parent = parent_map[n]

        if options['model_bounds']['rotation']['type'] == 'roll_pitch':
            rotation_angles = cas.vertcat(
                get_roll_expr(xd, n, parent_map[n], parent_map),
                get_pitch_expr(xd, n, parent_map[n], parent_map)
            )
            outputs['rotation']['max_n' + str(n) + str(parent)] = - expr + rotation_angles
            outputs['rotation']['min_n' + str(n) + str(parent)] = - expr - rotation_angles
            outputs['local_performance']['rot_angles' + str(n) + str(parent)] = cas.vertcat(
                cas.atan(rotation_angles[0]),
                cas.asin(rotation_angles[1])
            )

        elif options['model_bounds']['rotation']['type'] == 'yaw':
            yaw_expr, yaw_angle = get_yaw_expr(
                options, xd, n, parent_map[n], parent_map,
                parameters['theta0', 'model_bounds', 'rot_angles', 2]
            )
            outputs['rotation']['max_n' + str(n) + str(parent)] = - yaw_expr
            outputs['local_performance']['rot_angles' + str(n) + str(parent)] = yaw_angle

    # cross-tether
    if options['cross_tether'] and (number_of_nodes > 2):
        for l in architecture.layer_nodes:
            kites = architecture.kites_map[l]
            if len(kites) == 2:
                no_tethers = 1
            else:
                no_tethers = len(kites)

            for k in range(no_tethers):
                tether_name = '{}{}'.format(kites[k], kites[(k + 1) % len(kites)])
                tether_name2 = '{}{}'.format(kites[(k + 1) % len(kites)], kites[k])

                if options['tether']['cross_tether']['attachment'] is not 'wing_tip':

                    # get roll and pitch expressions at each end of the cross-tether
                    rotation_angles = cas.vertcat(
                        get_roll_expr(xd, kites[k], kites[(k + 1) % len(kites)], parent_map),
                        get_pitch_expr(xd, kites[k], kites[(k + 1) % len(kites)], parent_map)
                    )
                    rotation_angles2 = cas.vertcat(
                        get_roll_expr(xd, kites[(k + 1) % len(kites)], kites[k], parent_map),
                        get_pitch_expr(xd, kites[(k + 1) % len(kites)], kites[k], parent_map)
                    )
                    outputs['rotation']['max_n' + tether_name] = - expr + rotation_angles
                    outputs['rotation']['max_n' + tether_name2] = - expr + rotation_angles2
                    outputs['rotation']['min_n' + tether_name] = - expr - rotation_angles
                    outputs['rotation']['min_n' + tether_name2] = - expr - rotation_angles2
                    outputs['local_performance']['rot_angles' + tether_name] = cas.vertcat(
                        cas.atan(rotation_angles[0]),
                        cas.asin(rotation_angles[1])
                    )
                    outputs['local_performance']['rot_angles' + tether_name2] = cas.vertcat(
                        cas.atan(rotation_angles2[0]),
                        cas.asin(rotation_angles2[1])
                    )

                else:

                    # get angle between body span vector and cross-tether and related inequality
                    rotation_angle_expr, span = get_span_angle_expr(options, xd, kites[k], kites[(k + 1) % len(kites)],
                                                                    parent_map, parameters)
                    rotation_angle_expr2, span2 = get_span_angle_expr(options, xd, kites[(k + 1) % len(kites)],
                                                                      kites[k], parent_map, parameters)

                    outputs['rotation']['max_n' + tether_name] = rotation_angle_expr
                    outputs['rotation']['max_n' + tether_name2] = rotation_angle_expr2
                    outputs['local_performance']['rot_angles' + tether_name] = span
                    outputs['local_performance']['rot_angles' + tether_name2] = span2

    return outputs


def generate_constraints(options, variables, parameters, constraint_out):
    constraint_list = []

    # list all model equalities
    collection = options['model_constr']
    eq_struct, constraint_list = select_model_constraints(constraint_list, collection, constraint_out)

    # list all model inequalities
    collection = options['model_bounds']
    ineq_struct, constraint_list = select_model_constraints(constraint_list, collection, constraint_out)

    # make constraint dict
    model_constraints_dict = OrderedDict()
    model_constraints_dict['equality'] = eq_struct
    model_constraints_dict['inequality'] = ineq_struct

    # make entries if not empty
    constraint_entries = []
    if list(eq_struct.keys()):
        constraint_entries.append(cas.entry('equality', struct=eq_struct))

    if list(ineq_struct.keys()):
        constraint_entries.append(cas.entry('inequality', struct=ineq_struct))

    # generate model constraints - empty struct
    model_constraints_struct = cas.struct_symSX(constraint_entries)

    # fill in struct
    model_constraints = model_constraints_struct(cas.vertcat(*constraint_list))

    # constraints function options
    if options['jit_code_gen']['include']:
        opts = {'jit': True, 'compiler': options['jit_code_gen']['compiler']}
    else:
        opts = {}

    # create function
    model_constraints_fun = cas.Function('model_constraints_fun', [variables, parameters], [model_constraints.cat],
                                         opts)

    return model_constraints_struct, model_constraints_fun, model_constraints_dict


def select_model_constraints(constraint_list, collection, constraint_out):
    constraint_dict = OrderedDict()

    for constr_type in list(constraint_out.keys()):
        if constr_type in list(collection.keys()) and collection[constr_type]['include']:

            for name in struct_op.subkeys(constraint_out, constr_type):
                constraint_list.append(constraint_out[constr_type, name])

            constraint_dict[constr_type] = cas.struct_symSX(
                [cas.entry(name, shape=constraint_out[constr_type, name].size())
                 for name in struct_op.subkeys(constraint_out, constr_type)])

    constraint_struct = cas.struct_symSX([cas.entry(constr_type, struct=constraint_dict[constr_type])
                                          for constr_type in list(constraint_dict.keys())])

    return constraint_struct, constraint_list


def get_tether_segment_properties(options, architecture, variables_si, parameters, upper_node):
    kite_nodes = architecture.kite_nodes

    xd = variables_si['xd']
    theta = variables_si['theta']
    scaling = options['scaling']

    if upper_node == 1:
        vars_containing_length = xd
        vars_sym = 'xd'
        length_sym = 'l_t'
        diam_sym = 'diam_t'

    elif upper_node in kite_nodes:
        vars_containing_length = theta
        vars_sym = 'theta'
        length_sym = 'l_s'
        diam_sym = 'diam_s'

    else:
        vars_containing_length = theta
        vars_sym = 'theta'
        length_sym = 'l_i'
        diam_sym = 'diam_t'

    seg_length = vars_containing_length[length_sym]
    scaling_length = scaling[vars_sym][length_sym]

    seg_diam = theta[diam_sym]
    max_diam = options['system_bounds']['theta'][diam_sym][1]
    length_scaling = scaling[vars_sym][length_sym]
    scaling_diam = scaling['theta'][diam_sym]

    cross_section_area = np.pi * (seg_diam / 2.) ** 2.
    max_area = np.pi * (max_diam / 2.) ** 2.
    scaling_area = np.pi * (scaling_diam / 2.) ** 2.

    density = parameters['theta0', 'tether', 'rho']
    seg_mass = cross_section_area * density * seg_length
    scaling_mass = scaling_area * parameters['theta0', 'tether', 'rho'] * length_scaling

    props = {}
    props['seg_length'] = seg_length
    props['scaling_length'] = scaling_length

    props['seg_diam'] = seg_diam
    props['max_diam'] = max_diam
    props['scaling_diam'] = scaling_diam

    props['cross_section_area'] = cross_section_area
    props['max_area'] = max_area
    props['scaling_area'] = scaling_area

    props['seg_mass'] = seg_mass
    props['scaling_mass'] = scaling_mass

    return props
