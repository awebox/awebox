import collections
import pdb

import casadi.tools as cas

import awebox.mdl.lagr_dyn_dir.tools as tools

import awebox.tools.vector_operations as vect_op
import awebox.tools.struct_operations as struct_op
import awebox.tools.print_operations as print_op

import awebox.mdl.aero.tether_dir.tether_aero as tether_aero


from awebox.logger.logger import Logger as awelogger


def generate_holonomic_constraints(architecture, outputs, system_variables, parameters, options, scaling):
    number_of_nodes = architecture.number_of_nodes
    parent_map = architecture.parent_map
    kite_nodes = architecture.kite_nodes

    # extract necessary SI variables
    x_si = system_variables['SI']['x']
    z_si = system_variables['SI']['z']

    # scaled variables struct
    var_scaled = system_variables['scaled']

    if 'invariants' not in list(outputs.keys()):
        outputs['invariants'] = {}

    # build constraints with si variables
    g_dict = get_tether_length_constraint(options, system_variables['SI'], parameters, architecture)
    outputs['invariants'].update(g_dict)

    g = []
    gdot = []
    gddot = []
    holonomic_constraints = 0.0

    for node in range(1, number_of_nodes):
        parent = parent_map[node]

        # chain rule, based on scaled variables
        g_local = g_dict['c' + str(node) + str(parent)]
        g.append(g_local)

        # first-order derivative
        dg_local = tools.time_derivative(g_local, system_variables['scaled'], architecture, scaling)
        gdot.append(dg_local)

        # second-order derivative
        ddg_local = tools.time_derivative(dg_local, system_variables['scaled'], architecture, scaling)
        gddot.append(ddg_local)

        # outputs['invariants']['c' + str(node) + str(parent)] = g[-1]
        outputs['invariants']['dc' + str(node) + str(parent)] = dg_local
        outputs['invariants']['ddc' + str(node) + str(parent)] = ddg_local
        holonomic_constraints += z_si['lambda{}{}'.format(node, parent)] * g_local

        if node in kite_nodes:
            if 'r' + str(node) + str(parent) in list(x_si.keys()):
                r = cas.reshape(var_scaled['x', 'r' + str(node) + str(parent)], (3, 3))
                orthonormality = cas.mtimes(r.T, r) - cas.DM_eye(3)
                orthonormality = cas.reshape(orthonormality, (9, 1))

                outputs['invariants']['orthonormality' + str(node) + str(parent)] = orthonormality

                dr_dt = system_variables['SI']['xdot']['dr' + str(node) + str(parent)]
                dr_dt = cas.reshape(dr_dt, (3, 3))
                omega = system_variables['SI']['x']['omega' + str(node) + str(parent)]
                omega_skew = vect_op.skew(omega)
                dr = cas.mtimes(r, omega_skew)
                rot_kinematics = dr_dt - dr
                rot_kinematics = cas.reshape(rot_kinematics, (9, 1))
                outputs['invariants']['rot_kinematics10'] = rot_kinematics

    # add cross-tethers
    if options['cross_tether'] and len(kite_nodes) > 1:

        g_dict = get_cross_tether_length_constraint(options, system_variables['SI'], parameters, architecture)
        outputs['invariants'].update(g_dict)

        for ldx in architecture.layer_nodes:
            kite_children = architecture.kites_map[ldx]

            # dual kite system (per layer) only has one tether
            if len(kite_children) == 2:
                no_tethers = 1
            else:
                no_tethers = len(kite_children)

            # add cross-tether constraints
            for k in range(no_tethers):

                # set-up relevant node numbers
                n01 = '{}{}'.format(kite_children[k], kite_children[(k + 1) % len(kite_children)])

                length_constraint = g_dict['c{}'.format(n01)]

                # append constraint
                g_local = length_constraint
                g.append(length_constraint)

                dg_local = tools.time_derivative(g_local, system_variables['scaled'], architecture, scaling)
                gdot.append(dg_local)

                ddg_local = tools.time_derivative(dg_local, system_variables['scaled'], architecture, scaling)
                gddot.append(ddg_local)

                # save invariants to outputs
                outputs['invariants']['dc{}'.format(n01)] = dg_local
                outputs['invariants']['ddc{}'.format(n01)] = ddg_local

                # add to holonomic constraints
                holonomic_constraints += z_si['lambda{}'.format(n01)] * g_local

    g_cat = cas.vertcat(*g)
    gdot_cat = cas.vertcat(*gdot)
    gddot_cat = cas.vertcat(*gddot)

    return holonomic_constraints, outputs, g_cat, gdot_cat, gddot_cat

def get_cross_tether_length_constraint(options, vars_si, parameters, architecture):

    x_si = vars_si['x']
    theta_si = vars_si['theta']

    g_dict = {}

    kite_nodes = architecture.kite_nodes
    parent_map = architecture.parent_map

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
                    first_node = x_si['q{}'.format(n0)]
                    second_node = x_si['q{}'.format(n1)]

                # stick or wing-tip attachment
                else:

                    # only implemented for 6DOF
                    if int(options['kite_dof']) == 6:

                        # rotation matrices of relevant kites
                        dcm_first = cas.reshape(x_si['r{}'.format(n0)], (3, 3))
                        dcm_second = cas.reshape(x_si['r{}'.format(n1)], (3, 3))

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
                        first_node = x_si['q{}'.format(n0)] + cas.mtimes(dcm_first, r_tether)
                        second_node = x_si['q{}'.format(n1)] + cas.mtimes(dcm_second, r_tether)

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

                g_dict['c{}'.format(n01)] = length_constraint

    return g_dict

def get_tether_length_constraint(options, vars_si, parameters, architecture):

    x_si = vars_si['x']
    theta_si = vars_si['theta']

    g_dict = {}

    number_of_nodes = architecture.number_of_nodes
    parent_map = architecture.parent_map
    kite_nodes = architecture.kite_nodes

    com_attachment = (options['tether']['attachment'] == 'com')
    stick_attachment = (options['tether']['attachment'] == 'stick')
    kite_has_6dof = (int(options['kite_dof']) == 6)

    if not (com_attachment or stick_attachment):
        message = 'Unknown tether attachment option: {}'.format(options['tether']['attachment'])
        print_op.log_and_raise_error(message)

    for node in range(1, number_of_nodes):

        parent = parent_map[node]
        q_si = x_si['q' + str(node) + str(parent)]

        node_is_a_kite = (node in kite_nodes)
        has_extended_arm = node_is_a_kite and stick_attachment

        if has_extended_arm and not kite_has_6dof:
            message = 'Stick tether attachment option not implemented for 3DOF kites'
            print_op.log_and_raise_error(message)

        if has_extended_arm:
            dcm = cas.reshape(x_si['r{}{}'.format(node, parent)], (3, 3))
            current_node = q_si + cas.mtimes(dcm, parameters['theta0', 'geometry', 'r_tether'])
        else:
            current_node = q_si

        if node == 1:
            previous_node = cas.DM.zeros((3, 1))
            if 'l_t' in x_si.keys():
                segment_length = x_si['l_t']
            else:
                segment_length = theta_si['l_t']
        elif node in kite_nodes:
            grandparent = parent_map[parent]
            previous_node = x_si['q' + str(parent) + str(grandparent)]
            segment_length = theta_si['l_s']
        else:
            grandparent = parent_map[parent]
            previous_node = x_si['q' + str(parent) + str(grandparent)]
            segment_length = theta_si['l_i']

        # holonomic constraint
        seg_vector = (current_node - previous_node)
        length_constraint = 0.5 * (cas.mtimes(seg_vector.T, seg_vector) - segment_length ** 2.0)

        g_dict['c' + str(node) + str(parent)] = length_constraint

    return g_dict


def get_constraint_lhs(g, gdot, gddot, parameters):
    # todo: update baumgarte for dddl_t control.

    baumgarte = parameters['theta0', 'tether', 'kappa']
    lagrangian_lhs_constraints = gddot + 2. * baumgarte * gdot + baumgarte ** 2. * g

    return lagrangian_lhs_constraints


def generate_holonomic_scaling(options, architecture, scaling, variables, parameters):
    holonomic_scaling = []

    for n in range(1, architecture.number_of_nodes):
        seg_props = tether_aero.get_tether_segment_properties(options, architecture, scaling, variables, parameters, upper_node=n)

        scaling_length = seg_props['scaling_length']
        scaling_speed = seg_props['scaling_speed']
        scaling_acc = seg_props['scaling_acc']

        g_loc = scaling_length**2.
        gdot_loc = 2. * scaling_length * scaling_speed
        gddot_loc = 2. * scaling_length * scaling_acc + 2. * scaling_speed**2.

        # notice that if the scaling_length is large, loc_scaling easily ends up as a *massive* number
        loc_scaling = get_constraint_lhs(g_loc, gdot_loc, gddot_loc, parameters)
        holonomic_scaling = cas.vertcat(holonomic_scaling, loc_scaling)

    if architecture.number_of_kites > 1 and options['cross_tether']:
        dict_cross_tether = {}
        for theta_subkey in struct_op.subkeys(scaling, 'theta'):
            if ('l_c' in theta_subkey) and (theta_subkey[:3] == 'l_c'):
                dict_cross_tether[theta_subkey[3:]] = theta_subkey

        for cross_index in dict(sorted(dict_cross_tether.items())):
            local_scaling = scaling['theta', dict_cross_tether[cross_index]]**2
            holonomic_scaling = cas.vertcat(holonomic_scaling, local_scaling)

    return holonomic_scaling

