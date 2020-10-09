
import casadi.tools as cas

import awebox.mdl.lagr_dyn_dir.tether as tether_comp
import awebox.mdl.lagr_dyn_dir.tools as tools

import awebox.tools.vector_operations as vect_op
import awebox.tools.struct_operations as struct_op
import awebox.tools.print_operations as print_op


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
        gdot.append(tools.time_derivative(g[-1], cas.vertcat(xgc.cat, var['xd', 'l_t']), cas.vertcat(xgcdot.cat, var['xd', 'dl_t']), None))

        if int(options['kite_dof']) == 6:
            for k in kite_nodes:
                kparent = parent_map[k]
                gdot[-1] += 2 * cas.mtimes(
                    vect_op.jacobian_dcm(g[-1], xd_si, var, k, kparent),
                    var['xd', 'omega{}{}'.format(k, kparent)]
                )

        # second-order derivative
        gddot.append(tools.time_derivative(gdot[-1], cas.vertcat(xgc.cat, var['xd', 'l_t']), cas.vertcat(xgcdot.cat, var['xd', 'dl_t']), cas.vertcat(xgcddot.cat, ddl_t_scaled)))

        if int(options['kite_dof']) == 6:
            for kite in kite_nodes:
                kparent = parent_map[kite]

                # add time derivative due to angular velocity
                gddot[-1] += 2 * cas.mtimes(
                    vect_op.jacobian_dcm(gdot[-1], xd_si, var, kite, kparent),
                    var['xd', 'omega{}{}'.format(kite, kparent)]
                )

                # add time derivative due to angular acceleration
                gddot[-1] += 2 * cas.mtimes(
                    vect_op.jacobian_dcm(g[-1], xd_si, var, kite, kparent),
                    var['xddot', 'domega{}{}'.format(kite, kparent)]
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
                gdot.append(tools.time_derivative(g[-1], cas.vertcat(xgc.cat, var['xd', 'l_t']),
                                                  cas.vertcat(xgcdot.cat, var['xd', 'dl_t']), None))

                if int(options['kite_dof']) == 6:
                    for kite in kite_children:
                        kparent = parent_map[kite]
                        gdot[-1] += 2 * cas.mtimes(
                            vect_op.jacobian_dcm(g[-1], xd_si, var, kite, kparent),
                            var['xd', 'omega{}{}'.format(kite, kparent)]
                        )

                # second-order derivative
                gddot.append(tools.time_derivative(gdot[-1], cas.vertcat(xgc.cat, var['xd', 'l_t']),
                                                   cas.vertcat(xgcdot.cat, var['xd', 'dl_t']),
                                                   cas.vertcat(xgcddot.cat, ddl_t_scaled)))

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
                orthonormality = cas.mtimes(r.T, r) - cas.DM_eye(3)
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
        seg_props = tether_comp.get_tether_segment_properties(options, architecture, variables, parameters, upper_node=n)
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

