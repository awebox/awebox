import casadi.tools as cas

import awebox.tools.struct_operations as struct_op
import awebox.tools.print_operations as print_op
import awebox.tools.vector_operations as vect_op

import awebox.mdl.lagr_dyn_dir.tools as tools

import awebox.mdl.lagr_dyn_dir.holonomics as holonomic_comp
import awebox.mdl.lagr_dyn_dir.mass as mass_comp
import awebox.mdl.lagr_dyn_dir.energy as energy_comp
import awebox.mdl.lagr_dyn_dir.forces as forces_comp

import numpy as np

import pdb

def get_dynamics(options, atmos, wind, architecture, system_variables, variables_dict, system_gc, parameters, outputs):

    parent_map = architecture.parent_map
    number_of_nodes = architecture.number_of_nodes

    # generalized coordinates, velocities and accelerations
    generalized_coordinates = {}
    generalized_coordinates['scaled'] = generate_generalized_coordinates(system_variables['scaled'], system_gc)
    generalized_coordinates['SI'] = generate_generalized_coordinates(system_variables['SI'], system_gc)

    # -------------------------------
    # mass distribution in the system
    # -------------------------------
    node_masses_si, outputs = mass_comp.generate_m_nodes(options, system_variables['SI'], outputs, parameters, architecture)

    # --------------------------------
    # lagrangian
    # --------------------------------

    outputs = energy_comp.energy_outputs(options, parameters, outputs, node_masses_si, system_variables['SI'], architecture)
    e_kinetic = sum(outputs['e_kinetic'][nodes] for nodes in list(outputs['e_kinetic'].keys()))
    e_potential = sum(outputs['e_potential'][nodes] for nodes in list(outputs['e_potential'].keys()))

    holonomic_constraints, outputs, g, gdot, gddot = holonomic_comp.generate_holonomic_constraints(
        architecture,
        outputs,
        system_variables,
        parameters,
        options)

    # lagrangian function
    lag = e_kinetic - e_potential - holonomic_constraints

    # --------------------------------
    # generalized forces in the system
    # --------------------------------

    f_nodes, outputs = forces_comp.generate_f_nodes(options, atmos, wind, system_variables['SI'], parameters, outputs,
                                                    architecture)
    outputs = forces_comp.generate_tether_moments(options, system_variables['SI'], system_variables['scaled'], holonomic_constraints, outputs,
                                                  architecture)

    # --------------------------------
    # translational dynamics
    # --------------------------------

    # lhs of lagrange equations
    dlagr_dqdot = cas.jacobian(lag, generalized_coordinates['scaled']['xgcdot'].cat).T
    dlagr_dqdot_dt = tools.time_derivative(dlagr_dqdot, system_variables, architecture)

    dlagr_dq = cas.jacobian(lag, generalized_coordinates['scaled']['xgc'].cat).T

    lagrangian_lhs_translation = dlagr_dqdot_dt - dlagr_dq

    lagrangian_lhs_constraints = holonomic_comp.get_constraint_lhs(g, gdot, gddot, parameters)

    # lagrangian momentum correction
    if options['tether']['use_wound_tether']:
        lagrangian_momentum_correction = 0.
    else:
        lagrangian_momentum_correction = momentum_correction(options, generalized_coordinates, system_variables, node_masses_si,
                                                         outputs, architecture)

    # rhs of lagrange equations
    lagrangian_rhs_translation = cas.vertcat(
        *[f_nodes['f' + str(n) + str(parent_map[n])] for n in range(1, number_of_nodes)]) + \
                                 lagrangian_momentum_correction
    lagrangian_rhs_constraints = np.zeros(g.shape)

    # scaling
    holonomic_scaling = holonomic_comp.generate_holonomic_scaling(options, architecture, system_variables['SI'], parameters)
    node_masses_scaling = mass_comp.generate_m_nodes_scaling(options, system_variables['SI'], outputs, parameters, architecture)
    forces_scaling = node_masses_scaling * options['scaling']['other']['g'] * options['model_bounds']['acceleration']['acc_max']

    dynamics_translation = (lagrangian_lhs_translation - lagrangian_rhs_translation) / forces_scaling
    dynamics_constraints = (lagrangian_lhs_constraints - lagrangian_rhs_constraints) / holonomic_scaling

    # --------------------------------
    # rotational dynamics
    # --------------------------------

    rotation_dynamics, outputs = generate_rotational_dynamics(options, system_variables, f_nodes, parameters, outputs, architecture)

    # --------------------------------
    # trivial kinematics
    # --------------------------------


    trivial_dynamics_states = cas.vertcat(
        *[system_variables['scaled']['xddot', name] - system_variables['scaled']['xd', name] for name in
          list(system_variables['SI']['xddot'].keys()) if name in list(system_variables['SI']['xd'].keys())])
    trivial_dynamics_controls = cas.vertcat(
        *[system_variables['scaled']['xddot', name] - system_variables['scaled']['u', name] for name in
          list(system_variables['SI']['xddot'].keys()) if name in list(system_variables['SI']['u'].keys())])

    # --------------------------------
    # concatenation
    # --------------------------------

    # put the trivial constraints first, so they can be scaled according
    # to the radau coefficients, when using radau collocation (and no paralellization).
    lagr_dynamics = [
        trivial_dynamics_states,
        trivial_dynamics_controls,
        dynamics_translation,
        rotation_dynamics,
        dynamics_constraints
    ]

    return lagr_dynamics, outputs



def momentum_correction(options, generalized_coordinates, system_variables, node_masses, outputs, architecture):
    """Compute momentum correction for translational lagrangian dynamics of an open system.
    Here the system is "open" because the main tether mass is changing in time. During reel-out,
    momentum is injected in the system, and during reel-in, momentum is extracted.
    It is assumed that the tether mass is concentrated at the main tether node.

    See "Lagrangian Dynamics for Open Systems", R. L. Greene and J. J. Matese 1981, Eur. J. Phys. 2, 103.

    @return: lagrangian_momentum_correction - correction term that can directly be added to rhs of transl_dyn
    """

    # initialize
    xgcdot = generalized_coordinates['scaled']['xgcdot'].cat
    lagrangian_momentum_correction = cas.DM.zeros(xgcdot.shape)

    use_wound_tether = options['tether']['use_wound_tether']
    if not use_wound_tether:

        for node in range(1, architecture.number_of_nodes):
            label = str(node) + str(architecture.parent_map[node])
            mass = node_masses['m' + label]
            mass_flow = tools.time_derivative(mass, system_variables, architecture)

            # velocity of the mass particles leaving the system
            velocity = system_variables['SI']['xd']['dq' + label]
            # see formula in reference
            lagrangian_momentum_correction += mass_flow * cas.mtimes(velocity.T, cas.jacobian(velocity,
                                                                                              xgcdot)).T

    return lagrangian_momentum_correction


def generate_rotational_dynamics(options, variables, f_nodes, parameters, outputs, architecture):
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

            tether_moment = outputs['tether_moments']['n{}{}'.format(n, parent)]

            # moment = J dot(omega) + omega x (J omega) + [tether moment which is zero if holonomic constraints do not depend on omega]
            J_dot_omega = cas.mtimes(j_inertia, domega)
            omega_cross_J_omega = vect_op.cross(omega, cas.mtimes(j_inertia, omega))
            omega_derivative = J_dot_omega + omega_cross_J_omega - moment + tether_moment
            rotational_2nd_law = omega_derivative / vect_op.norm(cas.diag(j_inertia))

            # Rdot = R omega_skew -> R ( kappa/2 (I - R.T R) + omega_skew )
            baumgarte = parameters['theta0', 'kappa_r']
            orthonormality = baumgarte / 2. * (cas.DM_eye(3) - cas.mtimes(rlocal.T, rlocal))
            ref_frame_deriv_matrix = drlocal - cas.mtimes(rlocal, orthonormality + omega_skew)
            ref_frame_derivative = cas.reshape(ref_frame_deriv_matrix, (9, 1))

            # concatenate
            rotation_dynamics = cas.vertcat(rotation_dynamics, rotational_2nd_law, ref_frame_derivative)

    return rotation_dynamics, outputs


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
        # generalized_coordinates['xgcddot'] = cas.struct_SX(
        #     [cas.entry('dd' + name, expr=system_variables['xddot', 'dd' + name])
        #      for name in system_gc])
    else:
        generalized_coordinates = {}
        generalized_coordinates['xgc'] = cas.struct_SX(
            [cas.entry(name, expr=system_variables['xd'][name]) for name in system_gc])
        generalized_coordinates['xgcdot'] = cas.struct_SX(
            [cas.entry('d' + name, expr=system_variables['xd']['d' + name])
             for name in system_gc])
        # generalized_coordinates['xgcddot'] = cas.struct_SX(
        #     [cas.entry('dd' + name, expr=system_variables['xddot']['dd' + name])
        #      for name in system_gc])

    return generalized_coordinates


