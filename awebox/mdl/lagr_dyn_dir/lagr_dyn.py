import casadi.tools as cas

import awebox.tools.struct_operations as struct_op
import awebox.tools.print_operations as print_op
import awebox.tools.vector_operations as vect_op
import awebox.tools.constraint_operations as cstr_op

import awebox.mdl.lagr_dyn_dir.tools as tools

import awebox.mdl.lagr_dyn_dir.holonomics as holonomic_comp
import awebox.mdl.lagr_dyn_dir.mass as mass_comp
import awebox.mdl.lagr_dyn_dir.energy as energy_comp
import awebox.mdl.lagr_dyn_dir.forces as forces_comp

import numpy as np
import awebox.mdl.mdl_constraint as mdl_constraint

def get_dynamics(options, atmos, wind, architecture, system_variables, system_gc, parameters, outputs):

    parent_map = architecture.parent_map
    number_of_nodes = architecture.number_of_nodes

    # generalized coordinates, velocities and accelerations
    generalized_coordinates = {}
    generalized_coordinates['scaled'] = generate_generalized_coordinates(system_variables['scaled'], system_gc)
    generalized_coordinates['SI'] = generate_generalized_coordinates(system_variables['SI'], system_gc)

    # --------------------------------
    # tether and ground station masses
    # --------------------------------
    outputs = mass_comp.generate_mass_outputs(options, system_variables['SI'], outputs, parameters, architecture)

    # --------------------------------
    # lagrangian
    # --------------------------------

    outputs = energy_comp.energy_outputs(options, parameters, outputs, system_variables['SI'], architecture)
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

    cstr_list = mdl_constraint.MdlConstraintList()

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
        lagrangian_momentum_correction = momentum_correction(options, generalized_coordinates, system_variables,
                                                         outputs, architecture)

    # rhs of lagrange equations
    lagrangian_rhs_translation = cas.vertcat(
        *[f_nodes['f' + str(n) + str(parent_map[n])] for n in range(1, number_of_nodes)]) + \
                                 lagrangian_momentum_correction
    lagrangian_rhs_constraints = np.zeros(g.shape)

    # scaling
    holonomic_scaling = holonomic_comp.generate_holonomic_scaling(options, architecture, system_variables['SI'], parameters)
    node_masses_scaling = mass_comp.generate_m_nodes_scaling(options, system_variables['SI'], outputs, parameters, architecture)
    forces_scaling = options['scaling']['xl']['f_aero'] * (node_masses_scaling / parameters['theta0', 'geometry', 'm_k'])

    dynamics_translation = (lagrangian_lhs_translation - lagrangian_rhs_translation) / forces_scaling
    dynamics_translation_cstr = cstr_op.Constraint(expr=dynamics_translation,
                                                             cstr_type='eq',
                                                             name='dynamics_translation')
    cstr_list.append(dynamics_translation_cstr)


    # --------------------------------
    # rotational dynamics
    # --------------------------------

    kite_has_6dof = (int(options['kite_dof']) == 6)
    if kite_has_6dof:
        rotation_dynamics_cstr, outputs = generate_rotational_dynamics(options, system_variables, f_nodes, parameters, outputs, architecture)
        cstr_list.append(rotation_dynamics_cstr)

    # --------------------------------
    # trivial kinematics
    # --------------------------------

    for name in system_variables['SI']['xddot'].keys():

        name_in_xd = name in system_variables['SI']['xd'].keys()
        name_in_u = name in system_variables['SI']['u'].keys()

        if name_in_xd:
            trivial_dyn = cas.vertcat(*[system_variables['scaled']['xddot', name] - system_variables['scaled']['xd', name]])
        elif name_in_u:
            trivial_dyn = cas.vertcat(*[system_variables['scaled']['xddot', name] - system_variables['scaled']['u', name]])

        if name_in_xd or name_in_u:
            trivial_dyn_cstr = cstr_op.Constraint(expr=trivial_dyn,
                                                cstr_type='eq',
                                                name='trivial_' + name)
            cstr_list.append(trivial_dyn_cstr)

    # ---------------------------
    # holonomic constraints
    # ---------------------------
    dynamics_constraints = (lagrangian_lhs_constraints - lagrangian_rhs_constraints) / holonomic_scaling
    dynamics_constraint_cstr = cstr_op.Constraint(expr=dynamics_constraints,
                                                cstr_type='eq',
                                                name='dynamics_constraint')
    cstr_list.append(dynamics_constraint_cstr)

    return cstr_list, outputs



def momentum_correction(options, generalized_coordinates, system_variables, outputs, architecture):
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
            mass = outputs['masses']['m_tether{}'.format(node)]
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

    cstr_list = mdl_constraint.MdlConstraintList()

    for kite in kite_nodes:
        parent = parent_map[kite]
        moment = f_nodes['m' + str(kite) + str(parent)]

        rlocal = cas.reshape(xd['r' + str(kite) + str(parent)], (3, 3))
        drlocal = cas.reshape(xddot['dr' + str(kite) + str(parent)], (3, 3))

        omega = xd['omega' + str(kite) + str(parent)]
        omega_skew = vect_op.skew(omega)
        domega = xddot['domega' + str(kite) + str(parent)]

        tether_moment = outputs['tether_moments']['n{}{}'.format(kite, parent)]

        # moment = J dot(omega) + omega x (J omega) + [tether moment which is zero if holonomic constraints do not depend on omega]
        J_dot_omega = cas.mtimes(j_inertia, domega)
        omega_cross_J_omega = vect_op.cross(omega, cas.mtimes(j_inertia, omega))
        omega_derivative = moment - (J_dot_omega + omega_cross_J_omega + tether_moment)
        m_scale = options['scaling']['xl']['m_aero']
        rotational_2nd_law = omega_derivative / m_scale

        rotation_dynamics_cstr = cstr_op.Constraint(expr=rotational_2nd_law,
                                                    name='rotation_dynamics' + str(kite),
                                                    cstr_type='eq')
        cstr_list.append(rotation_dynamics_cstr)

        # Rdot = R omega_skew -> R ( kappa/2 (I - R.T R) + omega_skew )
        baumgarte = parameters['theta0', 'kappa_r']
        orthonormality = baumgarte / 2. * (cas.DM_eye(3) - cas.mtimes(rlocal.T, rlocal))
        ref_frame_deriv_matrix = drlocal - (cas.mtimes(rlocal, orthonormality + omega_skew))
        ref_frame_derivative = cas.reshape(ref_frame_deriv_matrix, (9, 1))

        ortho_cstr = cstr_op.Constraint(expr=ref_frame_derivative,
                                        name='ref_frame_deriv' + str(kite),
                                        cstr_type='eq')
        cstr_list.append(ortho_cstr)

    return cstr_list, outputs


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


