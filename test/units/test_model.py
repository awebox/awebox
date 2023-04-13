#!/usr/bin/python3
"""Test to check model functionality

@author: Jochem De Schutter,
edit: rachel leuthold, alu-fr 2020
"""
import copy
import pdb

import awebox as awe
import logging
import casadi as cas
import awebox.mdl.architecture as archi
import numpy as np
import awebox.mdl.system as system
import awebox.tools.constraint_operations as cstr_op
import awebox.tools.struct_operations as struct_op
import awebox.tools.vector_operations as vect_op
import awebox.opts.kite_data.ampyx_data as ampyx_data
import awebox.mdl.lagr_dyn_dir.tools as lagr_dyn_tools


logging.basicConfig(filemode='w', format='%(levelname)s:    %(message)s', level=logging.WARNING)


def test_architecture():
    """Test architecture construction routines
    """

    test_archi_dict = generate_architecture_dict()

    for archi_name in list(test_archi_dict.keys()):

        architecture = test_archi_dict[archi_name]
        test_archi = archi.Architecture(architecture['parent_map'])

        assert test_archi.kite_nodes      == architecture['kite_nodes']     , 'kite_nodes of '+archi_name
        assert test_archi.layer_nodes     == architecture['layer_nodes']    , 'layer nodes of '+archi_name
        assert test_archi.layers          == architecture['layers']         , 'layers of '+archi_name
        assert test_archi.siblings_map    == architecture['siblings_map']   , 'siblings_map of '+archi_name
        assert test_archi.number_of_nodes == architecture['number_of_nodes'], 'number_of_nodes of '+archi_name
        assert test_archi.children_map    == architecture['children_map']   , 'children map of '+archi_name
        assert test_archi.kites_map       == architecture['kites_map']   , 'kite-children map of '+archi_name

    return None


def generate_architecture_dict():
    """Generate dict containing tree-structured architectures with built
    attributes to be tested

    @return test_archi_dict  dict containing the test architectures
    """

    test_archi_dict = {}

    # single kites
    archi_dict = {'parent_map': {1:0},
                  'kite_nodes': [1],
                  'layer_nodes': [0],
                  'layers': 1,
                  'siblings_map': {1:[1]},
                  'number_of_nodes': 2,
                  'children_map': {0:[1]},
                  'kites_map': {0:[1]}}

    test_archi_dict['single_kite'] = archi_dict

    # dual kites
    archi_dict = {'parent_map': {1:0, 2:1, 3:1},
                  'kite_nodes': [2,3],
                  'layer_nodes': [1],
                  'layers': 1,
                  'siblings_map': {2:[2,3],3:[2,3]},
                  'number_of_nodes': 4,
                  'children_map': {0:[1], 1:[2,3]},
                  'kites_map': {0:[],1:[2,3]}}

    test_archi_dict['dual_kites'] = archi_dict

    # triple kites
    archi_dict = {'parent_map': {1:0, 2:1, 3:1, 4:1},
                  'kite_nodes': [2,3,4],
                  'layer_nodes': [1],
                  'layers': 1,
                  'siblings_map': {2:[2,3,4],3:[2,3,4],4:[2,3,4]},
                  'number_of_nodes': 5,
                  'children_map': {0:[1],1:[2,3,4]},
                  'kites_map': {0:[],1:[2,3,4]}}

    test_archi_dict['triple_kites'] = archi_dict

    # triple-dual kites
    archi_dict = {'parent_map': {1:0, 2:1, 3:1, 4:1, 5:1, 6:5, 7:5},
                  'kite_nodes': [2,3,4,6,7],
                  'layer_nodes': [1,5],
                  'layers': 2,
                  'siblings_map': {2:[2,3,4],3:[2,3,4],4:[2,3,4],6:[6,7],7:[6,7]},
                  'number_of_nodes': 8,
                  'children_map': {0:[1], 1:[2,3,4,5], 5:[6,7]},
                  'kites_map': {0:[],1:[2,3,4], 5:[6,7]}}


    test_archi_dict['triple_dual_kites'] = archi_dict

    return test_archi_dict

def test_drag_mode_model():
    """ Test drag mode construction routines
    """


    # single kite with point-mass model
    options = {}
    options['user_options.system_model.architecture'] = {1:0, 2:1, 3:1}
    options['user_options.system_model.kite_dof'] = 3
    options['user_options.kite_standard'] = awe.ampyx_data.data_dict()
    options['user_options.trajectory.type'] = 'power_cycle'
    options['user_options.trajectory.system_type'] = 'drag_mode'
    options['model.tether.use_wound_tether'] = False

    # don't include induction effects, use trivial tether drag
    options['user_options.induction_model'] = 'not_in_use'
    options['user_options.tether_drag_model'] = 'split'

    # build model
    trial_options = awe.Options()
    trial_options.fill_in_seed(options)
    architecture = archi.Architecture(options['user_options.system_model.architecture'])
    trial_options.build(architecture)
    model = awe.mdl.model.Model()
    model.build(trial_options['model'], architecture)

    # extract model info
    states = model.variables_dict['x']
    controls = model.variables_dict['u']
    outputs = model.outputs_dict

    # test states and controls
    assert('kappa10' not in list(states.keys()))
    assert('kappa21' in     list(states.keys()))
    assert('kappa31' in     list(states.keys()))

    assert('dkappa10' not in list(controls.keys()))
    assert('dkappa21' in     list(controls.keys()))
    assert('dkappa31' in     list(controls.keys()))

    # test outputs
    aero = outputs['aerodynamics']
    assert('f_gen1' not in list(aero.keys()))
    assert('f_gen2' in     list(aero.keys()))
    assert('f_gen3' in     list(aero.keys()))

    # test dynamics
    dynamics = model.dynamics(model.variables, model.parameters)
    assert (cas.jacobian(dynamics, model.variables['x', 'kappa21']).nnz() != 0)
    assert (cas.jacobian(dynamics, model.variables['x', 'kappa31']).nnz() != 0)
    assert (cas.jacobian(dynamics, model.variables['u', 'dkappa31']).nnz() != 0)
    assert (cas.jacobian(dynamics, model.variables['u', 'dkappa31']).nnz() != 0)
    assert (cas.jacobian(dynamics, model.variables['x', 'kappa21']).nnz() != 0)
    assert (cas.jacobian(dynamics, model.variables['x', 'kappa31']).nnz() != 0)
    assert (cas.jacobian(dynamics, model.variables['u', 'dkappa31']).nnz() != 0)
    assert (cas.jacobian(dynamics, model.variables['u', 'dkappa31']).nnz() != 0)

    # test power expression
    integral_outputs = model.integral_outputs_fun(model.variables, model.parameters)
    assert (cas.jacobian(integral_outputs, model.variables['x', 'kappa21']).nnz() != 0)
    assert (cas.jacobian(integral_outputs, model.variables['x', 'kappa31']).nnz() != 0)
    assert (cas.jacobian(integral_outputs, model.variables['theta', 'l_t']).nnz() == 0)
    assert (cas.jacobian(integral_outputs, model.variables['z', 'lambda10']).nnz() == 0)

    # test variable bounds
    lb_si = cas.DM(trial_options['model']['system_bounds']['u']['dkappa'][0])
    ub_si = cas.DM(trial_options['model']['system_bounds']['u']['dkappa'][1])

    lb21 = struct_op.var_si_to_scaled('u', 'dkappa21', lb_si, model.scaling)
    ub21 = struct_op.var_si_to_scaled('u', 'dkappa21', ub_si, model.scaling)

    lb31 = struct_op.var_si_to_scaled('u', 'dkappa31', lb_si, model.scaling)
    ub31 = struct_op.var_si_to_scaled('u', 'dkappa31', ub_si, model.scaling)

    assert(model.variable_bounds['u']['dkappa21']['lb'] == lb21)
    assert(model.variable_bounds['u']['dkappa31']['lb'] == lb31)
    assert(model.variable_bounds['u']['dkappa21']['ub'] == ub21)
    assert(model.variable_bounds['u']['dkappa31']['ub'] == ub31)

    if 'dddl_t' in model.variable_bounds['u'].keys():
        assert(model.variable_bounds['u']['dddl_t']['lb'] == 0.0)
        assert(model.variable_bounds['u']['dddl_t']['ub'] == 0.0)
    elif 'ddl_t' in model.variable_bounds['u'].keys():
        assert(model.variable_bounds['u']['ddl_t']['lb'] == 0.0)
        assert(model.variable_bounds['u']['ddl_t']['ub'] == 0.0)

    # test scaling
    assert(model.scaling['x', 'kappa21'] == trial_options['model']['scaling']['x']['kappa'])
    assert(model.scaling['x', 'kappa31'] == trial_options['model']['scaling']['x']['kappa'])
    assert (model.scaling['xdot', 'dkappa21'] == trial_options['model']['scaling']['x']['kappa'])
    assert (model.scaling['xdot', 'dkappa31'] == trial_options['model']['scaling']['x']['kappa'])
    assert(model.scaling['u', 'dkappa21'] == trial_options['model']['scaling']['u']['dkappa'])
    assert(model.scaling['u', 'dkappa21'] == trial_options['model']['scaling']['u']['dkappa'])

    return None

def test_cross_tether_model():
    """ Test cross-tether construction routines
    """

    # single kite with point-mass model
    options = {}
    options['user_options.system_model.architecture'] = {1: 0, 2: 1, 3: 1}
    options['user_options.system_model.kite_dof'] = 3
    options['user_options.kite_standard'] = awe.ampyx_data.data_dict()
    options['user_options.system_model.cross_tether'] = True
    options['model.tether.use_wound_tether'] = False

    # don't include induction effects, use trivial tether drag
    options['user_options.induction_model'] = 'not_in_use'
    options['user_options.tether_drag_model'] = 'split'

    # build model
    trial_options = awe.Options()
    trial_options.fill_in_seed(options)
    architecture = archi.Architecture(trial_options['user_options']['system_model']['architecture'])
    trial_options.build(architecture)
    model = awe.mdl.model.Model()
    model.build(trial_options['model'], architecture)

    # extract model info
    algvars = model.variables_dict['z']
    theta   = model.variables_dict['theta']
    outputs = model.outputs_dict
    constraints = model.constraints_dict

    # check variables
    assert('lambda10' in list(algvars.keys()))
    assert('lambda21' in list(algvars.keys()))
    assert('lambda31' in list(algvars.keys()))
    assert('lambda23' in list(algvars.keys()))
    assert('lambda32' not in list(algvars.keys()))
    assert('l_c1' in list(theta.keys()))
    assert('diam_c1' in list(theta.keys()))

    # check constraints
    assert('c10' in outputs['invariants'].keys())
    assert('c21' in outputs['invariants'].keys())
    assert('c31' in outputs['invariants'].keys())
    assert('c23' in outputs['invariants'].keys())
    assert('c32' not in outputs['invariants'].keys())

    assert (constraints['inequality']['tether_stress10'].shape[0] == 1)
    assert (constraints['inequality']['tether_stress21'].shape[0] == 1)
    assert (constraints['inequality']['tether_stress31'].shape[0] == 1)
    assert (constraints['inequality']['tether_stress23'].shape[0] == 1)

    assert ('tether_stress00' not in constraints['inequality'].keys())
    assert ('tether_stress32' not in constraints['inequality'].keys())

    return None

def test_tether_moments():
    """ Test moment contribution due to holonomic constraints """

    # single kite with point-mass model
    options = {}
    options['user_options.system_model.architecture'] = {1:0}
    options['user_options.system_model.kite_dof'] = 6
    options['user_options.kite_standard'] = awe.ampyx_data.data_dict()
    options['user_options.induction_model'] = 'not_in_use'
    options['user_options.tether_drag_model'] = 'split'
    options['user_options.trajectory.system_type'] = 'drag_mode'
    options['model.tether.use_wound_tether'] = False
    options['model.tether.use_wound_tether'] = False
    options['model.tether.lift_tether_force'] = False
    options['model.aero.lift_aero_force'] = False

    # tether attachment
    r_tether = np.array([0.0, 0.0, -0.1])
    options['model.tether.attachment'] = 'stick'
    options['model.geometry.overwrite.r_tether'] = r_tether

    # build model
    trial_options = awe.Options()
    trial_options.fill_in_seed(options)
    architecture = archi.Architecture(trial_options['user_options']['system_model']['architecture'])
    trial_options.build(architecture)
    model = awe.mdl.model.Model()
    model.build(trial_options['model'], architecture)

    # si variables
    var_si = model.variables(0.0)

    var_si['x', 'q10'] = np.array([130.644, 24.5223, 74.2863])
    var_si['x', 'dq10'] = np.array([-16.3061, -27.0514, 37.5959])
    var_si['x', 'omega10'] = np.array([-0.0181481, 0.275884, 1.5743])
    var_si['x', 'r10'] = np.array([0.271805, 0.334641, -0.902295,
                             0.0595685, 0.929945, 0.362839,
                             0.960506, -0.15237, 0.23283])
    var_si['x', 'delta10'] = np.array([0.0693676, 0.261684, -0.258425])
    var_si['x', 'kappa10'] = -0.146057

    if '[x,ddl_t,0]' in var_si.labels():
        var_si['x', 'ddl_t'] = 9.49347e-13

    var_si['z'] = 45.024
    var_si['theta'] = np.array([0.005, 3.93805, 152.184])

    # scaled variables
    scaling = model.scaling
    var_sc = system.scale_variable(model.variables, var_si, scaling)
    parameters = model.parameters(0.0)
    parameters['theta0','geometry','r_tether'] = r_tether

    # numerical result
    outputs = model.outputs(model.outputs_fun(var_sc, parameters))
    tether_moment = outputs['tether_moments', 'n10']

    # analytic expression
    dcm = cas.reshape(var_si['x', 'r10'],(3,3))
    tether_moment_true = var_si['z','lambda10'] * cas.cross(r_tether, cas.mtimes(dcm.T, var_si['x','q10']))

    # test implementation
    msg = 'Incorrect tether moment contribution for single kite systems'
    assert np.linalg.norm(tether_moment_true-tether_moment)/np.linalg.norm(tether_moment_true) < 1e-8, msg

    return None

def test_constraint_mechanism():
    rdx = 0
    results = {}

    # can we make a MdlConstraint?
    var = cas.SX.sym('var')
    expr = cas.vertcat(var ** 2. - 2., 8. * var)
    cstr_type = 'eq'
    name = 'cstr1'
    cstr1 = cstr_op.Constraint(expr=expr, name=name, cstr_type=cstr_type)

    # is the length of that constraint as expected?
    results[rdx] = (cstr1.expr.shape == (2, 1))
    rdx += 1

    # can we make a MdlConstraintList?
    cstr_list = cstr_op.MdlConstraintList()

    # are the lengths of the eq_list and ineq_list both zero?
    results[rdx] = (len(cstr_list.eq_list) == 0) and (len(cstr_list.ineq_list) == 0)
    rdx += 1

    # can we add non-empty constraints to cstr_list?
    expr2 = var + 4.
    cstr_type2 = 'eq'
    name2 = 'cstr2'
    cstr2 = cstr_op.Constraint(expr=expr2, name=name2, cstr_type=cstr_type2)

    cstr_list.append(cstr1)
    cstr_list.append(cstr2)

    # does the list record two equality constraints and 0 inequality constraints?
    results[rdx] = (len(cstr_list.eq_list) == 2) and (len(cstr_list.ineq_list) == 0)
    rdx += 1

    # is the number of expressions in the equality constraints == 3?
    results[rdx] = (cstr_list.get_expression_list('eq').shape == (3, 1))
    rdx += 1

    # can we add an empty list to the cstr_list?
    cstr_list.append([])

    # does the list still record two equality constraints and 0 inequality constraints?
    results[rdx] = (len(cstr_list.eq_list) == 2) and (len(cstr_list.ineq_list) == 0)
    rdx += 1

    # can we make an incomplete constraint?
    expr3 = []
    cstr_type3 = 'eq'
    name3 = 'cstr3'
    cstr3 = cstr_op.Constraint(expr=expr3, name=name3, cstr_type=cstr_type3)

    # can we add the incomplete constraint to the list?
    cstr_list.append(cstr3)

    # does the list still record two equality constraints and 0 inequality constraints?
    results[rdx] = (len(cstr_list.eq_list) == 2) and (len(cstr_list.ineq_list) == 0)
    rdx += 1

    # can we make an empty list?
    cstr_list_empty = cstr_op.MdlConstraintList()

    # can we add the empty list to the existing list?
    cstr_list.append(cstr_list_empty)

    # does the list still record two equality constraints and 0 inequality constraints?
    results[rdx] = (len(cstr_list.eq_list) == 2) and (len(cstr_list.ineq_list) == 0)
    rdx += 1

    # can we make a non-empty list, and append it to the main list?
    cstr_list_nonempty = cstr_op.MdlConstraintList()
    expr4 = var + 8.
    cstr_type4 = 'ineq'
    name4 = 'cstr4'
    cstr4 = cstr_op.Constraint(expr=expr4, name=name4, cstr_type=cstr_type4)
    cstr_list_nonempty.append(cstr4)
    cstr_list.append(cstr_list_nonempty)

    # does the list now record two equality constraints and 1 inequality constraints?
    results[rdx] = (len(cstr_list.eq_list) == 2) and (len(cstr_list.ineq_list) == 1)
    rdx += 1

    # can we make a constraint with a duplicate name, and append it to the main list?
    expr5 = cas.sin(var) + 8.
    cstr_type5 = 'ineq'
    name5 = 'cstr4'
    cstr5 = cstr_op.Constraint(expr=expr5, name=name5, cstr_type=cstr_type5)
    cstr_list.append(cstr5)

    # does the list still record two equality constraints and 1 inequality constraints?
    results[rdx] = (len(cstr_list.eq_list) == 2) and (len(cstr_list.ineq_list) == 1)
    rdx += 1

    # does the list still record 3 constraints together?
    results[rdx] = (len(cstr_list.all_list) == 3)
    rdx += 1

    # get functions
    params = cas.SX.sym('params', (2, 1))
    cstr5_fun = cstr5.get_function(variables=var, parameters=params)

    # do the functions give expected results?
    var_test = 30. * np.pi / 180.
    sol_test = np.sin(var_test) + 8.
    found_test = cstr5_fun(30. * np.pi / 180, params)
    results[rdx] = (found_test == sol_test)
    rdx += 1

    ##############
    # summarize results
    ##############

    print(results)
    for res_value in results.values():
        assert (res_value)

    return None


def get_consistent_inputs_for_inextensible_pendulum_problem(mass_si, gravity_si, length_si, pendulum_angle_rad, rod_mass_si=0., rod_diameter_si=0.):

    xhat = vect_op.xhat_np()
    zhat = vect_op.zhat_np()

    dl_t_initial_si = 0.
    ddl_t_initial_si = 0.
    dddl_t_initial_si = 0.

    theta = cas.DM(pendulum_angle_rad)
    sin_theta = cas.sin(theta)
    cos_theta = cas.cos(theta)

    dtheta = 0.
    num_dynamics = rod_mass_si / 2. + mass_si
    den_dynamics = rod_mass_si / 3. + mass_si
    ddtheta = -(gravity_si / length_si) * (num_dynamics / den_dynamics) * sin_theta

    q_initial_si = length_si * (sin_theta * xhat - cos_theta * zhat)
    dq_initial_si = dtheta * length_si * (cos_theta * xhat + sin_theta * zhat)
    ddq_initial_si = ddtheta * length_si * (cos_theta * xhat + sin_theta * zhat) - dtheta**2. * q_initial_si

    total_mass_si = mass_si + rod_mass_si
    center_of_mass = (1. / total_mass_si) * (mass_si * length_si + rod_mass_si * length_si / 2.)
    centripetal_tension = total_mass_si * dtheta**2. * center_of_mass
    gravitational_tension = (mass_si + rod_mass_si) * gravity_si * cos_theta
    expected_tension = gravitational_tension + centripetal_tension

    lambda_newtonian = expected_tension / length_si
    lambda_si = lambda_newtonian

    initial_si = {'q10': q_initial_si,
                 'dq10': dq_initial_si,
                 'ddq10': ddq_initial_si,
                 'l_t': length_si,
                 'dl_t': dl_t_initial_si,
                 'ddl_t': ddl_t_initial_si,
                 'dddl_t': dddl_t_initial_si,
                  'lambda10': lambda_si,
                  'diam_t': rod_diameter_si,
                  'l_t_full': length_si}
    return initial_si

def build_pendulum_test_model(mass_si, gravity_si, scaling_dict={}, use_wound_tether=False):
    options = {}
    options['user_options.system_model.architecture'] = {1: 0}
    options['user_options.system_model.kite_dof'] = 3
    options['user_options.wind.model'] = 'uniform'
    options['user_options.atmosphere'] = 'uniform'
    options['params.atmosphere.g'] = gravity_si
    options['user_options.induction_model'] = 'not_in_use'
    options['user_options.tether_drag_model'] = 'not_in_use'
    options['user_options.kite_standard'] = ampyx_data.data_dict()
    options['model.geometry.overwrite.m_k'] = mass_si
    options['model.tether.use_wound_tether'] = use_wound_tether

    if len(scaling_dict.keys()) > 0:
        for var_label, scaling_value in scaling_dict.items():
            options['model.scaling_overwrite.' + var_label] = scaling_value

    trial_options = awe.Options()
    trial_options.fill_in_seed(options)
    architecture = archi.Architecture(trial_options['user_options']['system_model']['architecture'])
    trial_options.build(architecture)
    model = awe.mdl.model.Model()
    model.build(trial_options['model'], architecture)
    return model

def populate_model_variables_and_parameters(model, gravity_si, mass_si, initial_si, rod_density_si=0.):

    variables = model.variables(0.)

    parameters = model.parameters(1.23)
    parameters['theta0', 'atmosphere', 'rho_ref'] = 0.
    parameters['theta0', 'atmosphere', 'g'] = gravity_si
    parameters['theta0', 'geometry', 'm_k'] = mass_si
    parameters['theta0', 'wind', 'u_ref'] = 0.
    parameters['theta0', 'tether', 'rho'] = rod_density_si
    parameters['theta0', 'ground_station', 'm_gen'] = 0.

    for var_type in model.variables_dict.keys():
        for var_name in model.variables_dict[var_type].keys():
            if var_name in initial_si.keys():
                variables[var_type, var_name] = struct_op.var_si_to_scaled(var_type, var_name, cas.DM(initial_si[var_name]), model.scaling)

    return variables, parameters


def test_that_lagrangian_dynamics_residual_is_zero_with_consistent_inputs_and_matching_scaling(epsilon=1e-8):
    # pendulum problem
    mass_si = 17.
    gravity_si = 11.
    length_si = 37.
    pendulum_angle_rad = np.pi/2. * 0.323

    initial_si = get_consistent_inputs_for_inextensible_pendulum_problem(mass_si, gravity_si, length_si, pendulum_angle_rad)

    model = build_pendulum_test_model(mass_si, gravity_si)
    variables, parameters = populate_model_variables_and_parameters(model, gravity_si, mass_si, initial_si)

    dynamics_residual = model.dynamics(variables, parameters)
    condition = cas.mtimes(dynamics_residual.T, dynamics_residual) < epsilon**2.

    if not condition:
        message = 'something went wrong when testing the lagrangian dynamics (pendulum motion, no extension, matching scaling)'
        raise Exception(message)

    return None


def test_that_lagrangian_dynamics_residual_is_zero_with_consistent_inputs_when_pendulum_rod_has_mass_without_wound_tether(epsilon=1e-8):
    run_lagrangian_dynamics_test_with_consistent_inputs_when_pendulum_rod_has_mass(epsilon=epsilon, use_wound_tether=False)
    return None


def test_that_lagrangian_dynamics_residual_is_zero_with_consistent_inputs_when_pendulum_rod_has_mass_with_wound_tether(epsilon=1e-8):
    run_lagrangian_dynamics_test_with_consistent_inputs_when_pendulum_rod_has_mass(epsilon=epsilon, use_wound_tether=True)
    return None


def run_lagrangian_dynamics_test_with_consistent_inputs_when_pendulum_rod_has_mass(epsilon=1e-8, use_wound_tether=False):
    # pendulum problem
    mass_si = 17.
    gravity_si = 11.
    length_si = 10.

    pendulum_angle_rad = np.pi/2. * 0.323
    rod_mass_si = 41.
    rod_diameter_si = 0.02
    rod_volume_si = np.pi * (rod_diameter_si / 2.)**2. * length_si
    rod_density_si = rod_mass_si / rod_volume_si

    initial_si = get_consistent_inputs_for_inextensible_pendulum_problem(mass_si, gravity_si, length_si, pendulum_angle_rad, rod_mass_si=rod_mass_si, rod_diameter_si=rod_diameter_si)

    model = build_pendulum_test_model(mass_si, gravity_si, use_wound_tether=use_wound_tether)

    variables, parameters = populate_model_variables_and_parameters(model, gravity_si, mass_si, initial_si, rod_density_si=rod_density_si)

    dynamics_residual = model.dynamics(variables, parameters)
    condition = cas.mtimes(dynamics_residual.T, dynamics_residual) < epsilon**2.

    if not condition:
        message = 'something went wrong when testing the lagrangian dynamics (pendulum motion, when rod has mass), but use_wound_tether is ' + str(use_wound_tether)
        raise Exception(message)

    return None



def test_that_lagrangian_dynamics_residual_is_nonzero_with_inconsistent_inputs_and_matching_scaling(epsilon=1.):
    # pendulum problem
    mass_si = 17.
    gravity_si = 11.
    length_si = 37.
    pendulum_angle_rad = np.pi/2. * 0.323

    initial_si = get_consistent_inputs_for_inextensible_pendulum_problem(mass_si, gravity_si, length_si, pendulum_angle_rad)
    initial_si['dq10'] = 50. * initial_si['q10']

    model = build_pendulum_test_model(mass_si, gravity_si)
    variables, parameters = populate_model_variables_and_parameters(model, gravity_si, mass_si, initial_si)

    dynamics_residual = model.dynamics(variables, parameters)
    condition = cas.mtimes(dynamics_residual.T, dynamics_residual) > epsilon**2.

    if not condition:
        message = 'something went wrong when testing the lagrangian dynamics (pendulum motion, inconsistent extension, matching scaling)'
        raise Exception(message)

    return None

def test_that_lagrangian_dynamics_residual_is_zero_with_consistent_inputs_and_nonmatching_scalar_scaling(epsilon=1e-8):
    # pendulum problem
    mass_si = 17.
    gravity_si = 11.
    length_si = 37.
    pendulum_angle_rad = np.pi/2. * 0.323

    initial_si = get_consistent_inputs_for_inextensible_pendulum_problem(mass_si, gravity_si, length_si, pendulum_angle_rad)

    scale1 = 7.1
    scale2 = 12.3

    scaling_dict = {'x.q': scale1, 'x.dq': scale2}

    model = build_pendulum_test_model(mass_si, gravity_si, scaling_dict=scaling_dict)

    if cas.diag(model.scaling.cat).is_eye():
        message = 'something went wrong when setting the non-matching scaling. the scaling vector is still returning unit-scaling.'
        raise Exception(message)

    variables, parameters = populate_model_variables_and_parameters(model, gravity_si, mass_si, initial_si)

    dynamics_residual = model.dynamics(variables, parameters)
    condition = cas.mtimes(dynamics_residual.T, dynamics_residual) < epsilon**2.

    if not condition:
        message = 'something went wrong when testing the lagrangian dynamics (pendulum motion, no extension, nonmatching-but-scalar scaling)'
        raise Exception(message)

    return None


def test_that_lagrangian_dynamics_residual_is_zero_with_consistent_inputs_and_nonmatching_vector_scaling(epsilon=1e-8):
    # pendulum problem
    mass_si = 17.
    gravity_si = 11.
    length_si = 37.
    pendulum_angle_rad = np.pi/2. * 0.323

    initial_si = get_consistent_inputs_for_inextensible_pendulum_problem(mass_si, gravity_si, length_si, pendulum_angle_rad)

    scaling_dict = {'x.q': [100., 10., 1000.], 'x.dq': [5., 7., 2.]}

    model = build_pendulum_test_model(mass_si, gravity_si, scaling_dict=scaling_dict)
    variables, parameters = populate_model_variables_and_parameters(model, gravity_si, mass_si, initial_si)

    dynamics_residual = model.dynamics(variables, parameters)
    condition = cas.mtimes(dynamics_residual.T, dynamics_residual) < epsilon**2.

    if not condition:
        message = 'something went wrong when testing the lagrangian dynamics (pendulum motion, no extension, nonmatching-vector scaling)'
        raise Exception(message)

    return None


def test_time_derivative_under_scaling():
    # pendulum problem
    mass_si = 17.
    gravity_si = 11.
    length_si = 37.
    pendulum_angle_rad = np.pi/2. * 0.323

    initial_si = get_consistent_inputs_for_inextensible_pendulum_problem(mass_si, gravity_si, length_si, pendulum_angle_rad)

    km = 1.e3
    cm = 1.e-2

    # if the time-derivative of scaled position state [km] is in [km/s]
    # then time-derivative of si position state [cm] should be 1e(3+2) * velocity state

    scaling_dict = {'x.q': km, 'x.dq': cm}

    model = build_pendulum_test_model(mass_si, gravity_si, scaling_dict=scaling_dict)
    variables, parameters = populate_model_variables_and_parameters(model, gravity_si, mass_si, initial_si)

    q10_scaled = model.variables['x', 'q10']
    q10_si = struct_op.var_scaled_to_si('x', 'q10', q10_scaled, model.scaling)
    dq10_si_dt = lagr_dyn_tools.time_derivative(q10_si, model.variables, model.architecture, model.scaling)
    dq10_scaled = model.variables['x', 'dq10']
    dq10_si = struct_op.var_scaled_to_si('x', 'dq10', dq10_scaled, model.scaling)
    diff_q = dq10_si_dt - 1.e5 * dq10_si

    fun_diff_q = cas.Function('fun_diff_q', [model.variables], [diff_q])
    condition_q = fun_diff_q(variables).is_zero()

    dq10_scaled = model.variables['x', 'dq10']
    dq10_si = struct_op.var_scaled_to_si('x', 'dq10', dq10_scaled, model.scaling)
    ddq10_si_dt = lagr_dyn_tools.time_derivative(dq10_si, model.variables, model.architecture, model.scaling)
    ddq10_scaled = model.variables['xdot', 'ddq10']
    ddq10_si = struct_op.var_scaled_to_si('xdot', 'ddq10', ddq10_scaled, model.scaling)
    diff_dq = ddq10_si_dt - 1. * ddq10_si

    fun_diff_dq = cas.Function('fun_diff_dq', [model.variables], [diff_dq])
    condition_dq = fun_diff_dq(variables).is_zero()

    criteria = condition_q and condition_dq
    if not criteria:
        message = 'something went wrong when computing time-derivatives with scaled variables'
        raise Exception(message)

    return None

# test_architecture()
# test_drag_mode_model()
# test_constraint_mechanism()
# test_cross_tether_model()
# test_tether_moments()
# test_time_derivative_under_scaling()
# test_that_lagrangian_dynamics_residual_is_zero_with_consistent_inputs_and_matching_scaling()
# test_that_lagrangian_dynamics_residual_is_nonzero_with_inconsistent_inputs_and_matching_scaling()
# test_that_lagrangian_dynamics_residual_is_zero_with_consistent_inputs_and_nonmatching_scalar_scaling()
# test_that_lagrangian_dynamics_residual_is_zero_with_consistent_inputs_and_nonmatching_vector_scaling()
test_that_lagrangian_dynamics_residual_is_zero_with_consistent_inputs_when_pendulum_rod_has_mass_without_wound_tether()
test_that_lagrangian_dynamics_residual_is_zero_with_consistent_inputs_when_pendulum_rod_has_mass_with_wound_tether()