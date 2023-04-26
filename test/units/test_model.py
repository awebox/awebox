#!/usr/bin/python3
"""Test to check model functionality

@author: Jochem De Schutter,
edit: rachel leuthold, alu-fr 2020
"""
import copy
import pdb

import casadi

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
import matplotlib.pyplot as plt

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


def get_consistent_inputs_for_pendulum_problem(pendulum_parameters, variable_length=False):

    xhat = vect_op.xhat_np()
    zhat = vect_op.zhat_np()

    dl_t_initial_si = 0.
    ddl_t_initial_si = 0.
    dddl_t_initial_si = 0.

    if variable_length:
        dl_t_initial_si = 0.01

    pendulum_angle_rad = pendulum_parameters['pendulum_angle_rad']
    pendulum_omega = pendulum_parameters['pendulum_omega']
    rod_mass_si = pendulum_parameters['rod_mass_si']
    mass_si = pendulum_parameters['mass_si']
    gravity_si = pendulum_parameters['gravity_si']
    length_si = pendulum_parameters['length_si']
    rod_diameter_si = pendulum_parameters['rod_diameter_si']

    theta = cas.DM(pendulum_angle_rad)
    sin_theta = cas.sin(theta)
    cos_theta = cas.cos(theta)

    dtheta = cas.DM(pendulum_omega)
    num_dynamics = rod_mass_si / 2. + mass_si
    den_dynamics = rod_mass_si / 3. + mass_si
    ddtheta = -(gravity_si / length_si) * (num_dynamics / den_dynamics) * sin_theta

    q_initial_si = length_si * (sin_theta * xhat - cos_theta * zhat)

    dq_initial_0_si = length_si * (cos_theta * xhat + sin_theta * zhat) * dtheta
    dq_initial_1_si = (sin_theta * xhat - cos_theta * zhat) * dl_t_initial_si
    dq_initial_si = dq_initial_0_si + dq_initial_1_si

    ddq_0_theta = q_initial_si * dtheta**2. + length_si * (cos_theta * xhat + sin_theta * zhat) * ddtheta
    ddq_0_lt = (cos_theta * xhat + sin_theta * zhat) * dtheta * dl_t_initial_si
    ddq_1_theta = (cos_theta * xhat + sin_theta * zhat) * dl_t_initial_si * dtheta
    ddq_1_lt = (sin_theta * xhat - cos_theta * zhat) * ddl_t_initial_si
    ddq_initial_si = ddq_0_theta + ddq_0_lt + ddq_1_theta + ddq_1_lt

    total_mass_si = mass_si + rod_mass_si
    center_of_mass = (1. / total_mass_si) * (mass_si * length_si + rod_mass_si * length_si / 2.)
    centripetal_tension = total_mass_si * dtheta**2. * center_of_mass
    gravitational_tension = (mass_si + rod_mass_si) * gravity_si * cos_theta
    expected_tension = gravitational_tension + centripetal_tension

    lambda_newtonian = expected_tension / length_si
    lambda_si = lambda_newtonian

    t_f = 2. * np.pi * (center_of_mass / gravity_si)**0.5

    initial_si = {
            'q10': q_initial_si,
            'dq10': dq_initial_si,
            'ddq10': ddq_initial_si,
            'l_t': length_si,
            'dl_t': dl_t_initial_si,
            'ddl_t': ddl_t_initial_si,
            'dddl_t': dddl_t_initial_si,
            'lambda10': lambda_si,
            'diam_t': rod_diameter_si,
            'l_t_full': 1e5 * length_si,
            't_f': t_f
            }
    return initial_si

def build_pendulum_test_model(pendulum_parameters, use_wound_tether=False, frictionless=True):
    if frictionless:
        tether_drag_model = 'not_in_use'
    else:
        tether_drag_model = 'split'

    options = {}
    options['user_options.system_model.architecture'] = {1: 0}
    options['user_options.system_model.kite_dof'] = 3
    options['user_options.wind.model'] = 'uniform'
    options['user_options.atmosphere'] = 'uniform'
    options['params.atmosphere.g'] = pendulum_parameters['gravity_si']
    options['params.ground_station.m_gen'] = 0.
    options['params.ground_station.r_gen'] = 0.25
    options['user_options.induction_model'] = 'not_in_use'
    options['user_options.tether_drag_model'] = tether_drag_model
    options['user_options.kite_standard'] = ampyx_data.data_dict()
    options['model.geometry.overwrite.m_k'] = pendulum_parameters['mass_si']
    options['model.tether.use_wound_tether'] = use_wound_tether
    options['quality.test_param.check_energy_summation'] = True

    trial_options = awe.Options()
    trial_options.fill_in_seed(options)
    architecture = archi.Architecture(trial_options['user_options']['system_model']['architecture'])
    trial_options.build(architecture)
    model = awe.mdl.model.Model()
    model.build(trial_options['model'], architecture)
    return model

def populate_model_variables_and_parameters(model, pendulum_parameters, initial_si, frictionless=True):

    if frictionless:
        rho_ref = 0.
    else:
        rho_ref = 1.

    variables = model.variables(0.)

    parameters = model.parameters(1.23)
    parameters['theta0', 'atmosphere', 'rho_ref'] = rho_ref
    parameters['theta0', 'atmosphere', 'g'] = pendulum_parameters['gravity_si']
    parameters['theta0', 'geometry', 'm_k'] = pendulum_parameters['mass_si']
    parameters['theta0', 'wind', 'u_ref'] = 0.
    parameters['theta0', 'tether', 'rho'] = pendulum_parameters['rod_density_si']
    parameters['theta0', 'ground_station', 'm_gen'] = 0.

    for var_type in model.variables_dict.keys():
        for var_name in model.variables_dict[var_type].keys():
            if var_name in initial_si.keys():
                variables[var_type, var_name] = struct_op.var_si_to_scaled(var_type, var_name, cas.DM(initial_si[var_name]), model.scaling)

    return variables, parameters


def get_arbitary_pendulum_problem_parameters(rod_has_mass=False):

    # pendulum problem
    mass_si = 17.
    gravity_si = 11.
    length_si = 37.
    pendulum_angle_rad = np.pi/2. * 0.7
    pendulum_omega = -1.e-12

    if rod_has_mass:
        rod_mass_si = 41.
    else:
        rod_mass_si = 0.

    rod_diameter_si = 0.02
    rod_volume_si = np.pi * (rod_diameter_si / 2.) ** 2. * length_si
    rod_density_si = rod_mass_si / rod_volume_si

    pendulum_parameters = {
        'mass_si': mass_si,
        'gravity_si': gravity_si,
        'length_si': length_si,
        'pendulum_angle_rad': pendulum_angle_rad,
        'pendulum_omega': pendulum_omega,
        'rod_mass_si': rod_mass_si,
        'rod_diameter_si': rod_diameter_si,
        'rod_density_si': rod_density_si}
    return pendulum_parameters


def test_that_lagrangian_dynamics_residual_is_nonzero_with_inconsistent_inputs(epsilon=1.):
    run_lagrangian_dynamics_pendulum_test(epsilon=epsilon,
                                          use_consistent_inputs=False,
                                          rod_has_mass=False,
                                          use_wound_tether=False,
                                          check_scaling_is_nontrivial=False)
    return None


def test_that_lagrangian_dynamics_residual_is_zero_with_consistent_inputs(epsilon=1e-8):
    run_lagrangian_dynamics_pendulum_test(epsilon=epsilon,
                                          use_consistent_inputs=True,
                                          rod_has_mass=False,
                                          use_wound_tether=False,
                                          check_scaling_is_nontrivial=False)
    return None


def test_that_lagrangian_dynamics_residual_is_zero_with_consistent_inputs_and_nontrivial_scaling(epsilon=1e-8):
    run_lagrangian_dynamics_pendulum_test(epsilon=epsilon,
                                          use_consistent_inputs=True,
                                          rod_has_mass=False,
                                          use_wound_tether=False,
                                          check_scaling_is_nontrivial=True)
    return None


def test_that_lagrangian_dynamics_residual_is_zero_with_consistent_inputs_when_pendulum_rod_has_mass_without_wound_tether(epsilon=1e-8):
    run_lagrangian_dynamics_pendulum_test(epsilon=epsilon,
                                          use_consistent_inputs=True,
                                          rod_has_mass=True,
                                          use_wound_tether=False,
                                          check_scaling_is_nontrivial=False)
    return None


def test_that_lagrangian_dynamics_residual_is_zero_with_consistent_inputs_when_pendulum_rod_has_mass_with_wound_tether(epsilon=1e-8):
    run_lagrangian_dynamics_pendulum_test(epsilon=epsilon,
                                          use_consistent_inputs=True,
                                          rod_has_mass=True,
                                          use_wound_tether=True,
                                          check_scaling_is_nontrivial=False)
    return None




def run_lagrangian_dynamics_pendulum_test(epsilon=1e-8, use_consistent_inputs=True, rod_has_mass=False, use_wound_tether=False, check_scaling_is_nontrivial=False):

    pendulum_parameters = get_arbitary_pendulum_problem_parameters(rod_has_mass=rod_has_mass)
    initial_si = get_consistent_inputs_for_pendulum_problem(pendulum_parameters, variable_length=False)

    if not use_consistent_inputs:
        initial_si['dq10'] = 50. * initial_si['q10']

    model = build_pendulum_test_model(pendulum_parameters, use_wound_tether=use_wound_tether)

    if check_scaling_is_nontrivial:
        break_if_scaling_is_trivial(model, 'q10', epsilon)

    variables, parameters = populate_model_variables_and_parameters(model, pendulum_parameters, initial_si)

    dynamics_residual = model.dynamics(variables, parameters)
    residual_counts_as_zero = cas.mtimes(dynamics_residual.T, dynamics_residual) < epsilon**2.

    condition_if_consistent = use_consistent_inputs and residual_counts_as_zero
    condition_if_inconsistent = (not use_consistent_inputs) and (not residual_counts_as_zero)
    condition = condition_if_consistent or condition_if_inconsistent

    if not condition:
        message = 'something went wrong when testing the lagrangian dynamics in a pendulum system: '
        message += 'use_consistent_inputs = ' + str(use_consistent_inputs) + ', '
        message += 'rod_has_mass = ' + str(rod_has_mass) + ', '
        message += 'use_wound_tether = ' + str(use_wound_tether) + ', '
        message += 'check_scaling_is_nontrivial = ' + str(check_scaling_is_nontrivial)
        raise Exception(message)

    return None


def break_if_scaling_is_trivial(model, var_name, epsilon):
    deriv_name = 'd' + var_name
    scaling_difference = model.scaling['x', var_name] - model.scaling['x', deriv_name]
    if cas.mtimes(scaling_difference.T, scaling_difference) < epsilon**2.:
        message = 'no difference in scaling between vars (' + var_name + ') and (' + deriv_name +'). this test is not going to be meaningful.'
        raise Exception(message)

    for local_name in [var_name, deriv_name]:
        if cas.diag(model.scaling['x', local_name]).is_eye():
            message = 'scaling for variable (' + local_name + ') appears to be unity. this test may not going to be meaningful'
            raise Exception(message)

    return None

def test_time_derivative_under_scaling(epsilon=1e-5):
    pendulum_parameters = get_arbitary_pendulum_problem_parameters()
    initial_si = get_consistent_inputs_for_pendulum_problem(pendulum_parameters)

    # if the time-derivative of scaled position state [km] is in [km/s]
    # then time-derivative of si position state [cm] should be 1e(3+2) * velocity state

    model = build_pendulum_test_model(pendulum_parameters)
    variables, parameters = populate_model_variables_and_parameters(model, pendulum_parameters, initial_si)

    dict_of_planned_tests = {'scalar': 'l_t', 'vector': 'q10'}
    dict_of_test_conclusions = {}

    for test_name, var_name in dict_of_planned_tests.items():
        deriv_name = 'd' + var_name

        break_if_scaling_is_trivial(model, var_name, epsilon)

        var_scaled = model.variables['x', var_name]
        var_si = struct_op.var_scaled_to_si('x', var_name, var_scaled, model.scaling)
        dot_var_si = lagr_dyn_tools.time_derivative(var_si, model.variables, model.architecture, model.scaling)

        x_dvar_scaled = model.variables['x', deriv_name]
        x_dvar_si = struct_op.var_scaled_to_si('x', deriv_name, x_dvar_scaled, model.scaling)
        x_diff = dot_var_si - x_dvar_si

        x_diff_fun = cas.Function('x_diff_fun', [model.variables, model.parameters], [x_diff])
        x_diff_eval = x_diff_fun(variables, parameters)
        x_condition = cas.mtimes(x_diff_eval.T, x_diff_eval) < epsilon**2.
        dict_of_test_conclusions[test_name + '_x'] = x_condition

        xdot_dvar_scaled = model.variables['xdot', deriv_name]
        xdot_dvar_si = struct_op.var_scaled_to_si('xdot', deriv_name, xdot_dvar_scaled, model.scaling)
        xdot_diff = dot_var_si - xdot_dvar_si

        xdot_diff_fun = cas.Function('xdot_diff_fun', [model.variables, model.parameters], [xdot_diff])
        xdot_diff_eval = xdot_diff_fun(variables, parameters)
        xdot_condition = cas.mtimes(xdot_diff_eval.T, xdot_diff_eval) < epsilon**2.
        dict_of_test_conclusions[test_name + '_xdot'] = xdot_condition

    criteria = all(dict_of_test_conclusions.values())
    if not criteria:
        message = 'something went wrong when computing time-derivatives with scaled variables'
        raise Exception(message)

    return None


def get_integration_test_setup(use_wound_tether=True, frictionless=True, variable_length=False):

    pendulum_parameters = get_arbitary_pendulum_problem_parameters(rod_has_mass=True)
    initial_si = get_consistent_inputs_for_pendulum_problem(pendulum_parameters, variable_length=variable_length)

    # x = cas.SX.sym('x')
    # z = cas.SX.sym('z')
    # p = cas.SX.sym('p')
    # dae = {'x': x, 'z': z, 'p':p, 'ode': x, 'alg':z-x}
    # F = cas.integrator('F', 'idas', dae)
    #
    # # x = exp(b t)
    # # xdot = b exp(b t)
    # # f = x - xdot = exp(b t) - b exp(b t)  => b = 1
    # # x(t = 2) = cas.exp(2)
    #
    # r = F(x0=1, z0=1, p=0.1)
    # r2 = F(x0=r['xf'], z0=r['zf'], p=0.1)
    # print(r2['xf'])

    model = build_pendulum_test_model(pendulum_parameters, use_wound_tether=use_wound_tether, frictionless=frictionless)
    t_f_simple = 2. * np.pi * (pendulum_parameters['length_si'] / pendulum_parameters['gravity_si'])**0.5

    total_time = t_f_simple / 2.
    number_of_steps = 1000
    delta_t = total_time / float(number_of_steps)

    dae = model.get_dae()
    dae.build_rootfinder()
    options = {
        'abstol': 1.e-10,
        'max_num_steps': 1e5,
        't0': 0,
        'tf': total_time
    }

    options_history = copy.deepcopy(options)
    options_history['tf'] = delta_t

    var_init_scaled, param_init_scaled = populate_model_variables_and_parameters(model, pendulum_parameters, initial_si, frictionless=frictionless)
    x0, z0, p = dae.fill_in_dae_variables(var_init_scaled, param_init_scaled)

    dae_dict = dae.dae

    F_idas = cas.integrator('F', 'idas', dae_dict, options)
    sol = F_idas(x0=x0, z0=z0, p=p)

    # F_idas_history = cas.integrator('F', 'idas', dae_dict, options_history)
    # x1 = x0
    # z1 = z0
    # q_history = []
    # for tdx in range(number_of_steps):
    #     sol_history = F_idas_history(x0=x1, z0=z1, p=p)
    #     x1 = sol_history['xf']
    #     z1 = sol_history['zf']
    #     local_variables = dae.reassemble_dae_outputs_into_model_variables(model.variables, sol_history, p)
    #     q_history = cas.horzcat(q_history, local_variables['x', 'q10'])
    #
    # qx = np.array(q_history[0, :]).T
    # qz = np.array(q_history[2, :]).T
    # plt.plot(qx, qz)
    # plt.show()

    integration_outputs = sol
    var_final_scaled = dae.reassemble_dae_outputs_into_model_variables(model.variables, integration_outputs, p)

    return model, var_init_scaled, param_init_scaled, var_final_scaled


def test_that_dae_integration_actually_does_something(epsilon=1.e-2, use_wound_tether=True, frictionless=True, variable_length=False):
    model, var_init_scaled, param_init_scaled, var_final_scaled = get_integration_test_setup(use_wound_tether=use_wound_tether, frictionless=frictionless, variable_length=variable_length)

    position_initial = var_init_scaled['x', 'q10']
    position_final = var_final_scaled['x', 'q10']

    diff = position_final - position_initial
    condition = cas.mtimes(diff.T, diff) > epsilon**2.
    if not condition:
        message = 'sanity check for pendulum integration tests does not work as expected'
        raise Exception(message)
    return None


def run_an_energy_conservation_test_in_a_pendulum_situation(epsilon=1.e-4, use_wound_tether=True, frictionless=True, variable_length=False):

    if frictionless:
        expect_conservation = True
    else:
        expect_conservation = False

    model, var_init_scaled, param_init_scaled, var_final_scaled = get_integration_test_setup(use_wound_tether=use_wound_tether,
                                                                                             frictionless=frictionless,
                                                                                             variable_length=variable_length)

    all_outputs = model.outputs(model.outputs_fun(model.variables, model.parameters))

    print('total_energy')
    total_energy = 0.
    types_of_energy = ['e_potential', 'e_kinetic']
    for e_type in types_of_energy:
        for source in struct_op.subkeys(all_outputs, e_type):
            local_energy = all_outputs[e_type, source]

            print(e_type + ' ' + source)
            local_energy_fun = casadi.Function('local_energy_fun', [model.variables, model.parameters], [local_energy])
            print(local_energy_fun(var_init_scaled, param_init_scaled))
            print(local_energy_fun(var_final_scaled, param_init_scaled))

            total_energy += local_energy

    print('total_energy')
    total_energy_fun = casadi.Function('total_energy_fun', [model.variables, model.parameters], [total_energy])
    print(total_energy_fun(var_init_scaled, param_init_scaled))
    print(total_energy_fun(var_final_scaled, param_init_scaled))

    total_energy_initial = total_energy_fun(var_init_scaled, param_init_scaled)
    total_energy_final = total_energy_fun(var_final_scaled, param_init_scaled)

    energy_diff = total_energy_final - total_energy_initial
    energy_error = energy_diff / total_energy_initial
    energy_is_conserved = energy_error**2. < epsilon**2.

    frictionless_condition = expect_conservation and energy_is_conserved
    friction_condition = (not expect_conservation) and (not energy_is_conserved)

    condition = frictionless_condition or friction_condition
    if not condition:
        message = 'the expectation that the pendulum-energy would'
        if not expect_conservation:
            message += ' not'
        message += ' be conserved,'
        message += " for pendulum test-case, with"

        if variable_length:
            message += ' a variable-length tether'
        else:
            message += ' a uniform-length tether'

        message += ' represented with'
        if use_wound_tether:
            message += ' a tether winding model'
        else:
            message += ' a Lagrangian-mechanics rhs. momentum-correction'

        if frictionless:
            message += ', when the air is assumed to be frictionless,'
        else:
            message += ', when energy is lost to air-resistance,'

        message += " is not met."
        raise Exception(message)
    return None


def test_that_energy_is_not_conserved_with_drag(epsilon=1.e-2):
    run_an_energy_conservation_test_in_a_pendulum_situation(epsilon=epsilon,
                                                            frictionless=False,
                                                            variable_length=False)
    return None


def test_that_energy_is_conserved_in_a_frictionless_pendulum_without_wound_tether(epsilon=1e-4):
    run_an_energy_conservation_test_in_a_pendulum_situation(epsilon=epsilon,
                                                            use_wound_tether=False,
                                                            variable_length=False)
    return None


def test_that_energy_is_conserved_in_a_frictionless_pendulum_with_wound_tether(epsilon=1e-4):
    run_an_energy_conservation_test_in_a_pendulum_situation(epsilon=epsilon,
                                                            use_wound_tether=True,
                                                            variable_length=False)
    return None


def test_that_energy_is_conserved_in_a_frictionless_variable_length_tether_situation_without_wound_tether(epsilon=1e-4):
    run_an_energy_conservation_test_in_a_pendulum_situation(epsilon=epsilon,
                                                            use_wound_tether=False,
                                                            variable_length=True)
    return None


def test_that_energy_is_conserved_in_a_frictionless_variable_length_tether_situation_with_wound_tether(epsilon=1e-4):
    run_an_energy_conservation_test_in_a_pendulum_situation(epsilon=epsilon,
                                                            use_wound_tether=True,
                                                            variable_length=True)
    return None


test_architecture()
test_drag_mode_model()
test_constraint_mechanism()
test_cross_tether_model()
test_tether_moments()

test_time_derivative_under_scaling()
test_that_lagrangian_dynamics_residual_is_nonzero_with_inconsistent_inputs()
test_that_lagrangian_dynamics_residual_is_zero_with_consistent_inputs()
test_that_lagrangian_dynamics_residual_is_zero_with_consistent_inputs_and_nontrivial_scaling()
test_that_lagrangian_dynamics_residual_is_zero_with_consistent_inputs_when_pendulum_rod_has_mass_with_wound_tether()
test_that_lagrangian_dynamics_residual_is_zero_with_consistent_inputs_when_pendulum_rod_has_mass_without_wound_tether()

test_that_dae_integration_actually_does_something()
test_that_energy_is_not_conserved_with_drag()
test_that_energy_is_conserved_in_a_frictionless_pendulum_without_wound_tether()
test_that_energy_is_conserved_in_a_frictionless_pendulum_with_wound_tether()
test_that_energy_is_conserved_in_a_frictionless_variable_length_tether_situation_without_wound_tether()
test_that_energy_is_conserved_in_a_frictionless_variable_length_tether_situation_with_wound_tether()
