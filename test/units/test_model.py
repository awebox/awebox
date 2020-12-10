#!/usr/bin/python3
"""Test to check model functionality

@author: Jochem De Schutter,
edit: rachel leuthold, alu-fr 2020
"""

import awebox as awe
import logging
import casadi as cas
import awebox.mdl.architecture as archi
import numpy as np
import awebox.mdl.system as system
import awebox.mdl.mdl_constraint as mdl_constraint
import awebox.tools.constraint_operations as cstr_op

logging.basicConfig(filemode='w',format='%(levelname)s:    %(message)s', level=logging.WARNING)
#
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

    # make default options object
    options = awe.Options(True)

    # single kite with point-mass model
    options['user_options']['system_model']['architecture'] = {1:0, 2:1, 3:1}
    options['user_options']['system_model']['kite_dof'] = 3
    options['user_options']['kite_standard'] = awe.ampyx_data.data_dict()
    options['user_options']['trajectory']['type'] = 'power_cycle'
    options['user_options']['trajectory']['system_type'] = 'drag_mode'
    options['model']['tether']['use_wound_tether'] = False

    # don't include induction effects, use trivial tether drag
    options['user_options']['induction_model'] = 'not_in_use'
    options['user_options']['tether_drag_model'] = 'split'

    # build model
    architecture = archi.Architecture(options['user_options']['system_model']['architecture'])
    options.build(architecture)
    model = awe.mdl.model.Model()
    model.build(options['model'], architecture)

    # extract model info
    states = model.variables_dict['xd']
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
    assert(cas.jacobian(dynamics,model.variables['xd','kappa21']).nnz()!=0)
    assert(cas.jacobian(dynamics,model.variables['xd','kappa31']).nnz()!=0)
    assert(cas.jacobian(dynamics,model.variables['u', 'dkappa31']).nnz()!=0)
    assert(cas.jacobian(dynamics,model.variables['u', 'dkappa31']).nnz()!=0)

    # test power expression
    integral_outputs = model.integral_outputs_fun(model.variables, model.parameters)
    assert(cas.jacobian(integral_outputs,model.variables['xd','kappa21']).nnz()!=0)
    assert(cas.jacobian(integral_outputs,model.variables['xd','kappa31']).nnz()!=0)
    assert(cas.jacobian(integral_outputs,model.variables['xd','l_t']).nnz()==0)
    assert(cas.jacobian(integral_outputs,model.variables['xd','dl_t']).nnz()==0)
    assert(cas.jacobian(integral_outputs,model.variables['xa','lambda10']).nnz()==0)

    # test variable bounds
    lb = options['model']['system_bounds']['u']['dkappa'][0]/options['model']['scaling']['xd']['kappa']
    ub = options['model']['system_bounds']['u']['dkappa'][1]/options['model']['scaling']['xd']['kappa']

    assert(model.variable_bounds['u']['dkappa21']['lb'] == lb)
    assert(model.variable_bounds['u']['dkappa31']['lb'] == lb)
    assert(model.variable_bounds['u']['dkappa21']['ub'] == ub)
    assert(model.variable_bounds['u']['dkappa31']['ub'] == ub)

    if 'dddl_t' in model.variable_bounds['u'].keys():
        assert(model.variable_bounds['u']['dddl_t']['lb'] == 0.0)
        assert(model.variable_bounds['u']['dddl_t']['ub'] == 0.0)
    elif 'ddl_t' in model.variable_bounds['u'].keys():
        assert(model.variable_bounds['u']['ddl_t']['lb'] == 0.0)
        assert(model.variable_bounds['u']['ddl_t']['ub'] == 0.0)

    # test scaling
    assert(model.scaling['xd']['kappa21'] == options['model']['scaling']['xd']['kappa'])
    assert(model.scaling['xd']['kappa31'] == options['model']['scaling']['xd']['kappa'])
    assert(model.scaling['u']['dkappa21'] == options['model']['scaling']['xd']['kappa'])
    assert(model.scaling['u']['dkappa21'] == options['model']['scaling']['xd']['kappa'])

    return None

def test_cross_tether_model():
    """ Test cross-tether construction routines
    """

    # make default options object
    options = awe.Options(True)

    # single kite with point-mass model
    options['user_options']['system_model']['architecture'] = {1:0, 2:1, 3:1}
    options['user_options']['system_model']['kite_dof'] = 3
    options['user_options']['kite_standard'] = awe.ampyx_data.data_dict()
    options['user_options']['system_model']['cross_tether'] = True
    options['model']['tether']['use_wound_tether'] = False

    # don't include induction effects, use trivial tether drag
    options['user_options']['induction_model'] = 'not_in_use'
    options['user_options']['tether_drag_model'] = 'split'

    # build model
    architecture = archi.Architecture(options['user_options']['system_model']['architecture'])
    options.build(architecture)
    model = awe.mdl.model.Model()
    model.build(options['model'], architecture)

    # extract model info
    algvars = model.variables_dict['xa']
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
    assert('c10' in outputs['tether_length'].keys())
    assert('c21' in outputs['tether_length'].keys())
    assert('c31' in outputs['tether_length'].keys())
    assert('c23' in outputs['tether_length'].keys())
    assert('c32' not in outputs['tether_length'].keys())

    assert (constraints['inequality']['tether_stress10'].shape[0] == 1)
    assert (constraints['inequality']['tether_stress21'].shape[0] == 1)
    assert (constraints['inequality']['tether_stress31'].shape[0] == 1)
    assert (constraints['inequality']['tether_stress23'].shape[0] == 1)

    assert ('tether_stress00' not in constraints['inequality'].keys())
    assert ('tether_stress32' not in constraints['inequality'].keys())

    return None

def test_tether_moments():
    """ Test moment contribution due to holonomic constraints """
    options = awe.Options(True)

    # single kite with point-mass model
    options['user_options']['system_model']['architecture'] = {1:0}
    options['user_options']['system_model']['kite_dof'] = 6
    options['user_options']['kite_standard'] = awe.ampyx_data.data_dict()
    options['user_options']['induction_model'] = 'not_in_use'
    options['user_options']['tether_drag_model'] = 'split'
    options['user_options']['trajectory']['system_type'] = 'drag_mode'
    options['model']['tether']['use_wound_tether'] = False

    # tether attachment
    r_tether = np.array([0.0, 0.0, -0.1])
    options['model']['tether']['attachment'] = 'stick'
    options['model']['geometry']['overwrite']['r_tether'] = r_tether

    # build model
    architecture = archi.Architecture(options['user_options']['system_model']['architecture'])
    options.build(architecture)
    model = awe.mdl.model.Model()
    model.build(options['model'], architecture)

    # si variables
    var_si = model.variables(0.0)

    var_si['xd', 'q10'] = np.array([130.644, 24.5223, 74.2863])
    var_si['xd', 'dq10'] = np.array([-16.3061, -27.0514, 37.5959])
    var_si['xd', 'omega10'] = np.array([-0.0181481, 0.275884, 1.5743])
    var_si['xd', 'r10'] = np.array([0.271805, 0.334641, -0.902295,
                             0.0595685, 0.929945, 0.362839,
                             0.960506, -0.15237, 0.23283])
    var_si['xd', 'delta10'] = np.array([0.0693676, 0.261684, -0.258425])
    var_si['xd', 'kappa10'] = -0.146057
    var_si['xd', 'l_t'] = 152.184
    var_si['xd', 'dl_t'] = 7.26866e-12

    if '[xd,ddl_t,0]' in var_si.labels():
        var_si['xd', 'ddl_t'] = 9.49347e-13

    var_si['xa'] = 45.024
    var_si['theta'] = np.array([50, 100, 0.005, 0.00492276, 3.93805])

    # scaled variables
    scaling = model.scaling
    var_sc = system.scale_variable(model.variables, var_si, scaling)
    parameters = model.parameters(0.0)
    parameters['theta0','geometry','r_tether'] = r_tether

    # numerical result
    outputs = model.outputs(model.outputs_fun(var_sc, parameters))
    tether_moment = outputs['tether_moments', 'n10']

    # analytic expression
    dcm = cas.reshape(var_si['xd', 'r10'],(3,3))
    tether_moment_true = var_si['xa','lambda10'] * cas.cross(r_tether, cas.mtimes(dcm.T, var_si['xd','q10']))

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
    cstr_list = mdl_constraint.MdlConstraintList()

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
    cstr_list_empty = mdl_constraint.MdlConstraintList()

    # can we add the empty list to the existing list?
    cstr_list.append(cstr_list_empty)

    # does the list still record two equality constraints and 0 inequality constraints?
    results[rdx] = (len(cstr_list.eq_list) == 2) and (len(cstr_list.ineq_list) == 0)
    rdx += 1

    # can we make a non-empty list, and append it to the main list?
    cstr_list_nonempty = mdl_constraint.MdlConstraintList()
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
