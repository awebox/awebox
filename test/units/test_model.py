#!/usr/bin/python3
"""Test to check model functionality

@author: Jochem De Schutter
"""

import awebox as awe
import logging
import casadi as cas
import awebox.mdl.architecture as archi
logging.basicConfig(filemode='w',format='%(levelname)s:    %(message)s', level=logging.WARNING)

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
    options['user_options']['trajectory']['type'] = 'drag_mode'

    # don't include induction effects, use trivial tether drag
    options['user_options']['induction_model'] = 'not_in_use'
    options['user_options']['tether_drag_model'] = 'trivial'

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