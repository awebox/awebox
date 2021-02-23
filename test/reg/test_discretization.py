#!/usr/bin/python3
"""Test of dae integrator implementation by comparing
against direct collocation solution of NLP

@author: Jochem De Schutter
"""

import awebox as awe
import logging
import awebox.tools.struct_operations as struct_op
from casadi.tools import *
import numpy as np
logging.basicConfig(filemode='w',format='%(levelname)s:    %(message)s', level=logging.WARNING)

from awebox.logger.logger import Logger as awelogger
awelogger.logger.setLevel(10)

def test_integrators():

    # ===========================================
    # SET-UP DIRECT COLLOCATION PROBLEM AND SOLVE
    # ===========================================

    # make default options object
    base_options = awe.Options(True) # True refers to internal access switch

    # choose simplest model
    base_options['user_options']['system_model']['architecture'] = {1:0}
    base_options['user_options']['system_model']['kite_dof'] = 3
    base_options['user_options']['kite_standard'] = awe.ampyx_data.data_dict()
    base_options['user_options']['tether_drag_model'] = 'split'
    base_options['user_options']['induction_model'] = 'not_in_use'
    
    # specify direct collocation options
    base_options['nlp']['n_k'] = 40
    base_options['nlp']['discretization'] = 'direct_collocation'
    base_options['nlp']['collocation']['u_param'] = 'zoh'
    base_options['nlp']['collocation']['scheme'] = 'radau'
    base_options['nlp']['collocation']['d'] = 4

    base_options['model']['tether']['control_var'] = 'dddl_t'

     # homotopy tuning
    base_options['solver']['mu_hippo'] = 1e-4
    base_options['solver']['tol_hippo'] = 1e-4

    # make trial, build and run
    trial = awe.Trial(name = 'test', seed = base_options)
    trial.build()
    trial.optimize()

    # extract solution data
    V_final = trial.optimization.V_opt
    P       = trial.optimization.p_fix_num
    Int_outputs = trial.optimization.integral_output_vals[1]
    model   = trial.model
    dae     = model.get_dae()

    # build dae variables for t = 0 within first shooting interval
    variables0 = struct_op.get_variables_at_time(base_options['nlp'], V_final, None, model.variables, 0)
    parameters = model.parameters(vertcat(P['theta0'], V_final['phi']))
    x0, z0, p  = dae.fill_in_dae_variables(variables0, parameters)

    # ===================================
    # TEST COLLOCATION INTEGRATOR
    # ===================================

    # set discretization to multiple shooting
    base_options['nlp']['discretization'] = 'multiple_shooting'
    base_options['nlp']['integrator']['type'] = 'collocation'
    base_options['nlp']['integrator']['collocation_scheme'] = base_options['nlp']['collocation']['scheme']
    base_options['nlp']['integrator']['interpolation_order'] = base_options['nlp']['collocation']['d']
    base_options['nlp']['integrator']['num_steps'] = 1

    # switch off expand to allow for use of integrator in NLP
    base_options['solver']['expand_overwrite'] = False

    # build MS trial
    trialColl = awe.Trial(name = 'testColl', seed = base_options)
    trialColl.build()

    # multiple shooting dae integrator
    F = trialColl.nlp.Multiple_shooting.F

    # integrate over one interval
    Ff = F(x0 = x0, z0 = z0, p = p)
    xf = Ff['xf']
    zf = Ff['zf']
    qf = Ff['qf']

    # evaluate integration error
    err_coll_x = np.max(np.abs(np.divide((xf - V_final['xd',1]), V_final['xd',1]).full()))
    xa = dae.z(zf)['xa']
    err_coll_z = np.max(np.abs(np.divide(dae.z(zf)['xa'] - V_final['coll_var',0, -1, 'xa'], V_final['coll_var',0, -1, 'xa']).full()))
    err_coll_q = np.max(np.abs(np.divide((qf - Int_outputs['int_out',1]), Int_outputs['int_out',1]).full()))

    tolerance = 1e-8

    # values should match up to nlp solver accuracy
    assert(err_coll_x < tolerance)
    assert(err_coll_z < tolerance)
    assert(err_coll_q < tolerance)

    # ===================================
    # TEST RK4-ROOT INTEGRATOR
    # ===================================

    # set discretization to multiple shooting
    base_options['nlp']['integrator']['type'] = 'rk4root'
    base_options['nlp']['integrator']['num_steps'] = 20

    # build MS trial
    trialRK = awe.Trial(name = 'testRK', seed = base_options)
    trialRK.build()

    # multiple shooting dae integrator
    F = trialRK.nlp.Multiple_shooting.F

    # integrate over one interval
    Ff = F(x0 = x0, z0 = z0, p = p)
    xf = Ff['xf']
    zf = Ff['zf']
    qf = Ff['qf']

    # evaluate 
    err_rk_x = np.max(np.abs(np.divide((xf - V_final['xd',1]), V_final['xd',1]).full()))
    xa = dae.z(zf)['xa']
    err_rk_z = np.max(np.abs(np.divide(dae.z(zf)['xa'] - V_final['coll_var',0, -1, 'xa'], V_final['coll_var',0, -1, 'xa']).full()))
    err_rk_q = np.max(np.abs(np.divide((qf - Int_outputs['int_out',1]), Int_outputs['int_out',1]).full()))

    # error should be below 1%
    assert(err_rk_x < 1e-2)
    assert(err_rk_z < 1e-2)
    assert(err_rk_q < 2e-2)
