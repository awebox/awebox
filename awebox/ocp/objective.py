#
#    This file is part of awebox.
#
#    awebox -- A modeling and optimization framework for multi-kite AWE systems.
#    Copyright (C) 2017-2021 Jochem De Schutter, Rachel Leuthold, Moritz Diehl,
#                            ALU Freiburg.
#    Copyright (C) 2018-2020 Thilo Bronnenmeyer, Kiteswarms Ltd.
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
objective code of the awebox
constructs an objective function from the various fictitious costs.
python-3.5 / casadi-3.4.5
- refactored from awebox code (elena malz, chalmers; jochem de schutter, alu-fr; rachel leuthold, alu-fr), 2018
- edited: rachel leuthold, jochem de schutter alu-fr 2018-2021
'''
from typing import Union, Dict

import casadi as ca
import casadi.tools as cas
import numpy as np
from . import collocation
from . import ocp_outputs

import collections

import awebox.tools.print_operations as print_op
import awebox.tools.struct_operations as struct_op
import awebox.tools.cached_functions as cf
from awebox.mdl.model import Model

def get_general_regularization_function(variables):

    weight_sym = cas.SX.sym('weight_sym', variables.cat.shape)
    var_sym = cas.SX.sym('var_sym', variables.cat.shape)
    ref_sym = cas.SX.sym('ref_sym', variables.cat.shape)

    diff = (var_sym - ref_sym)

    weight = cas.diag(weight_sym)

    diff_sq = cas.mtimes(cas.diag(diff), diff)
    reg = cas.mtimes(weight, diff_sq)

    reg_fun = cas.Function('reg_fun', [var_sym, ref_sym, weight_sym], [reg])

    return reg_fun

def get_general_reg_costs_function(nlp_options, variables, V):

    var_sym = cas.SX.sym('var_sym', variables.cat.shape)
    ref_sym = cas.SX.sym('ref_sym', variables.cat.shape)
    weight_sym = cas.SX.sym('weight_sym', variables.cat.shape)

    reg_fun = get_general_regularization_function(variables)
    regs = variables(reg_fun(var_sym, ref_sym, weight_sym))

    sorting_dict, reg_list = get_regularization_sorting_dict(nlp_options)
    reg_costs_dict = collections.OrderedDict()
    for cost in reg_list:
        reg_costs_dict[cost] = 0.0

    for var_type in set(variables.keys()):
        category = sorting_dict[var_type]['category']
        exceptions = sorting_dict[var_type]['exceptions']

        for var_name in set(struct_op.subkeys(variables, var_type)):
            name, _ = struct_op.split_name_and_node_identifier(var_name)

            if (not name in exceptions.keys()) and (not category == None):
                reg_costs_dict[category] = reg_costs_dict[category] + cas.sum1(regs[var_type, var_name])

            elif (name in exceptions.keys()) and (not exceptions[name] == None):
                exc_category = exceptions[name]
                reg_costs_dict[exc_category] = reg_costs_dict[exc_category] + cas.sum1(regs[var_type, var_name])

    reg_costs_list = cas.vertcat(*reg_costs_dict.values())
    reg_costs_fun = cas.Function('reg_costs_fun', [var_sym, ref_sym, weight_sym], [reg_costs_list])

    return reg_costs_fun, reg_costs_dict


def get_costs_struct(V):

    costs_struct = cas.struct_symSX([
        cas.entry("tracking_cost"),
        cas.entry("xdot_regularisation_cost"),
        cas.entry("u_regularisation_cost"),
        cas.entry("fictitious_cost"),
        cas.entry("theta_regularisation_cost"),
        cas.entry("beta_cost")] +
       [cas.entry(name + '_cost') for name in struct_op.subkeys(V, 'phi')] +
       [cas.entry("time_cost"),
        cas.entry("power_cost"),
        cas.entry("nominal_landing_cost"),
        cas.entry("transition_cost"),
        cas.entry("tracking_problem_cost"),
        cas.entry("power_problem_cost"),
        cas.entry("general_problem_cost"),
        cas.entry("objective")
        ])

    return costs_struct


def get_regularization_sorting_dict(nlp_options):

    # in general, regularization of the variables of type TYPE, enters the cost in the category CATEGORY,
    # with the exception of those variables named EXCLUDED_VARIABLE_NAME, which enter the cost in the category CATEGORY_FOR_EXCLUDED_VARIABLE
    #
    # sorting_dict[TYPE] = {'category': CATEGORY, 'exceptions': {EXCLUDED_VARIABLE_NAME: CATEGORY_FOR_EXCLUDED_VARIABLE}}
    #

    sorting_dict = {}
    sorting_dict['x'] = {'category': 'tracking_cost', 'exceptions': {'e': None} }
    sorting_dict['xdot'] = {'category': 'xdot_regularisation_cost', 'exceptions': {} }
    sorting_dict['u'] = {'category': 'u_regularisation_cost', 'exceptions': {'f_fict': 'fictitious_cost', 'm_fict': 'fictitious_cost'} }
    sorting_dict['z'] = {'category': 'tracking_cost', 'exceptions': {}}
    sorting_dict['theta'] = {'category': 'theta_regularisation_cost', 'exceptions': {'t_f': None}}

    for item in nlp_options['cost']['adjustments_to_general_regularization_distribution']:
        var_type = item[0]
        var_name = item[1]
        reassigment = item[2]
        sorting_dict[var_type]['exceptions'][var_name] = reassigment

    reg_list = ['tracking_cost', 'xdot_regularisation_cost', 'u_regularisation_cost', 'fictitious_cost', 'theta_regularisation_cost']

    return sorting_dict, reg_list


def get_regularization_weights(variables, P, nlp_options):

    sorting_dict, _ = get_regularization_sorting_dict(nlp_options)

    weights = variables(P['p', 'weights'])

    for var_type in set(variables.keys()):
        category = sorting_dict[var_type]['category']
        exceptions = sorting_dict[var_type]['exceptions']

        for var_name in set(struct_op.subkeys(variables, var_type)):
            name, _ = struct_op.split_name_and_node_identifier(var_name)

            if (name not in exceptions.keys()) and (category is not None):
                shortened_cat_name = category[:-5]
                normalization = nlp_options['cost']['normalization'][shortened_cat_name]
                factor = P['cost', shortened_cat_name]
                weights[var_type, var_name] = weights[var_type, var_name] * factor / normalization

            elif (name in exceptions.keys()) and (exceptions[name] is not None):
                shortened_cat_name = exceptions[name][:-5]
                normalization = nlp_options['cost']['normalization'][shortened_cat_name]
                factor = P['cost', shortened_cat_name]
                weights[var_type, var_name] = weights[var_type, var_name] * factor / normalization

    return weights


def get_coll_parallel_info(nlp_options, V, P, Xdot, model):

    n_k = nlp_options['n_k']
    d = nlp_options['collocation']['d']
    N_coll = n_k * d

    coll_weights = []
    int_weights = find_int_weights(nlp_options)
    p_weights = get_regularization_weights(model.variables, P, nlp_options).cat
    for ndx in range(n_k):
        for ddx in range(d):
            coll_weights = cas.horzcat(coll_weights, int_weights[ddx] * p_weights)

    coll_vars = struct_op.get_coll_vars(nlp_options, V, P, Xdot, model)
    coll_refs = struct_op.get_coll_vars(nlp_options, V(P['p', 'ref']), P, Xdot(0.0), model)

    return coll_vars, coll_refs, coll_weights, N_coll


def get_ms_parallel_info(nlp_options, V, P, Xdot, model):
    n_k = nlp_options['n_k']
    N_ms = n_k

    ms_weights = []
    p_weights = get_regularization_weights(model.variables, P, nlp_options).cat
    for ndx in range(n_k):
        ms_weights = cas.horzcat(ms_weights, p_weights)

    ms_vars = struct_op.get_ms_vars(nlp_options, V, P, Xdot, model)
    ms_refs = struct_op.get_ms_vars(nlp_options, V(P['p', 'ref']), P, Xdot, model)

    return ms_vars, ms_refs, ms_weights, N_ms


def find_general_regularisation(nlp_options, V, P, Xdot, model):

    variables = model.variables

    direct_collocation, multiple_shooting, _, _, _ = extract_discretization_info(nlp_options)
    if direct_collocation:
        vars, refs, weights, N_steps = get_coll_parallel_info(nlp_options, V, P, Xdot, model)
    elif multiple_shooting:
        vars, refs, weights, N_steps = get_ms_parallel_info(nlp_options, V, P, Xdot, model)
    parallellization = nlp_options['parallelization']['type']

    reg_costs_fun, reg_costs_dict = get_general_reg_costs_function(nlp_options, variables, V)
    if nlp_options['compile_subfunctions']:
        reg_costs_fun = cf.CachedFunction(nlp_options['compilation_file_name'], reg_costs_fun, do_compile=nlp_options['compile_subfunctions'])

    if nlp_options['parallelization']['map_type'] == 'for-loop':

        reg_costs_list = []
        for k in range(vars.shape[1]):
            reg_costs_list.append(reg_costs_fun(vars[:,k], refs[:,k], weights[:,k]))
        reg_costs = cas.horzcat(*reg_costs_list)

    elif nlp_options['parallelization']['map_type'] == 'map':

        reg_costs_map = reg_costs_fun.map('reg_costs_map', parallellization, N_steps, [], [])
        reg_costs = reg_costs_map(vars, refs, weights)

    summed_reg_costs = cas.sum2(reg_costs)

    idx = 0
    for cost in reg_costs_dict.keys():
        reg_costs_dict[cost] = summed_reg_costs[idx]
        idx += 1

    # initialize component costs
    component_costs = collections.OrderedDict()
    cost_struct = get_costs_struct(V)
    for cost in cost_struct.keys():
        if cost not in reg_costs_dict.keys():
            component_costs[cost] = 0.0
        else:
            component_costs[cost] = reg_costs_dict[cost]

    return component_costs

def find_int_weights(nlp_options):

    nk = nlp_options['n_k']
    d = nlp_options['collocation']['d']
    scheme = nlp_options['collocation']['scheme']
    Collocation = collocation.Collocation(nk, d, scheme)
    int_weights = Collocation.quad_weights

    return int_weights





def find_homotopy_parameter_costs(component_costs, V, P):

    for name in struct_op.subkeys(V, 'phi'):
        component_costs[name + '_cost'] = P['cost', name] * V['phi', name]

    return component_costs


def find_time_cost(nlp_options, V, P):

    time_period = ocp_outputs.find_time_period(nlp_options, V)
    tf_init = ocp_outputs.find_time_period(nlp_options, P.prefix['p', 'ref'])

    time_cost = P['cost', 't_f'] * (time_period - tf_init) * (time_period - tf_init)

    return time_cost


def find_power_cost(nlp_options, model, V, P, Integral_outputs):

    # maximization term for average power
    time_period = ocp_outputs.find_time_period(nlp_options, V)

    if not nlp_options['cost']['output_quadrature']:
        total_energy_scaled = V['x', -1, 'e']
    else:
        total_energy_scaled = Integral_outputs['int_out', -1, 'e']

    average_scaled_power = total_energy_scaled / time_period

    if nlp_options['cost']['P_max']:
        max_power_cost = (1.0 - P['cost', 'P_max']) * V['theta', 'P_max']
        power_cost = P['cost', 'power'] * (-1.) * average_scaled_power + max_power_cost
    elif nlp_options['cost']['PDGA']:
        power_cost = P['cost', 'power'] * (-1.) * average_scaled_power / (V['theta', 'ell_radius']**2)
    else:
        power_cost = P['cost', 'power'] * (-1.) * average_scaled_power

    return power_cost

def find_nominal_landing_cost(V, P, variables, nlp_options):
    pos_weight = nlp_options['landing']['cost']['position_weight']
    vel_weight = nlp_options['landing']['cost']['velocity_weight']

    q_end = {}
    dq_end = {}
    for name in struct_op.subkeys(variables, 'x'):
        if name[0] == 'q':
            q_end[name] = V['x',-1,name]
        elif name[:2] == 'dq':
            dq_end[name] = V['x',-1,name]

    velocity_end = 0.0
    position_end = 0.0
    for position in list(q_end.keys()):
        position_end += cas.mtimes(q_end[position].T,q_end[position])
    position_end *= 1./len(list(q_end.keys()))

    for velocity in list(dq_end.keys()):
        velocity_end += cas.mtimes(dq_end[velocity].T,dq_end[velocity])
    velocity_end *= 1./len(list(dq_end.keys()))

    nominal_landing_cost = P['cost', 'nominal_landing'] * (vel_weight * velocity_end + pos_weight * position_end)

    return nominal_landing_cost


#### problem costs
def find_transition_problem_cost(component_costs, P):

    xdot_regularisation = component_costs['xdot_regularisation_cost']
    u_regularisation = component_costs['u_regularisation_cost']

    transition_cost = xdot_regularisation + u_regularisation
    transition_cost = P['cost', 'transition'] * transition_cost

    return transition_cost


def find_tracking_problem_cost(component_costs, P):

    tracking_cost = component_costs['tracking_cost']
    tracking_problem_cost = tracking_cost

    return tracking_problem_cost


def find_power_problem_cost(component_costs):

    power_cost = component_costs['power_cost']

    power_problem_cost = power_cost

    return power_problem_cost

def find_nominal_landing_problem_cost(nlp_options, V, P, variables):

    nominal_landing_problem_cost = find_nominal_landing_cost(V, P, variables, nlp_options)

    return nominal_landing_problem_cost

def find_general_problem_cost(component_costs):

    u_regularisation_cost = component_costs['u_regularisation_cost']
    xdot_regularisation_cost = component_costs['xdot_regularisation_cost']
    theta_regularisation_cost = component_costs['theta_regularisation_cost']
    beta_cost = component_costs['beta_cost']
    time_cost = component_costs['time_cost']
    fictitious_cost = component_costs['fictitious_cost']

    general_problem_cost = fictitious_cost + u_regularisation_cost + xdot_regularisation_cost + theta_regularisation_cost + beta_cost + time_cost

    return general_problem_cost

def find_homotopy_cost(component_costs):

    gamma_cost = component_costs['gamma_cost']
    iota_cost = component_costs['iota_cost']
    tau_cost = component_costs['tau_cost']
    psi_cost = component_costs['psi_cost']
    eta_cost = component_costs['eta_cost']
    nu_cost = component_costs['nu_cost']
    upsilon_cost = component_costs['upsilon_cost']

    homotopy_cost = psi_cost + iota_cost + tau_cost + gamma_cost + eta_cost + nu_cost + upsilon_cost 

    return homotopy_cost


def find_beta_cost(nlp_options, model, Integral_outputs, P):

    if nlp_options['cost']['beta'] and model.kite_dof == 6:
        beta_cost = Integral_outputs['int_out', -1, 'beta_cost']
        beta_cost = P['cost', 'beta'] * beta_cost / nlp_options['cost']['normalization']['beta']
    else:
        beta_cost = 0

    return beta_cost


###### assemble the objective!

def find_objective(component_costs, V, nlp_options):

    # tracking disappears slowly in the cost function and energy maximising appears. at the final step, cost function
    # contains maximising energy, lift, sosc, and regularisation.

    tracking_problem_cost = component_costs['tracking_problem_cost']
    power_problem_cost = component_costs['power_problem_cost']
    nominal_landing_problem_cost = component_costs['nominal_landing_cost']
    transition_problem_cost = component_costs['tracking_problem_cost']
    general_problem_cost = component_costs['general_problem_cost']
    homotopy_cost = component_costs['homotopy_cost']

    # unpack the sam regularization costs, every key starts with 'SAM_Regularization_...'
    SAM_regularization = 0
    for key in component_costs.keys():
        if key.startswith('SAM_Regularization'):
            SAM_regularization += component_costs[key]

    trajectory_type = nlp_options['trajectory']['type']

    if trajectory_type == 'power_cycle':
        objective = V['phi', 'psi'] * tracking_problem_cost + \
                    (1. - V['phi', 'psi']) * power_problem_cost + \
                    general_problem_cost + \
                    homotopy_cost + \
                    SAM_regularization

    elif trajectory_type in ['transition', 'mpc', 'tracking']:
        objective = V['phi', 'upsilon'] * V['phi', 'nu'] * V['phi', 'eta'] * V['phi', 'psi'] * tracking_problem_cost + \
                    (1. - V['phi', 'psi']) * power_problem_cost + \
                    general_problem_cost + \
                    (1. - V['phi', 'eta']) * nominal_landing_problem_cost + \
                    (1. - V['phi', 'upsilon']) * transition_problem_cost + \
                    homotopy_cost
    else:
        message = 'unrecognized trajectory type (' + trajectory_type + ') requested.'
        print_op.log_and_raise_error(message)

    return objective


##### use the component_cost_dictionary to only do the calculation work once

def get_component_cost_dictionary(nlp_options, V, P, variables, parameters, xdot, Outputs, model, Integral_outputs):

    component_costs = find_general_regularisation(nlp_options, V, P, xdot, model)

    component_costs = find_homotopy_parameter_costs(component_costs, V, P)

    component_costs['time_cost'] = find_time_cost(nlp_options, V, P)
    component_costs['power_cost'] = find_power_cost(nlp_options, model, V, P, Integral_outputs)
    component_costs['nominal_landing_cost'] = find_nominal_landing_problem_cost(nlp_options, V, P, variables)
    component_costs['transition_cost'] = find_transition_problem_cost(component_costs, P)
    component_costs['beta_cost'] = find_beta_cost(nlp_options, model, Integral_outputs, P)
    component_costs['tracking_problem_cost'] = find_tracking_problem_cost(component_costs, P)
    component_costs['power_problem_cost'] = find_power_problem_cost(component_costs)
    component_costs['general_problem_cost'] = find_general_problem_cost(component_costs)
    component_costs['homotopy_cost'] = find_homotopy_cost(component_costs)

    sam_reg_dict = find_SAM_regularization(nlp_options, V, xdot, model)
    # unpack the SAM regularization
    for key, value in sam_reg_dict.items():
        component_costs[f'SAM_Regularization_{key}'] = value

    component_costs['objective'] = find_objective(component_costs, V, nlp_options)

    return component_costs


def find_SAM_regularization(nlp_options: dict, V: cas.struct, Xdot: cas.struct, model: Model) -> Dict[str,Union[ca.SX,float]]:
    """
    Compute the regularization cost to enforce the geometric assumptions of the Stroboscopy Average Method (SAM).
    This consists of penalizing:
        1. the first derivative of the average state
        2. the third derivative of the average state
        3. the third derivative of the algebraic variables (disabled by default)
        4. the similarity timescaling (period) of the micro-integrations
    here, d is the degree of the collocation scheme used for the SAM discretization (number of micro-integrations).

    :param nlp_options: dictionary containing the options of the NLP, i.e. trial.options['nlp']
    :param V: casidi symbolic struct containing the variables of the NLP
    :param model: awebox model objects
    :return: a dictionary of ca.SX for each type of regularization
    """
    if not nlp_options['SAM']['use']:
        return {'NotInUse':0}

    regularization_dict: dict = nlp_options['SAM']['Regularization']

    d_SAM = nlp_options['SAM']['d']
    N_SAM = nlp_options['SAM']['N']

    # add SAM cost: average dynamics should be minimized
    weights_state = model.variables_dict['x'](1E-14)

    # penalize changes is the variables that should not change much
    weights_dicts = regularization_dict['StateWeights']
    # (don't penalize the variables that can change (l_t, d_lt, e))

    # use weights for the correct nodes
    for key in weights_state.keys():
        if key[:-2] in weights_dicts.keys():
            weights_state[key] = weights_dicts[key[:-2]]
    W_x = cas.diag(weights_state.cat)

    from awebox.tools.sam_functionalities import CollocationIRK
    macro_int = CollocationIRK(np.array(cas.collocation_points(d_SAM, nlp_options['SAM']['MaInt_type'])))

    # third derivative of the average state
    DERIVATIVE_T0_REGULARIZE = 3
    V_matrix = cas.horzcat(*V['v_macro_coll'])
    sam_regularizaion_third_deriv_x_average = 0
    for i, c_i in enumerate(macro_int.c):
        # compute the 3rd derivative of the state (2nd derivative of the collocation poly)
        l_i_dot = ca.vertcat([l.deriv(DERIVATIVE_T0_REGULARIZE - 1)(c_i) for l in macro_int.polynomials])
        v_i_dot = V_matrix @ l_i_dot  # the value of the 3rd derivative of the state at the collocation point

        # compute the quadrature of the 3rd derivative of the state
        factor_time = N_SAM/N_SAM**(DERIVATIVE_T0_REGULARIZE*2)
        # factor_time = 1
        # sam_regularizaion_third_deriv_x_average += factor_time * macro_int.b[i] * v_i_dot.T @ W_x @ v_i_dot
        sam_regularizaion_third_deriv_x_average += factor_time * macro_int.b[i] * v_i_dot.T @ v_i_dot

    # third derivative of the of the algebraic variables (micro-integrations)
    sam_regularizaion_third_deriv_z = 0
    SAM_regions = struct_op.calculate_SAM_regions(nlp_options)  # get the indices of the SAM regions
    Z_matrix = cas.horzcat(*[cas.vertcat(*V['x', SAM_regions[i]]) for i in range(d_SAM)])
    W_z = cas.diag(cas.vertcat(*[weights_state.cat for n in range(len(SAM_regions[0]))]))
    for i, c_i in enumerate(macro_int.c):
        # compute the 3rd derivative of the polynomial for the algebraic variables
        l_i_dot = ca.vertcat([l.deriv(DERIVATIVE_T0_REGULARIZE)(c_i) for l in macro_int.polynomials])
        z_i_dot = Z_matrix @ l_i_dot  # value of the 3rd derivative of the algebraic variables at the collocation point

        # compute the quadrature of the squared 3rd derivative of the algebraic variables
        factor_time = N_SAM/N_SAM**(DERIVATIVE_T0_REGULARIZE*2)
        # factor_time = 1
        sam_regularizaion_third_deriv_z +=  factor_time * macro_int.b[i] * z_i_dot.T @ W_z @ z_i_dot

    # first derivative of the state
    sam_regularization_first_deriv_x_average = 0
    for i, c_i in enumerate(macro_int.c):
        factor_time = N_SAM/N_SAM**2
        # factor_time = 1
        sam_regularization_first_deriv_x_average += factor_time * macro_int.b[i] * V['v_macro_coll', i].T @ W_x @ V['v_macro_coll', i]

    # similar durations
    sam_regularization_similar_durations = 0
    tfs_cycles = V['theta', 't_f', 0:-1]
    for i, c_i in enumerate(macro_int.c):
        # compute the 3rd derivative of the polynomial for the algebraic variables
        l_i_dot = ca.vertcat([l.deriv(1)(c_i) for l in macro_int.polynomials])
        T_i_dot = tfs_cycles.T @ l_i_dot  # value of the first derivative of the cycle duration variables at the collocation point

        factor_time = N_SAM/N_SAM**2
        # factor_time = 1
        sam_regularization_similar_durations += factor_time * macro_int.b[i] * T_i_dot**2

    # invariants
    sam_regularization_invariants = 0
    for i in range(d_SAM +1):
        sam_regularization_invariants += V['lam_SAM', i].T @ V['lam_SAM', i] * 1E-8

    return {
        'X_dot': regularization_dict['AverageStateFirstDeriv'] * sam_regularization_first_deriv_x_average,
        'X_dddot': regularization_dict['AverageStateThirdDeriv'] * sam_regularizaion_third_deriv_x_average,
        'Z_dddot': regularization_dict['AverageAlgebraicsThirdDeriv'] * sam_regularizaion_third_deriv_z,
        'T_dot': regularization_dict['SimilarMicroIntegrationDuration'] * sam_regularization_similar_durations,
        'inv': sam_regularization_invariants
    }


def get_component_cost_function(component_costs, V, P):

    component_cost_fun = {}

    for cost in component_costs.keys():
        local_expr = component_costs[cost]
        component_cost_fun[cost + '_fun'] = cas.Function(cost + '_fun', [V, P], [local_expr])

    return component_cost_fun

def get_component_cost_structure(component_costs):

    list_of_entries = []
    for name in list(component_costs.keys()):
        list_of_entries += [cas.entry(name)]

    component_cost_struct = cas.struct_symMX(list_of_entries)

    return component_cost_struct

def get_cost_function_and_structure(nlp_options, V, P, variables, parameters, xdot, Outputs, model, Integral_outputs):

    component_costs = get_component_cost_dictionary(nlp_options, V, P, variables, parameters, xdot, Outputs, model, Integral_outputs)

    component_cost_function = get_component_cost_function(component_costs, V, P)
    component_cost_structure = get_component_cost_structure(component_costs)
    f_fun = make_cost_function(V, P, component_costs)

    return [component_cost_function, component_cost_structure, f_fun]

def make_cost_function(V, P, component_costs):

    f = component_costs['objective']

    # f = cas.sum1(component_costs.cat)
    f_fun = cas.Function('f', [V, P], [f])

    return f_fun

def get_cost_derivatives(V, P, f_fun):

    g = cas.jacobian(f_fun(V, P), V)
    H = cas.jacobian(g, V)
    f_jacobian_fun = cas.Function('f_jacobian', [V, P], [g])
    f_hessian_fun = cas.Function('f_hessian', [V, P], [H])

    return [f_fun, f_jacobian_fun, f_hessian_fun]


def extract_discretization_info(nlp_options):

    if nlp_options['discretization'] == 'direct_collocation':
        direct_collocation = True
        multiple_shooting = False
        d = nlp_options['collocation']['d']
        scheme = nlp_options['collocation']['scheme']
        int_weights = find_int_weights(nlp_options)
    elif nlp_options['discretization'] == 'multiple_shooting':
        direct_collocation = False
        multiple_shooting = True
        d = None
        scheme = None
        int_weights = None

    return direct_collocation, multiple_shooting, d, scheme, int_weights
