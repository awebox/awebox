#!/usr/bin/python3
"""Test to check objective functionality
@author: rachel leuthold, 2023
"""


import awebox as awe
import logging
import casadi as cas
import awebox.tools.print_operations as print_op
import awebox.ocp.objective as objective

logging.basicConfig(filemode='w', format='%(levelname)s:    %(message)s', level=logging.WARNING)


def build_trial(with_additional_exceptions=False, weight_of_var=None):

    var_to_find, var_type, usual_cost_type = get_relevant_variable_info()
    usual_cost_type_without_cost = usual_cost_type[:-5]

    # single kite with point-mass model
    options = {}
    options['user_options.system_model.architecture'] = {1:0, 2:1, 3:1}
    options['user_options.system_model.kite_dof'] = 3
    options['user_options.kite_standard'] = awe.ampyx_data.data_dict()
    options['user_options.trajectory.type'] = 'power_cycle'
    options['user_options.trajectory.system_type'] = 'lift_mode'
    options['nlp.n_k'] = 5

    options['solver.max_iter'] = 1.
    options['solver.homotopy_method.advance_despite_max_iter'] = False

    if weight_of_var is not None:
        options['solver.cost.' + usual_cost_type_without_cost + '.0'] = 1.
        options['solver.weights.l_t'] = weight_of_var

    # don't include induction effects, use trivial tether drag
    options['user_options.induction_model'] = 'not_in_use'
    options['user_options.tether_drag_model'] = 'split'

    if with_additional_exceptions:
        reassigned_cost_type = get_reassigned_cost_type()
        options['nlp.cost.adjustments_to_general_regularization_distribution'] = [(var_type, var_to_find, reassigned_cost_type)]

    # build model
    trial = awe.Trial(options, 'name')
    trial.build()

    return trial


def get_relevant_variable_info():
    var_to_find = 'l_t'
    var_type = 'x'
    usual_cost_type = 'tracking_cost'
    return var_to_find, var_type, usual_cost_type


def get_reassigned_cost_type():
    return 'u_regularisation_cost'


def we_expect_shooting_nodes_to_also_be_weighted(trial):
    u_param = trial.options['nlp']['collocation']['u_param']
    nlp_discretization = trial.options['nlp']['discretization']

     # for direct_collocation true-false, see ocp/discretization (# compute outputs for this time interval)
    if (nlp_discretization == 'multiple_shooting'):
        return True
    elif (nlp_discretization == 'direct_collocation') and (u_param == 'zoh'):
        return False
    elif (nlp_discretization == 'direct_collocation') and (u_param == 'poly'):
        return True
    else:
        message = 'unexpected discretization type'
        print_op.log_and_raise_error(message)

    return None


def get_the_indices_of_the_relevant_variable_instances(V, var_type, var_to_find, include_shooting_nodes=False):
    relevant_indices = []
    for vdx in range(V.cat.shape[0]):
        if ',' + var_to_find + ',0' in V.labels()[vdx]:
            this_is_a_shooting_node = V.labels()[vdx][:3] == "[" + var_type + ","
            if include_shooting_nodes or not this_is_a_shooting_node:
                relevant_indices += [vdx]
    return relevant_indices


def run_generalized_regularization_mechanism_test(with_additional_exceptions=False, epsilon=1.e-4):

    trial = build_trial(with_additional_exceptions)
    nlp_options = trial.options['nlp']
    V = trial.nlp.V
    P = trial.nlp.P(1.)
    Xdot = trial.nlp.Xdot(0.)
    model = trial.model

    include_shooting_nodes = we_expect_shooting_nodes_to_also_be_weighted(trial)

    component_costs = objective.find_general_regularisation(nlp_options, V, P, Xdot, model)

    var_to_find, var_type, usual_cost_type = get_relevant_variable_info()
    relevant_indices = get_the_indices_of_the_relevant_variable_instances(V, var_type, var_to_find, include_shooting_nodes=include_shooting_nodes)

    if not with_additional_exceptions:
        cost_type_where_we_expect_to_find_the_var_regularization = usual_cost_type
        cost_type_where_we_dont_expect_to_find_the_var_regularization = get_reassigned_cost_type()
    else:
        cost_type_where_we_expect_to_find_the_var_regularization = get_reassigned_cost_type()
        cost_type_where_we_dont_expect_to_find_the_var_regularization = usual_cost_type

    expected_cost = component_costs[cost_type_where_we_expect_to_find_the_var_regularization]
    expected_cost_jacobian = cas.jacobian(expected_cost, V)
    expected_cost_jacobian_fun = cas.Function('expected_cost_jacobian_fun', [V], [expected_cost_jacobian])
    expected_cost_jacobian_eval = expected_cost_jacobian_fun(V(2.))

    unexpected_cost = component_costs[cost_type_where_we_dont_expect_to_find_the_var_regularization]
    unexpected_cost_jacobian = cas.jacobian(unexpected_cost, V)
    unexpected_cost_jacobian_fun = cas.Function('unexpected_cost_jacobian_fun', [V], [unexpected_cost_jacobian])
    unexpected_cost_jacobian_eval = unexpected_cost_jacobian_fun(V(2.))

    zeros_in_expected_jacobian = [expected_cost_jacobian_eval[0, idx]**2. < epsilon**2. for idx in relevant_indices]
    zeros_in_unexpected_jacobian = [unexpected_cost_jacobian_eval[0, idx]**2. < epsilon**2. for idx in relevant_indices]

    expected_contains_zero_terms = any(zeros_in_expected_jacobian)
    if expected_contains_zero_terms:
        message = 'a state (' + var_to_find + ') that was expected to be regularized in the ' + cost_type_where_we_expect_to_find_the_var_regularization + ' has zeros at the following variables:'
        for rdx in range(len(relevant_indices)):
            if zeros_in_expected_jacobian[rdx]:
                message += " " + V.labels()[relevant_indices[rdx]]
        print_op.log_and_raise_error(message)

    unexpected_contains_nonzero_terms = not all(zeros_in_unexpected_jacobian)
    if unexpected_contains_nonzero_terms:
        message = 'a state (' + var_to_find + ') that was NOT expected to be regularized in the ' + cost_type_where_we_dont_expect_to_find_the_var_regularization + ' has zeros at the following variables:'
        for rdx in range(len(relevant_indices)):
            if not zeros_in_unexpected_jacobian[rdx]:
                message += " " + V.labels()[relevant_indices[rdx]]
        print_op.log_and_raise_error(message)

    return None


def test_that_cost_values_are_not_all_zero_in_default_problem(epsilon=1.e-4):
    options = {}
    options['user_options.kite_standard'] = awe.ampyx_data.data_dict()
    options['solver.max_iter'] = 1.
    options['solver.homotopy_method.advance_despite_max_iter'] = False
    options['nlp.n_k'] = 10

    trial = awe.Trial(options, 'name')
    trial.build()
    trial.optimize(final_homotopy_step='initial')
    P = trial.optimization.p_fix_num

    cost_weights = P['cost']

    resi = cas.mtimes(cost_weights.T, cost_weights)
    if resi < epsilon**2:
        message = 'something went wrong when setting the objective cost_terms: despite being in the initial homotopy step, they seem to all be zero.'
        print_op.log_and_raise_error(message)

    return None


def test_that_changing_the_weights_on_a_regularized_variable_influences_the_objective_predictably(epsilon=1.e-4):

    first_weight = 1.0
    factor = 17.3
    second_weight = factor * first_weight
    var_name, var_type, _ = get_relevant_variable_info()

    trial_1 = build_trial(weight_of_var=first_weight)
    trial_1.optimize(final_homotopy_step='initial')

    trial_2 = build_trial(weight_of_var=second_weight)
    trial_2.optimize(final_homotopy_step='initial')

    include_shooting_nodes = we_expect_shooting_nodes_to_also_be_weighted(trial_1)

    V = trial_1.nlp.V(2.)
    P1 = trial_1.optimization.p_fix_num
    P2 = trial_2.optimization.p_fix_num

    weights_diff = P1['p', 'weights', var_type, var_name] - P2['p', 'weights', var_type, var_name]
    if (not first_weight == second_weight) and (cas.mtimes(weights_diff.T, weights_diff) < epsilon**2.):
        message = 'something went wrong when passing the weights into the parameters. despite selection different weights in the nlp_options, the nlp parameters are the same'
        print_op.log_and_raise_error(message)

    [_, _, f_hessian_fun_1] = trial_1.nlp.get_f_jacobian_and_hessian_functions()
    [_, _, f_hessian_fun_2] = trial_2.nlp.get_f_jacobian_and_hessian_functions()

    f_hessian_1 = f_hessian_fun_1(V, P1)
    f_hessian_2 = f_hessian_fun_2(V, P2)

    hessian_sanity_check = (f_hessian_1.shape == f_hessian_2.shape) and (f_hessian_1.shape[0] == V.shape[0])
    if not hessian_sanity_check:
        message_hessian_sanity = 'something went wrong when re-extracting the weights from the objective. the hessian does not have the expected shape'
        print_op.log_and_raise_error(message_hessian_sanity)

    relevant_indices = get_the_indices_of_the_relevant_variable_instances(V, var_type, var_name, include_shooting_nodes=include_shooting_nodes)

    second_objective_term_relates_predictably_to_first_objective_term = True
    error_message = 'something went wrong with the weighting in a regularization-type objective term \n'
    error_message += 'unpredictable behavior exists at variables: \n'
    for idx in relevant_indices:
        local_term_1 = f_hessian_1[idx, idx]
        local_term_2 = f_hessian_2[idx, idx]

        trivial_condition = (first_weight**2. > epsilon**2.) and (local_term_1**2. < epsilon**2.)
        unpredictability_condition = (local_term_2 - local_term_1 * factor)**2. > epsilon**2.
        if trivial_condition or unpredictability_condition:
            local_name = V.labels()[idx]
            error_message += str(idx) + " " + local_name + " (value 1: {: 0.4G}, value 2: {: 0.4G}, expected value 2: {: 0.4G}".format(float(local_term_1), float(local_term_2), float(factor * local_term_1))
            error_message += '\n'
            second_objective_term_relates_predictably_to_first_objective_term = False

    if not second_objective_term_relates_predictably_to_first_objective_term:
        print_op.log_and_raise_error(error_message)

    return None


def test_generalized_regularization_mechanism_without_additional_exceptions():
    run_generalized_regularization_mechanism_test(with_additional_exceptions=False)
    return None


def test_generalized_regularization_mechanism_with_exception_mechanism():
    run_generalized_regularization_mechanism_test(with_additional_exceptions=True)
    return None

if __name__ == "__main__":
    test_generalized_regularization_mechanism_without_additional_exceptions()
    test_generalized_regularization_mechanism_with_exception_mechanism()
    test_that_cost_values_are_not_all_zero_in_default_problem()
    test_that_changing_the_weights_on_a_regularized_variable_influences_the_objective_predictably()
