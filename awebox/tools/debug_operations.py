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
file to provide debugging operations to the awebox,
_python-3.5 / casadi-3.4.5
- author: thilo bronnenmeyer, jochem de schutter, rachel leuthold, 2017-18
- edit, rachel leuthold 2018-21
'''

import casadi.tools as cas
import matplotlib.pylab as plt
import numpy as np
import numpy.ma as ma
import scipy.linalg as scila

import awebox.tools.vector_operations as vect_op
import awebox.tools.print_operations as print_op


from awebox.logger.logger import Logger as awelogger

def health_check(health_solver_options, nlp, solution, arg, stats, iterations):

    awelogger.logger.info('Checking health...')

    cstr_fun, lam_fun, cstr_labels = collect_equality_and_active_inequality_constraints(health_solver_options, nlp, solution, arg)

    cstr_jacobian_eval = get_jacobian_of_eq_and_active_ineq_constraints(nlp, solution, arg, cstr_fun)

    lagr_fun, lagr_jacobian_fun, lagr_hessian_fun = generate_lagrangian(health_solver_options, nlp, arg, solution)
    lagr_hessian_eval = get_lagr_hessian_eval(nlp, solution, arg, lagr_hessian_fun)

    kkt_matrix = get_kkt_matrix_eval(cstr_jacobian_eval, lagr_hessian_eval)

    reduced_hessian = get_reduced_hessian(health_solver_options, cstr_jacobian_eval, lagr_hessian_eval)

    if health_solver_options['spy_matrices']:
        vect_op.spy(np.absolute(np.array(kkt_matrix)), title='KKT Matrix')
        vect_op.spy(np.absolute(np.array(lagr_hessian_eval)), title='Hessian of the Lagrangian')
        vect_op.spy(np.absolute(np.array(cstr_jacobian_eval)), title='Jacobian of Active Constraints')
        vect_op.spy(np.absolute(np.array(reduced_hessian)), title='Reduced Hessian')
        plt.show()

    tractability = collect_tractability_indicators(stats, iterations, kkt_matrix, reduced_hessian)

    exact_licq_holds = is_matrix_full_rank(cstr_jacobian_eval, health_solver_options, tol=0.)
    if not exact_licq_holds:
        awelogger.logger.info('')
        message = 'linear independent constraint qualification is not satisfied at solution, with an exact computation'
        awelogger.logger.info(message)
        identify_largest_jacobian_entry(cstr_jacobian_eval, health_solver_options, cstr_labels, nlp)
        identify_dependent_constraint(cstr_jacobian_eval, health_solver_options, cstr_labels, nlp)

    licq_holds = is_matrix_full_rank(cstr_jacobian_eval, health_solver_options)
    if not licq_holds:
        awelogger.logger.info('')
        message = 'linear independent constraint qualification appears not to be satisfied at solution, given floating-point tolerance'
        awelogger.logger.error(message)
        identify_largest_jacobian_entry(cstr_jacobian_eval, health_solver_options, cstr_labels, nlp)
        identify_dependent_constraint(cstr_jacobian_eval, health_solver_options, cstr_labels, nlp)

    sosc_holds = is_reduced_hessian_positive_definite(tractability['min_reduced_hessian_eig'], health_solver_options)
    if not sosc_holds:
        awelogger.logger.info('')
        message = 'second order sufficient conditions appear not to be met at solution. please check if all ' \
                  'states/controls/parameters have enough regularization, and if all lifted variables are constrained.'
        awelogger.logger.error(message)

    problem_is_ill_conditioned = is_problem_ill_conditioned(tractability['condition'], health_solver_options)
    if problem_is_ill_conditioned:
        awelogger.logger.info('')
        message = 'problem appears to be ill-conditioned'
        awelogger.logger.info(message)

    problem_is_healthy = (not problem_is_ill_conditioned) and licq_holds and sosc_holds

    awelogger.logger.info('')
    if problem_is_healthy:
        message = 'OCP appears to be healthy'
        awelogger.logger.info(message)

    if not problem_is_healthy:
        identify_largest_kkt_element(kkt_matrix, cstr_labels, nlp)
        identify_smallest_normed_kkt_column(kkt_matrix, cstr_labels, nlp)

        awelogger.logger.info('')
        message = 'OCP appears to be unhealthy'
        awelogger.logger.info(message)

        if health_solver_options['raise_exception']:
            raise Exception(message)

    return problem_is_healthy


def print_cstr_info(cstr_jacobian_eval, cstr_labels, cdx, nlp):
    nonzero_string = get_nonzeros_as_strings(cstr_jacobian_eval, cdx, nlp)
    message = cstr_labels[cdx] + ' -> ' + nonzero_string
    awelogger.logger.info(message)
    return None


def get_nonzeros_as_strings(matrix, cdx, nlp):
    dict = {}
    nonzeros = np.flatnonzero(matrix[cdx, :])
    for ndx in nonzeros:

        var = nlp.V.labels()[ndx]
        value = '{:.2e}'.format(float(matrix[cdx, ndx]))
        dict[var] = value

    return repr(dict)

def collect_tractability_indicators(stats, iterations, kkt_matrix, reduced_hessian):

    awelogger.logger.info('collect tractability indicators...')
    tractability = {}

    # todo: add autoscaling_triggered? indicator
    tractability['local_iterations'] = get_local_iterations(stats)
    tractability['total_iterations'] = get_total_iterations(iterations)
    tractability['diagonality'] = get_pearson_diagonality(kkt_matrix)
    tractability['size'] = kkt_matrix.shape[0]
    tractability['condition'] = get_condition_number(kkt_matrix)
    tractability['min_reduced_hessian_eig'] = get_min_reduced_hessian_eigenvalue(reduced_hessian)

    awelogger.logger.info('tractability indicator report')
    print_op.print_dict_as_table(tractability)

    return tractability

def was_autoscale_triggered(stats):

    unscaled_obj = stats['iterations']['obj'][-1]
    scaled_dual_infeasibility = stats['iterations']['inf_du'][-1]
    unscaled_constraint_violation = stats['iterations']['inf_pr'][-1]

    maybe = 0.5
    return maybe

def get_local_iterations(stats):
    awelogger.logger.info('get local iterations...')
    return stats['iter_count']

def get_total_iterations(iterations):
    awelogger.logger.info('get total iterations...')
    total_iterations = 0.
    for step in iterations.keys():
        total_iterations += iterations[step]
    return total_iterations

def get_pearson_diagonality(kkt_matrix):

    awelogger.logger.info('compute Pearson diagonality...')

    matrA = cas.fabs(kkt_matrix)

    shape_dglt = (1, matrA.shape[0])
    j_dglt = np.ones(shape_dglt)
    r_dglt = np.reshape(np.arange(1., matrA.shape[0] + 1), shape_dglt)
    r2_dglt = np.square(r_dglt)

    # do this with casadi, not numpy, so that sparse zeros simplify memory use
    n_dglt = cas.mtimes(j_dglt, cas.mtimes(matrA, j_dglt.T))
    sum_x_dglt = cas.mtimes(r_dglt, cas.mtimes(matrA, j_dglt.T))
    sum_y_dglt = cas.mtimes(j_dglt, cas.mtimes(matrA, r_dglt.T))
    sum_x2_dglt = cas.mtimes(r2_dglt, cas.mtimes(matrA, j_dglt.T))
    sum_y2_dglt = cas.mtimes(j_dglt, cas.mtimes(matrA, r2_dglt.T))
    sum_xy_dglt = cas.mtimes(r_dglt, cas.mtimes(matrA, r_dglt.T))
    dglt_num = n_dglt * sum_xy_dglt - sum_x_dglt * sum_y_dglt
    dglt_den_x = cas.sqrt(n_dglt * sum_x2_dglt - sum_x_dglt ** 2.)
    dglt_den_y = cas.sqrt(n_dglt * sum_y2_dglt - sum_y_dglt ** 2.)
    dglt = float(dglt_num / dglt_den_x / dglt_den_y)

    return dglt

def get_condition_number(kkt_matrix):
    awelogger.logger.info('compute kkt matrix condition number...')

    cond_number = np.linalg.cond(np.array(kkt_matrix))
    return cond_number

def get_min_reduced_hessian_eigenvalue(reduced_hessian):
    awelogger.logger.info('get minimum reduced hessian eigenvalue...')

    if not reduced_hessian.shape == (0, 0):
        eigenvalues, _ = np.linalg.eig(np.array(reduced_hessian))
        return np.min(eigenvalues)
    else:
        return []





####### boolean tests

def is_reduced_hessian_positive_definite(min_reduced_hessian_eigenvalue, health_solver_options):
    reduced_hessian_eig_thesh = health_solver_options['thresh']['reduced_hessian_eig']
    return min_reduced_hessian_eigenvalue > reduced_hessian_eig_thesh


def is_matrix_full_rank(matrix, health_solver_options, tol=None):

    if tol is None:
        rank_tolerance = health_solver_options['tol']['constraint_jacobian_rank']
    else:
        rank_tolerance = tol

    matrix_rank = np.linalg.matrix_rank(matrix, rank_tolerance)
    required_rank = np.min(matrix.shape)

    return (required_rank == matrix_rank)


def identify_largest_jacobian_entry(cstr_jacobian_eval, health_solver_options, cstr_labels, nlp):
    message = '... largest absolute jacobian entry occurs at: '
    awelogger.logger.info(message)
    max_cdx = np.where(np.absolute(cstr_jacobian_eval) == np.amax(np.absolute(cstr_jacobian_eval)))[0][0]
    print_cstr_info(cstr_jacobian_eval, cstr_labels, max_cdx, nlp)
    return None

def identify_dependent_constraint(cstr_jacobian_eval, health_solver_options, cstr_labels, nlp):
    message = '... possible (floating-point) dependent constraints include: '
    awelogger.logger.info(message)

    search_dir = {}
    search_dir['forwards'] = {'cje': cstr_jacobian_eval, 'labels': cstr_labels}
    search_dir['backwards'] = {'cje': cstr_jacobian_eval[::-1], 'labels': cstr_labels[::-1]}

    for direction in search_dir.keys():

        local_cje = search_dir[direction]['cje']
        local_labels = search_dir[direction]['labels']

        while not is_matrix_full_rank(local_cje, health_solver_options):

            number_constraints = local_cje.shape[0]
            current_hunt = True

            for cdx in range(number_constraints):

                if current_hunt:

                    prev_matrix = local_cje[cdx:, :]
                    current_matrix = prev_matrix[1:, :]

                    prev_full_rank = is_matrix_full_rank(prev_matrix, health_solver_options)
                    current_full_rank = is_matrix_full_rank(current_matrix, health_solver_options)

                    if current_full_rank and (not prev_full_rank):
                        print_cstr_info(local_cje, local_labels, cdx, nlp)
                        local_cje, local_labels = pop_cstr_and_label(cdx, local_cje, local_labels)
                        current_hunt = False


    return None

def pop_cstr_and_label(cdx, local_cje, local_labels):

    number_constraints = local_cje.shape[0]

    if cdx == 0:
        local_cje = local_cje[1:, :]
        local_labels = local_labels[1:]

    elif cdx == number_constraints - 1:
        local_cje = local_cje[:-1, :]
        local_labels = local_labels[:-1]

    else:
        local_cje = cas.vertcat(local_cje[:cdx, :], local_cje[cdx + 1:, :])
        local_labels = local_labels[:cdx] + local_labels[cdx + 1:]

    return local_cje, local_labels


def is_problem_ill_conditioned(condition_number, health_solver_options):
    tol = health_solver_options['thresh']['condition_number']
    is_ill_conditioned = condition_number > tol
    return is_ill_conditioned




######## spy


def identify_smallest_normed_kkt_column(kkt_matrix, cstr_labels, nlp):

    smallest_norm = 1.e10
    smallest_idx = -1

    for idx in range(kkt_matrix.shape[1]):
        local_norm = vect_op.norm(kkt_matrix[:, idx])

        if local_norm < smallest_norm:
            smallest_idx = idx
            smallest_norm = float(local_norm)

    message = '... KKT column (' + str(smallest_idx) + ') with the smallest norm (' + str(smallest_norm) + ') is associated with:'
    awelogger.logger.info(message)

    number_variables = nlp.V.cat.shape[0]
    if smallest_idx < number_variables:
        relevant_variable_index = smallest_idx
        relevant_variable = nlp.V.getCanonicalIndex(relevant_variable_index)
        message = '{:>10}: {:>15} '.format('column', 'variable') + str(relevant_variable)
    else:
        relevant_multiplier_index = smallest_idx - number_variables
        relevant_multiplier = cstr_labels[relevant_multiplier_index]
        message = '{:>10}: {:>15} '.format('column', 'multiplier') + str(relevant_multiplier)
    awelogger.logger.info(message)

    return None


def identify_largest_kkt_element(kkt_matrix, cstr_labels, nlp):
    matrA = np.absolute(np.array(kkt_matrix))

    max_val = np.max(matrA)
    ind = np.unravel_index(np.argmax(matrA, axis=None), matrA.shape)

    associated_row = ind[0]
    associated_column = ind[1]

    number_variables = nlp.V.cat.shape[0]

    message = '... largest (absolute value sense) KKT matrix entry (' + str(max_val) + ') is associated with:'
    awelogger.logger.info(message)

    if associated_column < number_variables:
        relevant_variable_index = associated_column
        relevant_variable = nlp.V.getCanonicalIndex(relevant_variable_index)
        message = '{:>10}: {:>15} '.format('column', 'variable') + str(relevant_variable)
    else:
        relevant_multiplier_index = associated_column - number_variables
        relevant_multiplier = cstr_labels[relevant_multiplier_index]
        message = '{:>10}: {:>15} '.format('column', 'multiplier') + str(relevant_multiplier)
    awelogger.logger.info(message)


    if associated_row < number_variables:
        associated_variable_index = associated_row
        associated_variable = nlp.V.getCanonicalIndex(associated_variable_index)
        message = '{:>10}: {:>15} '.format('row', 'variable') + str(associated_variable)
    else:
        associated_constraint_index = associated_row - number_variables
        associated_constraint = cstr_labels[associated_constraint_index]
        message = '{:>10}: {:>15} '.format('row', 'constraint') + str(associated_constraint)
    awelogger.logger.info(message)

    return None


##### compute base elements

def get_reduced_hessian(health_solver_options, cstr_jacobian_eval, lagr_hessian_eval):
    awelogger.logger.info('compute reduced hessian...')

    null_tolerance = health_solver_options['tol']['reduced_hessian_null']
    null = scila.null_space(cstr_jacobian_eval, null_tolerance)

    reduced_hessian = cas.mtimes(cas.mtimes(null.T, lagr_hessian_eval), null)

    return reduced_hessian


def get_jacobian_of_eq_and_active_ineq_constraints(nlp, solution, arg, cstr_fun):
    awelogger.logger.info('compute jacobian of equality and active inequality constraints...')

    V = nlp.V
    P = nlp.P

    p_fix_num = nlp.P(arg['p'])

    relevant_constraints = cstr_fun(V, P, arg['lbx'], arg['ubx'])
    # relevant_constraints_eval = np.array(cstr_fun(V(solution['x']), p_fix_num, arg['lbx'], arg['ubx']))

    cstr_jacobian = cas.jacobian(relevant_constraints, V)
    cstr_jacobian_fun = cas.Function('cstr_jacobian_fun', [V, P], [cstr_jacobian])

    cstr_jacobian_eval = np.array(cstr_jacobian_fun(V(solution['x']), p_fix_num))

    return cstr_jacobian_eval


def get_lagr_hessian_eval(nlp, solution, arg, lagr_hessian_fun):
    awelogger.logger.info('evaluate hessian of the lagrangian...')

    V_opt = nlp.V(solution['x'])
    lambda_g = solution['lam_g']
    lambda_x = solution['lam_x']
    p_fix_num = nlp.P(arg['p'])
    lagr_hessian_eval = lagr_hessian_fun(V_opt, p_fix_num, lambda_g, lambda_x)
    return lagr_hessian_eval

def get_kkt_matrix_eval(cstr_jacobian_eval, lagr_hessian_eval):
    awelogger.logger.info('evaluate the kkt matrix...')

    lagr_block = lagr_hessian_eval
    constraint_block = cstr_jacobian_eval

    zeros_block = np.zeros((constraint_block.shape[0], constraint_block.shape[0]))

    # kkt_matrix =
    # nabla^2 lagr  nabla g^top     nabla hA^top
    # nabla g           0               0
    # nabla hA          0               0

    top_row = cas.horzcat(lagr_block, constraint_block.T)
    bottom_row = cas.horzcat(constraint_block, zeros_block)
    kkt_matrix = cas.vertcat(top_row, bottom_row)

    return kkt_matrix

def generate_lagrangian(health_solver_options, nlp, arg, solution):
    awelogger.logger.info('compute the lagrangian...')

    vars_sym = cas.SX.sym('vars_sym', nlp.V.shape)
    p_sym = cas.SX.sym('p_sym', nlp.P.shape)

    lam_x_sym = cas.SX.sym('lam_x_sym', nlp.V.shape)

    constraints = nlp.g_fun(vars_sym, p_sym)
    lam_g_sym = cas.SX.sym('lam_g_sym', constraints.shape)

    var_constraint_functions = collect_var_constraints(health_solver_options, nlp, arg, solution)
    rel_fun = var_constraint_functions['rel_fun']
    rel_lam_fun = var_constraint_functions['rel_lam_fun']

    bounds = rel_fun(vars_sym, arg['lbx'], arg['ubx'])
    lam_x_bounds = rel_lam_fun(lam_x_sym)

    f_fun = nlp.f_fun

    objective = f_fun(vars_sym, p_sym)

    lagrangian = objective + cas.mtimes(lam_g_sym.T, constraints) + cas.mtimes(lam_x_bounds.T, bounds)
    [lagr_hessian, lagr_jacobian] = cas.hessian(lagrangian, vars_sym)

    lagr_fun = cas.Function('lagr_fun', [vars_sym, p_sym, lam_g_sym, lam_x_sym], [lagrangian])
    lagr_jacobian_fun = cas.Function('lagr_jacobian_fun', [vars_sym, p_sym, lam_g_sym, lam_x_sym], [lagr_jacobian])
    lagr_hessian_fun = cas.Function('lagr_hessian_fun', [vars_sym, p_sym, lam_g_sym, lam_x_sym], [lagr_hessian])

    return lagr_fun, lagr_jacobian_fun, lagr_hessian_fun

def collect_equality_and_active_inequality_constraints(health_solver_options, nlp, solution, arg):
    awelogger.logger.info('collect the equality and active inequality constraints...')

    var_sym = cas.SX.sym('var_sym', nlp.V.shape)
    p_sym = cas.SX.sym('p_sym', nlp.P.shape)
    ubx_sym = cas.SX.sym('ubx_sym', arg['ubx'].shape)
    lbx_sym = cas.SX.sym('lbx_sym', arg['lbx'].shape)

    lam_x_sym = cas.SX.sym('lam_x_sym', solution['lam_x'].shape)
    lam_g_sym = cas.SX.sym('lam_g_sym', solution['lam_g'].shape)

    p_fix_num = nlp.P(arg['p'])

    _, eq_labels, eq_fun = collect_equality_constraints(nlp)
    equality_constraints = eq_fun(nlp.g_fun(var_sym, p_sym))
    equality_lambdas = eq_fun(lam_g_sym)

    _, active_ineq_labels, active_fun = collect_active_inequality_constraints(health_solver_options, nlp, solution, p_fix_num)
    active_inequality_constraints = active_fun(nlp.g_fun(var_sym, p_sym))
    active_inequality_lambdas = active_fun(lam_g_sym)

    var_constraint_functions = collect_var_constraints(health_solver_options, nlp, arg, solution)
    all_active_var_bounds = var_constraint_functions['all_act_fun'](var_sym, lbx_sym, ubx_sym)
    all_active_var_lambdas = var_constraint_functions['all_act_lam_fun'](lam_x_sym)
    all_active_var_labels = var_constraint_functions['all_act_labels']

    stacked_constraints = cas.vertcat(equality_constraints, active_inequality_constraints, all_active_var_bounds)
    stacked_cstr_fun = cas.Function('stacked_cstr_fun', [var_sym, p_sym, lbx_sym, ubx_sym], [stacked_constraints])

    stacked_lambdas = cas.vertcat(equality_lambdas, active_inequality_lambdas, all_active_var_lambdas)
    stacked_lam_fun = cas.Function('stacked_lam_fun', [lam_x_sym, lam_g_sym], [stacked_lambdas])

    stacked_labels = eq_labels + active_ineq_labels + all_active_var_labels

    return stacked_cstr_fun, stacked_lam_fun, stacked_labels

def collect_type_constraints(nlp, cstr_type):

    found_cstrs = []
    found_names = []
    found_syms = []

    # list the evaluated constraints at solution
    ocp_cstr_list = nlp.ocp_cstr_list

    name_list = ocp_cstr_list.get_name_list('all')
    g = ocp_cstr_list.get_expression_list('all')
    g_sym = cas.SX.sym('g_sym', g.shape)

    for cstr in ocp_cstr_list.get_list('all'):
        local_name = cstr.name
        if cstr.cstr_type == cstr_type:

            indices = [idx for idx,name in enumerate(name_list) if name == local_name]

            for idx in indices:
                found_cstrs = cas.vertcat(found_cstrs, g[idx])
                found_syms = cas.vertcat(found_syms, g_sym[idx])
                found_names += [local_name]

    cstr_fun = cas.Function('cstr_fun', [g_sym], [found_syms])

    return found_cstrs, found_names, cstr_fun

def collect_equality_constraints(nlp):
    return collect_type_constraints(nlp, 'eq')

def collect_inequality_constraints(nlp):
    return collect_type_constraints(nlp, 'ineq')

def collect_active_inequality_constraints(health_solver_options, nlp, solution, p_fix_num):

    active_threshold = health_solver_options['thresh']['active']
    v_vals = solution['x']

    active_constraints = []
    list_names = []
    active_sym = []

    [g_ineq, g_ineq_names, ineq_fun] = collect_inequality_constraints(nlp)

    # list the evaluated constraints at solution
    ocp_cstr_list = nlp.ocp_cstr_list

    g = nlp.g
    g_fun = nlp.g_fun
    g_vals = g_fun(v_vals, p_fix_num)
    g_sym = cas.SX.sym('g_sym', g.shape)
    g_names = ocp_cstr_list.get_name_list('all')

    # list the multipliers lambda at solution
    lam_vals = solution['lam_g']

    g_ineq_vals = ineq_fun(g_vals)
    lambda_ineq_vals = ineq_fun(lam_vals)
    g_ineq_sym = ineq_fun(g_sym)

    if not g_ineq_sym.shape[0] == 0:
        for gdx in range(g_ineq.shape[0]):

            local_g = g_ineq_vals[gdx]
            local_lam = lambda_ineq_vals[gdx]
            local_name = g_ineq_names[gdx]

            # if eval_constraint is small, then constraint is active. or.
            # if lambda >> eval_constraint, then: constraint is active
            if local_lam**2. > (active_threshold * local_g)**2.:

                # append active constraints to active_list
                active_constraints = cas.vertcat(active_constraints, local_g)

                list_names += [local_name]
                active_sym = cas.vertcat(active_sym, g_ineq_sym[gdx])

    active_fun = cas.Function('active_fun', [g_sym], [active_sym])

    # return active_list
    return active_constraints, list_names, active_fun

def collect_var_constraints(health_solver_options, nlp, arg, solution):

    active_threshold = health_solver_options['thresh']['active']

    lam = solution['lam_x']
    lbx = arg['lbx']
    ubx = arg['ubx']
    var = solution['x']

    lam_sym = cas.SX.sym('lam_sym', lam.shape)
    lbx_sym = cas.SX.sym('lbx_sym', lbx.shape)
    ubx_sym = cas.SX.sym('ubx_sym', ubx.shape)
    var_sym = cas.SX.sym('var_sym', var.shape)

    relevant_cstr = []
    relevant_lambdas = []
    relevant_vars = []
    relevant_labels = []

    equality_cstr = []
    equality_lambdas = []
    equality_vars = []
    equality_labels = []

    inequality_cstr = []
    inequality_lambdas = []
    inequality_vars = []
    inequality_labels = []

    active_cstr = []
    active_lambdas = []
    active_vars = []
    active_labels = []

    for idx in range(var.shape[0]):

        lam_idx = np.float(lam[idx])
        lbx_idx = np.float(lbx.cat[idx])
        ubx_idx = np.float(ubx.cat[idx])
        var_idx = np.float(var[idx])

        name_list_strings = list(map(str, nlp.V.getCanonicalIndex(idx)))
        name_underscore = [name + '_' for name in name_list_strings[:-1]] + [name_list_strings[-1]]
        name_idx = ''.join(name_underscore)

        # constraints are written in negative convention (as normal)
        lb_cstr = lbx_sym[idx] - var_sym[idx]
        ub_cstr = var_sym[idx] - ubx_sym[idx]
        lam_cstr = lam_sym[idx]
        var_cstr = var_sym[idx]

        if lam_idx == 0:
            # either there are no bounds, or equality constraints

            if lbx_idx == -cas.inf and ubx_idx == cas.inf:
                # var is not bounded
                # do not add constraint to relevant list -> do nothing
                32.0

            if ubx_idx == lbx_idx:
                # equality constraint
                # default to upper bound
                equality_cstr = cas.vertcat(equality_cstr, ub_cstr)
                equality_lambdas = cas.vertcat(equality_lambdas, lam_cstr)
                equality_vars = cas.vertcat(equality_vars, var_cstr)
                equality_labels += [name_idx]

                relevant_cstr = cas.vertcat(relevant_cstr, ub_cstr)
                relevant_lambdas = cas.vertcat(relevant_lambdas, lam_cstr)
                relevant_vars = cas.vertcat(relevant_vars, var_cstr)
                relevant_labels += [name_idx]

        else:
            # inequality constraints

            if lam_idx < 0:
                # the lower bound is the relevant bound
                cstr_here = lb_cstr
                lam_cstr_here = -1 * lam_cstr
                lam_idx_here = -1. * lam_idx
                eval_here = lbx_idx - var_idx

            else:
                # lam_idx > 0:
                # the upper bound is the relevant bound
                cstr_here = ub_cstr
                lam_cstr_here = lam_cstr
                lam_idx_here = lam_idx
                eval_here = var_idx - ubx_idx

            inequality_cstr = cas.vertcat(inequality_cstr, cstr_here)
            inequality_lambdas = cas.vertcat(inequality_lambdas, lam_cstr_here)
            inequality_vars = cas.vertcat(inequality_vars, var_cstr)
            inequality_labels += [name_idx]

            relevant_cstr = cas.vertcat(relevant_cstr, cstr_here)
            relevant_lambdas = cas.vertcat(relevant_lambdas, lam_cstr_here)
            relevant_vars = cas.vertcat(relevant_vars, var_cstr)
            relevant_labels += [name_idx]

            if lam_idx_here ** 2. > (active_threshold * eval_here) ** 2.:
                # this inequality constraint is active
                # because the constraint is "approximately" zero

                active_cstr = cas.vertcat(active_cstr, cstr_here)
                active_lambdas = cas.vertcat(active_lambdas, lam_cstr_here)
                active_vars = cas.vertcat(active_vars, var_cstr)
                active_labels += [name_idx]

    var_constraint_functions = {}

    var_constraint_functions['eq_fun'] = cas.Function('eq_fun', [var_sym, lbx_sym, ubx_sym], [equality_cstr])
    var_constraint_functions['eq_lam_fun'] = cas.Function('eq_lam_fun', [lam_sym], [equality_lambdas])
    var_constraint_functions['eq_vars_fun'] = cas.Function('eq_vars_fun', [var_sym], [equality_vars])
    var_constraint_functions['eq_labels'] = equality_labels

    var_constraint_functions['ineq_fun'] = cas.Function('ineq_fun', [var_sym, lbx_sym, ubx_sym], [inequality_cstr])
    var_constraint_functions['ineq_lam_fun'] = cas.Function('ineq_lam_fun', [lam_sym], [inequality_lambdas])
    var_constraint_functions['ineq_vars_fun'] = cas.Function('ineq_vars_fun', [var_sym], [inequality_vars])
    var_constraint_functions['ineq_labels'] = inequality_labels

    var_constraint_functions['act_ineq_fun'] = cas.Function('act_ineq_fun', [var_sym, lbx_sym, ubx_sym], [active_cstr])
    var_constraint_functions['act_ineq_lam_fun'] = cas.Function('act_ineq_lam_fun', [lam_sym], [active_lambdas])
    var_constraint_functions['act_ineq_vars_fun'] = cas.Function('act_ineq_vars_fun', [var_sym], [active_vars])
    var_constraint_functions['act_ineq_labels'] = active_labels

    var_constraint_functions['all_act_fun'] = cas.Function('all_act_fun', [var_sym, lbx_sym, ubx_sym], [cas.vertcat(equality_cstr, active_cstr)])
    var_constraint_functions['all_act_lam_fun'] = cas.Function('all_act_lam_fun', [lam_sym], [cas.vertcat(equality_lambdas, active_lambdas)])
    var_constraint_functions['all_act_vars_fun'] = cas.Function('all_act_vars_fun', [var_sym], [cas.vertcat(equality_vars, active_vars)])
    var_constraint_functions['all_act_labels'] = equality_labels + active_labels

    var_constraint_functions['rel_fun'] = cas.Function('rel_fun', [var_sym, lbx_sym, ubx_sym], [relevant_cstr])
    var_constraint_functions['rel_lam_fun'] = cas.Function('rel_lam_fun', [lam_sym], [relevant_lambdas])
    var_constraint_functions['rel_vars_fun'] = cas.Function('vars_fun', [var_sym], [relevant_vars])
    var_constraint_functions['rel_labels'] = relevant_labels

    return var_constraint_functions