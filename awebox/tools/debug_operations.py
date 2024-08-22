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
import copy
import datetime


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import casadi.tools as cas
import numpy as np
import numpy.ma as ma
import scipy.linalg as scila
import resource
import datetime as datetime
import time as time

import awebox.tools.vector_operations as vect_op
import awebox.tools.print_operations as print_op
import awebox.tools.save_operations as save_op
from awebox.logger.logger import Logger as awelogger

def health_check(trial_name, solver_options, nlp, model, solution, arg, stats, iterations, step_name, cumulative_max_memory):

    awelogger.logger.info('Checking health...')

    local_cumulative_max_memory = {'setup': cumulative_max_memory['setup'], 'optimization': cumulative_max_memory['optimization']}
    local_cumulative_max_memory['pre-health-check'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    V_opt = nlp.V(solution['x'])
    p_fix_num = nlp.P(arg['p'])

    health_solver_options = solver_options['health_check']
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

    tractability = collect_tractability_indicators(trial_name, step_name, solver_options, stats, iterations, nlp, model, kkt_matrix, reduced_hessian, local_cumulative_max_memory)

    if health_solver_options['save_health_indicators']:
        filename = health_solver_options['filename_identifier'].strip()
        if len(filename) > 0:
            filename += '_'
        filename += 'health_indicators.csv'
        save_op.write_or_append_two_column_dict_to_csv(tractability, filename)

    exact_licq_holds = is_matrix_full_rank(cstr_jacobian_eval, health_solver_options, tol=0.)
    licq_holds = is_matrix_full_rank(cstr_jacobian_eval, health_solver_options)
    sosc_holds = is_reduced_hessian_positive_definite(tractability['kkt: min_red_hessian_eig'], health_solver_options)
    problem_is_ill_conditioned = is_problem_ill_conditioned(tractability['kkt: condition'], health_solver_options)

    problem_is_healthy = (not problem_is_ill_conditioned) and licq_holds and sosc_holds

    if problem_is_healthy:
        awelogger.logger.info('')
        message = 'OCP appears to be healthy'
        awelogger.logger.info(message)

    elif (not problem_is_healthy) and health_solver_options['help_with_debugging']:

        if 'power' in step_name:
            awelogger.logger.warning('')
            message = 'this unhealthy behavior seems to appear in the power portion of the homotopy.'
            print_op.base_print(message, level='warning')
            message = 'problems of this type can frequently be resolved by increasing: solver.cost_factor.power and/or nlp.n_k'
            print_op.base_print(message, level='warning')

        if not (exact_licq_holds and licq_holds):
            if not exact_licq_holds:
                awelogger.logger.warning('')
                message = 'linear independent constraint qualification is not satisfied at solution, with an exact computation'
                awelogger.logger.warning(message)
            elif not licq_holds:
                awelogger.logger.warning('')
                message = 'linear independent constraint qualification appears not to be satisfied at solution, given floating-point tolerance'
                awelogger.logger.warning(message)

            identify_largest_jacobian_entry(cstr_jacobian_eval, health_solver_options, cstr_labels, nlp)
            identify_dependent_constraint(cstr_jacobian_eval, health_solver_options, cstr_labels, nlp)

        if not sosc_holds:
            awelogger.logger.warning('')
            message = 'second order sufficient conditions appear not to be met at solution. please check if all'
            print_op.base_print(message, level='warning')
            message = 'states/controls/parameters have enough regularization, and if all lifted variables are constrained.'
            print_op.base_print(message, level='warning')

        if problem_is_ill_conditioned:
            awelogger.logger.warning('')
            message = 'problem appears to be ill-conditioned'
            awelogger.logger.warning(message)

        awelogger.logger.warning('')
        identify_largest_kkt_element(kkt_matrix, cstr_fun, lam_fun, cstr_labels, nlp, solution=solution, arg=arg)
        identify_smallest_normed_kkt_column(kkt_matrix, cstr_labels, nlp)

    if not problem_is_healthy:
        awelogger.logger.info('')
        message = 'OCP appears to be unhealthy (step: ' + step_name + ')'

        if health_solver_options['raise_exception']:
            print_op.log_and_raise_error(message)
        else:
            awelogger.logger.warning(message)

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

def collect_tractability_indicators(trial_name, step_name, solver_options, stats, iterations, nlp, model, kkt_matrix, reduced_hessian, local_cumulative_max_memory):

    awelogger.logger.info('collect tractability indicators...')
    tractability = {}

    # trial identification
    tractability['trial_name'] = trial_name
    datetime_now = datetime.datetime.now()
    tractability['datetime_year'] = datetime_now.year
    tractability['datetime_month'] = datetime_now.month
    tractability['datetime_day'] = datetime_now.day
    tractability['datetime_hour'] = datetime_now.hour
    tractability['datetime_minute'] = datetime_now.minute
    tractability['datetime_second'] = datetime_now.second
    tractability['datetime_unix'] = datetime.datetime.timestamp(datetime_now)*1000
    tractability['step_name'] = step_name

    tractability['autoscale'] = solver_options['ipopt']['autoscale']
    # todo: find a way to determine this from ipopt output directly, just-in-case our autoscaling-block ever changes
    for stat_name in stats.keys():
        tractability['stats: ' + stat_name] = stats[stat_name]

    for stat_name in ['nlp_f', 'nlp_g', 'nlp_grad', 'nlp_grad_f', 'nlp_hess_l', 'nlp_jac_g']:
        for time_name in ['t_proc', 't_wall']:
            if ('n_call_' + stat_name in stats.keys()) and (time_name + '_' + stat_name in stats.keys()):
                tractability['avg: ' + time_name + '_' + stat_name] = float(stats[time_name + '_' + stat_name]) / vect_op.smooth_abs(stats['n_call_' + stat_name])

    tractability['total_iterations'] = get_total_iterations(iterations)

    for key in model.variables.keys():
        tractability['model: n_' + key] = model.variables[key].shape[0]

    for cstr_type in ['eq', 'ineq']:
        if hasattr(model.constraints_list.get_expression_list(cstr_type), 'nnz'):
            tractability['model: nnz_' + cstr_type] = model.constraints_list.get_expression_list(cstr_type).nnz()

    for key in model.variables.keys():
        tractability['model: nninf_bounds_' + key] = model.number_noninf_variable_bounds(key)

    tractability['ocp: n_k'] = nlp.n_k
    if hasattr(nlp, 'd'):
        tractability['ocp: d'] = nlp.d

    tractability['ocp: n_V'] = nlp.V.shape[0]
    tractability['ocp: n_theta'] = nlp.V['theta'].shape[0]

    for cstr_type in ['eq', 'ineq']:
        if hasattr(nlp.ocp_cstr_list.get_expression_list(cstr_type), 'nnz'):
            tractability['ocp: nnz_' + cstr_type] = nlp.ocp_cstr_list.get_expression_list(cstr_type).nnz()

    tractability['kkt: size'] = repr(kkt_matrix.shape)
    tractability['kkt: nnz'] = kkt_matrix.nnz()
    tractability['kkt: fraction non-zero'] = kkt_matrix.nnz() / (kkt_matrix.shape[0] * kkt_matrix.shape[1])
    tractability['kkt: diagonality'] = get_pearson_diagonality(kkt_matrix)

    tractability['kkt: condition'] = get_condition_number(kkt_matrix)
    tractability['kkt: min_red_hessian_eig'] = get_min_reduced_hessian_eigenvalue(reduced_hessian)

    local_cumulative_max_memory['at_indicator_report'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    for cmm_key in local_cumulative_max_memory.keys():
        tractability['memory: ' + cmm_key] = local_cumulative_max_memory[cmm_key]

    subset_of_tractability = {}
    for name in ['step_name', 'stats: success', 'stats: iter_count', 'stats: t_wall_total', 'memory: pre-health-check', 'avg: t_wall_nlp_f', 'avg: t_wall_nlp_g']:
        subset_of_tractability[name] = tractability[name]
    for name, value in tractability.items():
        if 'kkt:' in name:
            subset_of_tractability[name] = value

    awelogger.logger.info('tractability indicator report')
    print_op.print_dict_as_table(subset_of_tractability)

    return tractability


def get_local_iterations(stats):
    awelogger.logger.info('get local iterations...')
    return stats['iter_count']

def get_total_iterations(iterations):
    awelogger.logger.info('get total iterations...')
    total_iterations = 0
    for step in iterations.keys():
        total_iterations += int(iterations[step])
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
    max_abs_val = np.amax(np.absolute(cstr_jacobian_eval))
    message = "... largest absolute jacobian entry ({:0.4G}) occurs at:".format(max_abs_val)
    awelogger.logger.info(message)
    max_cdx = np.where(np.absolute(cstr_jacobian_eval) == max_abs_val)[0][0]
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
                    current_matrix = local_cje[cdx+1:, :]

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
    smallest_idx = np.nan

    for idx in range(kkt_matrix.shape[1]):
        local_norm = vect_op.norm(kkt_matrix[:, idx])

        if local_norm < smallest_norm:
            smallest_idx = idx
            smallest_norm = float(local_norm)

    message = '... KKT column (' + str(smallest_idx) + ') '
    message += "with the smallest norm ({:0.4G}) is associated with:".format(smallest_norm)
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


def identify_largest_kkt_element(kkt_matrix, cstr_fun, lam_fun, cstr_labels, nlp, solution=None, arg=None):

    V_opt = nlp.V(solution['x'])
    p_fix_num = nlp.P(arg['p'])

    matrA = np.absolute(np.array(kkt_matrix))

    max_val = np.max(matrA)
    ind = np.unravel_index(np.argmax(matrA, axis=None), matrA.shape)

    associated_row = ind[0]
    associated_column = ind[1]

    number_variables = nlp.V.cat.shape[0]

    message = "... largest (absolute value sense) KKT matrix entry ({:0.4G}) is associated with:".format(max_val)
    awelogger.logger.info(message)

    relevant_variable_index = None
    relevant_multiplier_index = None
    associated_variable_index = None
    associated_constraint_index = None

    if associated_column < number_variables:
        relevant_variable_index = associated_column
        relevant_variable = nlp.V.getCanonicalIndex(relevant_variable_index)
        message = '{:>10}: {:>15} '.format('column', 'variable') + str(relevant_variable)
    else:
        relevant_multiplier_index = associated_column - number_variables
        relevant_multiplier = cstr_labels[relevant_multiplier_index]
        message = '{:>10}: {:>15} '.format('column', 'multiplier') + str(relevant_multiplier)
    print_op.base_print(message, level='info')

    if associated_row < number_variables:
        associated_variable_index = associated_row
        associated_variable = nlp.V.getCanonicalIndex(associated_variable_index)
        message = '{:>10}: {:>15} '.format('row', 'variable') + str(associated_variable)
    else:
        associated_constraint_index = associated_row - number_variables
        associated_constraint = cstr_labels[associated_constraint_index]
        message = '{:>10}: {:>15} '.format('row', 'constraint') + str(associated_constraint)
    print_op.base_print(message, level='info')

    if (relevant_variable_index is not None) and (associated_variable_index is not None):
        if (V_opt is not None) and (p_fix_num is not None):
            [_, _, f_hessian_fun] = nlp.get_f_jacobian_and_hessian_functions()
            f_hessian = f_hessian_fun(V_opt, p_fix_num)
            local_f_hessian_entry = f_hessian[relevant_variable_index, associated_variable_index]
            message = 'the entry of the hessian of the objective-alone, corresponding to the:' + '\n'
            message += str(relevant_variable) + ' and ' + str(associated_variable) + '\n'
            message += 'is: ({:0.4G})'.format(float(local_f_hessian_entry))
            print_op.base_print(message, level='info')

            if (local_f_hessian_entry * 10.)**2. < max_val**2.:
                g_sym = cstr_fun(nlp.V, nlp.P, arg['lbx'], arg['ubx'])
                lam_vals = lam_fun(solution['lam_x'], solution['lam_g'])
                largest_abs_constraint_impact = 0.
                largest_abs_constraint_label = "none found"
                for gdx in range(g_sym.shape[0]):
                    local_lam = lam_vals[gdx]
                    local_g_jacobian_selected = cas.jacobian(g_sym[gdx], nlp.V)[relevant_variable_index]
                    local_g_hessian_selected = cas.jacobian(local_g_jacobian_selected, nlp.V)[associated_variable_index]
                    local_g_hessian_selected_fun = cas.Function('local_g_hessian_selected_fun', [nlp.V, nlp.P], [local_g_hessian_selected])
                    local_g_hessian_times_multiplier = local_lam * local_g_hessian_selected_fun(V_opt, p_fix_num)
                    if local_g_hessian_times_multiplier**2. > largest_abs_constraint_impact**2.:
                        largest_abs_constraint_impact = local_g_hessian_times_multiplier
                        largest_abs_constraint_label = cstr_labels[gdx]

                message = "the multiplier-constraint product that contributes most ({:0.4G})".format(float(largest_abs_constraint_impact))
                message += " to the hessian wrt " + str(associated_variable)
                message += " in the absolute-value sense is: " + str(largest_abs_constraint_label)
                print_op.base_print(message, level='info')

    print_op.base_print("", level='info')

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

    _, strongly_active_ineq_labels, strongly_active_fun = collect_strongly_active_inequality_constraints(health_solver_options, nlp, solution, p_fix_num)
    strongly_active_inequality_constraints = strongly_active_fun(nlp.g_fun(var_sym, p_sym))
    strongly_active_inequality_lambdas = strongly_active_fun(lam_g_sym)

    var_constraint_functions = collect_var_constraints(health_solver_options, nlp, arg, solution)
    all_nonzero_magnitude_var_bounds = var_constraint_functions['all_nonzero_magnitude_fun'](var_sym, lbx_sym, ubx_sym)
    all_nonzero_magnitude_var_lambdas = var_constraint_functions['all_nonzero_magnitude_lam_fun'](lam_x_sym)
    all_nonzero_magnitude_var_labels = var_constraint_functions['all_nonzero_magnitude_labels']

    stacked_constraints = cas.vertcat(equality_constraints, strongly_active_inequality_constraints, all_nonzero_magnitude_var_bounds)
    stacked_cstr_fun = cas.Function('stacked_cstr_fun', [var_sym, p_sym, lbx_sym, ubx_sym], [stacked_constraints])

    stacked_lambdas = cas.vertcat(equality_lambdas, strongly_active_inequality_lambdas, all_nonzero_magnitude_var_lambdas)
    stacked_lam_fun = cas.Function('stacked_lam_fun', [lam_x_sym, lam_g_sym], [stacked_lambdas])

    stacked_labels = eq_labels + strongly_active_ineq_labels + all_nonzero_magnitude_var_labels

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

def collect_strongly_active_inequality_constraints(health_solver_options, nlp, solution, p_fix_num):

    active_threshold = health_solver_options['thresh']['active']
    weak_threshold = health_solver_options['thresh']['weak']
    v_vals = solution['x']

    strongly_active_constraints = []
    list_names = []
    strongly_active_sym = []

    [g_ineq, g_ineq_names, ineq_fun] = collect_inequality_constraints(nlp)

    # list the evaluated constraints at solution
    # ocp_cstr_list = nlp.ocp_cstr_list

    g = nlp.g
    g_fun = nlp.g_fun
    g_vals = g_fun(v_vals, p_fix_num)
    g_sym = cas.SX.sym('g_sym', g.shape)
    # g_names = ocp_cstr_list.get_name_list('all')

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
            inequality_cstr_is_considered_strongly_active = is_inequality_cstr_considered_strongly_active(local_lam, local_g, active_threshold, weak_threshold)

            if inequality_cstr_is_considered_strongly_active:

                # append strongly active constraints to strongly active_list
                strongly_active_constraints = cas.vertcat(strongly_active_constraints, local_g)

                list_names += [local_name]
                strongly_active_sym = cas.vertcat(strongly_active_sym, g_ineq_sym[gdx])

    strongly_active_fun = cas.Function('strongly_active_fun', [g_sym], [strongly_active_sym])

    # return active_list
    return strongly_active_constraints, list_names, strongly_active_fun


def is_inequality_cstr_considered_active(evaluated_lambda, evaluated_expr, active_threshold):

    if active_threshold < 1:
        message = 'the threshold for an active inequality appears to be improperly set. '
        message += 'for a meaningful test, the threshold should be greater than one. it is currently: ' + str(active_threshold)
        print_op.base_print(message, level='warning')

    ineq_cstr_evaluates_to_approx_zero = (
                evaluated_lambda ** 2. > (active_threshold * evaluated_expr) ** 2.)
    return ineq_cstr_evaluates_to_approx_zero


def is_inequality_cstr_considered_strongly_active(evaluated_lambda, evaluated_expr, active_threshold, weak_threshold):

    ineq_cstr_evaluates_to_approx_zero = is_inequality_cstr_considered_active(evaluated_lambda, evaluated_expr, active_threshold)

    if weak_threshold < 1e-15 or weak_threshold > 1e-2:
        message = 'the threshold for a weakly-active inequality appears to be improperly set. '
        message += 'for a meaningful test, the threshold should be greater than small and positive. it is currently: ' + str(weak_threshold)
        print_op.base_print(message, level='warning')

    multiplier_evaluates_to_zero = (evaluated_lambda**2. < weak_threshold**2.)
    strongly_active = ineq_cstr_evaluates_to_approx_zero and (not multiplier_evaluates_to_zero)

    return strongly_active


def collect_var_constraints(health_solver_options, nlp, arg, solution):

    active_threshold = health_solver_options['thresh']['active']
    weak_threshold = health_solver_options['thresh']['weak']
    var_equidistant_to_bounds_threshold = health_solver_options['thresh']['var_equidistant_to_bounds']

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

    strongly_active_cstr = []
    strongly_active_lambdas = []
    strongly_active_vars = []
    strongly_active_labels = []
    weakly_active_labels = []

    selection_error_message = 'something went badly wrong with an if-elif statement expected to be mutually-exclusive'

    for idx in range(var.shape[0]):

        local_lambda = float(lam[idx])
        local_lbx = float(lbx.cat[idx])
        local_ubx = float(ubx.cat[idx])
        local_variable = float(var[idx])

        name_list_strings = list(map(str, nlp.V.getCanonicalIndex(idx)))
        name_underscore = [name + '_' for name in name_list_strings[:-1]] + [name_list_strings[-1]]
        name_idx = ''.join(name_underscore)

        # constraints are written in negative convention (as normal)
        symbolic_lb_expr = lbx_sym[idx] - var_sym[idx]
        symbolic_ub_expr = var_sym[idx] - ubx_sym[idx]
        symbolic_lambda = lam_sym[idx]
        symbolic_variable = var_sym[idx]

        local_variable_either_unbounded_or_equality_constrained_or_exactly_equidistant_between_bounds = (local_lambda == 0)
        local_variable_is_inequality_constrained = not local_variable_either_unbounded_or_equality_constrained_or_exactly_equidistant_between_bounds

        if local_variable_either_unbounded_or_equality_constrained_or_exactly_equidistant_between_bounds:

            local_variables_is_unbounded = (local_lbx == -cas.inf and local_ubx == cas.inf)
            local_variable_is_equality_constrained = (local_ubx == local_lbx)

            upper_distance = (local_ubx - local_variable)
            lower_distance = (local_variable - local_lbx)
            is_equidistant = (upper_distance - lower_distance)**2. < var_equidistant_to_bounds_threshold**2.
            local_variable_is_exactly_equidistant_between_bounds = (not local_variable_is_equality_constrained) and (not local_variables_is_unbounded) and is_equidistant

            if local_variables_is_unbounded:
                # do not add constraint to relevant list -> do nothing
                pass

            elif local_variable_is_exactly_equidistant_between_bounds:
                # do not add constraint to relevant list -> do nothing
                pass

            elif local_variable_is_equality_constrained:
                # default to upper bound
                equality_cstr = cas.vertcat(equality_cstr, symbolic_ub_expr)
                equality_lambdas = cas.vertcat(equality_lambdas, symbolic_lambda)
                equality_vars = cas.vertcat(equality_vars, symbolic_variable)
                equality_labels += [name_idx]

                relevant_cstr = cas.vertcat(relevant_cstr, symbolic_ub_expr)
                relevant_lambdas = cas.vertcat(relevant_lambdas, symbolic_lambda)
                relevant_vars = cas.vertcat(relevant_vars, symbolic_variable)
                relevant_labels += [name_idx]

            else:
                print_op.log_and_raise_error(selection_error_message)

        elif local_variable_is_inequality_constrained:

            relevant_bound_is_the_lower_bound = (local_lambda < 0)
            relevant_bound_is_the_upper_bound = not relevant_bound_is_the_lower_bound

            if relevant_bound_is_the_lower_bound:
                bound_selected_symbolic_expr = symbolic_lb_expr
                bound_selected_symbolic_lambda = -1 * symbolic_lambda
                bound_selected_evaluated_lambda = -1. * local_lambda
                bound_selected_evaluated_expr = local_lbx - local_variable

            elif relevant_bound_is_the_upper_bound:
                bound_selected_symbolic_expr = symbolic_ub_expr
                bound_selected_symbolic_lambda = symbolic_lambda
                bound_selected_evaluated_lambda = local_lambda
                bound_selected_evaluated_expr = local_variable - local_ubx

            else:
                print_op.log_and_raise_error(selection_error_message)

            inequality_cstr = cas.vertcat(inequality_cstr, bound_selected_symbolic_expr)
            inequality_lambdas = cas.vertcat(inequality_lambdas, bound_selected_symbolic_lambda)
            inequality_vars = cas.vertcat(inequality_vars, symbolic_variable)
            inequality_labels += [name_idx]

            relevant_cstr = cas.vertcat(relevant_cstr, bound_selected_symbolic_expr)
            relevant_lambdas = cas.vertcat(relevant_lambdas, bound_selected_symbolic_lambda)
            relevant_vars = cas.vertcat(relevant_vars, symbolic_variable)
            relevant_labels += [name_idx]

            inequality_cstr_is_considered_active = is_inequality_cstr_considered_active(bound_selected_evaluated_lambda, bound_selected_evaluated_expr, active_threshold)
            inequality_cstr_is_considered_strongly_active = is_inequality_cstr_considered_strongly_active(bound_selected_evaluated_lambda, bound_selected_evaluated_expr, active_threshold, weak_threshold)
            if inequality_cstr_is_considered_active:
                if inequality_cstr_is_considered_strongly_active:

                    strongly_active_cstr = cas.vertcat(strongly_active_cstr, bound_selected_symbolic_expr)
                    strongly_active_lambdas = cas.vertcat(strongly_active_lambdas, bound_selected_symbolic_lambda)
                    strongly_active_vars = cas.vertcat(strongly_active_vars, symbolic_variable)
                    strongly_active_labels += [name_idx]
                else: # weakly active
                    weakly_active_labels += [name_idx]

        else:
            print_op.log_and_raise_error(selection_error_message)

    var_constraint_functions = {}

    var_constraint_functions['eq_fun'] = cas.Function('eq_fun', [var_sym, lbx_sym, ubx_sym], [equality_cstr])
    var_constraint_functions['eq_lam_fun'] = cas.Function('eq_lam_fun', [lam_sym], [equality_lambdas])
    var_constraint_functions['eq_vars_fun'] = cas.Function('eq_vars_fun', [var_sym], [equality_vars])
    var_constraint_functions['eq_labels'] = equality_labels

    var_constraint_functions['ineq_fun'] = cas.Function('ineq_fun', [var_sym, lbx_sym, ubx_sym], [inequality_cstr])
    var_constraint_functions['ineq_lam_fun'] = cas.Function('ineq_lam_fun', [lam_sym], [inequality_lambdas])
    var_constraint_functions['ineq_vars_fun'] = cas.Function('ineq_vars_fun', [var_sym], [inequality_vars])
    var_constraint_functions['ineq_labels'] = inequality_labels

    var_constraint_functions['strongly_active_ineq_fun'] = cas.Function('strongly_active_ineq_fun', [var_sym, lbx_sym, ubx_sym], [strongly_active_cstr])
    var_constraint_functions['strongly_active_ineq_lam_fun'] = cas.Function('strongly_active_ineq_lam_fun', [lam_sym], [strongly_active_lambdas])
    var_constraint_functions['strongly_active_ineq_vars_fun'] = cas.Function('strongly_active_ineq_vars_fun', [var_sym], [strongly_active_vars])
    var_constraint_functions['strongly_active_ineq_labels'] = strongly_active_labels

    var_constraint_functions['all_nonzero_magnitude_fun'] = cas.Function('all_nonzero_magnitude_fun', [var_sym, lbx_sym, ubx_sym], [cas.vertcat(equality_cstr, strongly_active_cstr)])
    var_constraint_functions['all_nonzero_magnitude_lam_fun'] = cas.Function('all_nonzero_magnitude_lam_fun', [lam_sym], [cas.vertcat(equality_lambdas, strongly_active_lambdas)])
    var_constraint_functions['all_nonzero_magnitude_vars_fun'] = cas.Function('all_nonzero_magnitude_vars_fun', [var_sym], [cas.vertcat(equality_vars, strongly_active_vars)])
    var_constraint_functions['all_nonzero_magnitude_labels'] = equality_labels + strongly_active_labels

    var_constraint_functions['rel_fun'] = cas.Function('rel_fun', [var_sym, lbx_sym, ubx_sym], [relevant_cstr])
    var_constraint_functions['rel_lam_fun'] = cas.Function('rel_lam_fun', [lam_sym], [relevant_lambdas])
    var_constraint_functions['rel_vars_fun'] = cas.Function('vars_fun', [var_sym], [relevant_vars])
    var_constraint_functions['rel_labels'] = relevant_labels

    var_constraint_functions['weakly_active_labels'] = weakly_active_labels

    return var_constraint_functions

def test():
    # todo
    awelogger.logger.warning('no tests currently defined for debug_operations')
    return None
