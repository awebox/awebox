#
#    This file is part of awebox.
#
#    awebox -- A modeling and optimization framework for multi-kite AWE systems.
#    Copyright (C) 2017-2019 Jochem De Schutter, Rachel Leuthold, Moritz Diehl,
#                            ALU Freiburg.
#    Copyright (C) 2018-2019 Thilo Bronnenmeyer, Kiteswarms Ltd.
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
'''

import casadi.tools as cas
import numpy as np

import awebox.tools.vector_operations as vect_op

import awebox.tools.print_operations as print_op

from awebox.logger.logger import Logger as awelogger

def check_display(check_condition, check_name, label_pass, label_warning):

    awelogger.logger.debug('')
    awelogger.logger.debug(print_op.hline('*'))

    if check_condition:
        awelogger.logger.debug('{0:>15}:'.format(check_name + ' PASSED') + ' ' + label_pass)
    else:
        awelogger.logger.debug('{0:>15}:'.format(check_name + ' WARNING') + ' ' + label_warning)

    return None

def addon_disply(label, value):
    awelogger.logger.debug('{0:>40} = {1:5}'.format(label, value))

def sosc_check(health_solver_options, nlp, solution, arg):

    awelogger.logger.debug('sosc check...')

    V = nlp.V
    P = nlp.P

    V_vals = V(solution['x'])
    lambda_g_vals = solution['lam_g']
    lambda_x_vals = solution['lam_x']
    p_fix_num = nlp.P(arg['p'])

    awelogger.logger.debug('get constraints...')

    [stacked_cstr_fun, stacked_lam_fun, stacked_labels] = collect_equality_and_active_inequality_constraints(health_solver_options, nlp, solution, arg)
    relevant_constraints = stacked_cstr_fun(V, P, arg['lbx'], arg['ubx'])

    awelogger.logger.debug('get jacobian...')

    linearization = cas.jacobian(relevant_constraints, V)
    linearization_fun = cas.Function('linearization_fun', [V, P], [linearization])
    linearization_eval = np.array(linearization_fun(V_vals, p_fix_num))

    awelogger.logger.debug('find the null space...')


    null = vect_op.null(linearization_eval, health_solver_options['sosc']['reduced_hessian_null_tol'])

    awelogger.logger.debug('make lagrangian')

    [lagr_fun, lagr_jacobian_fun, lagr_hessian_fun] = generate_lagrangian(health_solver_options, nlp, arg, solution)
    lagr_hessian = lagr_hessian_fun(V_vals, p_fix_num, lambda_g_vals, lambda_x_vals)

    awelogger.logger.debug('get reduced hessian')

    reduced_hessian = cas.mtimes(cas.mtimes(null.T, lagr_hessian), null)

    awelogger.logger.debug('rank check on reduced hessian')

    full_rank_check(health_solver_options, reduced_hessian, 'reduced hessian', 'SOSC')

def spy_hessian(health_solver_options, nlp, solution, arg):
    [lagr_fun, lagr_jacobian_fun, lagr_hessian_fun] = generate_lagrangian(health_solver_options, nlp, arg, solution)

    ahess = np.array(lagr_hessian_fun(solution['x'], arg['p'], solution['lam_g'], solution['lam_x']))

    # vect_op.spy(ahess, tol=1., color=False)
    vect_op.spy(ahess, tol=0.1, color=False)


def full_rank_check(health_solver_options, matrix, matrix_name, check_name):

    rank_tolerance = health_solver_options['singular_values']['min_tol']

    matrix_rank = np.linalg.matrix_rank(matrix, rank_tolerance)
    required_rank = np.shape(matrix)[0]

    condition = (required_rank == matrix_rank)

    check_display(condition,
                  check_name,
                  matrix_name + ' has full row rank',
                  matrix_name + ' DOES NOT have full row rank')

    addon_disply(matrix_name + ' matrix rank', str(matrix_rank))
    addon_disply(matrix_name + ' row number', str(required_rank))

    return condition

def licq_check_on_equality_constraints(health_solver_options, nlp, solution, p_fix_num):

    V = nlp.V
    P = nlp.P

    [equality_constraints, eq_labels, eq_fun] = collect_equality_constraints(nlp)

    cstr_block = cas.jacobian(equality_constraints, V)
    cstr_block_fun = cas.Function('cstr_block_fun', [V, P], [cstr_block])

    cstr_block_eval = np.array(cstr_block_fun(V(solution['x']), p_fix_num))

    vect_op.spy(cstr_block_eval)
    poss_trouble_var = np.argmax(np.max(cstr_block_eval, axis=0))
    print(poss_trouble_var)
    print((nlp.V.getCanonicalIndex(poss_trouble_var)))
    print((np.max(cstr_block_eval[:, poss_trouble_var])))

    full_rank_check(health_solver_options, cstr_block_eval, 'equality constraint block', 'LICQ')

def licq_check(health_solver_options, nlp, solution, arg):

    V = nlp.V
    P = nlp.P

    p_fix_num = nlp.P(arg['p'])

    [stacked_cstr_fun, stacked_lam_fun, stacked_labels] = collect_equality_and_active_inequality_constraints(health_solver_options, nlp, solution, arg)

    relevant_constraints = stacked_cstr_fun(V, P, arg['lbx'], arg['ubx'])

    cstr_block = cas.jacobian(relevant_constraints, V)
    cstr_block_fun = cas.Function('cstr_block_fun', [V, P], [cstr_block])

    cstr_block_eval = np.array(cstr_block_fun(V(solution['x']), p_fix_num))

    full_rank_check(health_solver_options, cstr_block_eval, 'constraint block', 'LICQ')

def health_check(health_solver_options, nlp, solution, arg):

    eval_kkt = get_solution_kkt_matrix(health_solver_options, nlp, solution, arg)

    empty_cols_passes = empty_cols_check(health_solver_options, nlp, solution, arg, eval_kkt, 'kkt matrix')

    kkt_rank_passes = full_rank_check(health_solver_options, eval_kkt, 'kkt matrix', 'KKT')

    [uu, ss, vv] = np.linalg.svd(eval_kkt)

    ss_max = np.max(ss)
    ss_min = np.min(ss)

    singular_value_ratio_passes = singular_value_ratio_check(health_solver_options, ss_min, ss_max, 'KKT', 'kkt matrix')
    smallest_singular_value_passes = smallest_singular_value_check(health_solver_options, ss_min, 'KKT', 'kkt matrix')

    health_passes = kkt_rank_passes and singular_value_ratio_passes and smallest_singular_value_passes and empty_cols_passes

    spy_hessian(health_solver_options, nlp, solution, arg)

    return health_passes

def smallest_singular_value_check(health_solver_options, ss_min, check_name, matrix_name):

    tol = health_solver_options['singular_values']['min_tol']
    condition = ss_min > tol

    check_display(condition,
                  check_name,
                  'smallest singular values do not indicate ill-conditioned ' + matrix_name,
                  'smallest singular values indicate ill-conditioned ' + matrix_name)

    addon_disply('s_min', ss_min)

    return condition

def singular_value_ratio_check(health_solver_options, ss_min, ss_max, check_name, matrix_name):

    tol = health_solver_options['singular_values']['ratio_min_tol']
    ss_ratio = ss_max / ss_min

    condition = ss_ratio < tol

    check_display(condition,
                  check_name,
                  'singular value ratio does not indicate ill-conditioned ' + matrix_name,
                  'singular value ratio indicates ill-conditioned ' + matrix_name)

    addon_disply('s_max / s_min', ss_ratio)

    return condition

def get_solution_kkt_matrix(health_solver_options, nlp, solution, arg):
    V_opt = solution['x']
    lambda_g = solution['lam_g']
    lambda_x = solution['lam_x']

    p_fix_num = nlp.P(arg['p'])

    kkt_matrix_fun = make_kkt_matrix_function(health_solver_options, nlp, solution, arg)

    solution_kkt = np.array(kkt_matrix_fun(V_opt, p_fix_num, lambda_g, lambda_x))

    return np.array(solution_kkt)

def make_kkt_matrix_function(health_solver_options, nlp, solution, arg):

    V = nlp.V
    P = nlp.P

    lam_x_sym = cas.MX.sym('lam_x_sym', nlp.V.shape)
    lam_g_sym = cas.MX.sym('lam_g_sym', nlp.g.shape)

    [stacked_cstr_fun, stacked_lam_fun, stacked_labels] = collect_equality_and_active_inequality_constraints(health_solver_options, nlp, solution, arg)
    relevant_constraints = stacked_cstr_fun(V, P, arg['lbx'], arg['ubx'])

    [lagr_fun, lagr_jacobian_fun, lagr_hessian_fun] = generate_lagrangian(health_solver_options, nlp, arg, solution)

    lagr_block = lagr_hessian_fun(V, P, lam_g_sym, lam_x_sym)
    constraint_block = cas.jacobian(relevant_constraints, V)

    zeros_block = np.zeros((constraint_block.shape[0], constraint_block.shape[0]))

    # kkt_matrix =
    # nabla^2 lagr  nabla g^top     nabla hA^top
    # nabla g           0               0
    # nabla hA          0               0

    kkt_matrix = cas.horzcat(lagr_block, constraint_block.T)
    kkt_matrix = cas.vertcat(kkt_matrix, cas.horzcat(constraint_block, zeros_block))

    kkt_matrix_fun = cas.Function('kkt_matrix_fun', [V, P, lam_g_sym, lam_x_sym], [kkt_matrix])

    return kkt_matrix_fun

def generate_lagrangian(health_solver_options, nlp, arg, solution):

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

    var_sym = cas.SX.sym('var_sym', nlp.V.shape)
    p_sym = cas.SX.sym('p_sym', nlp.P.shape)
    ubx_sym = cas.SX.sym('ubx_sym', arg['ubx'].shape)
    lbx_sym = cas.SX.sym('lbx_sym', arg['lbx'].shape)

    lam_x_sym = cas.SX.sym('lam_x_sym', solution['lam_x'].shape)
    lam_g_sym = cas.SX.sym('lam_g_sym', solution['lam_g'].shape)

    p_fix_num = nlp.P(arg['p'])
    var_constraint_functions = collect_var_constraints(health_solver_options, nlp, arg, solution)

    [equality_constraints, eq_labels, eq_fun] = collect_equality_constraints(nlp)
    [active_inequality_constraints, active_ineq_labels, active_fun] = collect_active_inequality_constraints(health_solver_options, nlp, solution, p_fix_num)

    equality_constraints = eq_fun(nlp.g_fun(var_sym, p_sym))
    active_inequality_constraints = active_fun(nlp.g_fun(var_sym, p_sym))

    equality_lambdas = eq_fun(lam_g_sym)
    active_inequality_lambdas = active_fun(lam_g_sym)

    all_active_var_bounds = var_constraint_functions['all_act_fun'](var_sym, lbx_sym, ubx_sym)
    all_active_var_lambdas = var_constraint_functions['all_act_lam_fun'](lam_x_sym)
    all_active_var_labels = var_constraint_functions['all_act_labels']

    stacked_constraints = cas.vertcat(equality_constraints, active_inequality_constraints, all_active_var_bounds)
    stacked_cstr_fun = cas.Function('stacked_cstr_fun', [var_sym, p_sym, lbx_sym, ubx_sym], [stacked_constraints])

    stacked_lambdas = cas.vertcat(equality_lambdas, active_inequality_lambdas, all_active_var_lambdas)
    stacked_lam_fun = cas.Function('stacked_lam_fun', [lam_x_sym, lam_g_sym], [stacked_lambdas])

    stacked_labels = eq_labels + active_ineq_labels + all_active_var_labels

    return stacked_cstr_fun, stacked_lam_fun, stacked_labels

def collect_type_constraints(nlp, is_equality):

    constraints = []
    list_names = []
    constraint_sym = []

    # list the evaluated constraints at solution
    g = nlp.g

    g_sym = cas.SX.sym('g_sym', g.shape)

    for gdx in range(g.shape[0]):
        cstr_name = g.getCanonicalIndex(gdx)

        condition = 'inequality' in cstr_name
        if is_equality:
            condition = not condition

        if condition:
            constraints = cas.vertcat(constraints, g.cat[gdx])
            constraint_sym = cas.vertcat(constraint_sym, g_sym[gdx])

            name_list_strings = list(map(str, cstr_name))
            name_list = [name + '_' for name in name_list_strings[:-1]] + [name_list_strings[-1]]
            list_names += [''.join(name_list)]

    cstr_fun = cas.Function('cstr_fun', [g_sym], [constraint_sym])

    return constraints, list_names, cstr_fun

def collect_equality_constraints(nlp):
    return collect_type_constraints(nlp, True)

def collect_inequality_constraints(nlp):
    return collect_type_constraints(nlp, False)

def collect_active_inequality_constraints(health_solver_options, nlp, solution, p_fix_num):

    active_threshold = health_solver_options['active_threshold']
    v_vals = solution['x']

    active_constraints = []
    list_names = []
    active_sym = []

    [g_ineq, g_names, ineq_fun] = collect_inequality_constraints(nlp)

    # list the evaluated constraints at solution
    g = nlp.g
    g_fun = nlp.g_fun
    g_vals = g(g_fun(v_vals, p_fix_num))
    gsym = cas.SX.sym('gsym', g.shape)

    # list the multipliers lambda at solution
    lam_vals = g(solution['lam_g'])

    g_ineq_vals = ineq_fun(g_vals)
    lambda_ineq_vals = ineq_fun(lam_vals)
    g_ineq_sym = ineq_fun(gsym)

    if not g_ineq_sym.shape[0] == 0:
        for gdx in range(g_ineq.shape[0]):

            local_g = g_ineq_vals[gdx]
            local_lam = lambda_ineq_vals[gdx]
            local_name = g_names[gdx]

            # if eval_constraint is small, then constraint is active. or.
            # if lambda >> eval_constraint, then: constraint is active
            if local_lam**2. > (active_threshold * local_g)**2.:

                # append active constraints to active_list
                active_constraints = cas.vertcat(active_constraints, local_g)

                list_names += [local_name + '_' + str(gdx)]
                active_sym = cas.vertcat(active_sym, g_ineq_sym[gdx])

    active_fun = cas.Function('active_fun', [gsym], [active_sym])

    # return active_list
    return active_constraints, list_names, active_fun

def collect_var_constraints(health_solver_options, nlp, arg, solution):

    active_threshold = health_solver_options['active_threshold']

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

def empty_cols_check(health_solver_options, nlp, solution, arg, matrix, matrix_name):

    tol = health_solver_options['matrix_entry_zero_tol']

    zero_cols = vect_op.find_zero_cols(matrix, tol)
    test = zero_test_display(health_solver_options, nlp, solution, arg, zero_cols, matrix_name, 'col')

    return test

def empty_rows_check(health_solver_options, nlp, solution, arg, matrix, matrix_name):

    tol = health_solver_options['matrix_entry_zero_tol']

    zero_rows = vect_op.find_zero_rows(matrix, tol)
    test = zero_test_display(health_solver_options, nlp, solution, arg, zero_rows, matrix_name, 'row')

    return test

def zero_test_display(health_solver_options, nlp, solution, arg, zero_axis, matrix_name, axis):

    if (axis == 0) or ('row' in axis):
        hunt_name = 'empty rows'
    if (axis == 1) or ('col' in axis):
        hunt_name = 'empty columns'

    condition = (zero_axis.shape[0] == 0)

    check_display(condition,
                  'ZERO',
                  matrix_name + ' does not contain ' + hunt_name,
                  matrix_name + ' contains ' + hunt_name)

    if not condition:
        addon_disply('locations of ' + hunt_name, repr(zero_axis))

        for idx in range(zero_axis.shape[0]):
            interpret_kkt_index(health_solver_options, zero_axis[idx], nlp, solution, arg)

    return condition

def interpret_kkt_index(health_solver_options, idx, nlp, solution, arg):

    V = nlp.V
    numel_V = V.shape[0]

    [stacked_cstr_fun, stacked_lam_fun, stacked_labels] = collect_equality_and_active_inequality_constraints(health_solver_options, nlp, solution, arg)
    numel_constraints = stacked_labels.shape[0]

    if idx < numel_V:
        addon_disply('index ' + str(idx), V.getCanonicalIndex(idx))

    else:
        idx_remainder = numel_V - idx - 1

        if idx_remainder < numel_constraints:
            addon_disply('index ' + str(idx), stacked_labels[idx_remainder])

        else:
            awelogger.logger.error('index error in interpretation')

    return None
