#
#    This file is part of awebox.
#
#    awebox -- A modeling and optimization framework for multi-kite AWE systems.
#    Copyright (C) 2017-2020 Jochem De Schutter, Rachel Leuthold, Moritz Diehl,
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
constraints code of the awebox
takes model inequalities, generates constraint structures, and defines the nlp constraints
python-3.5 / casadi-3.4.5
- refactored from awebox code (elena malz, chalmers; jochem de schutter, alu-fr; rachel leuthold, alu-fr), 2018
- edited: jochem de schutter, rachel leuthold, alu-fr 2018 - 2019
'''

import casadi.tools as cas
import numpy as np
import awebox.mdl.aero.induction_dir.vortex_dir.fixing as vortex_fix
import awebox.mdl.aero.induction_dir.vortex_dir.strength as vortex_strength

import awebox.ocp.operation as operation
import awebox.ocp.ocp_constraint as ocp_constraint

import awebox.tools.print_operations as print_op
import awebox.tools.struct_operations as struct_op
import awebox.tools.constraint_operations as cstr_op
import awebox.tools.performance_operations as perf_op

from awebox.logger.logger import Logger as awelogger

import copy

def get_constraints(nlp_options, V, P, Xdot, model, dae, formulation, Integral_constraint_list, Collocation, Multiple_shooting, ms_z0, ms_xf, ms_vars, ms_params, Outputs):

    awelogger.logger.info('generate constraints...')

    direct_collocation = (nlp_options['discretization'] == 'direct_collocation')
    multiple_shooting = (nlp_options['discretization'] == 'multiple_shooting')

    ocp_cstr_list = ocp_constraint.OcpConstraintList()
    ocp_cstr_entry_list = []

    # model constraints structs
    mdl_path_constraints = model.constraints_dict['inequality']
    mdl_dyn_constraints = model.constraints_dict['equality']
    
    # get discretization information
    nk = nlp_options['n_k']
    
    # size of algebraic variables on interval nodes
    nz = model.variables['xa'].shape[0]
    if 'xl' in list(model.variables.keys()):
        nz += model.variables['xl'].shape[0]

    # add initial constraints
    var_initial = struct_op.get_variables_at_time(nlp_options, V, Xdot, model.variables, 0)
    var_ref_initial = struct_op.get_var_ref_at_time(nlp_options, P, V, Xdot, model, 0)
    init_cstr = operation.get_initial_constraints(nlp_options, var_initial, var_ref_initial, model, formulation.xi_dict)
    ocp_cstr_list.append(init_cstr)
    if len(init_cstr.eq_list) != 0:
        ocp_cstr_entry_list.append(cas.entry('initial', shape = init_cstr.eq_list[0].expr.shape))

    # entry tuple for nested constraints
    entry_tuple = ()
    entry_tuple += (cas.entry('dynamics', repeat = [nk], struct = model.variables_dict['xd']),)
    entry_tuple += (cas.entry('algebraic', repeat = [nk], shape = (nz,1)),)

    # add the path constraints.
    if multiple_shooting:
        ms_cstr = expand_with_multiple_shooting(nlp_options, V, model, dae, Multiple_shooting, ms_z0, ms_xf, ms_vars, ms_params)
        ocp_cstr_list.append(ms_cstr)
        entry_tuple += (cas.entry('path', repeat = [nk], struct = mdl_path_constraints),)

    elif direct_collocation:
        coll_cstr = expand_with_collocation(nlp_options, P, V, Xdot, model, Collocation)
        ocp_cstr_list.append(coll_cstr)
        d = nlp_options['collocation']['d']
        entry_tuple += (
            cas.entry('path',        repeat = [nk],    struct = mdl_path_constraints),
            cas.entry('collocation', repeat = [nk, d], struct = mdl_dyn_constraints),
            )

    else:
        message = 'unexpected ocp discretization method selected: ' + nlp_options['discretization']
        awelogger.logger.error(message)
        raise Exception(message)

    # continuity constraints
    entry_tuple += (
        cas.entry('continuity', repeat = [nk], struct = model.variables_dict['xd']),
    )

    # add stage and continuity constraints to list
    ocp_cstr_entry_list.append(entry_tuple)

    # add terminal constraints
    var_terminal = struct_op.get_variables_at_final_time(nlp_options, V, Xdot, model)
    var_ref_terminal = struct_op.get_var_ref_at_final_time(nlp_options, P, V, Xdot, model)
    terminal_cstr = operation.get_terminal_constraints(nlp_options, var_terminal, var_ref_terminal, model, formulation.xi_dict)
    ocp_cstr_list.append(terminal_cstr)
    if len(terminal_cstr.eq_list) != 0:
        ocp_cstr_entry_list.append(cas.entry('terminal', shape =  terminal_cstr.eq_list[0].expr.shape))

    # add periodic constraints
    periodic_cstr = operation.get_periodic_constraints(nlp_options, var_initial, var_terminal)
    ocp_cstr_list.append(periodic_cstr)
    if len(periodic_cstr.eq_list) != 0:
        ocp_cstr_entry_list.append(cas.entry('periodic', shape =  periodic_cstr.eq_list[0].expr.shape))

    vortex_fixing_cstr = vortex_fix.get_fixing_constraint(nlp_options, V, Outputs, model)
    ocp_cstr_list.append(vortex_fixing_cstr)
    if len(vortex_fixing_cstr.eq_list) != 0:
        ocp_cstr_entry_list.append(cas.entry('vortex_fix', shape =  vortex_fixing_cstr.eq_list[0].expr.shape))

    vortex_strength_cstr = vortex_strength.get_strength_constraint(nlp_options, V, Outputs, model)
    ocp_cstr_list.append(vortex_strength_cstr)
    if len(vortex_strength_cstr.eq_list) != 0:
        ocp_cstr_entry_list.append(cas.entry('vortex_strength', shape =  vortex_strength_cstr.eq_list[0].expr.shape))

    if direct_collocation:
        integral_cstr = get_integral_constraints(Integral_constraint_list, formulation.integral_constants)
        ocp_cstr_list.append(integral_cstr)
        if len(integral_cstr.eq_list) != 0:
            ocp_cstr_entry_list.append(cas.entry('integral', shape=integral_cstr.eq_list[0].expr.shape))

    # Constraints structure
    ocp_cstr_struct = cas.struct_symSX(ocp_cstr_entry_list)(ocp_cstr_list.get_expression_list('all'))

    return ocp_cstr_list, ocp_cstr_struct


def expand_with_collocation(nlp_options, P, V, Xdot, model, Collocation):

    cstr_list = ocp_constraint.OcpConstraintList()

    model_variables = model.variables
    model_parameters = model.parameters
    model_constraints_list = model.constraints_list

    n_k = nlp_options['n_k']
    d = nlp_options['collocation']['d']

    parallellization = nlp_options['parallelization']['type']

    # collect shooting variables
    shooting_nodes = struct_op.count_shooting_nodes(nlp_options)
    shooting_vars = struct_op.get_shooting_vars(nlp_options, V, P, Xdot, model)
    shooting_params = struct_op.get_shooting_params(nlp_options, V, P, model)

    # create maps of relevant functions
    mdl_ineq_fun = model_constraints_list.get_function(nlp_options, model_variables, model_parameters, 'ineq')
    mdl_ineq_map = mdl_ineq_fun.map('mdl_ineq_map', parallellization, shooting_nodes, [], [])

    mdl_eq_fun = model_constraints_list.get_function(nlp_options, model_variables, model_parameters, 'eq')
    mdl_eq_map = mdl_eq_fun.map('mdl_eq_map', parallellization, n_k * d, [], [])
    mdl_eq_map_shooting = mdl_eq_fun.map('mdl_eq_map', parallellization, shooting_nodes, [], [])

    # the inequality constraints
    ocp_ineqs_expr = mdl_ineq_map(shooting_vars, shooting_params)
    ineq_cstr = cstr_op.Constraint(expr=cas.reshape(ocp_ineqs_expr, (ocp_ineqs_expr.numel(), 1)),
                                   name='path_constraints',
                                   cstr_type='ineq')
    cstr_list.append(ineq_cstr)

    # the equality constraints
    coll_vars = struct_op.get_coll_vars(nlp_options, V, P, Xdot, model)
    coll_params = struct_op.get_coll_params(nlp_options, V, P, model)

    ocp_eqs_expr = mdl_eq_map(coll_vars, coll_params)
    eq_cstr = cstr_op.Constraint(expr=cas.reshape(ocp_eqs_expr, (ocp_eqs_expr.numel(), 1)),
                                 name='collocation_constraints',
                                 cstr_type='eq')
    cstr_list.append(eq_cstr)

    ocp_eqs_shooting_expr = mdl_eq_map_shooting(shooting_vars, shooting_params)
    expr = cas.reshape(ocp_eqs_shooting_expr, (ocp_eqs_shooting_expr.numel(), 1))
    eq_cstr_shooting = cstr_op.Constraint(expr = expr,
                                        name = 'algebraic_constraints',
                                        cstr_type='eq')
    cstr_list.append(eq_cstr_shooting)

    # the continuity constraints
    for kdx in range(n_k):
        # continuity condition between (kdx, -1) and (kdx + 1)
        continuity_cstr = Collocation.get_continuity_constraint(V, kdx)
        cstr_list.append(continuity_cstr)

    return cstr_list

def expand_with_multiple_shooting(nlp_options, V, model, dae, Multiple_shooting, ms_z0, ms_xf, ms_vars, ms_params):

    cstr_list = ocp_constraint.OcpConstraintList()

    model_variables = model.variables
    model_parameters = model.parameters
    model_constraints_list = model.constraints_list

    n_k = nlp_options['n_k']

    parallellization = nlp_options['parallelization']['type']

    # algebraic constraints
    z_from_V = []
    for kdx in range(n_k):
        if 'xl' in V.keys():
            local_z_from_V = cas.vertcat([V['xddot', kdx], V['xa', kdx], V['xl', kdx]])
        else:
            local_z_from_V = cas.vertcat([V['xddot', kdx], V['xa', kdx]])
        z_from_V = cas.horzcat(z_from_V, local_z_from_V)

    z_at_time_sym = copy.deepcopy(dae.z)
    z_from_V_sym = copy.deepcopy(dae.z)
    alg_fun = cas.Function('alg_fun', [z_at_time_sym, z_from_V_sym], [z_at_time_sym - z_from_V_sym])
    alg_map = alg_fun.map('alg_map', parallellization, n_k, [], [])

    alg_expr = alg_map(ms_z0, z_from_V)
    alg_cstr = cstr_op.Constraint(expr=cas.reshape(alg_expr, (alg_expr.shape[0] * alg_expr.shape[1], 1)),
                                   name='parallelized_algebraics',
                                   cstr_type='eq')
    cstr_list.append(alg_cstr)

    # the inequality constraints
    mdl_ineq_fun = model_constraints_list.get_function(nlp_options, model_variables, model_parameters, 'ineq')
    mdl_ineq_map = mdl_ineq_fun.map('mdl_ineq_map', parallellization, n_k, [], [])

    ocp_ineqs_expr = mdl_ineq_map(ms_vars, ms_params)
    ineq_cstr = cstr_op.Constraint(expr=cas.reshape(ocp_ineqs_expr, (ocp_ineqs_expr.shape[0] * ocp_ineqs_expr.shape[1], 1)),
                                   name='parallelized_model_inequalities',
                                   cstr_type='ineq')
    cstr_list.append(ineq_cstr)

    # continuity constraints
    for kdx in range(n_k):
        cont_cstr = Multiple_shooting.get_continuity_constraint(ms_xf, V, kdx)
        cstr_list.append(cont_cstr)

    return cstr_list



def get_algebraic_constraints(z_at_time, V, kdx):

    cstr_list = ocp_constraint.OcpConstraintList()

    if 'xddot' in list(V.keys()):
        xddot_at_time = z_at_time['xddot']
        expr = xddot_at_time - V['xddot', kdx]
        xddot_cstr = cstr_op.Constraint(expr=expr,
                                        name='xddot_' + str(kdx),
                                        cstr_type='eq')
        cstr_list.append(xddot_cstr)

    if 'xa' in list(V.keys()):
        xa_at_time = z_at_time['xa']
        expr = xa_at_time - V['xa',kdx]
        xa_cstr = cstr_op.Constraint(expr=expr,
                                     name='xa_' + str(kdx),
                                     cstr_type='eq')
        cstr_list.append(xa_cstr)

    if 'xl' in list(V.keys()):
        xl_at_time = z_at_time['xl']
        expr = xl_at_time - V['xl', kdx]
        xl_cstr = cstr_op.Constraint(expr=expr,
                                     name='xl_' + str(kdx),
                                     cstr_type='eq')
        cstr_list.append(xl_cstr)

    return cstr_list


def get_inequality_path_constraints(model, V, ms_vars, ms_params, kdx):

    cstr_list = ocp_constraint.OcpConstraintList()

    mdl_cstr_list = model.constraints_list
    model_variables = model.variables
    model_parameters = model.parameters

    vars_at_time = ms_vars[:, kdx]
    params_at_time = ms_params[:, kdx]

    # at each interval node, path constraints should be satisfied
    for cstr in mdl_cstr_list.get_list('ineq'):

        local_fun = cstr.get_function(model_variables, model_parameters)

        expr = local_fun(vars_at_time, params_at_time)
        local_cstr = cstr_op.Constraint(expr=expr,
                                        name=cstr.name + '_' + str(kdx),
                                        cstr_type=cstr.cstr_type)
        cstr_list.append(local_cstr)

    return cstr_list



def get_integral_constraints(integral_list, integral_constants):

    cstr_list = ocp_constraint.OcpConstraintList()

    # nu = V['phi','nu']
    integral_sum = {}

    for key_name in list(integral_constants.keys()):
        integral_t0 = integral_constants[key_name]
        integral_sum[key_name] = 0.
        for i in range(len(integral_list)):
            integral_sum[key_name] += integral_list[i][key_name]

        expr = (- integral_t0 - integral_sum[key_name]) / integral_t0
        cstr_type = translate_cstr_type(key_name)

        g_cstr = cstr_op.Constraint(expr=expr,
                                    name='integral_' + key_name,
                                    cstr_type=cstr_type)
        cstr_list.append(g_cstr)

    return cstr_list


def translate_cstr_type(constraint_type):

    # convention h(w) <= 0
    if constraint_type == 'inequality':
        return 'ineq'
    elif constraint_type == 'equality':
        return 'eq'
    else:
        raise ValueError('Wrong constraint type chosen. Possible values: "inequality" / "equality" ')

    return None

