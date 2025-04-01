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
constraints code of the awebox
takes model inequalities, generates constraint structures, and defines the nlp constraints
python-3.5 / casadi-3.4.5
- refactored from awebox code (elena malz, chalmers; jochem de schutter, alu-fr; rachel leuthold, alu-fr), 2018
- edited: jochem de schutter, rachel leuthold, alu-fr 2018 - 2021
'''

import casadi.tools as cas
from . import ocp_outputs
import numpy as np
import awebox.mdl.aero.induction_dir.vortex_dir.vortex as vortex

import awebox.ocp.operation as operation

import awebox.tools.print_operations as print_op
import awebox.tools.struct_operations as struct_op
import awebox.tools.constraint_operations as cstr_op
import awebox.tools.performance_operations as perf_op

import awebox.tools.cached_functions as cf

from awebox.logger.logger import Logger as awelogger


def get_constraints(nlp_options, V, P, Xdot, model, dae, formulation, Integral_constraint_list, Collocation, Multiple_shooting, ms_z0, ms_xf, ms_vars, ms_params, Outputs_structured, Integral_outputs, time_grids):

    ocp_cstr_list = cstr_op.OcpConstraintList()
    ocp_cstr_entry_list = []

    if nlp_options['generate_constraints']:
        awelogger.logger.info('Generating constraints...')

        direct_collocation = (nlp_options['discretization'] == 'direct_collocation')
        multiple_shooting = (nlp_options['discretization'] == 'multiple_shooting')

        # add initial constraints
        var_initial = struct_op.get_variables_at_time(nlp_options, V, Xdot, model.variables, 0)
        var_ref_initial = struct_op.get_var_ref_at_time(nlp_options, P, V, Xdot, model, 0)
        init_cstr = operation.get_initial_constraints(nlp_options, var_initial, var_ref_initial, model)
        ocp_cstr_list.append(init_cstr)
        if len(init_cstr.eq_list) != 0:
            ocp_cstr_entry_list.append(cas.entry('initial', shape=init_cstr.get_expression_list('all').shape))

        # add the path constraints.
        if multiple_shooting:
            ms_cstr, entry_tuple = expand_with_multiple_shooting(nlp_options, V, model, dae, Multiple_shooting, ms_z0, ms_xf, ms_vars, ms_params)
            ocp_cstr_list.append(ms_cstr)

        elif direct_collocation:
            coll_cstr, entry_tuple = expand_with_collocation(nlp_options, P, V, Xdot, model, Collocation)
            ocp_cstr_list.append(coll_cstr)

        else:
            message = 'unexpected ocp discretization method selected: ' + nlp_options['discretization']
            print_op.log_and_raise_error(message)

        # add stage and continuity constraints to list
        ocp_cstr_entry_list.append(entry_tuple)

        # add terminal constraints
        var_terminal = struct_op.get_variables_at_final_time(nlp_options, V, Xdot, model)
        var_ref_terminal = struct_op.get_var_ref_at_final_time(nlp_options, P, V, Xdot, model)
        terminal_cstr = operation.get_terminal_constraints(nlp_options, var_terminal, var_ref_terminal, model)
        ocp_cstr_list.append(terminal_cstr)
        if len(terminal_cstr.eq_list) != 0:
            ocp_cstr_entry_list.append(cas.entry('terminal', shape=terminal_cstr.get_expression_list('all').shape))

        # add periodic constraints
        periodic_cstr = operation.get_periodic_constraints(nlp_options, model, var_initial, var_terminal)
        ocp_cstr_list.append(periodic_cstr)
        if len(periodic_cstr.eq_list) != 0:
            ocp_cstr_entry_list.append(cas.entry('periodic', shape=periodic_cstr.get_expression_list('all').shape))

        vortex_ocp_cstr_list = vortex.get_ocp_constraints(nlp_options, V, Outputs_structured, Integral_outputs, model, time_grids)
        ocp_cstr_list.append(vortex_ocp_cstr_list)
        if len(vortex_ocp_cstr_list.eq_list) != 0:
            ocp_cstr_entry_list.append(cas.entry('vortex', shape=vortex_ocp_cstr_list.get_expression_list('all').shape))

        if direct_collocation:
            integral_cstr = get_integral_constraints(Integral_constraint_list, formulation.integral_constants)
            ocp_cstr_list.append(integral_cstr)
            if len(integral_cstr.eq_list) != 0:
                ocp_cstr_entry_list.append(cas.entry('integral', shape=integral_cstr.get_expression_list('all').shape))

    if nlp_options['induction']['induction_model'] == 'averaged':
        # todo: this belongs in induction_dir.averaged

        cstr_list = cstr_op.OcpConstraintList()
        t_f = ocp_outputs.find_time_period(nlp_options, V)
        nk_reelout = round(nlp_options['n_k'] * nlp_options['phase_fix_reelout'])
        F_avg = Integral_outputs['int_out', nk_reelout, 'tether_force_int']
        WdA_avg = Integral_outputs['int_out', nk_reelout, 'area_int']
        a = V['theta', 'a']*model.scaling['theta']['a']
        induction_expr = F_avg/t_f - 4*a*(1 - a)*WdA_avg
        induction_cstr = cstr_op.Constraint(expr=induction_expr,
                                    name='average_induction',
                                    cstr_type='eq')
        cstr_list.append(induction_cstr)

        ocp_cstr_list.append(cstr_list)
        ocp_cstr_entry_list.append(cas.entry('avg_induction', shape=(1, 1)))


    if (nlp_options['system_type'] == 'lift_mode') and (nlp_options['phase_fix'] == 'single_reelout'):
        t_f_cstr_list = get_t_f_bounds_contraints(nlp_options, V, model)
        shape = t_f_cstr_list.get_expression_list('ineq').shape
        ocp_cstr_list.append(t_f_cstr_list)
        ocp_cstr_entry_list.append(cas.entry('t_f_bounds', shape=shape))
    else:
        # period-length t_f constraint is set in ocp.var_bounds
        pass

    # Constraints structure
    ocp_cstr_struct = cas.struct_symMX(ocp_cstr_entry_list)

    # test constraints structure dimension with constraints list
    struct_length = ocp_cstr_struct.shape[0]
    vec_length = ocp_cstr_list.get_expression_list('all').shape[0]
    test_shapes = (struct_length == vec_length)
    assert test_shapes, f'Mismatch in dimension between constraint vector ({vec_length}) and constraint structure ({struct_length})!'

    return ocp_cstr_list, ocp_cstr_struct

def get_t_f_bounds_contraints(nlp_options, V, model):
    phase_fix_reelout = nlp_options['phase_fix_reelout']

    cstr_list = cstr_op.OcpConstraintList()
    t_f = ocp_outputs.find_time_period(nlp_options, V)
    upper_bound = model.variable_bounds['theta']['t_f']['ub']
    lower_bound = model.variable_bounds['theta']['t_f']['lb']

    scale = phase_fix_reelout

    t_f_max = (t_f - upper_bound) / scale
    t_f_min = (lower_bound - t_f) / scale

    t_f_max_cstr = cstr_op.Constraint(expr=t_f_max,
                                      name='t_f_max',
                                      cstr_type='ineq')
    cstr_list.append(t_f_max_cstr)
    t_f_min_cstr = cstr_op.Constraint(expr=t_f_min,
                                      name='t_f_min',
                                      cstr_type='ineq')
    cstr_list.append(t_f_min_cstr)
    return cstr_list

def get_subset_of_shooting_node_equalities_that_wont_cause_licq_errors(model):

    model_constraints_list = model.constraints_list
    model_variables = model.variables

    # remove those constraints that only depend on 'x', since continuity will duplicate those...
    relevant_shooting_vars = []
    for var_type in (set(model_variables.keys()) - set(['x'])):
        relevant_shooting_vars = cas.vertcat(relevant_shooting_vars, model_variables[var_type])

    for cstr in model_constraints_list.get_list('eq'):

        cstr_expr = cstr.expr

        for cdx in range(cstr_expr.shape[0]):
            local_expr = cstr_expr[cdx]
            local_jac = cas.jacobian(local_expr, relevant_shooting_vars)

            will_be_dependent = not (local_jac.nnz() > 0)
            if will_be_dependent:
                message = 'the ' + str(cdx) + 'th entry of the model ' + cstr.name + ' equality constraint ' \
                        'would be likely to trigger licq violations in direct collocation. we suggest re-considering '\
                        'this constraint formulation.'
                awelogger.logger.error(message)

    return None


def expand_with_collocation(nlp_options, P, V, Xdot, model, Collocation):

    cstr_list = cstr_op.OcpConstraintList()
    entry_tuple = ()     # entry tuple for nested constraints

    n_k = nlp_options['n_k']
    d = nlp_options['collocation']['d']

    model_variables = model.variables
    model_parameters = model.parameters
    model_constraints_list = model.constraints_list

    # todo: sort out influence of periodicity. currently: assume periodic trajectory
    n_shooting_cstr = model_constraints_list.get_expression_list('eq').shape[0]

    parallellization = nlp_options['parallelization']['type']

    # collect shooting variables
    shooting_nodes = struct_op.count_shooting_nodes(nlp_options)
    shooting_vars = struct_op.get_shooting_vars(nlp_options, V, P, Xdot, model)
    shooting_params = struct_op.get_shooting_params(nlp_options, V, P, model)

    # collect collocation variables
    coll_nodes = n_k*d
    coll_vars = struct_op.get_coll_vars(nlp_options, V, P, Xdot, model)
    coll_params = struct_op.get_coll_params(nlp_options, V, P, model)

    # create maps of relevant functions
    u_poly = (nlp_options['collocation']['u_param'] == 'poly')
    u_zoh_ineq_shoot = (nlp_options['collocation']['u_param'] == 'zoh') and (nlp_options['collocation']['ineq_constraints'] == 'shooting_nodes')
    u_zoh_ineq_coll = (nlp_options['collocation']['u_param'] == 'zoh') and (nlp_options['collocation']['ineq_constraints'] == 'collocation_nodes')
    inequalities_at_shooting_nodes = u_zoh_ineq_shoot
    inequalities_at_collocation_nodes = u_poly or u_zoh_ineq_coll
    mdl_ineq_fun = model_constraints_list.get_function(nlp_options, model_variables, model_parameters, 'ineq')
    if nlp_options['compile_subfunctions']:
        mdl_ineq_fun = cf.CachedFunction(nlp_options['compilation_file_name']+'_mdl_ineq', mdl_ineq_fun, do_compile=nlp_options['compile_subfunctions'])

    mdl_eq_fun = model_constraints_list.get_function(nlp_options, model_variables, model_parameters, 'eq')
    if nlp_options['compile_subfunctions']:
        mdl_eq_fun = cf.CachedFunction(nlp_options['compilation_file_name']+'_mdl_eq', mdl_eq_fun, do_compile=nlp_options['compile_subfunctions'])

    # evaluate constraint functions
    if nlp_options['parallelization']['map_type'] == 'for-loop':

        if inequalities_at_collocation_nodes:
            ocp_ineqs_list = []
            for k in range(coll_vars.shape[1]):
                ocp_ineqs_list.append(mdl_ineq_fun(coll_vars[:,k], coll_params[:,k]))
            ocp_ineqs_expr = cas.horzcat(*ocp_ineqs_list)

        elif inequalities_at_shooting_nodes:
            ocp_ineqs_list = []
            for k in range(shooting_vars.shape[1]):
                ocp_ineqs_list.append(mdl_ineq_fun(shooting_vars[:,k], shooting_params[:,k]))
            ocp_ineqs_expr = cas.horzcat(*ocp_ineqs_list)

        ocp_eqs_list = []
        for k in range(coll_vars.shape[1]):
            ocp_eqs_list.append(mdl_eq_fun(coll_vars[:,k], coll_params[:,k]))
        ocp_eqs_expr = cas.horzcat(*ocp_eqs_list)

        ocp_eqs_list = []
        for k in range(shooting_vars.shape[1]):
            ocp_eqs_list.append(mdl_eq_fun(shooting_vars[:,k], shooting_params[:,k]))
        ocp_eqs_shooting_expr = cas.horzcat(*ocp_eqs_list)

    elif nlp_options['parallelization']['map_type'] == 'map':

        if inequalities_at_shooting_nodes:
            mdl_ineq_map = mdl_ineq_fun.map('mdl_ineq_map', parallellization, shooting_nodes, [], [])
            ocp_ineqs_expr = mdl_ineq_map(shooting_vars, shooting_params)
        elif inequalities_at_collocation_nodes:
            mdl_ineq_map = mdl_ineq_fun.map('mdl_ineq_map', parallellization, coll_nodes, [], [])
            ocp_ineqs_expr = mdl_ineq_map(coll_vars, coll_params)

        mdl_eq_map = mdl_eq_fun.map('mdl_eq_map', parallellization, coll_nodes, [], [])
        mdl_shooting_eq_map = mdl_eq_fun.map('mdl_shooting_eq_map', parallellization, shooting_nodes, [], [])

        ocp_eqs_expr = mdl_eq_map(coll_vars, coll_params)
        ocp_eqs_shooting_expr = mdl_shooting_eq_map(shooting_vars, shooting_params)

    # sort constraints to obtain desired sparsity structure
    for kdx in range(n_k):

        if nlp_options['collocation']['u_param'] == 'zoh':

            # dynamics on shooting nodes
            if nlp_options['collocation']['name_constraints']:
                for cdx in range(ocp_eqs_shooting_expr[:, kdx].shape[0]):
                    cstr_list.append(cstr_op.Constraint(
                        expr=ocp_eqs_shooting_expr[cdx, kdx],
                        name='shooting_' + str(kdx) + '_' + model_constraints_list.get_name_list('eq')[
                            cdx] + '_' + str(cdx),
                        cstr_type='eq'
                        )
                    )
            else:
                cstr_list.append(cstr_op.Constraint(
                    expr=ocp_eqs_shooting_expr[:, kdx],
                    name='shooting_{}'.format(kdx),
                    cstr_type='eq'
                    )
                )

            # path constraints on shooting nodes
            if (ocp_ineqs_expr.shape != (0, 0)) and inequalities_at_shooting_nodes:
                if nlp_options['collocation']['name_constraints']:
                    for cdx in range(ocp_ineqs_expr[:, kdx].shape[0]):
                        cstr_list.append(cstr_op.Constraint(
                            expr=ocp_ineqs_expr[cdx, kdx],
                            name='path_' + str(kdx) + '_' + model_constraints_list.get_name_list('ineq')[
                                cdx] + '_' + str(cdx),
                            cstr_type='ineq'
                        )
                        )
                else:
                    cstr_list.append(cstr_op.Constraint(
                        expr=ocp_ineqs_expr[:, kdx],
                        name='path_{}'.format(kdx),
                        cstr_type='ineq'
                        )
                )

        # collocation constraints
        for jdx in range(d):
            ldx = kdx * d + jdx
            if inequalities_at_collocation_nodes:
                if ocp_ineqs_expr.shape != (0, 0):
                    cstr_list.append(cstr_op.Constraint(
                        expr = ocp_ineqs_expr[:,ldx],
                        name = 'path_{}_{}'.format(kdx,jdx),
                        cstr_type = 'ineq'
                        )
                    )

            if nlp_options['collocation']['name_constraints']:
                for cdx in range(ocp_eqs_expr[:, ldx].shape[0]):
                    cstr_list.append(cstr_op.Constraint(
                        expr=ocp_eqs_expr[cdx, ldx],
                        name='collocation_' + str(kdx) + '_' + str(jdx) + '_' + model_constraints_list.get_name_list('eq')[cdx] + '_' + str(cdx),
                        cstr_type='eq'
                        )
                    )
            else:
                cstr_list.append(cstr_op.Constraint(
                    expr=ocp_eqs_expr[:, ldx],
                    name='collocation_{}_{}'.format(kdx, jdx),
                    cstr_type='eq'
                    )
                )

        # continuity constraints
        cstr_list.append(Collocation.get_continuity_constraint(V, kdx))

    mdl_path_constraints = model.constraints_dict['inequality']
    mdl_dyn_constraints = model.constraints_dict['equality']
    
    if u_zoh_ineq_shoot:
        entry_tuple += (
            cas.entry('shooting',       repeat = [n_k],     struct = mdl_dyn_constraints),
            cas.entry('path',           repeat = [n_k],     struct = mdl_path_constraints),
        )
    
    elif u_zoh_ineq_coll:
        entry_tuple += (
            cas.entry('shooting',       repeat = [n_k],       struct = mdl_dyn_constraints),
            cas.entry('path',           repeat = [n_k,d],     struct = mdl_path_constraints),
        )

    elif u_poly:
        entry_tuple += (
            cas.entry('path',           repeat = [n_k, d],     struct = mdl_path_constraints),
        )

    entry_tuple += (
        cas.entry('collocation',    repeat = [n_k, d],  struct = mdl_dyn_constraints),
        cas.entry('continuity',     repeat = [n_k],     struct = model.variables_dict['x']),
    )

    return cstr_list, entry_tuple

def expand_with_multiple_shooting(nlp_options, V, model, dae, Multiple_shooting, ms_z0, ms_xf, ms_vars, ms_params):

    n_k = nlp_options['n_k']

    # number of algebraic variables that are restricted by model equalities (dynamics)
    nz = model.constraints_list.get_expression_list('eq').shape[0] - model.variables_dict['x'].shape[0]

    # entry tuple for nested constraints
    entry_tuple = ()

    entry_tuple += (cas.entry('dynamics', repeat=[n_k], struct=model.variables_dict['x']),)
    entry_tuple += (cas.entry('algebraic', repeat=[n_k], shape=(nz, 1)),)

    cstr_list = cstr_op.OcpConstraintList()

    model_variables = model.variables
    model_parameters = model.parameters
    model_constraints_list = model.constraints_list

    parallellization = nlp_options['parallelization']['type']

    # algebraic constraints
    z_from_V = []
    for kdx in range(n_k):
        local_z_from_V = cas.vertcat(V['xdot',kdx], V['z', kdx])
        z_from_V = cas.horzcat(z_from_V, local_z_from_V)

    z_at_time_sym = cas.MX.sym('z_at_time_sym',dae.z.shape)
    z_from_V_sym =  cas.MX.sym('z_from_V_sym',dae.z.shape)
    alg_fun = cas.Function('alg_fun', [z_at_time_sym, z_from_V_sym], [z_at_time_sym - z_from_V_sym])
    alg_map = alg_fun.map('alg_map', parallellization, n_k, [], [])

    # inequality constraints
    mdl_ineq_fun = model_constraints_list.get_function(nlp_options, model_variables, model_parameters, 'ineq')
    mdl_ineq_map = mdl_ineq_fun.map('mdl_ineq_map', parallellization, n_k, [], [])

    # evaluate mapped constraint functions
    alg_expr = alg_map(ms_z0, z_from_V)
    ocp_ineqs_expr = mdl_ineq_map(ms_vars, ms_params)

    for kdx in range(n_k):

        # algebraic constraints
        cstr_list.append(cstr_op.Constraint(
            expr=alg_expr[:,kdx],
            name='algebraic_{}'.format(kdx),
            cstr_type='eq'
            )
        )

        # path constraints
        cstr_list.append(cstr_op.Constraint(
            expr=ocp_ineqs_expr[:,kdx],
            name='path_{}'.format(kdx),
            cstr_type='ineq'
            )
        )

        # continuity constraints
        cstr_list.append(Multiple_shooting.get_continuity_constraint(ms_xf, V, kdx))

    mdl_path_constraints = model.constraints_dict['inequality']
    entry_tuple += (
        cas.entry('path', repeat=[n_k], struct=mdl_path_constraints),
        cas.entry('continuity', repeat = [n_k], struct = model.variables_dict['x']),
    )


    return cstr_list, entry_tuple

def get_integral_constraints(integral_list, integral_constants):

    cstr_list = cstr_op.OcpConstraintList()

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

