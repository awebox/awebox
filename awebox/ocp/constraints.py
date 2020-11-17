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

def get_constraints(nlp_options, V, P, Xdot, model, dae, formulation, Integral_constraint_list, Collocation, Multiple_shooting, ms_z0, ms_xf, ms_vars, ms_params, Outputs):

    awelogger.logger.info('generate constraints...')

    direct_collocation = (nlp_options['discretization'] == 'direct_collocation')
    radau_collocation = direct_collocation and (nlp_options['collocation']['scheme'] == 'radau')
    other_collocation = direct_collocation and (not radau_collocation)

    multiple_shooting = (nlp_options['discretization'] == 'multiple_shooting')

    ocp_cstr_list = ocp_constraint.OcpConstraintList()

    # add initial constraints
    var_initial = struct_op.get_variables_at_time(nlp_options, V, Xdot, model.variables, 0)
    var_ref_initial = struct_op.get_var_ref_at_time(nlp_options, P, V, Xdot, model, 0)
    init_cstr = operation.get_initial_constraints(nlp_options, var_initial, var_ref_initial, model, formulation.xi_dict)
    ocp_cstr_list.append(init_cstr)

    # add the path constraints.
    if multiple_shooting:
        ms_cstr = expand_with_multiple_shooting(nlp_options, V, model, dae, Multiple_shooting, ms_z0, ms_xf, ms_vars, ms_params)
        ocp_cstr_list.append(ms_cstr)

    elif radau_collocation:
        radau_cstr = expand_with_radau_collocation(nlp_options, P, V, Xdot, model, Collocation)
        ocp_cstr_list.append(radau_cstr)

    elif other_collocation:
        other_cstr = expand_with_other_collocation()
        ocp_cstr_list.append(other_cstr)

    else:
        message = 'unexpected ocp discretization method selected: ' + nlp_options['discretization']
        awelogger.logger.error(message)
        raise Exception(message)

    # add terminal constraints
    var_terminal = struct_op.get_variables_at_final_time(nlp_options, V, Xdot, model)
    var_ref_terminal = struct_op.get_var_ref_at_final_time(nlp_options, P, V, Xdot, model)
    terminal_cstr = operation.get_terminal_constraints(nlp_options, var_terminal, var_ref_terminal, model, formulation.xi_dict)
    ocp_cstr_list.append(terminal_cstr)

    # add periodic constraints
    periodic_cstr = operation.get_periodic_constraints(nlp_options, var_initial, var_terminal)
    ocp_cstr_list.append(periodic_cstr)

    if direct_collocation:
        integral_cstr = get_integral_constraints(Integral_constraint_list, formulation.integral_constants)
        ocp_cstr_list.append(integral_cstr)

    vortex_fixing_cstr = vortex_fix.get_fixing_constraint(nlp_options, V, Outputs, model)
    ocp_cstr_list.append(vortex_fixing_cstr)

    vortex_strength_cstr = vortex_strength.get_strength_constraint(nlp_options, V, Outputs, model)
    ocp_cstr_list.append(vortex_strength_cstr)

    return ocp_cstr_list


def expand_with_radau_collocation(nlp_options, P, V, Xdot, model, Collocation):

    cstr_list = ocp_constraint.OcpConstraintList()

    model_variables = model.variables
    model_parameters = model.parameters
    model_constraints_list = model.constraints_list

    mdl_ineq_list = model_constraints_list.ineq_list

    n_k = nlp_options['n_k']
    d = nlp_options['collocation']['d']

    for kdx in range(n_k):

        vars_at_time = struct_op.get_variables_at_time(nlp_options, V, Xdot, model_variables, kdx)
        params_at_time = struct_op.get_parameters_at_time(nlp_options, P, V, Xdot, model_variables,
                                                          model_parameters, kdx)

        # inequality constraints get enforced at control nodes
        for mdl_ineq in mdl_ineq_list:
            local_fun = mdl_ineq.get_function(model_variables, model_parameters)
            expr = local_fun(vars_at_time, params_at_time)

            local_cstr = cstr_op.Constraint(expr=expr,
                                            name=mdl_ineq.name + '_' + str(kdx),
                                            cstr_type=mdl_ineq.cstr_type)
            cstr_list.append(local_cstr)

        # equality constraints get enforced at collocation nodes
        for ddx in range(d):
            vars_at_time = struct_op.get_variables_at_time(nlp_options, V, Xdot, model_variables, kdx, ddx)
            params_at_time = struct_op.get_parameters_at_time(nlp_options, P, V, Xdot, model_variables,
                                                              model_parameters, kdx, ddx)

            middle_eq_cstr = get_equality_radau_constraints(nlp_options, model, Collocation, vars_at_time,
                                                              params_at_time, kdx, ddx)
            cstr_list.append(middle_eq_cstr)

        # continuity condition between (kdx, -1) and (kdx + 1)
        continuity_cstr = Collocation.get_continuity_constraint(V, kdx)
        cstr_list.append(continuity_cstr)

    periodic = perf_op.determine_if_periodic(nlp_options)
    if not periodic:
        # append inequality constraint at end, too.
        kdx = n_k
        vars_at_time = struct_op.get_variables_at_time(nlp_options, V, Xdot, model_variables, kdx)
        params_at_time = struct_op.get_parameters_at_time(nlp_options, P, V, Xdot, model_variables,
                                                          model_parameters, kdx)

        # inequality constraints get enforced at control nodes
        for mdl_ineq in mdl_ineq_list:
            local_fun = mdl_ineq.get_function(model_variables, model_parameters)
            expr = local_fun(vars_at_time, params_at_time)

            local_cstr = cstr_op.Constraint(expr=expr,
                                            name=mdl_ineq.name + '_' + str(kdx),
                                            cstr_type=mdl_ineq.cstr_type)
            cstr_list.append(local_cstr)

    return cstr_list

def get_equality_radau_constraints(nlp_options, model, Collocation, vars_at_time, params_at_time, kdx, ddx=None):

    model_variables = model.variables
    model_parameters = model.parameters
    model_constraints_list = model.constraints_list
    mdl_eq_list = model_constraints_list.eq_list
    n_k = nlp_options['n_k']

    cstr_list = ocp_constraint.OcpConstraintList()

    for mdl_eq in mdl_eq_list:

        local_fun = mdl_eq.get_function(model_variables, model_parameters)

        if (ddx is not None) and ('trivial' in mdl_eq.name):
            # compensate for the fact that the (derivatives of the basis functions) become
            # increasingly large as order increases
            t_f_estimate = nlp_options['normalization']['t_f']
            trivial_scaling = np.max(np.absolute(np.array(Collocation.coeff_collocation[:, ddx + 1]))) * (
                        n_k / t_f_estimate)
            expr = local_fun(vars_at_time, params_at_time) / trivial_scaling

        else:
            expr = local_fun(vars_at_time, params_at_time)

        if ddx is not None:
            name = mdl_eq.name + '_' + str(kdx) + '_' + str(ddx)
        else:
            name = mdl_eq.name + '_' + str(kdx)

        local_cstr = cstr_op.Constraint(expr=expr,
                                        name=name,
                                        cstr_type=mdl_eq.cstr_type)
        cstr_list.append(local_cstr)

    return cstr_list

def expand_with_other_collocation():

    # todo: add this.
    #  notice, that the logic flow of non-radau collection was *never* actually triggered in previous iterates.
    #  there would certainly have been an error otherwise, see, for example,
    #  the inclusion of the un-defined variable ms_z0[:, kdx]

    message = 'OCP discretization with non-Radau collection is not supported at present.'
    awelogger.logger.error(message)
    raise Exception(message)


def expand_with_multiple_shooting(nlp_options, V, model, dae, Multiple_shooting, ms_z0, ms_xf, ms_vars, ms_params):

    cstr_list = ocp_constraint.OcpConstraintList()

    n_k = nlp_options['n_k']
    for kdx in range(n_k):

        # at each interval node, algebraic constraints should be satisfied
        alg_cstr = get_algebraic_constraints(dae.z(ms_z0[:, kdx]), V, kdx)
        cstr_list.append(alg_cstr)

        ms_path_cstr = get_inequality_path_constraints(model, V, ms_vars, ms_params, kdx)
        cstr_list.append(ms_path_cstr)

        # endpoint should match next start point
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
    use_slack_formulation = ('us' in list(V.keys()))

    for cstr in mdl_cstr_list.get_list('ineq'):

        local_fun = cstr.get_function(model_variables, model_parameters)

        if use_slack_formulation:
            slacks = V['us', kdx]
            expr = local_fun(vars_at_time, params_at_time) - slacks
            local_cstr = cstr_op.Constraint(expr=expr,
                                            name=cstr.name + '_slack_' + str(kdx),
                                            cstr_type='eq')
            cstr_list.append(local_cstr)

        else:
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

