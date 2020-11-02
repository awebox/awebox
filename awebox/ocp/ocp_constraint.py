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
ocp constraint handling
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2020
'''

import casadi.tools as cas
from awebox.logger.logger import Logger as awelogger
import awebox.tools.constraint_operations as cstr_op
import awebox.tools.struct_operations as struct_op
import awebox.tools.performance_operations as perf_op
import awebox.tools.print_operations as print_op

import pdb

class OcpConstraintList(cstr_op.ConstraintList):
    def __init__(self):
        super().__init__(list_name='ocp constraints list')

    def expand_with_radau_collocation(self, nlp_options, P, V, Xdot, model, Collocation):

        model_variables = model.variables
        model_parameters = model.parameters
        model_constraints_list = model.constraints_list

        mdl_ineq_list = model_constraints_list.ineq_list
        mdl_eq_list = model_constraints_list.eq_list

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
                self.append(local_cstr)

            # equality constraints get enforced at collocation nodes
            for ddx in range(d):
                vars_at_time = struct_op.get_variables_at_time(nlp_options, V, Xdot, model_variables, kdx, ddx)
                params_at_time = struct_op.get_parameters_at_time(nlp_options, P, V, Xdot, model_variables,
                                                                  model_parameters, kdx, ddx)

                for mdl_eq in mdl_eq_list:
                    local_fun = mdl_eq.get_function(model_variables, model_parameters)
                    expr = local_fun(vars_at_time, params_at_time)

                    local_cstr = cstr_op.Constraint(expr=expr,
                                                    name=mdl_eq.name + '_' + str(kdx) + '_' + str(ddx),
                                                    cstr_type=mdl_eq.cstr_type)
                    self.append(local_cstr)

            # continuity condition between (kdx, -1) and (kdx + 1)
            continuity_cstr = Collocation.get_continuity_constraint(V, kdx)
            self.append(continuity_cstr)

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
                self.append(local_cstr)

        return None

    def expand_with_other_collocation(self):

        # todo: add this.
        #  notice, that the logic flow of non-radau collection was *never* actually triggered in previous iterates.
        #  there would certainly have been an error otherwise, see, for example,
        #  the inclusion of the un-defined variable ms_z0[:, kdx]

        message = 'OCP discretization with non-Radau collection is not supported at present.'
        awelogger.logger.error(message)
        raise Exception(message)

    def expand_with_multiple_shooting(self, nlp_options, P, V, Xdot, model, dae, Multiple_shooting, ms_z0, ms_xf):

        n_k = nlp_options['n_k']

        for kdx in range(n_k):

            # at each interval node, algebraic constraints should be satisfied
            alg_cstr = get_algebraic_constraints(dae.z(ms_z0[:, kdx]), V, kdx)
            self.append(alg_cstr)

            ms_path_cstr = get_inequality_path_constraints(nlp_options, model, V, P, Xdot, kdx)
            self.append(ms_path_cstr)

            # endpoint should match next start point
            cont_cstr = Multiple_shooting.get_continuity_constraint(ms_xf, V, kdx)
            self.append(cont_cstr)



def get_algebraic_constraints(z_at_time, V, kdx):

    cstr_list = OcpConstraintList()

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


def get_inequality_path_constraints(nlp_options, model, V, P, Xdot, kdx):

    cstr_list = OcpConstraintList()

    mdl_cstr_list = model.constraints_list
    model_variables = model.variables
    model_parameters = model.parameters

    vars_at_time = struct_op.get_variables_at_time(nlp_options, V, Xdot, model_variables, kdx)
    params_at_time = struct_op.get_parameters_at_time(nlp_options, P, V, Xdot, model_variables,
                                                      model_parameters, kdx)

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

def translate_cstr_type(constraint_type):

    # convention h(w) <= 0
    if constraint_type == 'inequality':
        return 'ineq'
    elif constraint_type == 'equality':
        return 'eq'
    else:
        raise ValueError('Wrong constraint type chosen. Possible values: "inequality" / "equality" ')

    return None