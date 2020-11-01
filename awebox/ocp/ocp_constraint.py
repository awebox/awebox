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

import pdb

class OcpConstraintList(cstr_op.ConstraintList):
    def __init__(self):
        super().__init__()

    def expand_with_radau_collocation(self, nlp_options, P, V, Xdot, model, collocation):

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
            continuity_cstr = collocation.get_continuity_constraint(V, kdx)
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