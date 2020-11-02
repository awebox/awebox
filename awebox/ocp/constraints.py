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
# import awebox.mdl.aero.induction_dir.vortex_dir.fixing as vortex_fix
# import awebox.mdl.aero.induction_dir.vortex_dir.strength as vortex_strength


import awebox.ocp.operation as operation
import awebox.ocp.ocp_constraint as ocp_constraint

import awebox.tools.print_operations as print_op
import awebox.tools.struct_operations as struct_op
import awebox.tools.constraint_operations as cstr_op

from awebox.logger.logger import Logger as awelogger

import pdb


def get_constraints(nlp_options, V, P, Xdot, model, dae, formulation, Integral_constraint_list, Collocation, Multiple_shooting, ms_z0, ms_constraints, ms_xf):

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

    if multiple_shooting:
        ocp_cstr_list.expand_with_multiple_shooting(nlp_options, P, V, Xdot, model, dae, Multiple_shooting, ms_z0, ms_xf)

    elif radau_collocation:
        ocp_cstr_list.expand_with_radau_collocation(nlp_options, P, V, Xdot, model, Collocation)

    elif other_collocation:
        ocp_cstr_list.expand_with_other_collocation()

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

    print_op.warn_about_temporary_funcationality_removal('discret.wake')
    # [g_list, g_bounds] = constraints.append_wake_fix_constraints(nlp_options, g_list, g_bounds, V, Outputs, model)
    # [g_list, g_bounds] = constraints.append_vortex_strength_constraints(nlp_options, g_list, g_bounds, V, Outputs, model)

    return ocp_cstr_list


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
