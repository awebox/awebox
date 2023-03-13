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
var_bounds code of the awebox
takes variable struct and options to and model inequalities, generates constraint structures, and defines the nlp constraints
python-3.5 / casadi-3.4.5
- refactored from awebox code (elena malz, chalmers; jochem de schutter, alu-fr; rachel leuthold, alu-fr), 2018
- edited: rachel leuthold, jochem de schutter alu-fr 2020
'''
import pdb

import casadi.tools as cas

import awebox.tools.struct_operations as struct_op
import awebox.tools.performance_operations as perf_op
import awebox.tools.print_operations as print_op

from awebox.logger.logger import Logger as awelogger
import awebox.ocp.operation as operation

def get_scaled_variable_bounds(nlp_options, V, model):

    # initialize
    vars_lb = V(-cas.inf)
    vars_ub = V(cas.inf)

    distinct_variables = struct_op.get_distinct_V_indices(V)

    n_k = nlp_options['n_k']
    d = nlp_options['collocation']['d']

    periodic = perf_op.determine_if_periodic(nlp_options)

    # fill in bounds
    for canonical in distinct_variables:

        [var_is_coll_var, var_type, kdx, ddx, name, dim] = struct_op.get_V_index(canonical)
        use_depending_on_periodicity = ((periodic and (not kdx is None) and (kdx < n_k)) or (not periodic))

        if (var_type == 'x'):

            if nlp_options['collocation']['u_param'] == 'poly' and var_is_coll_var:
                vars_lb['coll_var', kdx, ddx, var_type, name] = model.variable_bounds[var_type][name]['lb']
                vars_ub['coll_var', kdx, ddx, var_type, name] = model.variable_bounds[var_type][name]['ub']
            
            elif nlp_options['collocation']['u_param'] == 'zoh' and use_depending_on_periodicity and (not var_is_coll_var):
                vars_lb[var_type, kdx, name] = model.variable_bounds[var_type][name]['lb']
                vars_ub[var_type, kdx, name] = model.variable_bounds[var_type][name]['ub']

            [vars_lb, vars_ub] = assign_phase_fix_bounds(nlp_options, model, vars_lb, vars_ub, var_is_coll_var,
                                                            var_type, kdx, ddx, name)

        elif (var_type in {'z', 'u'}):
            if (var_type in V.keys()) and (not var_is_coll_var):

                vars_lb[var_type, kdx, name] = model.variable_bounds[var_type][name]['lb']
                vars_ub[var_type, kdx, name] = model.variable_bounds[var_type][name]['ub']

            elif nlp_options['collocation']['u_param'] == 'poly' and var_is_coll_var:
                vars_lb['coll_var', kdx, ddx, var_type, name] = model.variable_bounds[var_type][name]['lb']
                vars_ub['coll_var', kdx, ddx, var_type, name] = model.variable_bounds[var_type][name]['ub']

        elif (var_type == 'theta'):
            if name == 't_f':

                if nlp_options['phase_fix'] == 'single_reelout':
                    vars_lb[var_type, name] = cas.DM.zeros(vars_lb[var_type, name].shape)

                if nlp_options['phase_fix'] == 'simple':
                    vars_lb[var_type, name] = model.variable_bounds[var_type][name]['lb']
                    vars_ub[var_type, name] = model.variable_bounds[var_type][name]['ub']
            else:
                vars_lb[var_type, name] = model.variable_bounds[var_type][name]['lb']
                vars_ub[var_type, name] = model.variable_bounds[var_type][name]['ub']

        elif var_type == 'phi':
            vars_lb[var_type, name] = model.parameter_bounds[name]['lb']
            vars_ub[var_type, name] = model.parameter_bounds[name]['ub']

    return [vars_lb, vars_ub]


def assign_phase_fix_bounds(nlp_options, model, vars_lb, vars_ub, coll_flag, var_type, kdx, ddx, name):

    if nlp_options['system_type'] == 'drag_mode':
        # drag-mode phase fixing: fix y-speed of first system node
        if (kdx == 0) and (not coll_flag) and (name == 'dq10') and (var_type == 'x'):
            vars_lb[var_type, 0, name, 1] = 0.0
            vars_ub[var_type, 0, name, 1] = 0.0

    elif nlp_options['system_type'] == 'lift_mode':
        # lift-mode phase fixing

        if (name == 'dl_t') and not (var_type == 'x'):
            if coll_flag:
                vars_ub['coll_var', kdx, ddx, var_type, name] = cas.inf
                vars_lb['coll_var', kdx, ddx, var_type, name] = -cas.inf
            else:
                vars_ub[var_type, kdx, name] = cas.inf
                vars_lb[var_type, kdx, name] = -cas.inf

        if (name == 'dl_t') and (var_type == 'x'):
            n_k = nlp_options['n_k']
            d = nlp_options['collocation']['d']
            periodic, _, _, _, _, _, _ = operation.get_operation_conditions(nlp_options)
            radau_scheme = (nlp_options['collocation']['scheme'] == 'radau')
            poly_controls = (nlp_options['collocation']['u_param'] == 'poly')
            zoh_controls = (nlp_options['collocation']['u_param'] == 'zoh')

            given_max_value = model.variable_bounds[var_type][name]['ub']
            given_min_value = model.variable_bounds[var_type][name]['lb']

            at_initial_control_node = (kdx == 0) and (not coll_flag)

            if nlp_options['phase_fix'] == 'single_reelout':

                switch_kdx = round(nlp_options['n_k'] * nlp_options['phase_fix_reelout'])
                in_reelout_phase = (kdx < switch_kdx)
                in_reelin_phase = not in_reelout_phase

                at_periodic_initial_control_node = at_initial_control_node and periodic
                at_periodic_final_control_node = (kdx == n_k) and periodic and (not coll_flag)
                at_switching_control_node = (kdx == switch_kdx) and (not coll_flag)

                at_collocation_node_without_control_freedom = coll_flag and zoh_controls
                at_collocation_node_that_overlaps_with_control_node = coll_flag and (ddx == d-1) and radau_scheme
                at_collocation_node_with_control_freedom = coll_flag and poly_controls and (not at_collocation_node_that_overlaps_with_control_node)
                at_reelout_collocation_node_with_control_freedom = in_reelout_phase and at_collocation_node_with_control_freedom
                at_reelin_collocation_node_with_control_freedom = in_reelout_phase and at_collocation_node_with_control_freedom

                at_reelout_control_node = in_reelout_phase and (not coll_flag)
                at_reelin_control_node = in_reelin_phase and (not coll_flag)

                if at_periodic_initial_control_node:
                    max = cas.inf
                    min = -cas.inf

                elif at_periodic_final_control_node:
                    max = 0.
                    min = 0.

                elif at_switching_control_node:
                    max = 0.
                    min = 0.

                elif at_collocation_node_without_control_freedom:
                    max = cas.inf
                    min = -cas.inf

                elif at_collocation_node_that_overlaps_with_control_node:
                    max = cas.inf
                    min = -cas.inf

                elif at_reelout_collocation_node_with_control_freedom:
                    max = given_max_value
                    min = 0.

                elif at_reelin_collocation_node_with_control_freedom:
                    max = 0.
                    min = given_min_value

                elif at_reelout_control_node:
                    max = given_max_value
                    min = 0.

                elif at_reelin_control_node:
                    max = 0.
                    min = given_min_value
                else:
                    message = 'node classification within single reel-out phase-fixing, is undefined for node: \n'
                    message += 'coll_flag = ' + str(coll_flag) + ', kdx = ' + str(kdx) + ', ddx = ' + str(ddx)
                    print_op.log_and_raise_error(message)

                if coll_flag:
                    vars_ub['coll_var', kdx, ddx, var_type, name] = max
                    vars_lb['coll_var', kdx, ddx, var_type, name] = min
                else:
                    vars_ub[var_type, kdx, name] = max
                    vars_lb[var_type, kdx, name] = min

            elif nlp_options['phase_fix'] == 'single_reelout':
                if at_initial_control_node:
                    max = 0.
                    min = 0.
                    vars_ub[var_type, 0, name] = max
                    vars_lb[var_type, 0, name] = min

        pumping_range = nlp_options['pumping_range']
        if name == 'l_t' and (len(pumping_range) == 2) and (pumping_range[0] is not None) and (pumping_range[1] is not None):

            pumping_range_0_scaled = struct_op.var_si_to_scaled('x', 'l_t', nlp_options['pumping_range'][0], model.scaling)
            pumping_range_1_scaled = struct_op.var_si_to_scaled('x', 'l_t', nlp_options['pumping_range'][1], model.scaling)

            if kdx == 0 and (not coll_flag) and nlp_options['pumping_range'][0]:
                vars_lb[var_type, 0, name] = pumping_range_0_scaled
                vars_ub[var_type, 0, name] = pumping_range_0_scaled
            if kdx == switch_kdx and (not coll_flag) and nlp_options['pumping_range'][1]:
                vars_lb[var_type, kdx, name] = pumping_range_1_scaled
                vars_ub[var_type, kdx, name] = pumping_range_1_scaled

    return vars_lb, vars_ub
