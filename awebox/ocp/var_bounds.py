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
var_bounds code of the awebox
takes variable struct and options to and model inequalities, generates constraint structures, and defines the nlp constraints
python-3.5 / casadi-3.4.5
- refactored from awebox code (elena malz, chalmers; jochem de schutter, alu-fr; rachel leuthold, alu-fr), 2018
- edited: rachel leuthold, jochem de schutter alu-fr 2020
'''

import casadi.tools as cas

import awebox.tools.struct_operations as struct_op
import awebox.tools.performance_operations as perf_op
import awebox.tools.print_operations as print_op

import pdb


def get_variable_bounds(nlp_options, V, model):

    # initialize
    vars_lb = V(-cas.inf)
    vars_ub = V(cas.inf)

    distinct_variables = struct_op.get_distinct_V_indices(V)

    # fill in bounds
    for canonical in distinct_variables:

        [var_is_coll_var, var_is_us, var_type, kdx, ddx, name, dim] = struct_op.get_V_index(canonical)

        if var_is_us:
            # bounds on slacks (convention: h(x) < 0)
            vars_ub['us'] = 0.0

        elif var_is_coll_var:

            vars_lb['coll_var', kdx, ddx, var_type, name] = model.variable_bounds[var_type][name]['lb']
            vars_ub['coll_var', kdx, ddx, var_type, name] = model.variable_bounds[var_type][name]['ub']

            [vars_lb, vars_ub] = assign_phase_fix_bounds(nlp_options, model, vars_lb, vars_ub, var_is_coll_var, var_type, kdx, ddx, name)

        else:

            if var_type == 'xd':

                use_radau = nlp_options['collocation']['scheme'] == 'radau'
                coll_variables_already_bounded = ('coll_var' in V.keys()) and use_radau
                periodic = perf_op.determine_if_periodic(nlp_options)

                if coll_variables_already_bounded and periodic:
                    # do nothing. this prevents licq errors at active variable bounds
                    32.0

                elif coll_variables_already_bounded and (not periodic):
                    # only apply bounds at nodes that are not also described by collocation nodes:

                    if kdx == 0:
                        vars_lb[var_type, kdx, name] = model.variable_bounds[var_type][name]['lb']
                        vars_ub[var_type, kdx, name] = model.variable_bounds[var_type][name]['ub']

                        [vars_lb, vars_ub] = assign_phase_fix_bounds(nlp_options, model, vars_lb, vars_ub, var_is_coll_var,
                                                                     var_type, kdx, ddx, name)

                    else:
                        # do nothing
                        32.0

                else:
                    # apply the bounds at all kdx, because ddx=-1 does not already constrain these variables
                    vars_lb[var_type, kdx, name] = model.variable_bounds[var_type][name]['lb']
                    vars_ub[var_type, kdx, name] = model.variable_bounds[var_type][name]['ub']

                    [vars_lb, vars_ub] = assign_phase_fix_bounds(nlp_options, model, vars_lb, vars_ub, var_is_coll_var,
                                                                 var_type, kdx, ddx, name)


            if var_type in {'xl', 'xa','u'}:

                vars_lb[var_type, kdx, name] = model.variable_bounds[var_type][name]['lb']
                vars_ub[var_type, kdx, name] = model.variable_bounds[var_type][name]['ub']

                [vars_lb, vars_ub] = assign_phase_fix_bounds(nlp_options, model, vars_lb, vars_ub, var_is_coll_var, var_type, kdx, ddx, name)

            if var_type == 'theta':
                vars_lb[var_type, name] = model.variable_bounds[var_type][name]['lb']
                vars_ub[var_type, name] = model.variable_bounds[var_type][name]['ub']

            if var_type == 'phi':
                vars_lb[var_type, name] = model.parameter_bounds[name]['lb']
                vars_ub[var_type, name] = model.parameter_bounds[name]['ub']

    return [vars_lb, vars_ub]

def assign_phase_fix_bounds(nlp_options, model, vars_lb, vars_ub, coll_flag, var_type, kdx, ddx, name):

    if name == 'dl_t' and nlp_options['phase_fix']:
        if kdx < round(nlp_options['n_k'] * nlp_options['phase_fix_reelout']):
            if not coll_flag:
                vars_lb[var_type, kdx, name] = 0.0
                vars_ub[var_type, kdx, name] = model.variable_bounds[var_type][name]['ub']
            else:
                vars_lb['coll_var', kdx, ddx, var_type, name] = 0.0
                vars_ub['coll_var', kdx, ddx, var_type, name] = model.variable_bounds[var_type][name]['ub']

        else:
            if not coll_flag:
                vars_lb[var_type, kdx, name] = model.variable_bounds[var_type][name]['lb']
                vars_ub[var_type, kdx, name] = 0.0
            else:
                vars_lb['coll_var', kdx, ddx, var_type, name] = model.variable_bounds[var_type][name]['lb']
                vars_ub['coll_var', kdx, ddx, var_type, name] = 0.0

    if name == 'l_t' and nlp_options['pumping_range']:

        if kdx == 0 and (not coll_flag) and nlp_options['pumping_range'][0]:
            vars_lb[var_type, 0, name] = nlp_options['pumping_range'][0]
            vars_ub[var_type, 0, name] = nlp_options['pumping_range'][0]


        if kdx == round(nlp_options['n_k'] * nlp_options['phase_fix_reelout']) and (not coll_flag) and nlp_options['pumping_range'][1]:
            vars_lb[var_type, kdx, name] = nlp_options['pumping_range'][1]
            vars_ub[var_type, kdx, name] = nlp_options['pumping_range'][1]

    return vars_lb, vars_ub
