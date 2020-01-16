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
import pdb
import awebox.mdl.aero.vortex_dir.fixing as vortex_fix
import awebox.mdl.aero.vortex_dir.strength as vortex_strength

def setup_constraint_structure(nlp_numerics_options, model, formulation):

    constraints_entry_list = make_constraints_entry_list(nlp_numerics_options, formulation.constraints, model)

    # Constraints structure
    g_struct = cas.struct_symSX(constraints_entry_list)

    return g_struct

def make_constraints_entry_list(nlp_numerics_options, constraints, model):

    # get discretization information
    nk = nlp_numerics_options['n_k']

    # initialize entry list
    constraints_entry_list = []

    # size of algebraic variables on interval nodes
    nz = 0
    if nlp_numerics_options['lift_xddot']:
        nz += model.variables['xddot'].shape[0]
    if nlp_numerics_options['lift_xa']:
        nz += model.variables['xa'].shape[0]
        if 'xl' in list(model.variables.keys()):
            nz += model.variables['xl'].shape[0]

    # add initial constraints
    if list(constraints['initial'].keys()): # check if not empty
        constraints_entry_list.append(cas.entry('initial', struct = constraints['initial']))

    # empty tuple for nested constraints
    entry_tuple = ()

    if nlp_numerics_options['discretization'] == 'direct_collocation':

        # extract collocation parameters
        d = nlp_numerics_options['collocation']['d']
        scheme = nlp_numerics_options['collocation']['scheme']

        # make stage_constraints to be applied on the collocation nodes
        stage_constraints = make_stage_constraint_struct(model)

        if scheme != 'radau':
            # for legendre: add path constrains on interval nodes
            if list(model.constraints.keys()):  # check if not empty

                if nz > 0: # if there are any lifted algebraic vars on interval node
                    entry_tuple += (cas.entry('algebraic_constraints', repeat = [nk],    shape = (nz,1)),)

                entry_tuple += (
                    cas.entry('path_constraints',      repeat = [nk],    struct = model.constraints),
                    cas.entry('stage_constraints',     repeat = [nk, d], struct = stage_constraints),
                )

        else:
            # for radau: omit path constrains on interval nodes
            entry_tuple += (
                    cas.entry('stage_constraints', repeat = [nk, d], struct = stage_constraints),
            )

    elif nlp_numerics_options['discretization'] == 'multiple_shooting':

        # add path constraints at interval nodes
        if list(model.constraints.keys()):  # check if not empty
            if nz > 0: # if there are any lifted algebraic vars on interval node
                entry_tuple += (cas.entry('algebraic_constraints', repeat = [nk], shape = (nz,1)),)

            entry_tuple += (cas.entry('path_constraints',      repeat = [nk], struct = model.constraints),)

    # add continuity constraints
    entry_tuple += (
        cas.entry('continuity', repeat = [nk], shape = model.variables['xd'].size()),
    )

    # add stage and continuity constraints to list
    constraints_entry_list.append(entry_tuple)

    # add terminal constraints
    if list(constraints['terminal'].keys()): # check if not empty
        constraints_entry_list.append(cas.entry('terminal', struct = constraints['terminal']))

    # add periodicity constraints
    if list(constraints['periodic'].keys()):
        constraints_entry_list.append(cas.entry('periodic', struct = constraints['periodic']))

    if list(constraints['wake_fix'].keys()):
        constraints_entry_list.append(cas.entry('wake_fix', struct = constraints['wake_fix']))

    if list(constraints['vortex_strength'].keys()):
        constraints_entry_list.append(cas.entry('vortex_strength', struct = constraints['vortex_strength']))

    if list(constraints['integral'].keys()):
        constraints_entry_list.append(cas.entry('integral', struct=constraints['integral']))

    return constraints_entry_list

def make_stage_constraint_struct(model):

    # make entry list to check if not empty
    entry_list = [cas.entry('collocation', shape =model.dynamics(model.variables, model.parameters).size())]
    if list(model.constraints.keys()):  # check if not empty
        entry_list.append(cas.entry('path_constraints', struct = model.constraints))

    # stage constraints structure -- necessary for interleaving
    stage_constraints = cas.struct_symSX(entry_list)

    return stage_constraints

def create_constraint_outputs(g_list, g_bounds, g_struct, V, P):

    g = g_struct(cas.vertcat(*g_list))
    g_fun = cas.Function('g_fun', [V, P], [g.cat])
    g_jacobian_fun = cas.Function('g_jacobian_fun',[V,P],[g.cat, cas.jacobian(g.cat, V.cat)])

    g_bounds['lb'] = cas.vertcat(*g_bounds['lb'])
    g_bounds['ub'] = cas.vertcat(*g_bounds['ub'])

    return g, g_fun, g_jacobian_fun, g_bounds

def append_algebraic_constraints(g_list, g_bounds, z_at_time, V, kdx):

    # extract
    g_algebraic = np.array([])

    if 'xddot' in list(V.keys()):
        xddot_at_time = z_at_time['xddot']
        g_algebraic = cas.vertcat(g_algebraic, xddot_at_time - V['xddot', kdx])

    if 'xa' in list(V.keys()):
        xa_at_time = z_at_time['xa']
        g_algebraic = cas.vertcat(g_algebraic, xa_at_time - V['xa',kdx])

    if 'xl' in list(V.keys()):
        xl_at_time = z_at_time['xl']
        g_algebraic = cas.vertcat(g_algebraic, xl_at_time - V['xl', kdx])

    g_list.append(g_algebraic)
    g_bounds = append_constraint_bounds(g_bounds, 'equality', g_algebraic.shape[0])

    return [g_list, g_bounds]

def append_collocation_constraints(g_list, g_bounds, dynamics):

    # evaluate constraint
    g_list.append(dynamics)
    # add constraint bounds
    g_bounds = append_constraint_bounds(g_bounds, 'equality', dynamics.size()[0])

    return [g_list, g_bounds]

def append_path_constraints(g_list, g_bounds, path_constraints, path_constraints_values, slacks = None):

    if slacks is not None:

        path_constraints_struct = path_constraints(path_constraints_values)

        # append slacked constraints
        if 'equality' in list(path_constraints_struct.keys()):
            g_list.append(path_constraints_struct['equality'])
        if 'inequality' in list(path_constraints_struct.keys()):
            g_list.append(path_constraints_struct['inequality'] - slacks)

        # append constraint bounds
        for cstr_type in list(path_constraints.keys()):
            g_bounds = append_constraint_bounds(g_bounds, 'equality', path_constraints[cstr_type].size()[0])

    else:

        # append constraint
        g_list.append(path_constraints_values)
        # append constraint bounds
        for cstr_type in list(path_constraints.keys()):
            g_bounds = append_constraint_bounds(g_bounds, cstr_type, path_constraints[cstr_type].size()[0])

    return [g_list, g_bounds]

def append_terminal_constraints(g_list, g_bounds, constraints, constraints_fun, var_terminal, var_ref_terminal, xi):

    # evaluate constraint
    g_terminal = constraints_fun['terminal'](var_terminal, var_ref_terminal, xi)

    # append constraint
    g_list.append(g_terminal)
    # append constraint bounds
    for cstr_type in list(constraints['terminal'].keys()): # cstr_type = equality / inequality
        g_bounds = append_constraint_bounds(g_bounds, cstr_type, constraints['terminal'][cstr_type].size()[0])

    return [g_list, g_bounds]

def append_initial_constraints(g_list, g_bounds, constraints, constraints_fun, var_initial, var_ref_initial, xi):

    # evaluate constraint
    g_initial = constraints_fun['initial'](var_initial, var_ref_initial, xi)

    # append constraint
    g_list.append(g_initial)
    # append constraint bounds
    for cstr_type in list(constraints['initial'].keys()): # cstr_type = equality / inequality
        g_bounds = append_constraint_bounds(g_bounds, cstr_type, constraints['initial'][cstr_type].size()[0])

    return [g_list, g_bounds]

def append_wake_fix_constraints(options, g_list, g_bounds, V, Outputs, model):

    induction_model = options['induction']['induction_model']
    periods_tracked = options['induction']['vortex_periods_tracked']

    if induction_model == 'vortex':
        g_list, g_bounds = vortex_fix.fixing_constraints_on_zeroth_period(options, g_list, g_bounds, V, Outputs, model)

        for period in range(1, periods_tracked):
            g_list, g_bounds = vortex_fix.fixing_constraints_on_previous_period(options, g_list, g_bounds, V, Outputs, model, period)

    return [g_list, g_bounds]

def append_vortex_strength_constraints(options, g_list, g_bounds, V, Outputs, model):

    induction_model = options['induction']['induction_model']
    periods_tracked = options['induction']['vortex_periods_tracked']

    if induction_model == 'vortex':
        for period in range(periods_tracked):
            g_list, g_bounds = vortex_strength.fix_vortex_strengths(options, g_list, g_bounds, V, Outputs, model, period)

    return [g_list, g_bounds]


def append_periodic_constraints(g_list, g_bounds, constraints, constraints_fun, var_init, var_terminal):

    # evaluate constraint
    g_periodic = constraints_fun['periodic'](var_init, var_terminal)

    # append constraint
    g_list.append(g_periodic)
    # append constraint bounds
    for cstr_type in list(constraints['periodic'].keys()): # cstr_type = equality / inequality
        g_bounds = append_constraint_bounds(g_bounds, cstr_type, constraints['periodic'][cstr_type].size()[0])

    return [g_list, g_bounds]

def append_integral_constraints(nlp_numerics_options, g_list, g_bounds, integral_list, constraints, constraints_fun, V, Xdot, model, integral_constants):

    # nu = V['phi','nu']
    integral_sum = {}
    g_integral = {}

    for cstr_type in list(integral_constants.keys()):
        integral_t0 = integral_constants[cstr_type]
        integral_sum[cstr_type] = 0.
        for i in range(len(integral_list)):
            integral_sum[cstr_type] += integral_list[i][cstr_type]
        g_integral[cstr_type] = - integral_t0 - integral_sum[cstr_type]
        g_integral[cstr_type] /= integral_t0
        g_list.append(g_integral[cstr_type])

    for cstr_type in list(constraints['integral'].keys()):
        g_bounds = append_constraint_bounds(g_bounds, cstr_type, g_integral[cstr_type].size()[0])

    return [g_list, g_bounds]

def append_constraint_bounds(constraint_bounds, constraint_type, ndg):

    # convention h(w) <= 0
    if constraint_type == 'inequality':
        constraint_bounds['lb'].append(np.array([-cas.inf]*ndg))
    elif constraint_type == 'equality':
        constraint_bounds['lb'].append(np.zeros(ndg))
    else:
        raise ValueError('Wrong constraint type chosen. Possible values: "inequality" / "equality" ')

    # upper bound is always zero
    constraint_bounds['ub'].append(np.zeros(ndg))

    return constraint_bounds
