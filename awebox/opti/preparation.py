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
''' preparation for running the homotopy process.

python-3.5 / casadi-3.4.5
- authors: rachel leuthold, thilo bronnenmeyer, alu-fr 2018
'''

from .initialization_dir import modular as initialization_modular, initialization

from . import reference

import awebox.tools.struct_operations as struct_op

import copy

import casadi as cas

from awebox.logger.logger import Logger as awelogger
import awebox.tools.print_operations as print_op

def initialize_arg(nlp, formulation, model, options, warmstart_solution_dict = None):

    # V_init = initialization.get_initial_guess(nlp, model, formulation, options)
    if options['initialization']['initialization_type'] == 'default':
        V_init = initialization.get_initial_guess(nlp, model, formulation, options['initialization'])
    elif options['initialization']['initialization_type'] == 'modular':
        V_init = initialization_modular.get_initial_guess(nlp, model, formulation, options['initialization'])

    use_warmstart = not (warmstart_solution_dict == None)
    if use_warmstart:
        [V_init_proposed, _, _] = struct_op.setup_warmstart_data(nlp, warmstart_solution_dict)
        V_shape_matches = (V_init_proposed.cat.shape == nlp.V.cat.shape)
        if V_shape_matches:
            V_init = V_init_proposed
        else:
            raise ValueError('Variables of specified warmstart do not correspond to NLP requirements.')

    V_ref = reference.get_reference(nlp, model, V_init, options)

    p_fix_num = set_p_fix_num(V_ref, nlp, model, V_init, options)

    [V_bounds, g_bounds] = set_initial_bounds(nlp, model, formulation, options, V_init)

    V_bounds = fix_q_and_r_values_if_necessary(options, nlp, model, V_bounds, V_init)

    arg = {}
    # initial condition
    arg['x0'] = V_init

    # bounds on x
    arg['lbx'] = V_bounds['lb']
    arg['ubx'] = V_bounds['ub']

    # bounds on g
    arg['lbg'] = g_bounds['lb']
    arg['ubg'] = g_bounds['ub']
    arg['p'] = p_fix_num  # hand over the parameters to the solver

    return arg

def set_p_fix_num(V_ref, nlp, model, V_init, options):
    # --------------------
    # parameter values
    # --------------------
    # build reference parameters, references of cost function should match the
    # initial guess

    P = nlp.P
    p_fix_num = P(0.)
    p_fix_num['p', 'weights'] = 1.0e-8

    # weights and references
    for variable_type in set(model.variables.keys()):
        for name in struct_op.subkeys(model.variables, variable_type):
            # set weights
            var_name, _ = struct_op.split_name_and_node_identifier(name)

            if var_name[0] == 'w':
                # then, this is a vortex wake variable
                var_name = 'w'


            if var_name in list(options['weights'].keys()):  # global variable
                p_fix_num['p', 'weights', variable_type, name] = options['weights'][var_name]
            else:
                p_fix_num['p', 'weights', variable_type, name] = 1.0

            # set references
            if variable_type == 'u':
                if 'u' in V_ref.keys():
                    p_fix_num['p', 'ref', variable_type, :, name] = V_ref[variable_type, :, name]
                else:
                    p_fix_num['p', 'ref', 'coll_var', :, :, variable_type, name] = V_ref['coll_var', :, :, variable_type, name]

            elif variable_type == 'theta':
                p_fix_num['p', 'ref', variable_type, name] = V_ref[variable_type, name]
            elif variable_type in {'x','z'}:
                if variable_type in list(V_ref.keys()):
                    p_fix_num['p', 'ref', variable_type, :, name] = V_ref[variable_type, :, name]
                if 'coll_var' in list(V_ref.keys()):
                    p_fix_num['p', 'ref', 'coll_var', :, :, variable_type, name] = V_ref['coll_var', :, :, variable_type, name]

    # system parameters
    param_options = options['initialization']['sys_params_num']
    for param_type in list(param_options.keys()):
        if isinstance(param_options[param_type],dict):
            for param in list(param_options[param_type].keys()):
                if isinstance(param_options[param_type][param],dict):
                    for subparam in list(param_options[param_type][param].keys()):
                        p_fix_num['theta0',param_type,param,subparam] = param_options[param_type][param][subparam]

                else:
                    p_fix_num['theta0',param_type,param] = options['initialization']['sys_params_num'][param_type][param]

        else:
            p_fix_num['theta0',param_type] = param_options[param_type]

    use_vortex_linearization = 'lin' in P.keys()
    if use_vortex_linearization:
        p_fix_num['lin'] = V_init

    return p_fix_num

def set_initial_bounds(nlp, model, formulation, options, V_init):
    V_bounds = {}

    for name in list(nlp.V_bounds.keys()):
        V_bounds[name] = copy.deepcopy(nlp.V_bounds[name])

    g_bounds = copy.deepcopy(nlp.g_bounds)

    if 'ellipse_half21' in model.constraints_dict['inequality'].keys():
        g_ub = nlp.g(g_bounds['ub'])
        switch_kdx = round(nlp.options['n_k'] * nlp.options['phase_fix_reelout'])
        for k in range(switch_kdx+1, nlp.options['n_k']):
            g_ub['path', k, 'ellipse_half21'] = 1e5
            g_ub['path', k, 'ellipse_half31'] = 1e5
        g_bounds['ub'] = g_ub.cat

    # set homotopy parameters
    for name in list(model.parameters_dict['phi'].keys()):
        V_bounds['lb']['phi', name] = 1.
        V_bounds['ub']['phi', name] = 1.

    for name in list(formulation.xi_dict['xi_bounds'].keys()):
        xi_bounds = formulation.xi_dict['xi_bounds']
        V_bounds['lb']['xi', name] = xi_bounds[name][0]
        V_bounds['ub']['xi', name] = xi_bounds[name][1]

    for name in struct_op.subkeys(model.variables, 'theta'):
        if not name == 't_f' and not name[:3] == 'l_c' and not name[:6] == 'diam_c':
            initial_si_value = options['initialization']['theta'][name]
            initial_scaled_value = initial_si_value / model.scaling['theta'][name]

            V_bounds['ub']['theta', name] = initial_scaled_value
            V_bounds['lb']['theta', name] = initial_scaled_value

    initial_si_time = V_init['theta','t_f'] # * options['homotopy']['phase_fix'] #todo: move phase fixing to nlp
    initial_scaled_time = initial_si_time / model.scaling['theta']['t_f']

    # set theta parameters
    V_bounds['lb']['theta', 't_f'] = initial_scaled_time
    V_bounds['ub']['theta', 't_f'] = initial_scaled_time

    if 'P_max' in model.variables_dict['theta'].keys():
        if options['cost']['P_max'][0] == 1.0:
            V_bounds['lb']['theta', 'P_max'] = 1e3
            V_bounds['ub']['theta', 'P_max'] = 1e3
            nlp.V_bounds['lb']['theta', 'P_max'] = 1e3
            nlp.V_bounds['ub']['theta', 'P_max'] = 1e3

    # set fictitious forces bounds
    for name in list(model.variables_dict['u'].keys()):
        if 'fict' in name:
            if 'u' in V_init.keys():
                V_bounds['lb']['u', :, name] = -cas.inf
                V_bounds['ub']['u', :, name] = cas.inf
            else:
                V_bounds['lb']['coll_var', :, :, 'u', name] = -cas.inf
                V_bounds['ub']['coll_var', :, :, 'u', name] = cas.inf

    # if phase-fix, first free dl_t before introducing phase-fix in switch to power
    if nlp.V['theta','t_f'].shape[0] > 1:
        V_bounds['lb']['x',:,'dl_t'] = model.variable_bounds['x']['dl_t']['lb']
        V_bounds['ub']['x',:,'dl_t'] = model.variable_bounds['x']['dl_t']['ub']

        # make sure that pumping range fixing bounds are not imposed initially
        # TODO: write if test.....
        V_bounds['lb']['x',:,'l_t'] = V_bounds['lb']['x',1,'l_t']
        V_bounds['ub']['x',:,'l_t'] = V_bounds['ub']['x',1,'l_t']

        if 'coll_var' in list(nlp.V.keys()):
            V_bounds['lb']['coll_var',:,:,'x','dl_t'] = model.variable_bounds['x']['dl_t']['lb']
            V_bounds['ub']['coll_var',:,:,'x','dl_t'] = model.variable_bounds['x']['dl_t']['ub']

    return V_bounds, g_bounds


def generate_default_solver_options(options):

    opts = {}
    opts['expand'] = options['expand']

    if options['nlp_solver'] == 'ipopt':
        opts['ipopt.linear_solver'] = options['linear_solver']
        opts['ipopt.max_iter'] = options['max_iter']
        opts['ipopt.max_cpu_time'] = options['max_cpu_time']

        opts['ipopt.mu_target'] = options['mu_target']
        opts['ipopt.mu_init'] = options['mu_init']
        opts['ipopt.tol'] = options['tol']
        opts['ipopt.ma57_automatic_scaling'] = 'yes'

        if awelogger.logger.getEffectiveLevel() > 10:
            opts['ipopt.print_level'] = 0
            opts['print_time'] = 0
            opts['ipopt.sb'] = 'yes'

        if options['hessian_approximation']:
            opts['ipopt.hessian_approximation'] = 'limited-memory'

    opts['record_time'] = 1

    opts['jit'] = options['jit']
    if options['jit']:
        opts['compiler'] = options['compiler']
        opts['jit_options'] = {'flags': options['jit_flags']}

    return opts

def generate_solvers(awebox_callback, model, nlp, formulation, options):

    initial_opts = generate_default_solver_options(options)
    middle_opts = generate_default_solver_options(options)
    final_opts = generate_default_solver_options(options)

    if options['nlp_solver'] == 'ipopt':
        initial_opts['ipopt.mu_target'] = options['mu_hippo']
        initial_opts['ipopt.acceptable_iter'] = options['acceptable_iter_hippo']#5
        initial_opts['ipopt.tol'] = options['tol_hippo']

        middle_opts['ipopt.mu_init'] = options['mu_hippo']
        middle_opts['ipopt.mu_target'] = options['mu_hippo']
        middle_opts['ipopt.acceptable_iter'] = options['acceptable_iter_hippo']#5
        middle_opts['ipopt.tol'] = options['tol_hippo']
        middle_opts['ipopt.warm_start_init_point'] = 'yes'

        final_opts['ipopt.mu_init'] = options['mu_hippo']
        final_opts['ipopt.warm_start_init_point'] = 'yes'

    if options['callback']:
        initial_opts['iteration_callback'] = awebox_callback
        initial_opts['iteration_callback_step'] = options['callback_step']

        middle_opts['iteration_callback'] = awebox_callback
        middle_opts['iteration_callback_step'] = options['callback_step']

        final_opts['iteration_callback'] = awebox_callback
        final_opts['iteration_callback_step'] = options['callback_step']

    if 'lift_mode' == 'lift_mode':  # todo: get from formulation property
        # do whatever it is that depends on lift-mode here....
        32.0

    if options['nlp_solver'] == 'ipopt':
        initial_solver = cas.nlpsol('solver', 'ipopt', nlp.get_nlp(), initial_opts)
        middle_solver = cas.nlpsol('solver', 'ipopt', nlp.get_nlp(), middle_opts)
        final_solver = cas.nlpsol('solver', 'ipopt', nlp.get_nlp(), final_opts)
    elif options['nlp_solver'] == 'worhp':
        initial_solver = cas.nlpsol('solver', 'worhp', nlp.get_nlp(), final_opts)
        middle_solver = initial_solver
        final_solver = initial_solver

    solvers = {}
    solvers['initial'] = initial_solver
    solvers['middle'] = middle_solver
    solvers['final'] = final_solver

    return solvers

def fix_q_and_r_values_if_necessary(solver_options, nlp, model, V_bounds, V_init):

    if solver_options['fixed_q_r_values']:

        indices = nlp.V.f['x', :, 'q10'] + nlp.V.f['x', 0, 0, 'dq10']
        for idx in indices:
            V_bounds['lb'].cat[idx] = V_init.cat[idx]
            V_bounds['ub'].cat[idx] = V_init.cat[idx]

        if model.kite_dof == 6:
            indices = nlp.V.f['x', 0, 'omega10'] + nlp.V.f['x', 0, 'r10']
            for idx in indices:
                V_bounds['lb'].cat[idx] = V_init.cat[idx]
                V_bounds['ub'].cat[idx] = V_init.cat[idx]

            # lower triangular elements
            indices = nlp.V.f['x', 1:, 'r10', 1] + nlp.V.f['x', 1:, 'r10', 2] + nlp.V.f['x', 1:, 'r10', 5]
            for idx in indices:
                V_bounds['lb'].cat[idx] = V_init.cat[idx]
                V_bounds['ub'].cat[idx] = V_init.cat[idx]

    return V_bounds
