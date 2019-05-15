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
"""
Periodic MPC routines for awebox models

:author: Jochem De Schutter (ALU Freiburg 2019)
"""

import awebox as awe
import awebox.viz.tools as viz_tools
import casadi.tools as ct
import logging
import matplotlib.pyplot as plt
import numpy as np
import copy

class Pmpc(object):

    def __init__(self, mpc_options, ts, trial):
        """ Constructor.
        """

        logging.info("Creating a {} periodic MPC controller with horizon length {}...".format(
            mpc_options['cost_type'], mpc_options['N']))
        logging.info("Based on direct collocation with {} polynomials of order {}.".format(
            mpc_options['scheme'], mpc_options['d']))

        # store discretization data
        self.__N = mpc_options['N']
        self.__d = mpc_options['d']
        self.__scheme = mpc_options['scheme']
        self.__cost_type = mpc_options['cost_type']
        self.__pocp_trial = trial
        self.__ts = ts
        self.__mpc_options = mpc_options

        # store model data
        self.__nx = trial.model.variables['xd'].shape[0]
        self.__nu = trial.model.variables['u'].shape[0]
        self.__nz = trial.model.variables['xa'].shape[0]

        # create mpc trial
        options = copy.deepcopy(trial.options)
        options['user_options']['trajectory']['type'] = 'mpc'
        options['nlp']['discretization'] = 'direct_collocation'
        options['nlp']['n_k'] = self.__N
        options['nlp']['d'] = self.__d
        options['nlp']['scheme'] = self.__scheme
        options['nlp']['collocation']['u_param'] = 'poly'
        options['visualization']['cosmetics']['plot_ref'] = True
        fixed_params = {}
        for name in list(self.__pocp_trial.model.variables_dict['theta'].keys()):
            if name != 't_f':
                fixed_params[name] = self.__pocp_trial.optimization.V_final['theta',name].full()
        fixed_params['t_f'] = self.__N*self.__ts
        options['user_options']['trajectory']['fixed_params'] = fixed_params

        self.__trial = awe.Trial(seed = options)
        self.__build_trial()

        # construct mpc solver
        self.__construct_solver()

        # create time-varying reference to track
        if self.__cost_type == 'tracking':
            self.__create_reference_interpolator()

        # periodic indexing
        self.__index = 0

        # initialize
        self.__initialize_solver()

        logging.info("Periodic MPC controller built.")

        return None

    def __build_trial(self):
        """ Build options, model, formulation and nlp of mpc trial.
        """

        logging.info("Building MPC trial...")

        # build
        import awebox.mdl.architecture as archi
        architecture = archi.Architecture(self.__trial.options['user_options']['system_model']['architecture'])
        self.__trial.options.build(architecture)
        self.__trial.model.build(self.__trial.options['model'], architecture)
        self.__trial.formulation.build(self.__trial.options['formulation'], self.__trial.model) 
        self.__trial.nlp.build(self.__trial.options['nlp'], self.__trial.model, self.__trial.formulation)
        self.__trial.visualization.build(self.__trial.model, self.__trial.nlp, 'MPC control', self.__trial.options)

        return None

    def __construct_solver(self):
        """ Construct casadi.nlpsol Object based on MPC trial information.
        """

        logging.info("Constructing MPC solver object...")

        if self.__cost_type == 'economic':

            # parameters
            self.__p = ct.struct_symMX([
                ct.entry('x0', shape = (self.__nx,1))
            ])

        if self.__cost_type == 'tracking':
        
            # parameters
            self.__p = ct.struct_symMX([
                ct.entry('x0',  shape = (self.__nx,)),
                ct.entry('ref', struct = self.__trial.nlp.V)
            ])

        # create P evaluator for use in NLP arguments
        self.__create_P_fun()

        # generate mpc constraints
        g = self.__trial.nlp.g_fun(self.__trial.nlp.V, self.__P_fun(self.__p))

        # generate cost function
        f = self.__generate_objective()

        # fill in nlp dict
        nlp = {'x': self.__trial.nlp.V, 'p': self.__p, 'f': f, 'g': g}

        # store nlp bounds
        self.__trial.nlp.V_bounds['ub']['phi'] = 0.0
        for name in list(self.__trial.model.variables_dict['u'].keys()):
            if 'fict' in name:
                self.__trial.nlp.V_bounds['lb']['coll_var',:,:,'u',name] = 0.0
                self.__trial.nlp.V_bounds['ub']['coll_var',:,:,'u',name] = 0.0


        self.__lbw = self.__trial.nlp.V_bounds['lb']
        self.__ubw = self.__trial.nlp.V_bounds['ub']
        self.__lbg = self.__trial.nlp.g_bounds['lb']
        self.__ubg = self.__trial.nlp.g_bounds['ub']

        logging_level = logging.getLogger().getEffectiveLevel()
        opts = {}
        opts['expand'] = self.__mpc_options['expand']
        opts['ipopt.linear_solver'] = self.__mpc_options['linear_solver']
        opts['ipopt.max_iter'] = self.__mpc_options['max_iter']
        opts['ipopt.max_cpu_time'] = self.__mpc_options['max_cpu_time']
        opts['jit'] = self.__mpc_options['jit']

        if logging_level > 10:
            opts['ipopt.print_level'] = 0
            opts['print_time'] = 0

        self.__solver = ct.nlpsol('solver', 'ipopt', nlp, opts)

        return None

    def step(self, x0, plot_flag = False):

        """ Compute periodic MPC feedback control for given initial condition.
        """
        
        logging.info("Compute MPC feedback...")

        # update nlp parameters
        self.__p0 = self.__p(0.0)
        self.__p0['x0'] = x0

        if self.__cost_type == 'tracking':
            ref = self.__get_reference(self.__index)
            self.__p0['ref'] = ref

        self.__p_fix_num = self.__P_fun(self.__p0)

        # DEBUGGING
        w0 = self.__w0

        # MPC problem
        sol = self.__solver(
            x0  = self.__w0,
            lbx = self.__lbw,
            ubx = self.__ubw,
            lbg = self.__lbg,
            ubg = self.__ubg,
            p   = self.__p0
            )

        self.__index += 1

        if plot_flag == True:
            flags = ['states','controls','constraints']
            self.__plot(flags,self.__trial.nlp.V(sol['x']))

        self.__extract_solver_stats(sol)
        self.__shift_solution()

        # return zoh control
        u0 = ct.mtimes(self.__trial.nlp.Collocation.quad_weights[np.newaxis,:], ct.horzcat(*self.__trial.nlp.V(sol['x'])['coll_var',0,:,'u']).T)

        return u0

    def __generate_objective(self):
        """ Generate mpc objective.
        """

        logging.info("Generate MPC {} objective...".format(self.__cost_type))

        # initialize
        f = 0.0

        # weighting matrices
        Q = np.eye(self.__trial.model.variables['xd'].shape[0])
        R = np.eye(self.__trial.model.variables['u'].shape[0])
        Z = np.eye(self.__trial.model.variables['xa'].shape[0])

        # create tracking function
        from scipy.linalg import block_diag
        tracking_cost = self.__create_tracking_cost_fun(block_diag(Q,R,Z))
        cost_map = tracking_cost.map(self.__N)

        # cost function arguments
        cost_args = []
        V_list = [ct.horzcat(*self.__trial.nlp.V['coll_var',k,:]) for k in range(self.__N)]
        cost_args += [ct.horzcat(*V_list)]
        V_ref_list = [ct.horzcat(*self.__p['ref','coll_var',k,:]) for k in range(self.__N)]
        cost_args += [ct.horzcat(*V_ref_list)]

        # quadrature weights
        quad_weights = list(self.__trial.nlp.Collocation.quad_weights)*self.__N

        # integrate tracking cost function
        f = ct.mtimes(ct.vertcat(*quad_weights).T, cost_map(*cost_args).T)/self.__N

        return f

    def __extract_solver_stats(self, sol):
        """ Extract solver info to log.
        """

        info = self.__solver.stats()
        self.__log['cpu'].append(info['t_wall_solver'])
        self.__log['iter'].append(info['iter_count'])
        self.__log['status'].append(info['return_status'])
        self.__log['f'].append(sol['f'])
        self.__log['V_opt'].append(self.__trial.nlp.V(sol['x']))
        self.__log['lam_x'].append(sol['lam_x'])
        self.__log['lam_g'].append(sol['lam_g'])
        self.__log['u0'].append(self.__trial.nlp.V(sol['x'])['coll_var',0,0,'u'])

        return None

    def __compile_solver(self):
        """ Compile solver dependencies.
        """

        logging.info("Compiling solver...")

        # record compilation time
        import time
        ts = time.time()

        # generate solver dependencies
        self.__solver.generate_dependencies('solver.c')

        # compile dependencies and mpc script
        import os
        os.system("gcc -fPIC -shared -O3 solver.c -o solver.so")
        os.system("g++ -std=c++11 -o mpc.out mpc_codegen.cpp -lcasadi -ldl -Wl,-rpath,'$ORIGIN'")

        tc = time.time() - ts
        logging.info("Compilation time: {}s".format(tc))

        return None

    def __create_reference_interpolator(self):
        """ Create time-varying reference generator for tracking MPC based on interpolation of
            optimal periodic steady state.
        """

        # MPC time grid
        self.__t_grid_coll   = self.__trial.nlp.time_grids['coll'](self.__N*self.__ts)
        self.__t_grid_x_coll = self.__trial.nlp.time_grids['x_coll'](self.__N*self.__ts)

        # interpolate steady state solution
        self.__ref_dict = self.__pocp_trial.visualization.plot_dict
        nlp_options = self.__pocp_trial.options['nlp']
        V_opt = self.__pocp_trial.optimization.V_opt
        if self.__mpc_options['ref_interpolator'] == 'poly':
            self.__interpolator = self.__pocp_trial.nlp.Collocation.build_interpolator(nlp_options, V_opt)
        elif self.__mpc_options['ref_interpolator'] == 'spline':
            self.__interpolator = self.__build_spline_interpolator(nlp_options, V_opt)

        return None

    def __build_spline_interpolator(self, nlp_options, V_opt):

        variables_dict = self.__pocp_trial.model.variables_dict
        plot_dict = self.__pocp_trial.visualization.plot_dict
        cosmetics = self.__pocp_trial.options['visualization']['cosmetics']
        n_points = self.__t_grid_coll.numel()
        n_points_x = self.__t_grid_x_coll.numel()
        self.__spline_dict = {}

        for var_type in ['xd','u','xa']:
            self.__spline_dict[var_type] = {}
            for name in list(variables_dict[var_type].keys()):
                self.__spline_dict[var_type][name] = {}
                for j in range(variables_dict[var_type][name].shape[0]):
                    if var_type == 'xd':
                        values, time_grid = viz_tools.merge_xd_values(V_opt, name, j, plot_dict, cosmetics)
                        self.__spline_dict[var_type][name][j] = ct.interpolant(name+str(j), 'bspline', [[0]+time_grid], [values[-1]]+values, {}).map(n_points_x)
                    elif var_type == 'u':
                        values, time_grid = viz_tools.merge_xa_values(V_opt, var_type, name, j, plot_dict, cosmetics)
                        if all(v == 0 for v in values):
                            self.__spline_dict[var_type][name][j] = ct.Function(name+str(j), [ct.SX.sym('t',n_points)], [np.zeros((n_points,1))])
                        else:
                            self.__spline_dict[var_type][name][j] = ct.interpolant(name+str(j), 'bspline', [[0]+time_grid], [values[-1]]+values, {}).map(n_points)
                    elif var_type == 'xa':
                        values, time_grid = viz_tools.merge_xa_values(V_opt, var_type, name, j, plot_dict, cosmetics)
                        self.__spline_dict[var_type][name][j] = ct.interpolant(name+str(j), 'bspline', [[0]+time_grid], [values[-1]]+values, {}).map(n_points)

        def spline_interpolator(t_grid, name, j, var_type):

            values_ip = self.__spline_dict[var_type][name][j](t_grid)

            return values_ip

        return spline_interpolator

    def __get_reference(self, index):
        """ xxx
        """

        # current time grid
        Tref = self.__ref_dict['time_grids']['ip'][-1]
        t_grid = ct.reshape(self.__t_grid_coll.T, self.__t_grid_coll.numel(),1).full() + index*self.__ts
        t_grid = ct.vertcat(*list(map(lambda x: x % Tref, t_grid))).full().squeeze()

        t_grid_x = ct.reshape(self.__t_grid_x_coll.T, self.__t_grid_x_coll.numel(),1).full() + index*self.__ts
        t_grid_x = ct.vertcat(*list(map(lambda x: x % Tref, t_grid_x))).full().squeeze()

        # interpolate data
        variables_dict = self.__pocp_trial.model.variables_dict
        options = self.__pocp_trial.options
        values_ip_x, values_ip_u, values_ip_z = [], [], []
        for var_type in ['xd','u','xa']:
            for name in list(variables_dict[var_type].keys()):
                for j in range(variables_dict[var_type][name].shape[0]):
                    if var_type == 'xd':
                        values_ip_x.append(list(self.__interpolator(t_grid_x, name, j,var_type).full().squeeze()))
                    elif var_type == 'u':
                        values_ip_u.append(list(self.__interpolator(t_grid, name, j,var_type).full().squeeze()))
                    elif var_type == 'xa':
                        values_ip_z.append(list(self.__interpolator(t_grid, name, j,var_type).full().squeeze()))

        V_ref = self.__trial.nlp.V(0.0)
        for k in range(self.__N):
            V_ref['xd',k] = ct.vertcat(*[values_ip_x[i].pop(0) for i in range(self.__nx)])
            for j in range(self.__d):
                V_ref['coll_var',k,j,'xd'] = ct.vertcat(*[values_ip_x[i].pop(0) for i in range(self.__nx)])
                V_ref['coll_var',k,j,'u']  = ct.vertcat(*[values_ip_u[i].pop(0) for i in range(self.__nu)])
                V_ref['coll_var',k,j,'xa'] = ct.vertcat(*[values_ip_z[i].pop(0) for i in range(self.__nz)])
        V_ref['xd',-1] = ct.vertcat(*[values_ip_x[i].pop(0) for i in range(self.__nx)])

        for name in self.__trial.model.variables_dict['theta'].keys():
            if name != 't_f':
                V_ref['theta',name] = self.__pocp_trial.optimization.V_opt['theta',name]
        V_ref['theta','t_f'] = self.__N*self.__ts

        return V_ref

    def __initialize_solver(self):
        """ xxx
        """

        # initial guess
        self.__w0 = self.__trial.nlp.V(0.0)
        for name in self.__trial.model.variables_dict['theta'].keys():
            if name != 't_f':
                self.__w0['theta',name] = self.__pocp_trial.optimization.V_opt['theta',name]
        self.__w0['theta','t_f'] = self.__N*self.__ts

        ref = self.__get_reference(0)
        self.__w0['coll_var',:,:,'xd'] = ref['coll_var',:,:,'xd']
        self.__w0['coll_var',:,:,'u']  = ref['coll_var',:,:,'u']
        self.__w0['coll_var',:,:,'xa'] = ref['coll_var',:,:,'xa']
        for k in range(self.__N):
            self.__w0['xd',k+1] = ref['coll_var',k,-1,'xd']
        self.__w0['xd',0] = ref['coll_var',0,0,'xd']

        # intialize log
        self.__log = {
            'cpu':[],
            'iter':[],
            'status':[],
            'f':[],
            'V_opt':[],
            'lam_x':[],
            'lam_g':[],
            'u0': []
        }

        return None

    def __shift_solution(self):

        for k in range(self.__N-1):
            self.__w0['coll_var',k,:,'xd'] = self.__w0['coll_var',k+1,:,'xd']
            self.__w0['coll_var',k,:,'u']  = self.__w0['coll_var',k+1,:,'u']
            self.__w0['coll_var',k,:,'xa'] = self.__w0['coll_var',k+1,:,'xa']
            self.__w0['xd',k] = self.__w0['xd',k+1]

        return None

    def __plot(self, flags, V_opt):
        """ Plot MPC solution.
        """

        # reference trajectory
        V_ref = self.__trial.nlp.V(self.__p0['ref'])

        # generate system outputs
        [nlp_outputs, nlp_output_fun] = self.__trial.nlp.output_components
        outputs_init = nlp_outputs(nlp_output_fun(self.__w0, self.__p_fix_num))
        outputs_opt = nlp_outputs(nlp_output_fun(V_opt, self.__p_fix_num))
        outputs_ref = nlp_outputs(nlp_output_fun(V_ref, self.__p_fix_num))
        output_vals = [outputs_init, outputs_opt, outputs_ref]

        # generate integral outputs
        [nlp_integral_outputs, nlp_integral_outputs_fun] = self.__trial.nlp.integral_output_components
        integral_outputs_final = nlp_integral_outputs(nlp_integral_outputs_fun(V_opt, self.__p_fix_num))

        # time grids
        time_grids = {}
        for grid in self.__trial.nlp.time_grids:
            time_grids[grid] = self.__trial.nlp.time_grids[grid](V_opt['theta','t_f'])
        time_grids['ref'] = time_grids

        # cost function
        cost_fun = self.__trial.nlp.cost_components[0]
        import awebox.tools.struct_operations as struct_op
        cost = struct_op.evaluate_cost_dict(cost_fun, V_opt, self.__p_fix_num)

        # reference trajectory
        self.__trial.visualization.plot(
            V_opt,
            self.__trial.options,
            output_vals,
            integral_outputs_final,
            flags,
            time_grids,
            cost,
            'MPC solution',
            False,
            V_ref)

        plt.show()

        return None

    def __create_P_fun(self):

        # initialize
        pp = self.__trial.nlp.P(0.0)

        # fill in system parameters
        param_options = self.__trial.options['solver']['initialization']['sys_params_num']
        for param_type in list(param_options.keys()):
            if isinstance(param_options[param_type],dict):
                for param in list(param_options[param_type].keys()):
                    if isinstance(param_options[param_type][param],dict):
                        for subparam in list(param_options[param_type][param].keys()):
                            pp['theta0',param_type,param,subparam] = param_options[param_type][param][subparam]

                    else:
                        pp['theta0',param_type,param] = param_options[param_type][param]

            else:
                pp['theta0',param_type] = param_options[param_type]
        p_sym = self.__trial.nlp.P(ct.vertcat(self.__p['x0'],pp.cat[self.__nx:]))
        self.__P_fun = ct.Function('P_fun',[self.__p], [p_sym])

    def __create_tracking_cost_fun(self, W):
        """ Create casadi.Function to compute tracking cost at one time instant.
        """

        w = ct.SX.sym('w',W.shape[0])
        w_ref = ct.SX.sym('w_ref', W.shape[0])

        return ct.Function('tracking_cost', [w, w_ref], [ct.mtimes((w-w_ref).T,(w-w_ref))])

    @property
    def trial(self):
        """ awebox.Trial attribute containing model and OCP info.
        """
        return self.__trial

    @trial.setter
    def trial(self, value):
        print('Cannot set trial object.')

    @property
    def log(self):
        """ log attribute containing MPC info.
        """
        return self.__log

    @log.setter
    def log(self, value):
        print('Cannot set log object.')

    @property
    def solver(self):
        """ casadi.nlpsol attribute containing nonlinear optimization solver.
        """
        return self.__solver

    @solver.setter
    def solver(self, value):
        print('Cannot set solver object.')