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
"""
Periodic MPC routines for awebox models

:author: Jochem De Schutter (ALU Freiburg 2019)
"""

import awebox as awe
import awebox.viz.tools as viz_tools
import casadi.tools as ct
from awebox.logger.logger import Logger as awelogger
import matplotlib.pyplot as plt
import numpy as np
import copy

class Pmpc(object):

    def __init__(self, mpc_options, ts, trial):
        """ Constructor.
        """

        awelogger.logger.info("Creating a {} periodic MPC controller with horizon length {}...".format(
            mpc_options['cost_type'], mpc_options['N']))
        awelogger.logger.info("Based on direct collocation with {} polynomials of order {}.".format(
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
        self.__var_list = ['xd', 'xa']
        if 'xl' in self.__pocp_trial.model.variables_dict.keys():
            self.__var_list.append('xl')
        self.__var_list.append('u')
        self.__nx = trial.model.variables['xd'].shape[0]
        self.__nu = trial.model.variables['u'].shape[0]
        self.__nz = trial.model.variables['xa'].shape[0]
        if 'xl' in self.__var_list:
            self.__nl = trial.model.variables['xl'].shape[0]

        # create mpc trial
        options = copy.deepcopy(trial.options)
        options['user_options']['trajectory']['type'] = 'mpc'
        options['nlp']['discretization'] = 'direct_collocation'
        options['nlp']['n_k'] = self.__N
        options['nlp']['d'] = self.__d
        options['nlp']['scheme'] = self.__scheme
        options['nlp']['collocation']['u_param'] = self.__mpc_options['u_param']
        options['formulation']['mpc']['terminal_point_constr'] = self.__mpc_options['terminal_point_constr']
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

        awelogger.logger.info("Periodic MPC controller built.")

        return None

    def __build_trial(self):
        """ Build options, model, formulation and nlp of mpc trial.
        """

        awelogger.logger.info("Building MPC trial...")

        # build
        import awebox.mdl.architecture as archi
        architecture = archi.Architecture(self.__trial.options['user_options']['system_model']['architecture'])
        self.__trial.options.build(architecture)
        self.__trial.model.build(self.__trial.options['model'], architecture)
        self.__trial.formulation.build(self.__trial.options['formulation'], self.__trial.model)
        self.__trial.nlp.build(self.__trial.options['nlp'], self.__trial.model, self.__trial.formulation)
        self.__trial.visualization.build(self.__trial.model, self.__trial.nlp, 'MPC control', self.__trial.options)

        # remove state constraints at k = 0
        self.__trial.nlp.V_bounds['lb']['xd',0] = - np.inf
        self.__trial.nlp.V_bounds['ub']['xd',0] = np.inf
        if self.__mpc_options['terminal_point_constr']:
            self.__trial.nlp.V_bounds['lb']['xd',-1] = - np.inf
            self.__trial.nlp.V_bounds['ub']['xd',-1] = np.inf
        g_ub = self.__trial.nlp.g(self.__trial.nlp.g_bounds['ub'])
        for constr in self.__trial.model.constraints_dict['inequality'].keys():
            if constr != 'dcoeff_actuation':
                if self.__mpc_options['u_param'] == 'poly':
                    g_ub['path',0,:,constr] = np.inf
                else:
                    g_ub['path',0, constr] = np.inf
        self.__trial.nlp.g_bounds['ub'] = g_ub.cat

        return None

    def __construct_solver(self):
        """ Construct casadi.nlpsol Object based on MPC trial information.
        """

        awelogger.logger.info("Constructing MPC solver object...")

        if self.__cost_type == 'economic':

            # parameters
            self.__p = ct.struct_symMX([
                ct.entry('x0', shape = (self.__nx,1)),
                ct.entry('u_ref', shape = (1,1)),
                ct.entry('Q', shape = (self.__nx, 1)),
                ct.entry('R', shape = (self.__nu, 1)),
                ct.entry('P', shape = (self.__nx, 1))
            ])

        if self.__cost_type == 'tracking':

            # parameters
            self.__p = ct.struct_symMX([
                ct.entry('x0',  shape = (self.__nx,)),
                ct.entry('ref', struct = self.__trial.nlp.V),
                ct.entry('u_ref', shape = (1,1)),
                ct.entry('Q', shape = (self.__nx, 1)),
                ct.entry('R', shape = (self.__nu, 1)),
                ct.entry('P', shape = (self.__nx, 1))
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
        self.__trial.nlp.V_bounds['lb']['xi'] = 0.0
        self.__trial.nlp.V_bounds['ub']['xi'] = 0.0

        for name in list(self.__trial.model.variables_dict['u'].keys()):
            if 'fict' in name:
                if self.__mpc_options['u_param'] == 'poly':
                    self.__trial.nlp.V_bounds['lb']['coll_var',:,:,'u',name] = 0.0
                    self.__trial.nlp.V_bounds['ub']['coll_var',:,:,'u',name] = 0.0
                elif self.__mpc_options['u_param'] == 'zoh':
                    self.__trial.nlp.V_bounds['lb']['u',:,name] = 0.0
                    self.__trial.nlp.V_bounds['ub']['u',:,name] = 0.0

        self.__lbw = self.__trial.nlp.V_bounds['lb']
        self.__ubw = self.__trial.nlp.V_bounds['ub']
        self.__lbg = self.__trial.nlp.g_bounds['lb']
        self.__ubg = self.__trial.nlp.g_bounds['ub']

        awelogger.logger.level = awelogger.logger.getEffectiveLevel()
        opts = {}
        opts['expand'] = self.__mpc_options['expand']
        opts['ipopt.linear_solver'] = self.__mpc_options['linear_solver']
        opts['ipopt.max_iter'] = self.__mpc_options['max_iter']
        opts['ipopt.max_cpu_time'] = self.__mpc_options['max_cpu_time']
        opts['jit'] = self.__mpc_options['jit']
        opts['record_time'] = 1
        opts['ipopt.tol'] = 1e-6
        if awelogger.logger.level > 10:
            opts['ipopt.print_level'] = 0
            opts['print_time'] = 0

        if self.__mpc_options['homotopy_warmstart']:
            opts['ipopt.mu_init'] = 1e-3
            homotopy_opts = copy.deepcopy(opts)
            homotopy_opts['ipopt.mu_target'] = 1e-3
            homotopy_opts['ipopt.tol'] = 1e-4
            homotopy_opts['ipopt.max_iter'] = 2

            self.__homotopy_solver = ct.nlpsol('solver', 'ipopt', nlp, homotopy_opts)

        self.__solver = ct.nlpsol('solver', 'ipopt', nlp, opts)

        return None

    def step(self, x0, plot_flag = False, u_ref = None, Q = None, R = None, P = None):

        """ Compute periodic MPC feedback control for given initial condition.
        """

        awelogger.logger.info("Compute MPC feedback...")

        # update nlp parameters
        self.__p0 = self.__p(0.0)
        self.__p0['x0'] = x0

        # update reference
        if self.__cost_type == 'tracking':
            ref = self.get_reference(*self.__compute_time_grids(self.__index))
            self.__p0['ref'] = ref

        # update model wind speed
        if u_ref == None:
            self.__p0['u_ref'] = self.__trial.options['solver']['initialization']['sys_params_num']['wind']['u_ref']
        else:
            self.__p0['u_ref'] = u_ref

        # update tracking weights
        if Q == None:
            self.__p0['Q'] = self.__weights['Q']
        if R == None:
            self.__p0['R'] = self.__weights['R']
        if P == None:
            self.__p0['P'] = self.__weights['P']

        if self.__mpc_options['homotopy_warmstart']:
            sol = self.__homotopy_solver(
                x0  = self.__w0,
                lbx = self.__lbw,
                ubx = self.__ubw,
                lbg = self.__lbg,
                ubg = self.__ubg,
                p   = self.__p0
                )

            self.__w0 = self.__trial.nlp.V(sol['x'])

        # MPC problem
        sol = self.__solver(
            x0  = self.__w0,
            lbx = self.__lbw,
            ubx = self.__ubw,
            lbg = self.__lbg,
            ubg = self.__ubg,
            p   = self.__p0
            )

        if not self.__mpc_options['homotopy_warmstart']:
            self.__w0 = self.__trial.nlp.V(sol['x'])

        self.__index += 1

        if plot_flag == True:
            self.__p_fix_num = self.__P_fun(self.__p0)
            flags = ['states','controls','constraints']
            self.__plot(flags,self.__trial.nlp.V(sol['x']))

        self.__extract_solver_stats(sol)

        # return zoh control
        if self.__mpc_options['u_param'] == 'poly':
            u0 = ct.mtimes(
                    self.__trial.nlp.Collocation.quad_weights[np.newaxis,:],
                    ct.horzcat(*self.__trial.nlp.V(sol['x'])['coll_var',0,:,'u']).T
                    )
        elif self.__mpc_options['u_param'] == 'zoh':
            u0 = self.__trial.nlp.V(sol['x'])['u',0]

        # initial guess for higher level simulation
        xa0 = self.__trial.nlp.V(sol['x'])['coll_var',0,0,'xa']
        xdot0 = self.__trial.nlp.Xdot(self.__trial.nlp.Xdot_fun(sol['x']))['xd',0]
        self.__z0 = ct.vertcat(xdot0, xa0)

        # shift solution (note: after control assignment!)
        self.__shift_solution()

        return u0

    def __generate_objective(self):
        """ Generate mpc objective.
        """

        awelogger.logger.info("Generate MPC {} objective...".format(self.__cost_type))

        # initialize
        f = 0.0

        # weighting matrices
        weights = {}
        for weight in ['Q', 'R', 'P']:
            if weight in self.__mpc_options.keys():
                weights[weight] = self.__mpc_options[weight]
            else:
                if weight in ['Q', 'P']:
                    weights[weight] = np.ones((self.__nx,1))
                elif weight == 'R':
                    weights[weight] = np.ones((self.__nu,1))
        weights['Z'] = np.ones((self.__nz,1))
        if 'xl' in self.__var_list:
            weights['L'] = 1e-5*np.ones((self.__nl,1))
        self.__weights = weights

        # create tracking function
        tracking_cost = self.__create_tracking_cost_fun()
        cost_map = tracking_cost.map(self.__N)

        # cost function arguments
        cost_args = []
        if self.__mpc_options['u_param'] == 'poly':
            V_list = [ct.horzcat(*self.__trial.nlp.V['coll_var',k,:]) for k in range(self.__N)]
            V_ref_list = [ct.horzcat(*self.__p['ref','coll_var',k,:]) for k in range(self.__N)]

        elif self.__mpc_options['u_param'] == 'zoh':
            V_list = []
            V_ref_list = []
            for k in range(self.__N):
                for d in range(self.__d):
                    V_list.append(ct.vertcat(self.__trial.nlp.V['coll_var',k,d],self.__trial.nlp.V['u',k]))
                    V_ref_list.append(ct.vertcat(self.__p['ref','coll_var',k,d],self.__p['ref','u',k]))

        cost_args += [ct.horzcat(*V_list)]
        cost_args += [ct.horzcat(*V_ref_list)]
        cost_args += [ct.vertcat(self.__p['Q'], self.__weights['Z'], self.__p['R'])]

        # quadrature weights
        quad_weights = list(self.__trial.nlp.Collocation.quad_weights)*self.__N

        # integrate tracking cost function
        f = ct.mtimes(ct.vertcat(*quad_weights).T, cost_map(*cost_args).T)/self.__N

        # terminal cost
        dxN = self.__trial.nlp.V['xd',-1] - self.__p['ref','xd',-1]
        f += ct.mtimes(ct.mtimes((dxN.T, ct.diag(self.__p['P']))),dxN)

        return f

    def __extract_solver_stats(self, sol):
        """ Extract solver info to log.
        """

        info = self.__solver.stats()

        if self.__mpc_options['homotopy_warmstart']:
            info_homotopy = self.__homotopy_solver.stats()
            self.__log['cpu'].append(info['t_wall_total']+info_homotopy['t_wall_total'])
            self.__log['iter'].append(info['iter_count']+info_homotopy['iter_count'])
        else:
            self.__log['cpu'].append(info['t_wall_total'])
            self.__log['iter'].append(info['iter_count'])
        self.__log['status'].append(info['return_status'])
        self.__log['f'].append(sol['f'])
        self.__log['V_opt'].append(self.__trial.nlp.V(sol['x']))
        self.__log['lam_x'].append(sol['lam_x'])
        self.__log['lam_g'].append(sol['lam_g'])
        if self.__mpc_options['u_param'] == 'poly':
            self.__log['u0'].append(self.__trial.nlp.V(sol['x'])['coll_var',0,0,'u'])
        else:
            self.__log['u0'].append(self.__trial.nlp.V(sol['x'])['u',0])

        return None

    def __compile_solver(self):
        """ Compile solver dependencies.
        """

        awelogger.logger.info("Compiling solver...")

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
        awelogger.logger.info("Compilation time: {}s".format(tc))

        return None

    def __create_reference_interpolator(self):
        """ Create time-varying reference generator for tracking MPC based on interpolation of
            optimal periodic steady state.
        """

        # MPC time grid
        self.__t_grid_coll = self.__trial.nlp.time_grids['coll'](self.__N*self.__ts)
        self.__t_grid_coll = ct.reshape(self.__t_grid_coll.T, self.__t_grid_coll.numel(),1).full()
        self.__t_grid_x_coll = self.__trial.nlp.time_grids['x_coll'](self.__N*self.__ts)
        self.__t_grid_x_coll = ct.reshape(self.__t_grid_x_coll.T, self.__t_grid_x_coll.numel(),1).full()
        self.__t_grid_u = self.__trial.nlp.time_grids['u'](self.__N*self.__ts)
        self.__t_grid_u = ct.reshape(self.__t_grid_u.T, self.__t_grid_u.numel(),1).full()

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
        """ Build spline-based reference interpolating method.
        """

        variables_dict = self.__pocp_trial.model.variables_dict
        plot_dict = self.__pocp_trial.visualization.plot_dict
        cosmetics = self.__pocp_trial.options['visualization']['cosmetics']
        n_points = self.__t_grid_coll.shape[0]
        n_points_x = self.__t_grid_x_coll.shape[0]
        self.__spline_dict = {}

        for var_type in self.__var_list:
            self.__spline_dict[var_type] = {}
            for name in list(variables_dict[var_type].keys()):
                self.__spline_dict[var_type][name] = {}
                for j in range(variables_dict[var_type][name].shape[0]):
                    if var_type == 'xd':
                        values, time_grid = viz_tools.merge_xd_values(V_opt, name, j, plot_dict, cosmetics)
                        self.__spline_dict[var_type][name][j] = ct.interpolant(name+str(j), 'bspline', [[0]+time_grid], [values[-1]]+values, {}).map(n_points_x)
                    elif var_type in ['xa', 'xl'] or (var_type == 'u' and self.__mpc_options['u_param'] == 'poly'):
                        values, time_grid = viz_tools.merge_xa_values(V_opt, var_type, name, j, plot_dict, cosmetics)
                        if all(v == 0 for v in values) or 'fict' in name:
                            self.__spline_dict[var_type][name][j] = ct.Function(name+str(j), [ct.SX.sym('t',n_points)], [np.zeros((1,n_points))])
                        else:
                            self.__spline_dict[var_type][name][j] = ct.interpolant(name+str(j), 'bspline', [[0]+time_grid], [values[-1]]+values, {}).map(n_points)
                    elif var_type == 'u' and self.__mpc_options['u_param'] == 'zoh':
                        values, time_grid = viz_tools.merge_xa_values(V_opt, var_type, name, j, plot_dict, cosmetics)
                        if all(v == 0 for v in values) or 'fict' in name:
                            self.__spline_dict[var_type][name][j] = ct.Function(name+str(j), [ct.SX.sym('t',self.__N)], [np.zeros((1,self.__N))])
                        else:
                            self.__spline_dict[var_type][name][j] = ct.interpolant(name+str(j), 'bspline', [[0]+time_grid], [values[-1]]+values, {}).map(self.__N)

        def spline_interpolator(t_grid, name, j, var_type):
            """ Interpolate reference on specific time grid for specific variable.
            """

            values_ip = self.__spline_dict[var_type][name][j](t_grid)

            return values_ip

        return spline_interpolator

    def __compute_time_grids(self, index):
        """ Compute NLP time grids based in periodic index
        """

        Tref = self.__ref_dict['time_grids']['ip'][-1]
        t_grid = self.__t_grid_coll + index*self.__ts
        t_grid = ct.vertcat(*list(map(lambda x: x % Tref, t_grid))).full().squeeze()

        t_grid_x = self.__t_grid_x_coll + index*self.__ts
        t_grid_x = ct.vertcat(*list(map(lambda x: x % Tref, t_grid_x))).full().squeeze()

        t_grid_u = self.__t_grid_u + index*self.__ts
        t_grid_u = ct.vertcat(*list(map(lambda x: x % Tref, t_grid_u))).full().squeeze()

        return t_grid, t_grid_x, t_grid_u

    def get_reference(self, t_grid, t_grid_x, t_grid_u):
        """ Interpolate reference on NLP time grids.
        """

        ip_dict = {}
        V_ref = self.__trial.nlp.V(0.0)
        for var_type in self.__var_list:
            ip_dict[var_type] = []
            for name in list(self.__trial.model.variables_dict[var_type].keys()):
                for dim in range(self.__trial.model.variables_dict[var_type][name].shape[0]):
                    if var_type == 'xd':
                        ip_dict[var_type].append(self.__interpolator(t_grid_x, name, dim,var_type))
                    elif (var_type == 'u') and self.__mpc_options['u_param']:
                        ip_dict[var_type].append(self.__interpolator(t_grid_u, name, dim,var_type))
                    else:
                        ip_dict[var_type].append(self.__interpolator(t_grid, name, dim,var_type))
            if self.__mpc_options['ref_interpolator'] == 'poly':
                ip_dict[var_type] = ct.horzcat(*ip_dict[var_type]).T
            elif self.__mpc_options['ref_interpolator'] == 'spline':
                ip_dict[var_type] = ct.vertcat(*ip_dict[var_type])

        counter = 0
        counter_x = 0
        counter_u = 0
        V_list = []

        for name in self.__trial.model.variables_dict['theta'].keys():
            if name != 't_f':
                V_list.append(self.__pocp_trial.optimization.V_opt['theta',name])
            else:
                V_list.append(self.__N*self.__ts)
        V_list.append(np.zeros(V_ref['phi'].shape))
        V_list.append(np.zeros(V_ref['xi'].shape))

        for k in range(self.__N):
            for j in range(self.__trial.nlp.d+1):
                if j == 0:
                    V_list.append(ip_dict['xd'][:,counter_x])
                    counter_x += 1

                    if self.__mpc_options['u_param'] == 'zoh':
                        V_list.append(ip_dict['u'][:, counter_u])
                        V_list.append(np.zeros((self.__nx, 1)))
                        V_list.append(np.zeros((self.__nz, 1)))
                        counter_u += 1
                else:
                    for var_type in self.__var_list:
                        if var_type == 'xd':
                            V_list.append(ip_dict[var_type][:,counter_x])
                            counter_x += 1
                        elif var_type in ['xa','xl'] or (var_type == 'u' and self.__mpc_options['u_param']=='poly'):
                            V_list.append(ip_dict[var_type][:,counter])
                    counter += 1

        V_list.append(ip_dict['xd'][:,counter_x])

        V_ref = V_ref(ct.vertcat(*V_list))

        return V_ref

    def __initialize_solver(self):
        """ Initialize solver with reference solution.
        """

        # initial guess
        self.__w0 = self.get_reference(*self.__compute_time_grids(0.0))

        for name in self.__trial.model.variables_dict['theta'].keys():
            if name != 't_f':
                self.__w0['theta',name] = self.__pocp_trial.optimization.V_opt['theta',name]
        self.__w0['theta','t_f'] = self.__N*self.__ts

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
        """ Shift NLP solution one stage to the left.
        """

        for k in range(self.__N-1):
            self.__w0['coll_var',k,:,'xd'] = self.__w0['coll_var',k+1,:,'xd']
            if self.__mpc_options['u_param'] == 'poly':
                self.__w0['coll_var',k,:,'u']  = self.__w0['coll_var',k+1,:,'u']
            elif self.__mpc_options['u_param'] == 'zoh':
                self.__w0['u',k] = self.__w0['u',k+1]
            self.__w0['coll_var',k,:,'xa'] = self.__w0['coll_var',k+1,:,'xa']
            if 'xl' in self.__var_list:
                self.__w0['coll_var',k,:,'xl'] = self.__w0['coll_var',k+1,:,'xl']
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
        """ Create function that maps MPC parameters to periodic OCP parameters.
        """

        # initialize
        import awebox.opti.preparation as prep
        V_dummy = self.__trial.nlp.V(0.0)
        p_fix_num = prep.set_p_fix_num(V_dummy, self.__trial.nlp, self.__trial.model, V_dummy, self.__trial.options['solver'])
        x0 = self.__trial.model.variables_dict['xd'](self.__p['x0'])
        xN = self.__trial.model.variables_dict['xd'](self.__trial.nlp.V(self.__p['ref'])['xd',-1])

        # fill in P
        p_list = []
        for k in range(self.__trial.nlp.P.shape[0]):

            canonical = self.__trial.nlp.P.getCanonicalIndex(k)
            
            if canonical[0] == 'p' and canonical[1] == 'ref' and canonical[2] == 'xd':
                if canonical[3] == 0:
                    name = canonical[4]
                    dim = canonical[5]
                    p_list.append(x0[name, dim])
                elif canonical[3] == self.__N:
                    name = canonical[4]
                    dim = canonical[5]
                    p_list.append(xN[name, dim])
                else:
                    p_list.append(0.0)
            
            # fill in params
            elif canonical[0] == 'theta0':

                if canonical[1] == 'wind' and canonical[2] == 'u_ref':
                    p_list.append(self.__p['u_ref'])
                elif len(canonical) == 4:
                    p_list.append(p_fix_num[canonical[0],canonical[1],canonical[2],canonical[3]])
                elif len(canonical) == 5:
                    p_list.append(p_fix_num[canonical[0],canonical[1],canonical[2],canonical[3],canonical[4]])
                elif len(canonical) == 3:
                    p_list.append(p_fix_num[canonical[0],canonical[1],canonical[2]])

            else:
                p_list.append(0.0)

        p_sym = self.__trial.nlp.P(ct.vertcat(*p_list))
        self.__P_fun = ct.Function('P_fun',[self.__p], [p_sym])

    def __create_tracking_cost_fun(self):
        """ Create casadi.Function to compute tracking cost at one time instant.
        """

        w = ct.SX.sym('w', self.__nx+self.__nu+self.__nz)
        w_ref = ct.SX.sym('w_ref', self.__nx+self.__nu+self.__nz)
        W = ct.SX.sym('W', self.__nx+self.__nu+self.__nz)
        f_t = ct.mtimes(
                ct.mtimes(
                    (w-w_ref).T,
                    ct.diag(W)
                ),
                (w-w_ref)
        )

        return ct.Function('tracking_cost', [w, w_ref, W], [f_t])

    @property
    def trial(self):
        """ awebox.Trial attribute containing model and OCP info.
        """
        return self.__trial

    @trial.setter
    def trial(self, value):
        awelogger.logger.info('Cannot set trial object.')

    @property
    def log(self):
        """ log attribute containing MPC info.
        """
        return self.__log

    @log.setter
    def log(self, value):
        awelogger.logger.info('Cannot set log object.')

    @property
    def solver(self):
        """ casadi.nlpsol attribute containing nonlinear optimization solver.
        """
        return self.__solver

    @solver.setter
    def solver(self, value):
        awelogger.logger.info('Cannot set solver object.')

    @property
    def solver_bounds(self):
        """ Solver variable and constraints bounds vectors
        """
        return {'lbw': self.__lbw, 'ubw': self.__ubw, 'lbg': self.__lbg, 'ubg': self.__ubg}

    @solver_bounds.setter
    def solver_bounds(self, value):
        awelogger.logger.info('Cannot set solver_bounds object.')

    @property
    def w0(self):
        """ Solver initial guess vector
        """
        return self.__w0

    @w0.setter
    def w0(self, value):
        awelogger.logger.info('Cannot set w0 object.')

    @property
    def t_grid_coll(self):
        """ Collocation grid time vector
        """
        return self.__t_grid_coll

    @t_grid_coll.setter
    def t_grid_coll(self, value):
        awelogger.logger.info('Cannot set t_grid_coll object.')

    @property
    def t_grid_u(self):
        """ ZOH control grid time vector
        """
        return self.__t_grid_u

    @t_grid_u.setter
    def t_grid_u(self, value):
        awelogger.logger.info('Cannot set t_grid_u object.')

    @property
    def t_grid_x_coll(self):
        """ Collocation grid time vector
        """
        return self.__t_grid_x_coll

    @t_grid_x_coll.setter
    def t_grid_x_coll(self, value):
        awelogger.logger.info('Cannot set t_grid_x_coll object.')

    @property
    def interpolator(self):
        """ interpolator
        """
        return self.__interpolator

    @interpolator.setter
    def interpolator(self, value):
        awelogger.logger.info('Cannot set interpolator object.')

    @property
    def z0(self):
        """ algebraic dae variables initial guess
        """
        return self.__z0

    @z0.setter
    def z0(self, value):
        awelogger.logger.info('Cannot set z0 object.')
