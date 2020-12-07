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
Simulation class for open-loop and closed-loop simulations based on awebox reference trajectories
and related models.

:author: Jochem De Schutter - ALU Freiburg 2019
"""


import casadi.tools as ct
import awebox.pmpc as pmpc
import awebox.tools.integrator_routines as awe_integrators
import awebox.viz.visualization as visualization
import awebox.viz.tools as viz_tools
import copy
import numpy as np

class Simulation:
    def __init__(self, trial, sim_type, ts, options):
        """ Constructor.
        """

        if sim_type not in ['closed_loop', 'open_loop']:
            raise ValueError('Chosen simulation type not valid: {}'.format(sim_type))

        self.__sim_type = sim_type
        self.__trial = trial
        self.__ts = ts
        self.__sim_options = options['sim']
        self.__mpc_options = options['mpc']
        self.__build()

        return None

    def __build(self):
        """ Build simulation
        """

        # generate plant model
        sys_params = self.__sim_options['sys_params']
        if sys_params is None:
            sys_params = self.__trial.options['solver']['initialization']['sys_params_num']
        model = self.__trial.generate_optimal_model(sys_params)
        self.__F = awe_integrators.rk4root(
            'F',
            model['dae'],
            model['rootfinder'],
            {'tf': self.__ts/model['t_f'],
            'number_of_finite_elements':self.__sim_options['number_of_finite_elements']}
            )
        self.__dae = self.__trial.model.get_dae()
        self.__dae.build_rootfinder()

        # generate mpc controller
        if self.__sim_type == 'closed_loop':

            self.__mpc = pmpc.Pmpc(self.__mpc_options, self.__ts, self.__trial)


        #  initialize and build visualization
        self.__visualization = visualization.Visualization()
        self.__visualization.build(self.__trial.model, self.__trial.nlp, 'simulation', self.__trial.options)
        for var_type in set(self.__trial.model.variables_dict.keys()) - set(['theta','xddot']):
            self.__visualization.plot_dict[var_type] = {}
            for name in list(self.__trial.model.variables_dict[var_type].keys()):
                self.__visualization.plot_dict[var_type][name] = []
                for dim in range(self.__trial.model.variables_dict[var_type][name].shape[0]):
                    self.__visualization.plot_dict[var_type][name].append([])

        self.__visualization.plot_dict['outputs'] = {}
        for output_type in list(self.__trial.model.outputs.keys()):
            self.__visualization.plot_dict['outputs'][output_type] = {}
            for name in list(self.__trial.model.outputs_dict[output_type].keys()):
                self.__visualization.plot_dict['outputs'][output_type][name] = []
                for dim in range(self.__trial.model.outputs_dict[output_type][name].shape[0]):
                    self.__visualization.plot_dict['outputs'][output_type][name].append([])

        self.__visualization.plot_dict['integral_outputs'] = {}
        for name in self.__visualization.plot_dict['integral_variables']:
            self.__visualization.plot_dict['integral_outputs'][name] = [[]]

        self.__visualization.plot_dict['V_plot'] = None

        return None

    def run(self, n_sim, x0 = None, u_sim = None):
        """ Run simulation
        """

        # TODO: check consistency of initial conditions and give warning

        x0 = self.__initialize_sim(n_sim, x0, u_sim)

        for i in range(n_sim):

            # get (open/closed-loop) controls
            if self.__sim_type == 'closed_loop':
                u0 = self.__mpc.step(x0, self.__mpc_options['plot_flag'])

            elif self.__sim_type == 'open_loop':
                u0 = self.__u_sim[:,i]

            # simulate
            var_next = self.__F(x0 = x0, p = u0, z0 = self.__z0)
            self.__store_results(x0, u0, var_next['qf'])

            # shift initial state
            x0 = var_next['xf']

        self.__postprocess_sim()

        return None

    def __initialize_sim(self, n_sim, x0, u_sim):
        """ Initialize simulation.
        """

        # take first state of optimization trial
        if x0 is None:
            x0 = self.__trial.optimization.V_opt['xd',0]

        # set-up open loop controls
        if self.__sim_type == 'open_loop':
            values_ip_u = []
            interpolator = self.__trial.nlp.Collocastion.build_interpolator(
                self.__trial.options['nlp'],
                self.__trial.optimization.V_opt)
            T_ref = self.__trial.visualization.plot_dict['time_grids']['ip'][-1]
            t_grid = np.linspace(0, n_sim*self.__ts, n_sim)
            self.__t_grid = ct.vertcat(*list(map(lambda x: x % Tref, t_grid))).full().squeeze()
            for name in list(self.__trial.model.variables_dict['u'].keys()):
                for j in range(self.__trial.variables_dict['u'][name].shape[0]):
                    values_ip_u.append(list(interpolator(t_grid, name, j,'u').full()))

            self.__u_sim = ct.horzcat(*values_ip_u)

        # initialize algebraic variables for integrator
        self.__z0 = 0.1

        # initialize numerical parameters
        self.__theta = self.__trial.model.variables_dict['theta'](0.0)
        for name in self.__trial.model.variables_dict['theta'].keys():
            if name != 't_f':
                self.__theta[name] = self.__trial.optimization.V_opt['theta',name]
        self.__theta['t_f'] = self.__ts
        self.__parameters_num = self.__trial.model.parameters(0.0)
        self.__parameters_num['theta0'] = self.__trial.optimization.p_fix_num['theta0']

        # time grids
        self.__visualization.plot_dict['time_grids'] = {}
        self.__visualization.plot_dict['time_grids']['ip'] = np.linspace(0,n_sim*self.__ts, n_sim)
        self.__visualization.plot_dict['time_grids']['u']  = np.linspace(0,n_sim*self.__ts, n_sim)

        # create reference
        T_ref = self.__trial.visualization.plot_dict['time_grids']['ip'][-1]
        trial_plot_dict = copy.deepcopy(self.__trial.visualization.plot_dict)
        tgrid_ip = copy.deepcopy(self.__visualization.plot_dict['time_grids']['ip'])
        trial_plot_dict['time_grids']['ip'] = ct.vertcat(*list(map(lambda x: x % T_ref, tgrid_ip))).full().squeeze()
        trial_plot_dict['V_ref'] = self.__trial.visualization.plot_dict['V_plot']
        trial_plot_dict['output_vals'][2] =  self.__trial.visualization.plot_dict['output_vals'][1]
        trial_plot_dict = viz_tools.interpolate_ref_data(trial_plot_dict, self.__trial.options['visualization']['cosmetics'])
        self.__visualization.plot_dict['ref'] = trial_plot_dict['ref']
        self.__visualization.plot_dict['time_grids']['ref'] = trial_plot_dict['time_grids']['ref']
        self.__visualization.plot_dict['time_grids']['ref']['ip'] = self.__visualization.plot_dict['time_grids']['ip']

        return x0

    def __store_results(self, x0, u0, qf):

        x = self.__trial.model.variables_dict['xd'](x0)
        variables = self.__trial.model.variables(0.1)
        variables['xd'] = x0
        variables['u']  = u0
        variables['theta'] = self.__theta
        x, z, p = self.__dae.fill_in_dae_variables(variables, self.__parameters_num)
        z0 = self.__dae.dae['z'](self.__dae.rootfinder(z, x, p))

        variables['xa'] = self.__trial.model.variables_dict['xa'](z0['xa'])
        if 'xl' in list(variables.keys()):
            variables['xl'] = self.__trial.model.variables_dict['xl'](z0['xl'])
        variables['xddot'] = self.__trial.model.variables_dict['xddot'](z0['xddot'])

        # evaluate system outputs
        outputs = self.__trial.model.outputs(self.__trial.model.outputs_fun(variables, self.__parameters_num))
        qf = self.__trial.model.integral_outputs(qf)

        # store results
        for var_type in set(self.__trial.model.variables_dict.keys()) - set(['theta','xddot']):
            for name in list(self.__trial.model.variables_dict[var_type].keys()):
                for dim in range(self.__trial.model.variables_dict[var_type][name].shape[0]):
                    self.__visualization.plot_dict[var_type][name][dim].append(variables[var_type,name,dim]*self.__trial.model.scaling[var_type][name])

        for output_type in list(self.__trial.model.outputs.keys()):
            for name in list(self.__trial.model.outputs_dict[output_type].keys()):
                for dim in range(self.__trial.model.outputs_dict[output_type][name].shape[0]):
                    self.__visualization.plot_dict['outputs'][output_type][name][dim].append(outputs[output_type,name,dim])

        for name in self.__visualization.plot_dict['integral_variables']:
            self.__visualization.plot_dict['integral_outputs'][name][0].append(qf[name])

        return None

    def plot(self, flags):
        """ plot visualization
        """

        self.__trial.options['visualization']['cosmetics']['plot_ref'] = True
        self.__visualization.plot(None, self.__trial.options, None, None, flags, None, None, 'simulation', False, None, 'plot', recalibrate = False)

        return None

    def __postprocess_sim(self):
        """ Postprocess simulation results.
        """

        # vectorize result lists for plotting
        for var_type in set(self.__trial.model.variables_dict.keys()) - set(['theta','xddot']):
            for name in list(self.__trial.model.variables_dict[var_type].keys()):
                for dim in range(self.__trial.model.variables_dict[var_type][name].shape[0]):
                    self.__visualization.plot_dict[var_type][name][dim] = ct.vertcat(*self.__visualization.plot_dict[var_type][name][dim])

        for output_type in list(self.__trial.model.outputs.keys()):
            for name in list(self.__trial.model.outputs_dict[output_type].keys()):
                for dim in range(self.__trial.model.outputs_dict[output_type][name].shape[0]):
                    self.__visualization.plot_dict['outputs'][output_type][name][dim] = ct.vertcat(*self.__visualization.plot_dict['outputs'][output_type][name][dim]).full()

        for name in self.__visualization.plot_dict['integral_variables']:
            self.__visualization.plot_dict['integral_outputs'][name][0] = ct.vertcat(*self.__visualization.plot_dict['integral_outputs'][name][0])

    @property
    def trial(self):
        """ awebox.Trial attribute containing model and OCP info.
        """
        return self.__trial

    @trial.setter
    def trial(self, value):
        print('Cannot set trial object.')

    @property
    def mpc(self):
        """ awebox.pmpc.Pmpc attribute containing MPC info.
        """
        return self.__mpc

    @mpc.setter
    def mpc(self, value):
        print('Cannot set mpc object.')

    @property
    def F(self):
        """ integrator attribute containing simulation info.
        """
        return self.__F

    @F.setter
    def F(self, value):
        print('Cannot set F object.')

    @property
    def visualization(self):
        """ awebox.pmpc.Visualization attribute containing MPC info.
        """
        return self.__visualization

    @visualization.setter
    def visualization(self, value):
        print('Cannot set visualization object.')