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
Simulation class for open-loop and closed-loop simulations based on awebox reference trajectories
and related models.

:author: Jochem De Schutter - ALU Freiburg 2019
"""


import casadi.tools as ct
import awebox.pmpc as pmpc
import awebox.tools.integrator_routines as awe_integrators
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
        model = self.__trial.generate_optimal_model()
        self.__F = awe_integrators.rk4root(
            'F',
            model['dae'],
            model['rootfinder'],
            {'tf': self.__ts/model['t_f'],
            'number_of_finite_elements':self.__sim_options['number_of_finite_elements']}
            )

        if self.__sim_type == 'closed_loop':

            self.__mpc = pmpc.Pmpc(self.__mpc_options, self.__ts, self.__trial)

        return None

    def run(self, n_sim, x0 = None, u_sim = None):
        """ Run simulation
        """

        # TODO: check consistency of initial conditions and give warning

        x0 = self.__initialize_sim(n_sim, x0, u_sim)

        for i in range(n_sim):

            # get (open/closed-loop) controls
            if self.__sim_type == 'closed_loop':
                u0 = self.__mpc.step(x0)

            elif self.__sim_type == 'open_loop':
                u0 = self.__u_sim[:,i]

            # simulate
            x0 = self.__F(x0 = x0, p = u0, z0 = self.__z0)

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

        # initialize plot_dict
        self.__plot_dict = {
            'variables_dict': self.__trial.model.variables_dict,
            'integral_variables': self.__trial.model.integral_outputs
        }

        return x0

    @property
    def trial(self):
        """ awebox.Trial attribute containing model and OCP info.
        """
        return self.__trial

    @trial.setter
    def trial(self, value):
        print('Cannot set trial object.')