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
###################################
# Class Simulation runs a forward in time simulation of the multi-kite system from an initial position
###################################

import awebox.tools.struct_operations as struct_op
import casadi.tools as cas

class Simulation:
    def __init__(self, options):
        self.__status = 'Simulation not yet solved.'
        self.__N_sim = options['Nsim']
        self.__integrator_type = options['integrator']['type']

        self.__outputs = None

    def build_integrator(self, options, model):
        print('Building integrator...')

        # get model variables
        variables = model.variables
        # construct the DAE variables
        x = cas.struct_SX([cas.entry('xd', expr = variables['xd'])]) # differential states
        z = cas.struct_SX([cas.entry('xddot', expr = variables['xddot']), # state derivatives
                       cas.entry('xa', expr = variables['xa']), # algebraic variables
                       cas.entry('xl', expr = variables['xl']), # lifted variables
                      ])
        p = cas.struct_SX([cas.entry('u', expr = variables['u']), # dae parameters
                       cas.entry('theta', expr = variables['theta']),
                       cas.entry('phi', expr = model.parameters)])

        # scale xddot with t_f
        time_scaled_variables = scale_xddot(variables)

        # model equations
        alg = model.dynamics(time_scaled_variables, model.parameters)
        ode = variables['xddot']

        # create dae
        dae = {'x': x.cat, 'z': z.cat, 'p': p.cat, 'alg': alg,'ode': ode}
        # system dynamics
        f = cas.Function('f', [x, z, p], [ode, alg], ['x', 'z', 'p'], ['ode', 'alg'])

        # create integrator and rootfinder
        if cas.sprank(cas.jacobian(alg,z)) < z.cat.size()[0]:  # check dae index
            raise ValueError('jacobian of dynamics is structurally rank-deficient: DAE is not of index 1!')
        else:
            # create integrator
            I = cas.integrator('I', options['integrator']['type'], dae, {'tf': 1.0 / self.__N_sim})
            # create rootfinder
            g = cas.Function('g',[z.cat,x.cat,p.cat],[alg])
            G = cas.rootfinder('G', 'newton', g, {'linear_solver': 'csparse'})
            self.__integrator = I
            self.__rootfinder = G
            self.__variables_dict = model.variables_dict
            self.__phi = model.parameters
            self.__dae = dae
            self.__f = f
            self.__x = x
            self.__z = z
            self.__p = p

    def run(self, x0, u_sim, theta_sim, phi_sim):
        # check consistency of initial conditions:
        # to do: check values of g, gdot...
        consistency = True
        if consistency == False:
            raise ValueError('provided initial conditions are not consistent!')
        else:
            # horizon length
            N = len(u_sim)

            # V_sim / simulation output structure
            V_sim = cas.struct_symMX([
            (
                cas.entry('xd', repeat=[N, self.__N_sim],   struct=self.__variables_dict['xd']),
                cas.entry('xa', repeat=[N, self.__N_sim-1], struct=self.__variables_dict['xa']),
                cas.entry('xl', repeat=[N, self.__N_sim-1], struct=self.__variables_dict['xl']),
                cas.entry('u', repeat=[N], struct=self.__variables_dict['u']),
            ),
            cas.entry('theta', struct=self.__variables_dict['theta']),
            cas.entry('phi', struct=self.__phi)
            ])

            # initialize solution vector
            V0 = V_sim(0.)

            # fill in controls and parameters
            for i in range(N):
                for name in list(self.__variables_dict['u'].keys()):
                    V0['u',i,name] = self.__variables_dict['u'](u_sim[i])[name]
            V0['theta'] = theta_sim
            # adjust time-scaling factor for integrator step size
            V0['theta','t_f'] = V0['theta','t_f'] / N
            V0['phi'] = phi_sim

            # integrate / fill in states and alg vars

            # initial state
            x_sim = self.__x(x0)['xd']
            z_sim = self.__z(0.0)
            for i in range(N):

                # dae parameters for this time step
                p_sim = self.__p(0.)
                p_sim['u'] = u_sim[i]
                p_sim['theta'] = V0['theta']
                p_sim['phi'] = V0['phi']

                # state on initial time of shooting node
                for name in list(self.__variables_dict['xd'].keys()):
                    V0['xd',i,0,name] = self.__variables_dict['xd'](x_sim)[name]

                # integrate up to (including) the final time of the shooting node
                for j in range(self.__N_sim-1):
                    print(j)
                    ### TESTING
                    # [ode_test, alg_test] = self.__f(x_sim,0.,p_sim.cat)
                    # z_test = self.__rootfinder(0.,x_sim,p_sim.cat)
                    # xddot_test = self.__variables_dict['xddot'](self.__z(z_test)['xddot'])

                    # perform integration
                    res = self.__integrator(x0= x_sim, p = p_sim.cat, z0 = z_sim.cat)

                    # set new initial guess for algebraic variable
                    z_sim = self.__z(res['zf'])

                    # set algebraic variables
                    for name in list(self.__variables_dict['xa'].keys()):
                        V0['xa',i,j,name] = self.__variables_dict['xa'](z_sim['xa'])[name]
                    # set lifted variables
                    for name in list(self.__variables_dict['xl'].keys()):
                        V0['xl',i,j,name] = self.__variables_dict['xl'](z_sim['xl'])[name]

                    # set-up next state
                    x_sim = self.__x(res['xf'])['xd']

                    # set differential states
                    for name in list(self.__variables_dict['xd'].keys()):
                        V0['xd',i,j+1,name] = self.__variables_dict['xd'](x_sim)[name]


            self.__status = 'I am a simulation.'
            self.__V0 = V0
            print('Simulation solved.')

            return None


    @property
    def status(self):
        return self.__status

    @status.setter
    def status(self, value):
        print('Cannot set status object.')

    @property
    def V0(self):
        return self.__V0

    @V0.setter
    def V0(self, value):
        print('Cannot set V0 object')

    @property
    def f(self):
        return self.__f

    @f.setter
    def f(self, value):
        print('Cannot set f object')

    @property
    def x(self):
        return self.__x

    @x.setter
    def x(self, value):
        print('Cannot set x object')

    @property
    def p(self):
        return self.__p

    @p.setter
    def p(self, value):
        print('Cannot set p object')

    @property
    def z(self):
        return self.__z

    @z.setter
    def z(self, value):
        print('Cannot set z object')

    @property
    def dae(self):
        return self.__dae

    @dae.setter
    def dae(self, value):
        print('Cannot set dae object')

    @property
    def rootfinder(self):
        return self.__rootfinder

    @rootfinder.setter
    def rootfinder(self, value):
        print('Cannot set rootfinder object')

    @property
    def outputs(self):
        return self.__outputs

    @outputs.setter
    def outputs(self, value):
        print('Cannot set outputs object.')

def scale_xddot(variables):

    time_scaled_variables = variables(variables.cat)
    for name in struct_op.subkeys(variables, 'xddot'):
        time_scaled_variables['xddot',name] = variables['xddot',name]/variables['theta','t_f']

    return time_scaled_variables
