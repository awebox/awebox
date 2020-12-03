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
DAE routines for awebox models
python-3.5 / casadi 3.0.0
- author: jochem de schutter, alu-fr 2018
"""

import awebox.tools.integrator_routines as int_rout
import casadi.tools as cas

class Dae(object):
    """
    Dae object that serves as an interface to CasADi's
    `rootfinder <http://casadi.sourceforge.net/api/html/d3/d65/group__rootfinder.html>`_ and
    `integrator <http://casadi.sourceforge.net/api/html/dd/d1b/group__integrator.html>`_ 
    solvers for awebox Model objects .
    """

    def __init__(self, variables, parameters, dynamics, integral_outputs_fun, param = 'sym'):
        """ Constructor.
    
        :type variables: casadi.tools.structure.ssymStruct
        :param variables: model variables
    
        :type parameters: casadi.tools.structure.ssymStruct
        :param parameters: model parameters
    
        :type dynamics: casadi.Function
        :param dynamics: fully implicit dae dynamics
    
        :type integral_outputs_fun: casadi.Function
        :param integral_outputs_fun: quadrature state dynamics
    
        :raises ValueError: if the DAE-index is higher than 1
    
        :rtype: None
        """    

        # construct the DAE variables
        x, z, p = self.__build_dae_variables(variables, parameters, param)

        # model equations
        alg = dynamics(variables, parameters)
        ode = variables['theta','t_f']*variables['xddot']
        quad = variables['theta','t_f']*integral_outputs_fun(variables, parameters)

        # create dae dictionary
        dae = {'x': x, 'z': z, 'p': p, 'alg': alg,'ode': ode, 'quad': quad}

        if cas.sprank(cas.jacobian(alg,z)) < z.cat.size()[0]:  # check dae index
            raise ValueError('jacobian of dynamics is structurally rank-deficient: DAE is not of index 1!')

        self.__x = x
        self.__z = z
        self.__p = p
        self.__dae = dae

        return None

    def build_rootfinder(self):
        """ Create rootfinder function for fully implicit dae object.
        """

        # create rootfinder
        g = cas.Function('g',[self.__z.cat,self.__x.cat,self.__p.cat],[self.__dae['alg']])
        G = cas.rootfinder('G', 'fast_newton', g, {'jit': True})

        self.__rootfinder = G

        return None

    def build_integrator(self, options, time_step):
        """ Create integrator for fully implicit dae object.

        @param options - options including integrator type
        @param time_step - time horizon of one integrator step
        """

        # set options
        opts =  {'tf': time_step,'jit': options['jit'],'expand':True}

        if options['type'] is not 'rk4root':

            # collocation options
            if options['type'] == 'collocation':

                opts['number_of_finite_elements'] = options['num_steps']
                opts['collocation_scheme'] = options['collocation_scheme']
                opts['interpolation_order'] = options['interpolation_order']
                opts['rootfinder'] = 'fast_newton'

            # create integrator
            I = cas.integrator('I', options['type'], self.__dae, opts)

        else:

            opts['number_of_finite_elements'] = options['num_steps']
            I = int_rout.rk4root('I', self.__dae, self.__rootfinder, opts)

        return I

    def __build_dae_variables(self, variables, parameters, param):
        """ Create dae variables based on awebox model variables and parameters.
        
        @ param variables - model variables struct
        @ param parameters - parameters struct
        @ return - dae states 'x', algebraic vars 'z' and parameters 'p'
        """

        # differential states
        x = cas.struct_SX([cas.entry('xd', expr = variables['xd'])])

        # algebraic variables
        if 'xl' in list(variables.keys()):
            z = cas.struct_SX([cas.entry('xddot', expr = variables['xddot']), # state derivatives
                            cas.entry('xa', expr = variables['xa']), # algebraic variables
                            cas.entry('xl', expr = variables['xl']), # lifted variables
                            ])
        else:
            z = cas.struct_SX([cas.entry('xddot', expr = variables['xddot']), # state derivatives
                        cas.entry('xa', expr = variables['xa']), # algebraic variables
                    ])

        # parameters
        if param == 'sym':
            p = cas.struct_SX([cas.entry('u', expr = variables['u']), # controls
                            cas.entry('theta', expr = variables['theta']), # free parameters
                            cas.entry('param', expr = parameters)]) # fixed parameters
        elif param == 'num':
            p = cas.struct_SX([cas.entry('u', expr = variables['u'])]) # controls

        return x, z, p

    def fill_in_dae_variables(self, variables, parameters):

        x = self.__x(variables['xd'])

        if 'xl' in list(variables.keys()):
            z = self.__z(cas.vertcat(variables['xddot'],
                        variables['xa'],
                        variables['xl']))
        else:
            z = self.__z(cas.vertcat(variables['xddot'],
                        variables['xa']))

        p = self.__p(cas.vertcat(
                variables['u'], variables['theta'], parameters.cat
            )
        )

        return x, z, p

    @property
    def rootfinder(self):
        """Newton-type rootfinder"""
        return self.__rootfinder

    @property
    def z(self):
        """algebraic variable struct"""
        return self.__z

    @property
    def dae(self):
        """dae dictionary"""
        return self.__dae

