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
multiple shooting code
python-3.5 / casadi-3.4.5
- author: jochem de schutter alu-fr 2018
'''

import casadi.tools as cas
import numpy as np
import awebox.tools.struct_operations as struct_op
from . import constraints

import awebox.tools.constraint_operations as cstr_op

class Multiple_shooting(object):
    """Multiple shooting class with methods for optimal control
    """

    def __init__(self, n_k, dae, options):
        """Constructor

        @param n_k number of intervals
        @param dae model equations
        @param options integrator options
        """

        # save discretization info
        self.__n_k = n_k
        self.__dae = dae

        # contruct integrator
        self.__F = self.__dae.build_integrator(options, 1./self.__n_k)

        return None

    def discretize_constraints(self, options, model, formulation, V, P):
        """Discretize dynamics and path constraints in a parallelizable fashion

        @param options nlp options
        @param model awebox model
        @param formulation awebox formulation
        @param V decision variables
        @param P nlp parameters
        """

        # rearrange nlp variables
        self.__ms_nlp_vars(options, model, V, P)

        # implicit values of algebraic variables at interval nodes
        ms_z0 = self.__ms_z0

        # use function map for parallellization
        parallellization = options['parallelization']['type']
        F_map = self.__F.map('F_map', parallellization, self.__n_k, [], [])
        path_constraints_fun = model.constraints_fun.map('constraints_map', parallellization, self.__n_k, [], []) # notice that these are the model inequality constraints
        outputs_fun = model.outputs_fun.map('outputs_fun', parallellization, self.__n_k, [], [])

        # integrate
        ms_dynamics = F_map(x0= self.__ms_x, z0 = self.__ms_z, p = self.__ms_p)
        ms_xf = ms_dynamics['xf']
        ms_qf = cas.horzcat(np.zeros(self.__dae.dae['quad'].size()), ms_dynamics['qf'])
        ms_constraints = path_constraints_fun(self.__ms_vars, self.__ms_params) # evaluate the model ineqs. at
        ms_outputs = outputs_fun(self.__ms_vars, self.__ms_params)

        # integrate quadrature outputs
        for i in range(self.__n_k):
            ms_qf[:,i+1] = ms_qf[:,i+1] + ms_qf[:,i]

        # integral outputs and constraints
        Integral_outputs_list = self.__build_integral_outputs(ms_qf, model.integral_outputs)
        Integral_constraints_list = None

        # construct state derivative struct
        Xdot = struct_op.construct_Xdot_struct(options, model.variables_dict)
        Xdot = self.__fill_in_Xdot(Xdot)

        return ms_xf, ms_z0, self.__ms_vars, self.__ms_params, Xdot, ms_constraints, ms_outputs, Integral_outputs_list, Integral_constraints_list

    def __ms_nlp_vars(self, options, model, V, P):
        """Rearrange decision variables to dae-compatible form,
        allowing for parallel function evaluations

        @param model awebox model
        @param V nlp decision variables
        @param P nlp parameters
        """

        # interval parameters
        param_at_time = model.parameters(cas.vertcat(P['theta0'], V['phi']))
        ms_params = cas.repmat(param_at_time, 1, self.__n_k)

        # use function map for rootfinder parallellization
        G_map = self.__dae.rootfinder.map('G_map', options['parallelization']['type'], self.__n_k, [], [])
        x_root = []
        z_root = []
        p_root = []

        # compute explicit values of implicit variables
        ms_vars0 = []
        for kdx in range(self.__n_k):
            # get vars at time
            var_at_time = struct_op.get_variables_at_time(options, V, None, model.variables, kdx)
            ms_vars0 += [var_at_time]
            # get dae vars at time
            x, z, p = self.__dae.fill_in_dae_variables(var_at_time, param_at_time)

 
            # store vars for parallelization
            x_root = cas.horzcat(x_root, x)
            z_root = cas.horzcat(z_root, z)
            p_root = cas.horzcat(p_root, p)

        # compute implicit vars in parallel fashion
        z_implicit = G_map(z_root, x_root, p_root)

        # construct list of all interval variables
        # ms_vars = []
        # ms_x = []
        # ms_z = []
        # ms_p = []

        # for kdx in range(self.__n_k):
        #     # fill in non-lifted vars (not applicable)
        #     #var_at_time = self.__set_implicit_variables(options, ms_vars0[kdx], param_at_time, self.__dae.z(z_implicit[:,kdx]))
        #     # update dae vars at time
        #     x, z, p = self.__dae.fill_in_dae_variables(var_at_time, param_at_time)

        #     # store result
        #     ms_vars = cas.horzcat(ms_vars, var_at_time)
        #     ms_x = cas.horzcat(ms_x, x)
        #     ms_z = cas.horzcat(ms_z, z)
        #     ms_p = cas.horzcat(ms_p, p)

        self.__ms_params = ms_params
        self.__ms_vars = cas.horzcat(*ms_vars0)
        self.__ms_x = x_root
        self.__ms_z = z_root
        self.__ms_z0 = z_implicit
        self.__ms_p = p_root

        return None

    def __set_implicit_variables(self, options, variables, parameters, z_at_time):
        """Set non-lifted implicit variables xa, xl and xddot to value computed
        using rootfinder

        @param options nlp options
        @param variables vars at a specific time
        @param parameters params at a specific time
        @param z_at_time alg vars computed with rootfinder
        @return variables variables struct containing implicit variable values
        """

        if (not options['lift_xddot'] or not options['lift_xa']):

            # fill in result if not lifted
            var_list = []
            for var_type in list(variables.keys()):
                if var_type == 'xddot':
                    if not options['lift_xddot']:
                        var_list.append(z_at_time['xddot'])
                    else:
                        var_list.append(variables['xddot'])
                elif var_type in set(['xa','xl']):
                    if not options['lift_xa']:
                        var_list.append(z_at_time[var_type])
                    else:
                        var_list.append(variables[var_type])
                else:
                    var_list.append(variables[var_type])

            variables = variables(cas.vertcat(*var_list))

        return variables

    def __build_integral_outputs(self, ms_qf, integral_outputs):
        """Build integral outputs list based on integrator quadrature outputs

        @param ms_dynamics integrator quadrature outputs
        @param integral_outputs model integral outputs
        """

        # initialize quadratures at zero
        Integral_outputs_list = []

        for i in range(ms_qf.size()[1]):
            Integral_outputs_list.append(ms_qf[:,i])

        return Integral_outputs_list


    def get_continuity_constraint(self, ms_xf, V, kdx):
        """Append multiple shooting continuity constraint to list of constraints

        @param g_list current list of constraints
        @param g_bounds corresponding list of constraint bounds
        @param ms_xf integrator output
        @param V nlp decision variables
        @param kdx interval index
        """

        # add continuity equation to nlp
        g_continuity = V['xd', kdx + 1] - ms_xf[:,kdx]

        cont_cstr = cstr_op.Constraint(expr=g_continuity,
                                  name='continuity_{}'.format(kdx),
                                  cstr_type='eq')
        return cont_cstr

    def __fill_in_Xdot(self, Xdot):
        """Construct state derivatives at all interval nodes

        @param Xdot empty Xdot struct
        @return Xdot state derivatives at interval nodes
        """

        xdot = []
        for i in range(self.__ms_z[1,:].shape[1]):
            z_at_time = self.__dae.z(self.__ms_z[:,i])
            xdot = cas.vertcat(xdot, z_at_time['xddot'])

        Xdot = Xdot(xdot)

        return Xdot

    @property
    def F(self):
        """Dae integrator"""
        return self.__F
