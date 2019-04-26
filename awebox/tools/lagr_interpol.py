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
LAGRANGE INTERPOLATION to generate a polyomial fit to a series of datapoints
The lagrange polyonomial is penalized on its second derivative in order to get a smoother and more realistic fit of the wind shear.
Python Version 2.7 / Casadi version 3.3.0
- Author: Elena Malz, Chalmers 2019, elenama@chalmers.se
'''

import casadi.tools as cas

def lagrange_poly(x, y):
    "creates an lagrange polynomial through each data point based on x and y data"
    t = cas.SX.sym('t')
    d = x.shape[0]  # amount of parameters

    poly = 0
    for j in range(d):  # for all data points ...
        L = y[j]  # parameter = fct output
        for r in range(d):
            if r != j:
                L *= (t - x[r]) / (x[j] - x[r])
        poly += L
    lfcn = cas.Function('lfcn', [t], [poly])
    return lfcn


def smooth_lagrange_poly(x, y):
    t    = cas.SX.sym('t')
    d    = len(x)                       # amount of parameters
    tau  = cas.SX.sym('tau',d)              # parameter as minimisation variable
    poly = 0

    for j in range(d):                  # for all data points ...
        L = tau[j]
        for r in range(d):
            if r != j:
                L *= (t-x[r])/(x[j]-x[r])
        poly+=L
    L_fun   = cas.Function('L_fun', [t,tau],[poly])
    ddL,_     = cas.hessian(poly,t)
    ddL_fun = cas.Function('ddL_fun', [t,tau], [ddL])
    # ddL_fun = L_fun.hessian(0)          # second order derivative to
    # [ddL,_,_]  = ddL_fun([t,tau])

    # minimise tau = fct output, incl penalize curvature
    res = 0.1 *  sum([(L_fun(x[k],tau) - y[k])**2 for k in range(d)])[0]
    res += sum([ddL_fun(x[k],tau)[0]**2 * 1e5 for k in range(d)])[0]

    Cost= cas.Function('cost',[tau],[res])
    nlp = {'x': tau, 'f': res}
    opts = {}
    opts['ipopt.print_level'] = 0
    solver = cas.nlpsol("solver", "ipopt", nlp,opts)
    sol = solver(**{})
    opts['ipopt.print_level'] = 0
    tau_opt = sol['x']                  # optimal parameter for polynomial
    return L_fun, tau_opt
