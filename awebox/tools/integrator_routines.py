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
self-written integrator routines for DAE systems
python-3.5 / casadi-3.4.5
- author: jochem de schutter alu-fr 2018
"""

import casadi.tools as cas

def rk4root(name, dae, rootfinder, options):
    """explicit RK4 integrator for DAE systems, using rootfinder
    for the algebraic equations

    @param name integrator name
    @param dae dae object
    @param rootfinder newton solver for algebraic equations
    @param options integrator options
    @return I rk4root integrator function
    """

    # discretization info
    N = options['number_of_finite_elements']
    h = options['tf']/N

    # dae functions
    odef  = cas.Function('odef',[dae['x'],dae['p'],dae['z']], [dae['ode']])
    quadf = cas.Function('quadf',[dae['x'],dae['p'],dae['z']], [dae['quad']])

    # initialize
    x0 = dae['x'](cas.MX.sym('x',dae['x'].shape))
    z0  = dae['z'](cas.MX.sym('z',dae['z'].shape))
    p   = dae['p'](cas.MX.sym('p',dae['p'].shape))
    qf = 0.0

    for i in range(N):

        # first iteration starts with x0
        if i == 0:
            xf = x0

        #rk4 with rootfinder step
        [xf, zf, qf] = rk4root_step(odef, rootfinder, quadf, h, xf, z0, p, qf)

    I = cas.Function(name, [x0, z0, p], [xf, zf, qf], ['x0','z0','p'],['xf','zf','qf'])

    return I


def rk4root_step(my_ode, rootfinder, quad, h, x0, z_guess, p, q0):

   # RK4 function evaluations
   z   = rootfinder(z_guess, x0, p)
   k1  = my_ode(x0, p, z)
   qk1 = quad(x0, p, z)

   z   = rootfinder(z, x0 + h*k1/2,p)
   k2  = my_ode(x0 + h * k1 / 2 , p, z)
   qk2 = quad(x0 + h * k1 / 2, p, z)

   z   = rootfinder(z, x0 + h*k2/2, p)
   k3  = my_ode(x0 + h * k2 / 2 , p, z)
   qk3 = quad(x0 + h * k2 / 2, p, z)

   z   = rootfinder(z, x0+ h*k3, p)
   k4  = my_ode(x0 + h * k3, p, z)
   qk4 = quad(x0 + h * k3 / 2, p, z)

   # Output state, algebraic and quadrature variables
   xout = (x0 + h * (k1 + 2 * k2 + 2* k3 + k4) / 6)
   zout = rootfinder(z, xout, p)
   qout = (q0 + h * (qk1 + 2 * qk2 + 2* qk3 + qk4) / 6)

   return [xout, zout, qout]
