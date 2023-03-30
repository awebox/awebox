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
trajectory averaged induction model
python-3.5 / casadi-3.4.5
- authors: jochem de schutter, 2021-2022
- edit: rachel leuthold, 2022
'''
import pdb

import casadi.tools as cas
import numpy as np
import awebox.tools.vector_operations as vect_op
import awebox.tools.struct_operations as struct_op
import awebox.tools.print_operations as print_op
import awebox.tools.constraint_operations as cstr_op

def get_dictionary_of_derivatives(system_variables, parameters, atmos, wind, outputs, architecture):
    derivative_dict = {}

    tether_forces = 0.0
    WdA = 0.0
    for kite in architecture.kite_nodes:
        tether_forces += outputs['local_performance']['tether_force{}'.format(architecture.node_label(kite))]
        q = system_variables['SI']['x']['q{}'.format(architecture.node_label(kite))]
        rho = atmos.get_density(q[2])[0]
        u_inf = wind.get_velocity(q[2])[0]
        dq = system_variables['SI']['x']['dq{}'.format(architecture.node_label(kite))]
        b = parameters['theta0', 'geometry', 'b_ref']
        WdA += 0.5 * b * vect_op.norm(dq) * rho * u_inf ** 2

    tether_force_int_scaling = 1.0
    area_int_scaling = 1.
    derivative_dict['tether_force_int'] = (tether_forces, tether_force_int_scaling)
    derivative_dict['area_int'] = (WdA, area_int_scaling)

    return derivative_dict

def get_ellipse_half_constraint(options, variables_si, parameters, architecture, outputs):

    cstr_list = cstr_op.MdlConstraintList()

    alpha = parameters['theta0', 'model_bounds', 'ellipsoidal_flight_region', 'alpha']
    if options['model_bounds']['ellipsoidal_flight_region']['include']:
        ell_theta = variables_si['theta']['ell_theta']
        for kite in architecture.kite_nodes:
            q = variables_si['x']['q{}'.format(architecture.node_label(kite))]

            yy = q[1]
            zz = - q[0]*np.sin(alpha) + q[2]*np.cos(alpha)
            if kite == 2:
                ellipse_half_ineq = np.cos(ell_theta)*zz - np.sin(ell_theta)*yy
            elif kite == 3:
                ellipse_half_ineq = np.sin(ell_theta)*yy - np.cos(ell_theta)*zz

            ellipse_half_cstr = cstr_op.Constraint(expr=ellipse_half_ineq,
                                        name='ellipse_half' + architecture.node_label(kite),
                                        cstr_type='ineq')
            cstr_list.append(ellipse_half_cstr)

    return outputs, cstr_list
