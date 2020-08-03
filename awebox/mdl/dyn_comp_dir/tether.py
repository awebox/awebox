#
#    This file is part of awebox.
#
#    awebox -- A modeling and optimization framework for multi-kite AWE systems.
#    Copyright (C) 2017-2020 Jochem De Schutter, Rachel Leuthold, Moritz Diehl,
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
tether properties
_python-3.5 / casadi-3.4.5
- edited: rachel leuthold, jochem de schutter alu-fr 2017-20
'''

import casadi.tools as cas
import numpy as np

import awebox.tools.vector_operations as vect_op
import awebox.tools.struct_operations as struct_op
import awebox.tools.print_operations as print_op

from awebox.logger.logger import Logger as awelogger



def get_tether_segment_properties(options, architecture, variables_si, parameters, upper_node):
    kite_nodes = architecture.kite_nodes

    xd = variables_si['xd']
    theta = variables_si['theta']
    scaling = options['scaling']

    if upper_node == 1:
        vars_containing_length = xd
        vars_sym = 'xd'
        length_sym = 'l_t'
        diam_sym = 'diam_t'

    elif upper_node in kite_nodes:
        vars_containing_length = theta
        vars_sym = 'theta'
        length_sym = 'l_s'
        diam_sym = 'diam_s'

    else:
        vars_containing_length = theta
        vars_sym = 'theta'
        length_sym = 'l_i'
        diam_sym = 'diam_t'

    seg_length = vars_containing_length[length_sym]
    scaling_length = scaling[vars_sym][length_sym]

    seg_diam = theta[diam_sym]
    max_diam = options['system_bounds']['theta'][diam_sym][1]
    length_scaling = scaling[vars_sym][length_sym]
    scaling_diam = scaling['theta'][diam_sym]

    cross_section_area = np.pi * (seg_diam / 2.) ** 2.
    max_area = np.pi * (max_diam / 2.) ** 2.
    scaling_area = np.pi * (scaling_diam / 2.) ** 2.

    density = parameters['theta0', 'tether', 'rho']
    seg_mass = cross_section_area * density * seg_length
    scaling_mass = scaling_area * parameters['theta0', 'tether', 'rho'] * length_scaling

    props = {}
    props['seg_length'] = seg_length
    props['scaling_length'] = scaling_length

    props['seg_diam'] = seg_diam
    props['max_diam'] = max_diam
    props['scaling_diam'] = scaling_diam

    props['cross_section_area'] = cross_section_area
    props['max_area'] = max_area
    props['scaling_area'] = scaling_area

    props['seg_mass'] = seg_mass
    props['scaling_mass'] = scaling_mass

    return props
