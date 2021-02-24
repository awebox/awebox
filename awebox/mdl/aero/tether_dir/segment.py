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
drag on a tether segment (between two nodes)
_python-3.5 / casadi-3.4.5
- edited: rachel leuthold, jochem de schutter alu-fr 2017-20
'''

import awebox.tools.vector_operations as vect_op
import casadi.tools as cas
import numpy as np

import awebox.mdl.aero.tether_dir.element as element
import awebox.mdl.aero.tether_dir.reynolds as reynolds

def get_segment_drag(n_elements, variables, upper_node, architecture, element_drag_fun):

    combined_info = []
    for elem in range(n_elements):
        elem_info = element.get_element_info_column(variables, upper_node, architecture, elem, n_elements)
        combined_info = cas.horzcat(combined_info, elem_info)

    drag_map = element_drag_fun.map(n_elements, 'openmp')
    
    combined_drag = drag_map(combined_info)

    return combined_drag

def get_distributed_segment_forces(n_elements, variables, upper_node, architecture, element_drag_fun):

    q_upper, q_lower, _, _ = element.get_upper_and_lower_pos_and_vel(variables, upper_node, architecture)
    combined_drag = get_segment_drag(n_elements, variables, upper_node, architecture, element_drag_fun)

    # integration step size
    ds = 1.0/n_elements

    # integration grid (midpoint rule)
    s_grid = np.linspace(0.5*ds, 1 - 0.5*ds, n_elements)

    # numerical evaluation of analytic drag force expressions
    force_upper = sum([s_grid[k]*combined_drag[:, k] for k in range(n_elements)])
    force_lower = sum([(1-s_grid[k])*combined_drag[:, k] for k in range(n_elements)])

    return force_lower, force_upper


def get_kite_only_segment_forces(atmos, outputs, variables, upper_node, architecture, cd_tether_fun):

    force_lower = cas.DM.zeros((3, 1))
    force_upper = cas.DM.zeros((3, 1))

    if upper_node in architecture.kite_nodes:

        kite = upper_node

        ehat_1 = outputs['aerodynamics']['ehat_chord' + str(kite)]
        ehat_3 = outputs['aerodynamics']['ehat_span' + str(kite)]
        alpha = outputs['aerodynamics']['alpha' + str(kite)]
        d_hat = cas.cos(alpha) * ehat_1 + cas.sin(alpha) * ehat_3

        kite_dyn_pressure = outputs['aerodynamics']['dyn_pressure' + str(kite)]
        q_upper, q_lower, dq_upper, dq_lower = element.get_upper_and_lower_pos_and_vel(variables, upper_node,
                                                                                       architecture)
        length = vect_op.norm(q_upper - q_lower)
        diam = element.get_element_diameter(variables, upper_node, architecture)

        air_velocity = outputs['aerodynamics']['air_velocity' + str(kite)]
        re_number = reynolds.get_reynolds_number(atmos, air_velocity, diam, q_upper, q_lower)
        cd_tether = cd_tether_fun(re_number)

        d_mag = (1./4.) * cd_tether * kite_dyn_pressure * diam * length

        force_upper = d_mag * d_hat

    return force_lower, force_upper

def get_segment_reynolds_number(variables, atmos, wind, upper_node, architecture):
    diam = element.get_element_diameter(variables, upper_node, architecture)

    q_upper, q_lower, dq_upper, dq_lower = element.get_upper_and_lower_pos_and_vel(variables, upper_node, architecture)

    ua = element.get_uapp(q_upper, q_lower, dq_upper, dq_lower, wind)

    re_number = reynolds.get_reynolds_number(atmos, ua, diam, q_upper, q_lower)

    return re_number
