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
tether aerodynamics model of an awe system
takes states, finds approximate total force and moment for a tether element
finds equivalent forces corresponding to the total force and moment.
_python-3.5 / casadi-3.4.5
- author: elena malz, chalmers 2016
- edited: rachel leuthold, jochem de schutter alu-fr 2017
'''

import casadi.tools as cas
import numpy as np

import awebox.tools.vector_operations as vect_op
import awebox.mdl.aero.tether_dir.reynolds as reynolds
import awebox.mdl.aero.tether_dir.segment as segment
import awebox.mdl.aero.tether_dir.element as element


def get_scale(options, atmos, wind, upper_node):

    if upper_node == 1:
        diam = options['scaling']['theta']['diam_t']
        length = options['scaling']['xd']['l_t']
    else:
        diam = options['scaling']['theta']['diam_s']
        length = options['scaling']['theta']['l_s']

    area = diam * length

    rho = atmos.get_density_ref()
    u = wind.get_velocity_ref()
    dyn_press = 0.5 * rho * u**2.

    scale = area * dyn_press

    return scale

def get_tether_model_types(options):

    selected_model = options['tether']['tether_drag']['model_type']

    if selected_model == 'not_in_use':
        tether_models = []

    elif selected_model == 'kite_only':
        tether_models = ['kite_only']

    elif selected_model == 'split':
        tether_models = ['split', 'single', 'multi']

    else:
        tether_models = ['split'] + [selected_model]

    return tether_models


def get_force_vars(variables, upper_node, options, atmos, wind):

    tether_models = get_tether_model_types(options)
    sides = ['upper', 'lower']

    scale = get_scale(options, atmos, wind, upper_node)

    forces = {}
    for model in tether_models:
        for side in sides:
            name = model + '_' + side + str(upper_node)
            var = variables['xl']['f_' + name]
            forces[name] = var * scale

    return forces

def get_residuals(outputs, variables, architecture, options, atmos, wind):

    tether_models = get_tether_model_types(options)
    sides = ['upper', 'lower']

    all_resi = []

    n_nodes = architecture.number_of_nodes
    for n in range(1, n_nodes):
        force_vars = get_force_vars(variables, n, options, atmos, wind)
        force_outputs = outputs['tether_aero']

        scale = get_scale(options, atmos, wind, n)

        for model in tether_models:
            for side in sides:

                name = model + '_' + side + str(n)

                f_var = force_vars[name]
                f_out = force_outputs[name]

                resi = (f_var - f_out) / scale
                all_resi = cas.vertcat(all_resi, resi)

    return all_resi




def get_force_outputs(model_options, variables, atmos, wind, upper_node, tether_cd_fun, outputs, architecture):

    split_distribution_fun = segment.get_segment_half_fun()
    equivalent_distribution_fun = segment.get_segment_equiv_fun()

    element_drag_fun, element_moment_fun = element.get_element_drag_and_moment_fun(wind, atmos, tether_cd_fun)

    kite_only_lower, kite_only_upper = segment.get_kite_only_segment_forces(atmos, outputs, variables, upper_node, architecture, tether_cd_fun)

    split_lower, split_upper = segment.get_distributed_segment_forces(1, variables, upper_node, architecture, element_drag_fun,
                                   element_moment_fun, split_distribution_fun)

    single_lower, single_upper = segment.get_distributed_segment_forces(1, variables, upper_node, architecture, element_drag_fun,
                                   element_moment_fun, equivalent_distribution_fun)

    n_elements = model_options['tether']['aero_elements']
    multi_lower, multi_upper = segment.get_distributed_segment_forces(n_elements, variables, upper_node, architecture, element_drag_fun,
                                   element_moment_fun, equivalent_distribution_fun)

    re_number = segment.get_segment_reynolds_number(variables, atmos, wind, upper_node, architecture)

    if 'tether_aero' not in list(outputs.keys()):
        outputs['tether_aero'] = {}

    outputs['tether_aero']['multi_upper' + str(upper_node)] = multi_upper
    outputs['tether_aero']['multi_lower' + str(upper_node)] = multi_lower
    outputs['tether_aero']['single_upper' + str(upper_node)] = single_upper
    outputs['tether_aero']['single_lower' + str(upper_node)] = single_lower
    outputs['tether_aero']['split_upper' + str(upper_node)] = split_upper
    outputs['tether_aero']['split_lower' + str(upper_node)] = split_lower
    outputs['tether_aero']['kite_only_upper' + str(upper_node)] = kite_only_upper
    outputs['tether_aero']['kite_only_lower' + str(upper_node)] = kite_only_lower

    outputs['tether_aero']['reynolds' + str(upper_node)] = re_number

    return outputs

