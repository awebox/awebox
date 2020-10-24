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
tether aerodynamics model of an awe system
takes states, finds approximate total force and moment for a tether element
finds equivalent forces corresponding to the total force and moment.
_python-3.5 / casadi-3.4.5
- author: elena malz, chalmers 2016
- edited: rachel leuthold, jochem de schutter alu-fr 2020
'''

import casadi.tools as cas
import numpy as np

import awebox.tools.vector_operations as vect_op
import awebox.mdl.aero.tether_dir.reynolds as reynolds
import awebox.mdl.aero.tether_dir.segment as segment
import awebox.mdl.aero.tether_dir.element as element
import awebox.tools.print_operations as print_op
import pdb


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


def get_force_var(variables_si, upper_node, architecture):

    lower_node = architecture.parent_map[upper_node]
    name = str(upper_node) + str(lower_node)
    var = variables_si['xl']['f_tether' + name]
    return var

def get_tether_resi(options, variables_si, atmos, wind, architecture, parameters, outputs):

    # homotopy parameters
    p_dec = parameters.prefix['phi']
    
    # initialize dictionary
    tether_drag_forces = {}
    for node in range(1, architecture.number_of_nodes):
        parent = architecture.parent_map[node]
        tether_drag_forces['f' + str(node) + str(parent)] = cas.SX.zeros((3, 1))

    for node in range(1, architecture.number_of_nodes):
        parent = architecture.parent_map[node]
    
        multi_upper = outputs['tether_aero']['multi_upper' + str(node)]
        multi_lower = outputs['tether_aero']['multi_lower' + str(node)]
        single_upper = outputs['tether_aero']['single_upper' + str(node)]
        single_lower = outputs['tether_aero']['single_lower' + str(node)]
        split_upper = outputs['tether_aero']['split_upper' + str(node)]
        split_lower = outputs['tether_aero']['split_lower' + str(node)]
        kite_only_upper = outputs['tether_aero']['kite_only_upper' + str(node)]
        kite_only_lower = outputs['tether_aero']['kite_only_lower' + str(node)]
    
        tether_model = options['tether']['tether_drag']['model_type']
    
        if tether_model == 'multi':
            drag_node = p_dec['tau'] * split_upper + (1. - p_dec['tau']) * multi_upper
            drag_parent = p_dec['tau'] * split_lower + (1. - p_dec['tau']) * multi_lower
    
        elif tether_model == 'single':
            drag_node = p_dec['tau'] * split_upper + (1. - p_dec['tau']) * single_upper
            drag_parent = p_dec['tau'] * split_lower + (1. - p_dec['tau']) * single_lower
    
        elif tether_model == 'split':
            drag_node = split_upper
            drag_parent = split_lower
    
        elif tether_model == 'kite_only':
            drag_node = kite_only_upper
            drag_parent = kite_only_lower
    
        elif tether_model == 'not_in_use':
            drag_parent = cas.DM.zeros((3, 1))
            drag_node = cas.DM.zeros((3, 1))
    
        else:
            raise ValueError('tether drag model not supported.')
    
        # attribute portion of segment drag to parent
        if node > 1:
            grandparent = architecture.parent_map[parent]
            tether_drag_forces['f' + str(parent) + str(grandparent)] += drag_parent
    
        # attribute portion of segment drag to node
        tether_drag_forces['f' + str(node) + str(parent)] += drag_node

    resi = []
    for node in range(1, architecture.number_of_nodes):

        parent = architecture.parent_map[node]
        f_tether_var = get_force_var(variables_si, node, architecture)
        f_tether_val = tether_drag_forces['f' + str(node) + str(parent)]
        local_resi_unscaled = (f_tether_var - f_tether_val)

        scale = options['scaling']['xl']['f_tether']
        print_op.warn_about_temporary_funcationality_removal(location='tether-aero')

        local_resi = local_resi_unscaled / scale
        
        resi = cas.vertcat(resi, local_resi)

    return resi






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

