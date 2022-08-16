#
#    This file is part of awebox.
#
#    awebox -- A modeling and optimization framework for multi-kite AWE systems.
#    Copyright (C) 2017-2021 Jochem De Schutter, Rachel Leuthold, Moritz Diehl,
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
constraints to create "intermediate condition" fixing constraints on the positions of the wake nodes,
to be referenced/used from ocp.constraints
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2020-21
'''
import copy
import pdb

import numpy as np
import awebox.mdl.aero.induction_dir.general_dir.tools as general_tools
import awebox.mdl.aero.induction_dir.vortex_dir.tools as vortex_tools
import awebox.mdl.aero.induction_dir.vortex_dir.alg_repr_dir.structure as alg_structure
import awebox.mdl.aero.induction_dir.vortex_dir.alg_repr_dir.fixing as alg_fixing

import awebox.tools.struct_operations as struct_op
import awebox.tools.print_operations as print_op
import awebox.tools.vector_operations as vect_op
import awebox.tools.constraint_operations as cstr_op

import casadi.tools as cas
from awebox.logger.logger import Logger as awelogger


################# define the actual constraint

def get_initialization(nlp_options, V_init, p_fix_num, nlp, model):

    time_grids = nlp.time_grids

    V_init_si = copy.deepcopy(V_init)
    V_init_scaled = struct_op.si_to_scaled(V_init_si, model.scaling)

    Outputs_init = nlp.Outputs(nlp.Outputs_fun(V_init_scaled, p_fix_num))

    integral_output_components = nlp.integral_output_components

    Integral_outputs_struct = integral_output_components[0]
    Integral_outputs_fun = integral_output_components[1]
    Integral_outputs_scaled = Integral_outputs_struct(Integral_outputs_fun(V_init_scaled, p_fix_num))

    V_init_scaled = append_specific_initialization('wx', nlp_options, V_init_scaled, Outputs_init, Integral_outputs_scaled, model, time_grids)
    V_init_scaled = append_specific_initialization('wg', nlp_options, V_init_scaled, Outputs_init, Integral_outputs_scaled, model, time_grids)

    far_wake_element_type = general_tools.get_option_from_possible_dicts(nlp_options, 'far_wake_element_type', 'vortex')
    if far_wake_element_type == 'semi_infinite_cylinder':
        V_init_scaled = append_specific_initialization('wx_center', nlp_options, V_init_scaled, Outputs_init, Integral_outputs_scaled, model, time_grids)
        V_init_scaled = append_specific_initialization('wh', nlp_options, V_init_scaled, Outputs_init, Integral_outputs_scaled, model, time_grids)

    V_init_si = struct_op.scaled_to_si(V_init_scaled, model.scaling)

    return V_init_si

def append_specific_initialization(abbreviated_var_name, nlp_options, V_init_scaled, Outputs, Integral_outputs_scaled, model, time_grids):

    n_k = nlp_options['n_k']
    d = nlp_options['collocation']['d']

    wake_nodes = general_tools.get_option_from_possible_dicts(nlp_options, 'wake_nodes', 'vortex')
    rings = general_tools.get_option_from_possible_dicts(nlp_options, 'rings', 'vortex')
    kite_nodes = model.architecture.kite_nodes
    wingtips = ['ext', 'int']

    if abbreviated_var_name == 'wx':
        kite_shed_or_parent_shed_list = kite_nodes
        tip_list = wingtips
        wake_node_or_ring_list = range(wake_nodes)
    elif abbreviated_var_name == 'wg':
        kite_shed_or_parent_shed_list = kite_nodes
        tip_list = [None]
        wake_node_or_ring_list = range(rings)
    elif abbreviated_var_name == 'wh':
        kite_shed_or_parent_shed_list = kite_nodes
        tip_list = [None]
        wake_node_or_ring_list = [None]
    elif abbreviated_var_name == 'wx_center':
        kite_shed_or_parent_shed_list = set([model.architecture.parent_map[kite] for kite in model.architecture.kite_nodes])
        tip_list = [None]
        wake_node_or_ring_list = [None]
    else:
        message = 'get_specific_constraint function is not set up for this abbreviation (' + abbreviated_var_name + ') yet.'
        awelogger.logger.error(message)
        raise Exception(message)

    for kite_shed_or_parent_shed in kite_shed_or_parent_shed_list:
        for tip in tip_list:
            for wake_node_or_ring in wake_node_or_ring_list:

                for ndx in range(n_k):
                    V_init_scaled = get_specific_local_initialization(abbreviated_var_name, nlp_options, V_init_scaled, Outputs,
                                                               Integral_outputs_scaled, model, time_grids,
                                                               kite_shed_or_parent_shed, tip, wake_node_or_ring, ndx)

                    for ddx in range(d):
                        V_init_scaled = get_specific_local_initialization(abbreviated_var_name, nlp_options, V_init_scaled, Outputs,
                                                                       Integral_outputs_scaled, model,
                                                                       time_grids, kite_shed_or_parent_shed, tip,
                                                                       wake_node_or_ring, ndx, ddx)

    return V_init_scaled


def get_specific_local_initialization(abbreviated_var_name, nlp_options, V_init_scaled, Outputs, Integral_outputs_scaled, model, time_grids, kite_shed_or_parent_shed, tip,
                                      wake_node_or_ring, ndx, ddx=None):

    var_name = vortex_tools.get_var_name(abbreviated_var_name, kite_shed_or_parent_shed=kite_shed_or_parent_shed, tip=tip, wake_node_or_ring=wake_node_or_ring)

    if ddx is None:
        var_val_scaled = V_init_scaled['coll_var', ndx - 1, -1, 'xl', var_name]

    else:
        # look-up the actual value from the Outputs. Keep the computing here minimal.
        if abbreviated_var_name == 'wx':
            var_val_si = alg_fixing.get_local_convected_position_value(nlp_options, V_init_scaled, Outputs, model, time_grids, kite_shed_or_parent_shed, tip, wake_node_or_ring, ndx, ddx)
        elif abbreviated_var_name == 'wg':
            var_val_si = alg_fixing.get_local_average_circulation_value(nlp_options, V_init_scaled, Integral_outputs_scaled, model, time_grids, kite_shed_or_parent_shed, wake_node_or_ring, ndx, ddx)
        elif abbreviated_var_name == 'wh':
            var_val_si = alg_fixing.get_local_cylinder_pitch_value(nlp_options, Outputs, kite_shed_or_parent_shed, wake_node_or_ring, ndx, ddx)
        elif abbreviated_var_name == 'wx_center':
            var_val_si = alg_fixing.get_local_cylinder_center_value(nlp_options, Outputs, kite_shed_or_parent_shed, wake_node_or_ring, ndx, ddx)
        else:
            message = 'get_specific_local_constraint function is not set up for this abbreviation (' + abbreviated_var_name + ') yet.'
            awelogger.logger.error(message)
            raise Exception(message)

        var_val_scaled = struct_op.var_si_to_scaled('xl', var_name, var_val_si, model.scaling)

    if ddx is None:
        V_init_scaled['xl', ndx, var_name] = var_val_scaled
    else:
        V_init_scaled['coll_var', ndx, ddx, 'xl', var_name] = var_val_scaled

    return V_init_scaled
