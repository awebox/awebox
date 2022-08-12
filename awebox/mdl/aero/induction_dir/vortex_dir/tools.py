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
various structural tools for the vortex model
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2019-2021
'''
import pdb

import casadi.tools as cas
import numpy as np
from awebox.logger.logger import Logger as awelogger
import awebox.tools.struct_operations as struct_op



def get_variable_si(variables, var_type, var_name, scaling=None):
    var = struct_op.get_variable_from_model_or_reconstruction(variables, var_type, var_name)
    if scaling is not None:
        return struct_op.var_scaled_to_si(var_type, var_name, var, scaling)
    else:
        return var

def vortices_are_modelled(options):
    comparison_labels = get_option_from_possible_dicts(options, 'comparison_labels')
    any_vor = any(label[:3] == 'vor' for label in comparison_labels)
    return any_vor

def check_positive_vortex_wake_nodes(options):
    wake_nodes = options['induction']['vortex_wake_nodes']
    if wake_nodes < 0:
        message = 'insufficient wake nodes for creating a filament list: wake_nodes = ' + str(wake_nodes)
        awelogger.logger.error(message)
        raise Exception(message)
    return None

def get_option_from_possible_dicts(options, name):

    vortex_name = 'vortex_' + name

    has_induction_toplevel = 'induction' in options.keys()
    name_in_induction_keys = has_induction_toplevel and (name in options['induction'].keys())
    vortex_name_in_induction_keys = has_induction_toplevel and (vortex_name in options['induction'].keys())

    has_aero_toplevel = 'aero' in options.keys()
    has_vortex_midlevel = has_aero_toplevel and ('vortex' in options['aero'].keys())
    name_in_aero_vortex_keys = has_vortex_midlevel and (name in options['aero']['vortex'].keys())

    has_model_toplevel = 'model' in options.keys()
    has_aero_midlevel = has_model_toplevel and ('aero' in options['model'].keys())
    has_vortex_lowlevel = has_aero_midlevel and ('vortex' in options['model']['aero'].keys())
    name_in_model_aero_vortex_keys = has_vortex_lowlevel and (name in options['model']['aero']['vortex'].keys())

    if vortex_name_in_induction_keys:
        value = options['induction'][vortex_name]
    elif name_in_induction_keys:
        value = options['induction'][name]
    elif name_in_aero_vortex_keys:
        value = options['aero']['vortex'][name]
    elif name_in_model_aero_vortex_keys:
        value = options['model']['aero']['vortex'][name]
    else:
        message = 'no available information about the desired option (' + name + ') found.'
        awelogger.logger.error(message)
        raise Exception(message)

    return value

def evaluate_symbolic_on_segments_and_sum(filament_fun, segment_list):

    n_filaments = segment_list.shape[1]
    filament_map = filament_fun.map(n_filaments, 'openmp')
    all = filament_map(segment_list)

    total = cas.sum2(all)

    return total

def get_epsilon(options, parameters):
    c_ref = parameters['theta0','geometry','c_ref']
    epsilon = options['aero']['vortex']['epsilon_to_chord_ratio'] * c_ref
    return epsilon

def get_r_core(options, parameters):

    core_to_chord_ratio = options['aero']['vortex']['core_to_chord_ratio']
    if core_to_chord_ratio == 0.:
        r_core = 0.

    else:
        c_ref = parameters['theta0','geometry','c_ref']
        r_core = core_to_chord_ratio * c_ref

    return r_core

def get_PE_wingtip_name():
    return 'ext'

def get_NE_wingtip_name():
    return 'int'

def get_wingtip_name_and_strength_direction_dict():
    dict = {
        get_NE_wingtip_name(): -1.,
        get_PE_wingtip_name(): +1.
    }
    return dict