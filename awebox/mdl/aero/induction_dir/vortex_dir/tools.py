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
import awebox.tools.vector_operations as vect_op
from awebox.logger.logger import Logger as awelogger
import awebox.tools.print_operations as print_op
import awebox.tools.struct_operations as struct_op
import awebox.mdl.aero.induction_dir.vortex_dir.element_list as vortex_element_list
import awebox.mdl.aero.induction_dir.vortex_dir.element as vortex_element


def get_wake_node_position_si(options, variables, kite, tip, wake_node, scaling=None):

    vortex_representation = options['induction']['vortex_representation']
    if vortex_representation == 'state':
        var_type = 'xd'
    elif vortex_representation == 'alg':
        var_type = 'xl'
    else:
        message = 'unexpected vortex representation'
        raise Exception(message)

    coord_name = 'wx_' + str(kite) + '_' + tip + '_' + str(wake_node)
    dwx_local = struct_op.get_variable_from_model_or_reconstruction(variables, var_type, coord_name)

    if scaling is not None:
        return struct_op.var_scaled_to_si(var_type, coord_name, dwx_local, scaling)

    return dwx_local

def check_positive_vortex_wake_nodes(options):
    wake_nodes = options['induction']['vortex_wake_nodes']
    if wake_nodes < 0:
        message = 'insufficient wake nodes for creating a filament list: wake_nodes = ' + str(wake_nodes)
        awelogger.logger.error(message)
        raise Exception(message)
    return None

def get_option_from_possible_dicts(options, name):
    if ('induction' in options.keys()) and (name in options['induction'].keys()):
        value = options['induction']['vortex_' + name]
    elif ('aero' in options.keys()) and ('vortex' in options['aero'].keys()) and (name in options['aero']['vortex'].keys()):
        value = options['aero']['vortex'][name]
    elif ('model' in options.keys()) and ('aero' in options['model'].keys()) and ('vortex' in options['model']['aero'].keys()) and (name in options['model']['aero']['vortex'].keys()):
        value = options['model']['aero']['vortex'][name]
    else:
        message = 'no available information about the desired option (' + name + ') found.'
        awelogger.logger.error(message)
        raise Exception(message)

    return value

def get_ring_strength_si(variables, kite, ring, scaling=None):
    coord_name = 'wg_' + str(kite) + '_' + str(ring)
    wg_local = struct_op.get_variable_from_model_or_reconstruction(variables, 'xl', coord_name)

    if scaling is not None:
        return struct_op.var_scaled_to_si('xl', coord_name, wg_local, scaling)

    return wg_local

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

def get_strength_scale(variables_dict, scaling):
    var_type = 'xl'
    for var_name in variables_dict[var_type].labels():
        if 'wg' == var_name[:2]:
            wg_scale = struct_op.var_scaled_to_si(var_type, var_name, 1., scaling)
            return wg_scale

    wg_scale = 1.
    return wg_scale


def append_bounds(g_bounds, fix):

    if (type(fix) == type([])) and fix == []:
        return g_bounds

    else:
        try:
            fix_shape = fix.shape
        except:
            message = 'An attempt to append bounds was passed a vortex-related constraint with an unaccepted structure.'
            awelogger.logger.error(message)
            raise Exception(message)

        g_bounds['ub'].append(cas.DM.zeros(fix_shape))
        g_bounds['lb'].append(cas.DM.zeros(fix_shape))

        return g_bounds


def get_shedding_ndx_and_ddx(options, ndx, ddx=None):

    vortex_representation = options['induction']['vortex_representation']

    n_k = options['n_k']

    wake_nodes = options['induction']['vortex_wake_nodes']
    wake_node = wake_nodes - 1

    if (vortex_representation == 'state'):

        shooting_ndx = n_k - wake_node
        collocation_ndx = shooting_ndx - 1
        modular_ndx = np.mod(collocation_ndx, n_k)

        shedding_ndx = modular_ndx
        shedding_ddx = -1

    elif (vortex_representation == 'alg'):

        if ddx is None:
            ndx_collocation = ndx - 1
            ddx_collocation = -1
        else:
            ndx_collocation = ndx
            ddx_collocation = ddx

        subtracted_ndx = ndx_collocation - wake_node
        shedding_ndx = np.mod(subtracted_ndx, n_k)

        if wake_node == 0:
            shedding_ddx = ddx_collocation
        else:
            shedding_ddx = -1

    else:
        message = 'unknown vortex representation specified: ' + vortex_representation
        awelogger.logger.error(message)
        raise Exception(message)

    return shedding_ndx, shedding_ddx