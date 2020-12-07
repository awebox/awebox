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
initialization functions related to the landing scenarios, shared between nominal and compromised scenarios
_python _version 2.7 / casadi-3.4.5
- _author: rachel leuthold, jochem de schutter, thilo bronnenmeyer (alu-fr, 2017 - 20)
'''

import numpy as np
import casadi.tools as cas
import awebox.tools.vector_operations as vect_op
from awebox.logger.logger import Logger as awelogger
import awebox.tools.struct_operations as struct_op
import awebox.opti.initialization_dir.induction as induction_init
import awebox.opti.initialization_dir.tools as tools_init

def guess_final_time(initialization_options, formulation, ntp_dict):
    n_min = ntp_dict['n_min']
    d_min = ntp_dict['d_min']
    l_0 = formulation.xi_dict['V_pickle_initial']['coll_var', n_min, d_min, 'xd', 'l_t'] * \
          initialization_options['xd']['l_t']
    tf_guess = l_0 / initialization_options['landing_velocity']
    return tf_guess

def guess_values_at_time(t, init_options, model, formulation, tf_guess, ntp_dict):
    n_min = ntp_dict['n_min']
    d_min = ntp_dict['d_min']

    l_s = model.variable_bounds['theta']['l_s']['lb'] * init_options['theta']['l_s']
    l_i = model.variable_bounds['theta']['l_i']['lb'] * init_options['theta']['l_i']

    ret = {}
    for name in struct_op.subkeys(model.variables, 'xd'):
        ret[name] = 0.0

    kite_nodes = model.architecture.kite_nodes
    parent_map = model.architecture.parent_map
    parent_map[0] = -1
    number_of_nodes = model.architecture.number_of_nodes
    V_0 = formulation.xi_dict['V_pickle_initial']
    q_n = {}
    q_0 = {}
    x_t = {}
    dq_n = {}

    q_0['q0-1'] = cas.DM([0.0, 0.0, 0.0])
    for node in range(1, number_of_nodes):
        parent = parent_map[node]
        node_str = 'q' + str(node) + str(parent)
        q_0[node_str] = V_0['coll_var', n_min, d_min, 'xd', node_str]

    for node in range(1, number_of_nodes):
        parent = parent_map[node]
        grandparent = parent_map[parent]
        tether_str = 't' + str(node) + str(parent)
        x_t[tether_str] = vect_op.normalize(
            q_0['q' + str(node) + str(parent)] - q_0['q' + str(parent) + str(grandparent)])

    dl_0 = formulation.xi_dict['V_pickle_initial']['coll_var', n_min, d_min, 'xd', 'dl_t'] * init_options['xd']['l_t']
    l_0 = formulation.xi_dict['V_pickle_initial']['coll_var', n_min, d_min, 'xd', 'l_t'] * init_options['xd']['l_t']
    l_f = model.variable_bounds['xd']['q10']['lb'][2] / x_t['t10'][2]
    dl_f = 0

    c1 = (l_f - l_0 - 0.5 * tf_guess * dl_0 - 0.5 * tf_guess * dl_f) / (1. / 6. * tf_guess ** 3 - 0.25 * tf_guess ** 3)
    c2 = (dl_f - dl_0 - 0.5 * tf_guess ** 2 * c1) / tf_guess
    c3 = dl_0
    c4 = l_0
    l_t = 1. / 6. * t ** 3 * c1 + 0.5 * t ** 2 * c2 + t * c3 + c4
    dl_t = 0.5 * t ** 2 * c1 + t * c2 + c3
    ddl_t = t ** 2 * c2 + t * c1

    q_n['q0-1'] = cas.DM([0.0, 0.0, 0.0])
    for node in range(1, number_of_nodes):
        parent = parent_map[node]
        grandparent = parent_map[parent]
        if node == 1:
            seg_length = l_t
        elif node in kite_nodes:
            seg_length = l_s
        else:
            seg_length = l_i
        q_n['q' + str(node) + str(parent)] = x_t['t' + str(node) + str(parent)] * seg_length + q_n[
            'q' + str(parent) + str(grandparent)]
        if node == 1:
            dq_n['dq' + str(node) + str(parent)] = dl_t * x_t['t' + str(node) + str(parent)]
        else:
            dq_n['dq' + str(node) + str(parent)] = dq_n['dq10']

    for node in range(1, number_of_nodes):
        parent = parent_map[node]
        ret['q' + str(node) + str(parent)] = q_n['q' + str(node) + str(parent)]
        ret['dq' + str(node) + str(parent)] = dq_n['dq' + str(node) + str(parent)]

    ret['l_t'] = l_t
    ret['dl_t'] = dl_t
    # ret['ddl_t'] = ddl_t

    return ret



######################### compromised landing

def get_compromised_landing_normalized_time_param_dict(ntp_dict, formulation):
    xi_0_init = formulation.xi_dict['xi_bounds']['xi_0'][0]
    nk_xi = len(formulation.xi_dict['V_pickle_initial']['coll_var', :, :, 'xd'])
    d_xi = len(formulation.xi_dict['V_pickle_initial']['coll_var', 0, :, 'xd'])
    n_min = int(xi_0_init * nk_xi)
    d_min = int((xi_0_init * nk_xi - int(xi_0_init * nk_xi)) * (d_xi))

    ntp_dict['n_min'] = n_min
    ntp_dict['d_min'] = d_min


def set_compromised_landing_normalized_time_params(formulation, V_init):
    xi_0_init = formulation.xi_dict['xi_bounds']['xi_0'][0]
    xi_f_init = 0.0
    V_init['xi', 'xi_0'] = xi_0_init
    V_init['xi', 'xi_f'] = xi_f_init

    return V_init

######################### nominal landing

def get_nominal_landing_normalized_time_param_dict(ntp_dict, formulation):
    V_0 = formulation.xi_dict['V_pickle_initial']
    min_dl_t_arg = np.argmin(np.array(V_0['coll_var', :, :, 'xd', 'dl_t']))
    n0 = len(V_0['coll_var', :, :, 'xd'])
    d0 = len(V_0['coll_var', 0, :, 'xd']) - 1
    n_min = min_dl_t_arg / (d0 + 1)
    d_min = min_dl_t_arg - min_dl_t_arg / (d0 + 1) * (d0 + 1)

    ntp_dict['n0'] = n0
    ntp_dict['n_min'] = n_min
    ntp_dict['d_min'] = d_min

    return ntp_dict

def set_nominal_landing_normalized_time_params(formulation, V_init):
    V_0 = formulation.xi_dict['V_pickle_initial']
    min_dl_t_arg = np.argmin(np.array(V_0['coll_var', :, :, 'xd', 'dl_t']))
    xi_0_init = min_dl_t_arg / float(np.array(V_0['coll_var', :, :, 'xd', 'dl_t']).size)
    xi_f_init = 0.0
    V_init['xi', 'xi_0'] = xi_0_init
    V_init['xi', 'xi_f'] = xi_f_init

    return V_init
