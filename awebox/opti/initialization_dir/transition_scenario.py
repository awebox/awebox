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
initialization functions specific to the transition scenario
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

def get_normalized_time_param_dict(ntp_dict, formulation):

    d = ntp_dict['d']

    V_0 = formulation.xi_dict['V_pickle_initial']
    min_dl_t_0_arg = np.argmin(np.array(V_0['coll_var', :, :, 'xd', 'dl_t']))
    n_min_0 = min_dl_t_0_arg / (d + 1)
    d_min_0 = min_dl_t_0_arg - min_dl_t_0_arg / (d + 1) * (d + 1)

    V_f = formulation.xi_dict['V_pickle_terminal']
    min_dl_t_f_arg = np.argmin(np.array(V_f['coll_var', :, :, 'xd', 'dl_t']))
    n_min_f = min_dl_t_f_arg / (d + 1)
    d_min_f = min_dl_t_f_arg - min_dl_t_f_arg / (d + 1) * (d + 1)

    n_min = n_min_0
    d_min = d_min_0

    ntp_dict['n_min'] = n_min
    ntp_dict['d_min'] = d_min
    ntp_dict['n_min_f'] = n_min_f
    ntp_dict['d_min_f'] = d_min_f
    ntp_dict['n_min_0'] = n_min_0
    ntp_dict['d_min_0'] = d_min_0

    return ntp_dict

def set_normalized_time_params(formulation, V_init):
    V_0 = formulation.xi_dict['V_pickle_initial']
    min_dl_t_0_arg = np.argmin(np.array(V_0['coll_var', :, :, 'xd', 'dl_t']))
    xi_0_init = min_dl_t_0_arg / float(np.array(V_0['coll_var', :, :, 'xd', 'dl_t']).size)

    V_f = formulation.xi_dict['V_pickle_terminal']
    min_dl_t_f_arg = np.argmin(np.array(V_f['coll_var', :, :, 'xd', 'dl_t']))
    xi_f_init = min_dl_t_f_arg / float(np.array(V_0['coll_var', :, :, 'xd', 'dl_t']).size)

    V_init['xi', 'xi_0'] = xi_0_init
    V_init['xi', 'xi_f'] = xi_f_init

    return V_init


def guess_final_time(init_options, formulation, ntp_dict):
    tf_guess = 30.
    return tf_guess

def guess_values_at_time(t, init_options, model, formulation, tf_guess, ntp_dict):
    n_min_0 = ntp_dict['n_min_0']
    d_min_0 = ntp_dict['d_min_0']
    n_min_f = ntp_dict['n_min_f']
    d_min_f = ntp_dict['d_min_f']

    l_s = model.variable_bounds['theta']['l_s']['lb'] * init_options['theta']['l_s']
    l_i = model.variable_bounds['theta']['l_i']['lb'] * init_options['theta']['l_i']

    ret = {}
    for name in struct_op.subkeys(model.variables,'xd'):
        ret[name] = 0.0

    parent_map = model.architecture.parent_map
    parent_map[0] = -1
    number_of_nodes = model.architecture.number_of_nodes
    V_0 = formulation.xi_dict['V_pickle_initial']
    V_f = formulation.xi_dict['V_pickle_terminal']
    q_n = {}
    q_0 = {}
    q_f = {}
    phi_0 = {}
    phi_f = {}
    phi_n = {}
    theta_0 = {}
    theta_f = {}
    theta_n = {}
    dphi_n = {}
    dtheta_n = {}
    x_t_0 = {}
    x_t_f = {}
    dq_n = {}

    dl_0 = formulation.xi_dict['V_pickle_initial']['coll_var',n_min_0,d_min_0,'xd','dl_t'] * init_options['xd']['l_t']
    l_0 = formulation.xi_dict['V_pickle_initial']['coll_var',n_min_0,d_min_0,'xd','l_t'] * init_options['xd']['l_t']
    l_f = formulation.xi_dict['V_pickle_terminal']['coll_var',n_min_f,d_min_f,'xd','l_t'] * init_options['xd']['l_t']
    dl_f = formulation.xi_dict['V_pickle_terminal']['coll_var',n_min_f,d_min_f,'xd','dl_t'] * init_options['xd']['l_t']

    c1 = (l_f - l_0 - 0.5*tf_guess*dl_0 - 0.5*tf_guess*dl_f)/(1./6.*tf_guess**3 - 0.25*tf_guess**3)
    c2 = (dl_f - dl_0 - 0.5*tf_guess**2*c1)/tf_guess
    c3 = dl_0
    c4 = l_0
    l_t = 1./6.*t**3*c1 + 0.5*t**2*c2 + t*c3 + c4
    dl_t = 0.5*t**2*c1 + t*c2 + c3
    ddl_t = t**2*c2 + t*c1

    q_0['q0-1'] = cas.DM([0.0,0.0,0.0])
    q_f['q0-1'] = cas.DM([0.0,0.0,0.0])

    for node in range(1,number_of_nodes):
        parent = parent_map[node]
        node_str = 'q' + str(node) + str(parent)
        q_0[node_str] = V_0['coll_var',n_min_0,d_min_0,'xd',node_str]
        q_f[node_str] = V_f['coll_var',n_min_f,d_min_f,'xd',node_str]

    for node in range(1,number_of_nodes):
        parent = parent_map[node]
        grandparent = parent_map[parent]
        tether_str = 'q' + str(node) + str(parent)
        x_t_0[tether_str] = vect_op.normalize(q_0['q' + str(node) + str(parent)] - q_0['q' + str(parent) + str(grandparent)])
        x_t_f[tether_str] = vect_op.normalize(q_f['q' + str(node) + str(parent)] - q_f['q' + str(parent) + str(grandparent)])

    vec_plus = vect_op.normalize(q_0['q31'] - q_0['q21'])
    vec_par = vect_op.normalize(x_t_0['q21']*l_s + 0.5*(q_0['q31'] - q_0['q21']))
    vec_orth = vect_op.normed_cross(vec_par,vec_plus)

    for node in range(2,number_of_nodes):
        x_0 = cas.mtimes(vec_par.T,l_s*x_t_0['q' + str(node) + str(parent)])
        y_0 = cas.mtimes(vec_plus.T,l_s*x_t_0['q' + str(node) + str(parent)])
        z_0 = cas.mtimes(vec_orth.T,l_s*x_t_0['q' + str(node) + str(parent)])
        theta_0['q' + str(node) + str(parent)] = np.arcsin(z_0/l_s)
        phi_0['q' + str(node) + str(parent)] = np.arctan2(y_0,x_0)
        x_f = cas.mtimes(vec_par.T,l_s*x_t_f['q' + str(node) + str(parent)])
        y_f = cas.mtimes(vec_plus.T,l_s*x_t_f['q' + str(node) + str(parent)])
        z_f = cas.mtimes(vec_orth.T,l_s*x_t_f['q' + str(node) + str(parent)])
        theta_f['q' + str(node) + str(parent)] = np.arcsin(z_f / l_s)
        phi_f['q' + str(node) + str(parent)] = np.arctan2(y_f,x_f)

    for node in range(2, number_of_nodes):
        parent = parent_map[node]
        phi_n['q' + str(node) + str(parent)] = phi_0['q' + str(node) + str(parent)] + t/tf_guess*(phi_f['q' + str(node) + str(parent)] - phi_0['q' + str(node) + str(parent)])
        dphi_n['q' + str(node) + str(parent)] = 1./tf_guess*(phi_f['q' + str(node) + str(parent)] - phi_0['q' + str(node) + str(parent)])
        theta_n['q' + str(node) + str(parent)] = theta_0['q' + str(node) + str(parent)] + t/tf_guess * (theta_f['q' + str(node) + str(parent)] - theta_0['q' + str(node) + str(parent)])
        dtheta_n['q' + str(node) + str(parent)] = 1./tf_guess * (theta_f['q' + str(node) + str(parent)] - theta_0['q' + str(node) + str(parent)])

    a = cas.mtimes((q_f['q10'] - q_0['q10']).T,(q_f['q10'] - q_0['q10']))
    b = 2*cas.mtimes(q_0['q10'].T,(q_f['q10'] - q_0['q10']))
    c = cas.mtimes(q_0['q10'].T,q_0['q10']) - l_t**2
    dc = -2*l_t*dl_t
    D = b**2 - 4*a*c
    dD = -4*a*dc
    x1 = (-b + np.sqrt(D))/(2*a)
    x2 = (-b - np.sqrt(D))/(2*a)
    s = x2
    ds = - 1./(2*a) * 1./(2*np.sqrt(D)) * dD
    e_t = 1./l_t*(q_0['q10'] + s*(q_f['q10'] - q_0['q10']))
    de_t = 1./l_t*(ds*(q_f['q10'] - q_0['q10'])) - 1./l_t**2 * dl_t * (q_0['q10'] + s*(q_f['q10'] - q_0['q10']))
    q_n['q10'] = l_t*e_t
    dq_n['q10'] = dl_t*e_t + l_t*de_t

    for node in range(2, number_of_nodes):
        parent = parent_map[node]
        nstr = 'q' + str(node) + str(parent)
        q_n[nstr] = q_n['q10'] + (np.sin(phi_n[nstr])*np.cos(theta_n[nstr])*vec_plus + np.cos(phi_n[nstr])*np.cos(theta_n[nstr])*vec_par + np.sin(theta_n[nstr])*vec_orth)*l_s
        dq_n[nstr] = dq_n['q10'] + (- np.cos(phi_n[nstr])*np.sin(theta_n[nstr])*dtheta_n[nstr]*dphi_n[nstr]*vec_plus + np.sin(phi_n[nstr])*np.sin(theta_n[nstr])*dtheta_n[nstr]*dphi_n[nstr]*vec_par + np.cos(theta_n[nstr])*dtheta_n[nstr]*vec_orth)*l_s

    for node in range(1, number_of_nodes):
        parent = parent_map[node]
        ret['q' + str(node) + str(parent)] = q_n['q' + str(node) + str(parent)]
        ret['dq' + str(node) + str(parent)] = dq_n['q' + str(node) + str(parent)]

    ret['l_t'] = l_t
    ret['dl_t'] = dl_t

    return ret
