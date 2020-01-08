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
simple initialization intended for awe systems
initializes to a simple uniform circle path for kites, and constant location for tether nodes
no reel-in or out, as inteded for base of tracking problem
_python _version 2.7 / casadi-3.4.5
- _author: rachel leuthold, jochem de schutter, thilo bronnenmeyer (alu-fr, 2017 - 19)
'''

import numpy as np
import casadi.tools as cas
import awebox.tools.vector_operations as vect_op
from awebox.logger.logger import Logger as awelogger
import awebox.tools.struct_operations as struct_op

import pdb

def get_initial_guess(nlp, model, formulation, options):
    V_init_si = build_si_initial_guess(nlp, model, formulation, options)
    if True in np.isnan(np.array(V_init_si.cat)):
        raise ValueError('NaN detected in V_init_si')
    V_init = struct_op.si_to_scaled(model, V_init_si)
    return V_init

def build_si_initial_guess(nlp, model, formulation, options):

    initialization_options = options

    awelogger.logger.info('build si initial guess...')

    V = nlp.V
    V_init = V(0.0)

    # set lagrange multipliers different from zero to avoid singularity
    if 'xa' in list(V_init.keys()):
        V_init['xa', :] = 1.
    if 'coll_var' in list(V_init.keys()):
        V_init['coll_var',:,:,'xa'] = 1.

    V_init, n_d_list = set_xi_initial_values(nlp, formulation, initialization_options, V_init)
    n_min = n_d_list[1]
    d_min = n_d_list[2]

    tf_init = guess_tf(initialization_options, model, formulation, n_min, d_min)
    if V_init['theta','t_f'].shape[0] > 1:
        tf_init = cas.vertcat(tf_init, tf_init)

    V_init = extract_time_grid(model, nlp, formulation, initialization_options, V_init, tf_init, n_d_list)

    V_init = initial_guess_induction(options, nlp, formulation, model, V_init)

    # specified initial values for system parameters
    for name in set(struct_op.subkeys(model.variables, 'theta')) - set(['t_f']):
        if name in list(initialization_options['theta'].keys()):
            V_init['theta', name] = initialization_options['theta'][name]
        elif name[:3] == 'l_c':
            layer = int(name[3:])
            kites = model.architecture.kites_map[layer]
            q_first = V_init['xd',0,'q{}{}'.format(kites[0],model.architecture.parent_map[kites[0]])]
            q_second = V_init['xd',0,'q{}{}'.format(kites[1],model.architecture.parent_map[kites[1]])]
            V_init['theta', name] = np.linalg.norm(q_first - q_second)
            if options['cross_tether_attachment'] == 'wing_tip':
                V_init['theta', name] += - options['sys_params_num']['geometry']['b_ref']
        elif name[:6] == 'diam_c':
            V_init['theta', name] = initialization_options['theta']['diam_c']
        else:
            raise ValueError("please specify an initial value for variable '" + name + "' of type 'theta'")

    # initial time guess (same for both intervals in case of phase fixing)
    V_init['theta', 't_f'] = tf_init

    # initial values for homotopy parameters
    for name in list(model.parameters_dict['phi'].keys()):
        V_init['phi', name] = 1.

    return V_init


def set_xi_initial_values(nlp, formulation, initialization_options, V_init):

    d = nlp.d

    n0 = -999
    n_min = -999
    d_min = -999
    n_min_f = -999
    d_min_f = -999
    n_min_0 = -999
    d_min_0 = -999

    # set xi initial values
    if initialization_options['type'] in ['nominal_landing']:
        V_0 = formulation.xi_dict['V_pickle_initial']
        min_dl_t_arg = np.argmin(np.array(V_0['coll_var',:,:,'xd','dl_t']))
        n0 = len(V_0['coll_var',:,:,'xd'])
        d0 = len(V_0['coll_var',0,:,'xd']) - 1
        n_min = min_dl_t_arg/(d0 + 1)
        d_min = min_dl_t_arg - min_dl_t_arg/(d0 + 1) * (d0 + 1)
        xi_0_init = min_dl_t_arg/float(np.array(V_0['coll_var',:,:,'xd','dl_t']).size)
        xi_f_init = 0.0
    elif initialization_options['type'] in ['compromised_landing']:
        xi_0_init = formulation.xi_dict['xi_bounds']['xi_0'][0]
        nk_xi = len(formulation.xi_dict['V_pickle_initial']['coll_var',:,:,'xd'])
        d_xi = len(formulation.xi_dict['V_pickle_initial']['coll_var',0,:,'xd'])
        n_min = int(xi_0_init*nk_xi)
        d_min = int((xi_0_init*nk_xi - int(xi_0_init*nk_xi))*(d_xi))
        xi_f_init = 0.0
    elif initialization_options['type'] in ['transition']:
        V_0 = formulation.xi_dict['V_pickle_initial']
        min_dl_t_0_arg = np.argmin(np.array(V_0['coll_var', :, :,'xd', 'dl_t']))
        n_min_0 = min_dl_t_0_arg / (d + 1)
        d_min_0 = min_dl_t_0_arg - min_dl_t_0_arg / (d + 1) * (d + 1)
        xi_0_init = min_dl_t_0_arg / float(np.array(V_0['coll_var', :, :, 'xd','dl_t']).size)
        V_f = formulation.xi_dict['V_pickle_terminal']
        min_dl_t_f_arg = np.argmin(np.array(V_f['coll_var', :, :, 'xd', 'dl_t']))
        n_min_f = min_dl_t_f_arg / (d + 1)
        d_min_f = min_dl_t_f_arg - min_dl_t_f_arg / (d + 1) * (d + 1)
        xi_f_init = min_dl_t_f_arg / float(np.array(V_0['coll_var', :, :, 'xd', 'dl_t']).size)
        n_min = n_min_0
        d_min = d_min_0
    else:
        xi_0_init = 0.0
        xi_f_init = 0.0
        n_min = 0
        d_min = 0
    V_init['xi', 'xi_0'] = xi_0_init
    V_init['xi', 'xi_f'] = xi_f_init

    n_d_list = [n0, n_min, d_min, n_min_f, d_min_f, n_min_0, d_min_0]

    return V_init, n_d_list

def extract_time_grid(model, nlp, formulation, initialization_options, V_init, tf_init, n_d_list):

    n0 = n_d_list[0]
    n_min = n_d_list[1]
    d_min = n_d_list[2]
    n_min_f = n_d_list[3]
    d_min_f = n_d_list[4]
    n_min_0 = n_d_list[5]
    d_min_0 = n_d_list[6]

    # extract time grid
    tgrid_xd = nlp.time_grids['x'](tf_init)
    if 'coll' in list(nlp.time_grids.keys()):
        tgrid_coll = nlp.time_grids['coll'](tf_init)

    d = nlp.d
    n_k = nlp.n_k

    for k in range(n_k+1):

        t = tgrid_xd[k]

        # initialize kite(s) on trajectory
        if initialization_options['type'] in ['nominal_landing','compromised_landing']:
            guess = initial_guess_landing(t, initialization_options, model, formulation, tf_init, n_min, d_min)

        elif initialization_options['type'] in ['transition']:
            guess = initial_guess_transition(t, initialization_options, model, formulation, tf_init, n_min_0, d_min_0, n_min_f, d_min_f)

        elif initialization_options['type'] == 'aero_test':
            guess = initial_guess_aero_test(t, initialization_options, model, formulation)

        else:
            guess = initial_guess(t, initialization_options, nlp, model, formulation)

        for name in struct_op.subkeys(model.variables, 'xd'):
            V_init['xd', k, name] = guess[name]

        if nlp.discretization == 'direct_collocation' and (k < n_k):
            for j in range(d):
                t = tgrid_coll[k,j]

                # initialize kite(s) on trajectory
                if initialization_options['type'] in ['nominal_landing','compromised_landing']:
                    guess = initial_guess_landing(t, initialization_options, model, formulation, tf_init, n_min, d_min)

                elif initialization_options['type'] in ['transition']:
                    guess = initial_guess_transition(t, initialization_options, model, formulation, tf_init, n_min_0, d_min_0, n_min_f, d_min_f)

                elif initialization_options['type'] == 'aero_test':
                    guess = initial_guess_aero_test(t, initialization_options, model, formulation)

                else:
                    guess = initial_guess(t, initialization_options, nlp, model, formulation)

                for name in struct_op.subkeys(model.variables, 'xd'):
                    V_init['coll_var', k, j, 'xd', name] = guess[name]

    return V_init


def guess_tf(initialization_options, model, formulation, n_min = None, d_min = None):

    if initialization_options['type'] in ['nominal_landing','compromised_landing']:
        tf_guess = guess_tf_nominal_or_compromised(initialization_options, formulation, n_min, d_min)

    elif initialization_options['type'] == 'transition':
        tf_guess = guess_tf_transition()

    elif initialization_options['type'] == 'aero_test':
        _, tf_guess = guess_radius_and_tf_aero_test(initialization_options)

    else:
        _, tf_guess = guess_radius_and_tf_standard(initialization_options, model)

    return tf_guess


def guess_tf_transition():
    tf_guess = 30.
    return tf_guess


def guess_tf_nominal_or_compromised(initialization_options, formulation, n_min, d_min):
    l_0 = formulation.xi_dict['V_pickle_initial']['coll_var', n_min, d_min, 'xd', 'l_t'] * initialization_options['xd']['l_t']
    tf_guess = l_0 / initialization_options['landing_velocity']
    return tf_guess


def guess_radius_and_tf_aero_test(initialization_options):
    radius = 999.

    if initialization_options['aero_test']['omega'] > 0:
        tf_guess = initialization_options['aero_test']['total_periods'] * (2. * np.pi / initialization_options['aero_test']['omega'])
    else:
        tf_guess = initialization_options['aero_test']['total_periods']

    return radius, tf_guess


def guess_radius_and_tf_standard(initialization_options, model):

    tether_length, max_cone_angle = get_hypotenuse_and_max_cone_angle(model, initialization_options)

    windings = initialization_options['windings']
    winding_period = initialization_options['winding_period']
    ua_norm = initialization_options['ua_norm']

    winding_period = clip_winding_period(initialization_options, winding_period)
    tf_guess = windings * winding_period

    total_distance = ua_norm * tf_guess
    circumference = total_distance / windings
    radius = circumference / 2. / np.pi

    radius = clip_radius(initialization_options, max_cone_angle, tether_length, radius)

    total_distance = 2. * np.pi * radius * windings
    tf_guess = total_distance / ua_norm

    return radius, tf_guess


def clip_winding_period(initialization_options, winding_period):
    # acc = omega * ua = 2 pi ua / winding_period < hardware_limit
    acc_max = initialization_options['acc_max']
    ua_norm = initialization_options['ua_norm']

    winding_period_min = 2. * np.pi * ua_norm / acc_max
    if winding_period < winding_period_min:
        winding_period = winding_period_min
    else:
        winding_period = winding_period

    return winding_period

def clip_radius(initialization_options, max_cone_angle, tether_length, radius):
    b_ref = initialization_options['sys_params_num']['geometry']['b_ref']
    min_radius = initialization_options['min_rel_radius'] * b_ref
    max_radius = np.sin(max_cone_angle * np.pi / 180.) * tether_length

    if radius < min_radius:
        radius = min_radius

    if radius > max_radius:
        radius = max_radius

    return radius

def get_hypotenuse_and_max_cone_angle(model, initialization_options):
    max_cone_angle_multi = initialization_options['max_cone_angle_multi']
    max_cone_angle_single = initialization_options['max_cone_angle_single']

    number_kites = model.architecture.number_of_kites
    if number_kites == 1:
        tether_length = initialization_options['xd']['l_t']
        max_cone_angle = max_cone_angle_single
    else:
        tether_length = initialization_options['theta']['l_s']
        max_cone_angle = max_cone_angle_multi

    return tether_length, max_cone_angle






def get_cone_height_and_radius(options, model, l_t):

    # get radius and height of the cones in use
    # two cone types specified, based on main tether (single kite option) and secondary tether (multi-kite option)
    # radius is dependent on flight velocity
    # height is a dependent
    hypotenuse_list = cas.vertcat(l_t, options['theta']['l_s'])
    [radius, _] = guess_radius_and_tf_standard(options, model)

    height_list = []
    for hdx in range(hypotenuse_list.shape[0]):
        hypotenuse = hypotenuse_list[hdx]
        height = (hypotenuse**2. - radius**2.)**0.5
        height_list = cas.vertcat(height_list, height)

    return height_list, radius



def get_tether_node_position(options, parent_position, node, l_t):

    ehat_tether = get_ehat_tether(options)

    seg_length = options['theta']['l_i']
    if node == 1:
        seg_length = l_t

    position = parent_position + seg_length * ehat_tether

    return position

def get_azimuthal_angle(t, level_siblings, node, parent, omega_norm):

    number_of_siblings = len(level_siblings[parent])
    if number_of_siblings == 1:
        psi0 = 0.
    else:
        idx = level_siblings[parent].index(node)
        psi0 = np.float(idx) / np.float(number_of_siblings) * 2. * np.pi

    psi = psi0 + omega_norm * t

    return psi




def initial_guess_transition(t, initialization_options, model, formulation, t_f, n_min_0, d_min_0, n_min_f, d_min_f):

    l_s = model.variable_bounds['theta']['l_s']['lb'] * initialization_options['theta']['l_s']
    l_i = model.variable_bounds['theta']['l_i']['lb'] * initialization_options['theta']['l_i']

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

    dl_0 = formulation.xi_dict['V_pickle_initial']['coll_var',n_min_0,d_min_0,'xd','dl_t'] * initialization_options['xd']['l_t']
    l_0 = formulation.xi_dict['V_pickle_initial']['coll_var',n_min_0,d_min_0,'xd','l_t'] * initialization_options['xd']['l_t']
    l_f = formulation.xi_dict['V_pickle_terminal']['coll_var',n_min_f,d_min_f,'xd','l_t'] * initialization_options['xd']['l_t']
    dl_f = formulation.xi_dict['V_pickle_terminal']['coll_var',n_min_f,d_min_f,'xd','dl_t'] * initialization_options['xd']['l_t']

    c1 = (l_f - l_0 - 0.5*t_f*dl_0 - 0.5*t_f*dl_f)/(1./6.*t_f**3 - 0.25*t_f**3)
    c2 = (dl_f - dl_0 - 0.5*t_f**2*c1)/t_f
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
        phi_n['q' + str(node) + str(parent)] = phi_0['q' + str(node) + str(parent)] + t/t_f*(phi_f['q' + str(node) + str(parent)] - phi_0['q' + str(node) + str(parent)])
        dphi_n['q' + str(node) + str(parent)] = 1./t_f*(phi_f['q' + str(node) + str(parent)] - phi_0['q' + str(node) + str(parent)])
        theta_n['q' + str(node) + str(parent)] = theta_0['q' + str(node) + str(parent)] + t/t_f * (theta_f['q' + str(node) + str(parent)] - theta_0['q' + str(node) + str(parent)])
        dtheta_n['q' + str(node) + str(parent)] = 1./t_f * (theta_f['q' + str(node) + str(parent)] - theta_0['q' + str(node) + str(parent)])

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

def initial_guess_landing(t, initialization_options, model, formulation, t_f, n_min, d_min):

    l_s = model.variable_bounds['theta']['l_s']['lb'] * initialization_options['theta']['l_s']
    l_i = model.variable_bounds['theta']['l_i']['lb'] * initialization_options['theta']['l_i']

    ret = {}
    for name in struct_op.subkeys(model.variables,'xd'):
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

    q_0['q0-1'] = cas.DM([0.0,0.0,0.0])
    for node in range(1,number_of_nodes):
        parent = parent_map[node]
        node_str = 'q' + str(node) + str(parent)
        q_0[node_str] = V_0['coll_var',n_min,d_min,'xd',node_str]

    for node in range(1,number_of_nodes):
        parent = parent_map[node]
        grandparent = parent_map[parent]
        tether_str = 't' + str(node) + str(parent)
        x_t[tether_str] = vect_op.normalize(q_0['q' + str(node) + str(parent)] - q_0['q' + str(parent) + str(grandparent)])

    dl_0 = formulation.xi_dict['V_pickle_initial']['coll_var',n_min,d_min,'xd','dl_t'] * initialization_options['xd']['l_t']
    l_0 = formulation.xi_dict['V_pickle_initial']['coll_var',n_min,d_min,'xd','l_t'] * initialization_options['xd']['l_t']
    l_f = model.variable_bounds['xd']['q10']['lb'][2]/x_t['t10'][2]
    dl_f = 0

    c1 = (l_f - l_0 - 0.5*t_f*dl_0 - 0.5*t_f*dl_f)/(1./6.*t_f**3 - 0.25*t_f**3)
    c2 = (dl_f - dl_0 - 0.5*t_f**2*c1)/t_f
    c3 = dl_0
    c4 = l_0
    l_t = 1./6.*t**3*c1 + 0.5*t**2*c2 + t*c3 + c4
    dl_t = 0.5*t**2*c1 + t*c2 + c3
    ddl_t = t**2*c2 + t*c1

    q_n['q0-1'] = cas.DM([0.0,0.0,0.0])
    for node in range(1, number_of_nodes):
        parent = parent_map[node]
        grandparent = parent_map[parent]
        if node == 1:
            seg_length = l_t
        elif node in kite_nodes:
            seg_length = l_s
        else:
            seg_length = l_i
        q_n['q' + str(node) + str(parent)] = x_t['t' + str(node) + str(parent)] * seg_length + q_n['q' + str(parent) + str(grandparent)]
        if node == 1:
            dq_n['dq' + str(node) + str(parent)] = dl_t*x_t['t' + str(node) + str(parent)]
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

def initial_guess_aero_test(t, initialization_options, model, formulation):
    ret = {}
    for name in struct_op.subkeys(model.variables, 'xd'):
        ret[name] = 0.0
    ret['e'] = 0.0

    l_t, dl_t, _, _, _, _ = get_aero_test_values(t, initialization_options)

    ret['l_t'] = l_t
    ret['dl_t'] = dl_t

    ret = initial_node_variables_for_aero_test_test(t, initialization_options, model, formulation, ret)

    return ret

def initial_guess(t, options, nlp, model, formulation):
    ret = {}
    for name in struct_op.subkeys(model.variables, 'xd'):
        ret[name] = 0.0
    ret['e'] = 0.0

    ret['l_t'] = options['xd']['l_t']
    ret['dl_t'] = 0.0

    ret = initial_node_variables_for_standard_path(t, options, model, formulation, ret)

    return ret


def initial_node_variables_for_aero_test_test(t, options, model, formulation, ret={}):

    kite_nodes = model.architecture.kite_nodes

    if len(kite_nodes) > 1:
        awelogger.logger.error('ERROR: pitch-plunge test only defined (to date) for one wing')

    _, _, q10, dq10, r_dcm, omega = get_aero_test_values(t, options)

    ret['q10'] = q10
    ret['dq10'] = dq10

    ret['r10'] = r_dcm
    ret['omega10'] = omega

    return ret

def get_ehat_radial(t, options, model, kite, ret={}):
    parent_map = model.architecture.parent_map
    level_siblings = model.architecture.get_all_level_siblings()

    ua_norm = options['ua_norm']

    if ret == {}:
        l_t = options['xd']['l_t']
    else:
        l_t = ret['l_t']

    height_list, radius = get_cone_height_and_radius(options, model, l_t)

    parent = parent_map[kite]

    omega_norm = ua_norm / radius
    psi = get_azimuthal_angle(t, level_siblings, kite, parent, omega_norm)

    _, y_rot_hat, z_rot_hat = get_rotor_reference_frame(options)

    cospsi_var = np.cos(psi)
    sinpsi_var = np.sin(psi)

    # for positive yaw(turns around +zhat, normal towards +yhat):
    #     rhat = zhat * cos(psi) - yhat * sin(psi)
    ehat_radial = z_rot_hat * cospsi_var - y_rot_hat * sinpsi_var

    return ehat_radial


def initial_node_variables_for_standard_path(t, options, model, formulation, ret={}):

    number_of_nodes = model.architecture.number_of_nodes
    parent_map = model.architecture.parent_map
    kite_nodes = model.architecture.kite_nodes

    ua_norm = options['ua_norm']
    kite_dof = model.kite_dof

    height_list, radius = get_cone_height_and_radius(options, model, ret['l_t'])

    for node in range(1, number_of_nodes):

        parent = parent_map[node]
        if parent == 0:
            parent_position = np.zeros((3, 1))
        else:
            grandparent = parent_map[parent]
            parent_position = ret['q' + str(parent) + str(grandparent)]

        if not node in kite_nodes:
            ret['q' + str(node) + str(parent)] = get_tether_node_position(options, parent_position, node, ret['l_t'])
            ret['dq' + str(node) + str(parent)] = np.zeros((3, 1))

        else:
            if parent == 0:
                height = height_list[0]
            else:
                height = height_list[1]

            omega_norm = ua_norm / radius

            n_rot_hat, y_rot_hat, z_rot_hat = get_rotor_reference_frame(options)

            ehat_normal = n_rot_hat
            ehat_radial = get_ehat_radial(t, options, model, node, ret)
            ehat_tangential = vect_op.normed_cross(ehat_normal, ehat_radial)

            omega_vector = ehat_normal * omega_norm

            tether_vector = ehat_radial * radius + ehat_normal * height
            position = parent_position + tether_vector

            velocity = ua_norm * ehat_tangential

            ehat1 = -1. * ehat_tangential
            ehat3 = n_rot_hat
            ehat2 = vect_op.normed_cross(ehat3, ehat1)

            dcm = cas.horzcat(ehat1, ehat2, ehat3)
            if options['cross_tether']:
                if options['cross_tether_attachment'] in ['com','stick']:
                    dcm = get_cross_tether_dcm(options, dcm)
            dcm_column = cas.reshape(dcm, (9, 1))

            ret['q' + str(node) + str(parent)] = position
            ret['dq' + str(node) + str(parent)] = velocity

            if int(kite_dof) == 6:
                ret['omega' + str(node) + str(parent)] = omega_vector
                ret['r' + str(node) + str(parent)] = dcm_column

    return ret

def get_cross_tether_dcm(options, dcm):
    ang = -options['rotation_bounds'] * 1.05
    rotx = np.array([[1, 0, 0], [0, np.cos(ang), -np.sin(ang)], [0, np.sin(ang), np.cos(ang)]])
    dcm = cas.mtimes(dcm, rotx)
    return dcm


def get_ehat_tether(options):
    inclination = options['incid_deg'] * np.pi / 180.
    ehat_tether = np.cos(inclination) * vect_op.xhat() + np.sin(inclination) * vect_op.zhat()
    return ehat_tether

def get_rotor_reference_frame(initialization_options):
    n_rot_hat = get_ehat_tether(initialization_options)
    u_hat = vect_op.xhat_np()
    z_rot_hat = vect_op.normed_cross(u_hat, n_rot_hat)
    y_rot_hat = vect_op.normed_cross(z_rot_hat, n_rot_hat)
    return n_rot_hat, y_rot_hat, z_rot_hat

def get_local_wind_reference_frame(initialization_options):
    u_hat = vect_op.xhat_np()
    n_rot_hat = get_ehat_tether(initialization_options)
    w_hat = vect_op.normed_cross(u_hat, n_rot_hat)
    v_hat = vect_op.normed_cross(w_hat, u_hat)
    return u_hat, v_hat, w_hat

def get_gamma_angle(initialization_options):
    n_rot_hat = get_ehat_tether(initialization_options)
    u_hat = vect_op.xhat_np()
    gamma = vect_op.angle_between(u_hat, n_rot_hat)
    return gamma

def get_chi_angle(options, var_type):
    a = options['xd']['a']
    gamma = get_gamma_angle(options)
    chi = (0.6 * a + 1.) * gamma
    return chi


def get_aero_test_values(t, options):

    h_0 = options['aero_test']['h_0']
    phi_0 = options['aero_test']['phi_0']
    omega = options['aero_test']['omega']

    h = h_0 * np.sin(omega * t)
    dh = h_0 * np.cos(omega * t) * omega
    dh = -1. * h_0 * np.sin(omega * t) * omega**2.

    phi = phi_0 * np.cos(omega * t)
    dphi = -1. * phi_0 * np.sin(omega * t) * omega
    ddphi = -1. * phi_0 * np.cos(omega * t) * omega**2.

    l_t = h + 3.
    dl_t = dh

    q10 = l_t * vect_op.zhat_np()
    dq10 = dl_t * vect_op.zhat_np()

    ehat1 = np.cos(phi) * vect_op.xhat_np() - np.sin(phi) * vect_op.zhat_np()
    ehat2 = vect_op.yhat_np()
    ehat3 = np.cos(phi) * vect_op.zhat_np() + np.sin(phi) * vect_op.xhat_np()

    r_dcm = cas.horzcat(ehat1, ehat2, ehat3)
    r_dcm = cas.reshape(r_dcm, (9, 1))

    omega = dphi * ehat2

    return l_t, dl_t, q10, dq10, r_dcm, omega

def initial_guess_induction(options, nlp, formulation, model, V_init):

    induction_model = options['model']['induction_model']
    if induction_model == 'actuator':
        V_init = initial_guess_actuator(options, nlp, formulation, model, V_init)

    if induction_model == 'vortex':
        V_init = initial_guess_vortex(options, nlp, formulation, model, V_init)

    return V_init

def initial_guess_vortex(options, nlp, formulation, model, V_init):

    tf_init = guess_tf(options, model, formulation)
    tgrid_xd = nlp.time_grids['x'](tf_init)
    if 'coll' in list(nlp.time_grids.keys()):
        tgrid_coll = nlp.time_grids['coll'](tf_init)

    U_ref = options['sys_params_num']['wind']['u_ref'] * vect_op.xhat_np()
    b_ref = options['sys_params_num']['geometry']['b_ref']
    n_k = nlp.n_k
    d = nlp.d
    n_nodes = n_k * d
    dims = ['x', 'y', 'z']
    tips = ['ext']
    signs = [+1.]
    kite_nodes = model.architecture.kite_nodes
    parent_map = model.architecture.parent_map

    # create space for vortex nodes
    dict_coll = {}
    dict_xd = {}
    for kite in kite_nodes:
        dict_coll[kite] = {}
        dict_xd[kite] = {}
        for ext_int in tips:
            dict_coll[kite][ext_int] = {}
            dict_xd[kite][ext_int] = {}
            for dim in dims:
                dict_coll[kite][ext_int][dim] = {}
                dict_xd[kite][ext_int][dim] = {}
                for ndx in range(n_k):
                    dict_coll[kite][ext_int][dim][ndx] = {}
                    dict_xd[kite][ext_int][dim][ndx] = np.zeros((n_k, d))

                    for ddx in range(d):
                        dict_coll[kite][ext_int][dim][ndx][ddx] = np.zeros((n_k, d))

    # fix values
    for kite in kite_nodes:
        parent = parent_map[kite]
        for sdx in range(len(tips)):
            sign = signs[sdx]
            ext_int = tips[sdx]

            for jdx in range(len(dims)):
                dim = dims[jdx]
                for ndx in range(n_k):

                    for ndx_shed in range(n_k):
                        for ddx_shed in range(d):

                            q_kite = V_init['coll_var', ndx_shed, ddx_shed, 'xd', 'q' + str(kite) + str(parent)]

                            t_shed = tgrid_coll[ndx_shed, ddx_shed]
                            ehat_radial = get_ehat_radial(t_shed, options, model, kite)
                            q_tip = q_kite + b_ref * sign * ehat_radial / 2.

                            t_local_xd = tgrid_xd[ndx]
                            time_convected_xd = t_local_xd - t_shed
                            q_convected_xd = q_tip + U_ref * time_convected_xd

                            dict_xd[kite][ext_int][dim][ndx][ndx_shed][ddx_shed] = q_convected_xd[jdx]

                            for ddx in range(d):
                                t_local_coll = tgrid_coll[ndx, ddx]
                                time_convected_coll = t_local_coll - t_shed
                                q_convected_coll = q_tip + U_ref * time_convected_coll

                                dict_coll[kite][ext_int][dim][ndx][ddx][ndx_shed][ddx_shed] = q_convected_coll[jdx]

    # save values to V_init
    for kite in kite_nodes:
        parent = parent_map[kite]
        for ext_int in tips:
            for jdx in range(len(dims)):
                dim = dims[jdx]
                var_name = 'w' + dim + '_' + ext_int + str(kite) + str(parent)

                for ndx in range(n_k):
                    dict_xd[kite][ext_int][dim][ndx] = cas.reshape(dict_xd[kite][ext_int][dim][ndx], (n_nodes, 1))
                    V_init['xd', ndx, var_name] = dict_xd[kite][ext_int][dim][ndx]

                    # V_init['xd', ndx, 'd' + var_name] = np.ones((n_nodes, 1)) * U_ref[jdx]

                    for ddx in range(d):
                        dict_coll[kite][ext_int][dim][ndx][ddx] = cas.reshape(dict_coll[kite][ext_int][dim][ndx][ddx], (n_nodes, 1))
                        V_init['coll_var', ndx, ddx, 'xd', var_name] = dict_coll[kite][ext_int][dim][ndx][ddx]

                        # V_init['coll_var', ndx, ddx, 'xd', 'd' + var_name] = np.ones((n_nodes, 1)) * U_ref[jdx]

    return V_init


def initial_guess_actuator(options, nlp, formulation, model, V_init):

    V_init = initial_guess_actuator_xd(options, nlp, formulation, model, V_init)
    V_init = initial_guess_actuator_xl(options, nlp, formulation, model, V_init)

    return V_init





def initial_guess_actuator_xd(options, nlp, formulation, model, V_init):

    level_siblings = model.architecture.get_all_level_siblings()

    tf_init = guess_tf(options, model, formulation)
    tgrid_coll = nlp.time_grids['coll'](tf_init)

    ua_norm = options['ua_norm']
    l_t_temp = options['xd']['l_t']
    _, radius = get_cone_height_and_radius(options, model, l_t_temp)
    omega_norm = ua_norm / radius

    dict = {}
    dict['a'] = cas.DM(options['xd']['a'])
    dict['asin'] = cas.DM(0.)
    dict['acos'] = cas.DM(0.)
    dict['ct'] = 4. * dict['a'] * (1. - dict['a'])
    dict['bar_varrho'] = cas.DM(1.)

    var_type = 'xd'
    for name in struct_op.subkeys(model.variables, var_type):
        name_stripped = struct_op.get_node_variable_name(name)

        if 'psi' in name_stripped:
            V_init = set_azimuth_variables(V_init, name, model, nlp, tgrid_coll, level_siblings, omega_norm)

        elif name_stripped in dict.keys():
            V_init = insert_dict(dict, var_type, name, name_stripped, V_init)

    return V_init


def initial_guess_actuator_xl(options, nlp, formulation, model, V_init):

    level_siblings = model.architecture.get_all_level_siblings()

    tf_init = guess_tf(options, model, formulation)
    tgrid_coll = nlp.time_grids['coll'](tf_init)

    ua_norm = options['ua_norm']
    l_t_temp = options['xd']['l_t']
    _, radius = get_cone_height_and_radius(options, model, l_t_temp)
    omega_norm = ua_norm / radius

    n_rot_hat, y_rot_hat, z_rot_hat = get_rotor_reference_frame(options)
    rot_matr = cas.horzcat(n_rot_hat, y_rot_hat, z_rot_hat)
    rot_matr_cols = cas.reshape(rot_matr, (9, 1))

    u_hat, v_hat, w_hat = get_local_wind_reference_frame(options)
    uzero_matr = cas.horzcat(u_hat, v_hat, w_hat)
    uzero_matr_cols = cas.reshape(uzero_matr, (9, 1))

    dict = {}
    dict['area'] = cas.DM(1.)
    dict['cmy'] = cas.DM(0.)
    dict['cmz'] = cas.DM(0.)
    dict['rot_matr'] = rot_matr_cols
    dict['uzero_matr'] = uzero_matr_cols
    dict['n_vec_length'] = cas.DM(1.)
    dict['g_vec_length'] = cas.DM(1.)
    dict['z_vec_length'] = cas.DM(1.)
    dict['u_vec_length'] = cas.DM(1.)
    dict['varrho'] = cas.DM(1.)
    dict['qzero'] = cas.DM(1.)
    dict['gamma'] = get_gamma_angle(options)
    dict['cosgamma'] = np.cos(dict['gamma'])
    dict['singamma'] = np.sin(dict['gamma'])
    dict['chi'] = get_chi_angle(options, 'xl')
    dict['coschi'] = np.cos(dict['chi'])
    dict['sinchi'] = np.sin(dict['chi'])
    dict['tanhalfchi'] = np.tan(dict['chi']/2.)
    dict['sechalfchi'] = 1./np.cos(dict['chi']/2.)
    dict['LL'] = cas.DM([0.375, 0., 0., 0., -1., 0., 0., -1., 0.])
    dict['corr'] = 1. - options['xd']['a']

    var_type = 'xl'
    for name in struct_op.subkeys(model.variables, 'xl'):
        name_stripped = struct_op.get_node_variable_name(name)

        if 'psi' in name_stripped:
            V_init = set_azimuth_variables(V_init, name, model, nlp, tgrid_coll, level_siblings, omega_norm)

        elif name_stripped in dict.keys():
            V_init = insert_dict(dict, var_type, name, name_stripped, V_init)

    return V_init


def insert_dict(dict, var_type, name, name_stripped, V_init):
    init_val = dict[name_stripped]

    for idx in range(init_val.shape[0]):
        V_init = insert_val(V_init, var_type, name, init_val[idx], idx)

    return V_init


def insert_val(V_init, var_type, name, init_val, idx = 0):

    # initialize on collocation nodes
    V_init['coll_var', :, :, var_type, name, idx] = init_val

    if var_type == 'xd':
        # initialize on interval nodes
        # V_init[var_type, :, :, name] = init_val

        V_init[var_type, :, name, idx] = init_val

    return V_init

def set_psi_variables(V_init, name, model, nlp, tgrid_coll, level_siblings, omega_norm):
    kite_parent = name[3:]
    kite, parent = struct_op.split_kite_and_parent(kite_parent, model.architecture)

    idx = 0
    for nn in range(nlp.n_k):
        for dd in range(nlp.d):

            t = tgrid_coll[nn, dd]
            psi_scale = 2. * np.pi
            psi = get_azimuthal_angle(t, level_siblings, kite, parent, omega_norm)

            init_val = psi / psi_scale
            V_init['coll_var', nn, dd, 'xd', name] = init_val

            if dd == 0:
                V_init['xd', nn, name] = init_val

            idx = idx + 1

    return V_init

def set_dpsi_variables(V_init, name, model, nlp, tgrid_coll, level_siblings, omega_norm):
    V_init = insert_val(V_init, 'xd', name, omega_norm / (2. * np.pi))
    return V_init


def set_sinpsi_variables(V_init, name, model, nlp, tgrid_coll, level_siblings, omega_norm):
    kite_parent = name[6:]
    kite, parent = struct_op.split_kite_and_parent(kite_parent, model.architecture)

    idx = 0
    for nn in range(nlp.n_k):
        for dd in range(nlp.d):
            t = tgrid_coll[nn, dd]
            psi = get_azimuthal_angle(t, level_siblings, kite, parent, omega_norm)

            init_val = np.sin(psi)
            V_init['coll_var', nn, dd, 'xl', name] = init_val

            idx = idx + 1

    return V_init

def set_cospsi_variables(V_init, name, model, nlp, tgrid_coll, level_siblings, omega_norm):
    kite_parent = name[6:]
    kite, parent = struct_op.split_kite_and_parent(kite_parent, model.architecture)

    idx = 0
    for nn in range(nlp.n_k):
        for dd in range(nlp.d):
            t = tgrid_coll[nn, dd]
            psi = get_azimuthal_angle(t, level_siblings, kite, parent, omega_norm)

            init_val = np.cos(psi)
            V_init['coll_var', nn, dd, 'xl', name] = init_val

            idx = idx + 1

    return V_init

def set_azimuth_variables(V_init, name, model, nlp, tgrid_coll, level_siblings, omega_norm):

    if 'cos' in name:
        V_init = set_cospsi_variables(V_init, name, model, nlp, tgrid_coll, level_siblings, omega_norm)
    elif 'sin' in name:
        V_init = set_sinpsi_variables(V_init, name, model, nlp, tgrid_coll, level_siblings, omega_norm)
    elif 'dpsi' in name:
        V_init = set_dpsi_variables(V_init, name, model, nlp, tgrid_coll, level_siblings, omega_norm)
    else:
        V_init = set_psi_variables(V_init, name, model, nlp, tgrid_coll, level_siblings, omega_norm)

    return V_init