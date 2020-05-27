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
initialization of induction variables
_python _version 2.7 / casadi-3.4.5
- _author: rachel leuthold, jochem de schutter, thilo bronnenmeyer (alu-fr, 2017 - 20)
'''


import numpy as np
import casadi.tools as cas
import awebox.tools.vector_operations as vect_op
from awebox.logger.logger import Logger as awelogger
import awebox.tools.struct_operations as struct_op
import awebox.opti.initialization_tools as tools_init


def initial_guess_induction(options, nlp, formulation, model, V_init):

    comparison_labels = options['model']['comparison_labels']

    if comparison_labels:
        V_init = initial_guess_general(options, nlp, formulation, model, V_init)

    any_act = any(label[:3] == 'act' for label in comparison_labels)
    if any_act:
        V_init = initial_guess_actuator(options, nlp, formulation, model, V_init)

    any_vor = any(label[:3] == 'vor' for label in comparison_labels)
    if any_vor:
        V_init = initial_guess_vortex(options, nlp, formulation, model, V_init)

    return V_init


def initial_guess_general(options, nlp, formulation, model, V_init):
    return V_init


def initial_guess_vortex(options, nlp, formulation, model, V_init):
    if not nlp.discretization == 'direct_collocation':
        awelogger.logger.error('vortex induction model is only defined for direct-collocation model, at this point')

    # create the dictionaries
    dict_xd, dict_coll = reserve_space_in_wake_node_position_dicts(options, nlp, model)

    # save values into dictionaries
    dict_xd = save_starting_wake_node_position_into_xd_dict(dict_xd, options, nlp, formulation, model, V_init)
    dict_coll = save_starting_wake_node_position_into_coll_dict(dict_coll, options, nlp, formulation, model, V_init)
    dict_xd, dict_coll = save_regular_wake_node_position_into_dicts(dict_xd, dict_coll, options, nlp, formulation,
                                                                    model, V_init)

    # set dictionary values into V_init
    V_init = set_wake_node_positions_from_dict(dict_xd, dict_coll, options, nlp, model, V_init)

    return V_init


def reserve_space_in_wake_node_position_dicts(options, nlp, model):
    n_k = nlp.n_k
    d = nlp.d
    dims = ['x', 'y', 'z']
    wingtips = ['ext', 'int']
    kite_nodes = model.architecture.kite_nodes
    periods_tracked = options['model']['vortex_periods_tracked']

    # create space for vortex nodes
    dict_coll = {}
    dict_xd = {}
    for kite in kite_nodes:
        dict_coll[kite] = {}
        dict_xd[kite] = {}
        for tip in wingtips:
            dict_coll[kite][tip] = {}
            dict_xd[kite][tip] = {}
            for dim in dims:
                dict_coll[kite][tip][dim] = {}
                dict_xd[kite][tip][dim] = {}

                for period in range(periods_tracked):
                    dict_coll[kite][tip][dim][period] = {}
                    dict_xd[kite][tip][dim][period] = {}

                    for ndx in range(n_k):
                        dict_coll[kite][tip][dim][period][ndx] = {}
                        dict_xd[kite][tip][dim][period][ndx] = {}

                        dict_xd[kite][tip][dim][period][ndx]['start'] = 0.
                        dict_xd[kite][tip][dim][period][ndx]['reg'] = np.zeros((n_k, d))

                        for ddx in range(d):
                            dict_coll[kite][tip][dim][period][ndx][ddx] = {}

                            dict_coll[kite][tip][dim][period][ndx][ddx]['start'] = np.zeros((n_k, d))
                            dict_coll[kite][tip][dim][period][ndx][ddx]['reg'] = np.zeros((n_k, d))

    return dict_xd, dict_coll


def set_wake_node_positions_from_dict(dict_xd, dict_coll, options, nlp, model, V_init):
    n_k = nlp.n_k
    d = nlp.d
    dims = ['x', 'y', 'z']
    wingtips = ['ext', 'int']

    kite_nodes = model.architecture.kite_nodes
    parent_map = model.architecture.parent_map
    periods_tracked = options['model']['vortex_periods_tracked']

    for kite in kite_nodes:
        parent = parent_map[kite]
        for tip in wingtips:
            for jdx in range(len(dims)):
                dim = dims[jdx]

                for period in range(periods_tracked):
                    var_name = 'w' + dim + '_' + tip + '_' + str(period) + '_' + str(kite) + str(parent)

                    for ndx in range(n_k):

                        start_xd = dict_xd[kite][tip][dim][period][ndx]['start']
                        regular_xd = cas.reshape(dict_xd[kite][tip][dim][period][ndx]['reg'], (n_k * d, 1))
                        all_xd = cas.vertcat(start_xd, regular_xd)
                        V_init['xd', ndx, var_name] = all_xd

                        # V_init['xd', ndx, 'd' + var_name] = np.ones((n_nodes, 1)) * U_ref[jdx]

                        for ddx in range(d):
                            start_coll = dict_coll[kite][tip][dim][period][ndx][ddx]['start']
                            regular_coll = cas.reshape(dict_coll[kite][tip][dim][period][ndx][ddx]['reg'], (n_k * d, 1))
                            all_coll = cas.vertcat(start_coll, regular_coll)
                            V_init['coll_var', ndx, ddx, 'xd', var_name] = all_coll

                            # V_init['coll_var', ndx, ddx, 'xd', 'd' + var_name] = np.ones((n_nodes, 1)) * U_ref[jdx]
    return V_init


def save_starting_wake_node_position_into_xd_dict(dict_xd, options, nlp, formulation, model, V_init):
    U_ref = options['sys_params_num']['wind']['u_ref'] * vect_op.xhat_np()
    b_ref = options['sys_params_num']['geometry']['b_ref']
    n_k = nlp.n_k
    dims = ['x', 'y', 'z']
    wingtips = ['ext', 'int']
    signs = [+1., -1.]
    kite_nodes = model.architecture.kite_nodes
    parent_map = model.architecture.parent_map
    periods_tracked = options['model']['vortex_periods_tracked']

    tf_init = tools_init.guess_tf(options, model, formulation)
    tgrid_xd = nlp.time_grids['x'](tf_init)

    # fix values
    for kite in kite_nodes:
        parent = parent_map[kite]
        for sdx in range(len(wingtips)):
            sign = signs[sdx]
            tip = wingtips[sdx]
            for jdx in range(len(dims)):
                dim = dims[jdx]
                for period in range(periods_tracked):

                    q_kite = V_init['xd', 0, 'q' + str(kite) + str(parent)]
                    t_shed = period * tf_init

                    for ndx in range(n_k):
                        t_local = tgrid_xd[ndx]

                        q_convected = guess_vortex_node_position(t_shed, t_local, q_kite, options, model, kite, b_ref,
                                                                 sign, U_ref)
                        dict_xd[kite][tip][dim][period][ndx]['start'] = q_convected[jdx]

    return dict_xd


def save_starting_wake_node_position_into_coll_dict(dict_coll, options, nlp, formulation, model, V_init):
    U_ref = options['sys_params_num']['wind']['u_ref'] * vect_op.xhat_np()
    b_ref = options['sys_params_num']['geometry']['b_ref']
    n_k = nlp.n_k
    d = nlp.d
    dims = ['x', 'y', 'z']
    wingtips = ['ext', 'int']
    signs = [+1., -1.]
    kite_nodes = model.architecture.kite_nodes
    parent_map = model.architecture.parent_map
    periods_tracked = options['model']['vortex_periods_tracked']

    tf_init = tools_init.guess_tf(options, model, formulation)
    tgrid_coll = nlp.time_grids['coll'](tf_init)

    # fix values
    for kite in kite_nodes:
        parent = parent_map[kite]
        for sdx in range(len(wingtips)):
            sign = signs[sdx]
            tip = wingtips[sdx]
            for jdx in range(len(dims)):
                dim = dims[jdx]
                for period in range(periods_tracked):

                    q_kite = V_init['xd', 0, 'q' + str(kite) + str(parent)]
                    t_shed = period * tf_init

                    for ndx in range(n_k):
                        for ddx in range(d):
                            t_local = tgrid_coll[ndx, ddx]

                            q_convected = guess_vortex_node_position(t_shed, t_local, q_kite, options, model, kite,
                                                                     b_ref, sign, U_ref)
                            dict_coll[kite][tip][dim][period][ndx][ddx]['start'] = q_convected[jdx]
    return dict_coll


def save_regular_wake_node_position_into_dicts(dict_xd, dict_coll, options, nlp, formulation, model, V_init):
    U_ref = options['sys_params_num']['wind']['u_ref'] * vect_op.xhat_np()
    b_ref = options['sys_params_num']['geometry']['b_ref']
    n_k = nlp.n_k
    d = nlp.d
    dims = ['x', 'y', 'z']
    wingtips = ['ext', 'int']
    signs = [+1., -1.]
    kite_nodes = model.architecture.kite_nodes
    parent_map = model.architecture.parent_map
    periods_tracked = options['model']['vortex_periods_tracked']

    tf_init = tools_init.guess_tf(options, model, formulation)
    tgrid_xd = nlp.time_grids['x'](tf_init)
    tgrid_coll = nlp.time_grids['coll'](tf_init)

    for kite in kite_nodes:
        parent = parent_map[kite]
        for sdx in range(len(wingtips)):
            sign = signs[sdx]
            tip = wingtips[sdx]
            for jdx in range(len(dims)):
                dim = dims[jdx]

                for period in range(periods_tracked):
                    for ndx_shed in range(n_k):
                        for ddx_shed in range(d):
                            q_kite = V_init['coll_var', ndx_shed, ddx_shed, 'xd', 'q' + str(kite) + str(parent)]
                            t_shed = period * tf_init + tgrid_coll[ndx_shed, ddx_shed]

                            for ndx in range(n_k):
                                t_local_xd = tgrid_xd[ndx]
                                q_convected_xd = guess_vortex_node_position(t_shed, t_local_xd, q_kite, options, model,
                                                                            kite, b_ref, sign, U_ref)
                                dict_xd[kite][tip][dim][period][ndx]['reg'][ndx_shed][ddx_shed] = q_convected_xd[jdx]

                                for ddx in range(d):
                                    t_local_coll = tgrid_coll[ndx, ddx]
                                    q_convected_coll = guess_vortex_node_position(t_shed, t_local_coll, q_kite, options,
                                                                                  model, kite, b_ref, sign, U_ref)
                                    dict_coll[kite][tip][dim][period][ndx][ddx]['reg'][ndx_shed][ddx_shed] = \
                                    q_convected_coll[jdx]

    return dict_xd, dict_coll


def guess_vortex_node_position(t_shed, t_local, q_kite, options, model, kite, b_ref, sign, U_ref):
    ehat_radial = tools_init.get_ehat_radial(t_shed, options, model, kite)
    q_tip = q_kite + b_ref * sign * ehat_radial / 2.

    time_convected = t_local - t_shed
    q_convected = q_tip + U_ref * time_convected

    return q_convected


def initial_guess_actuator(options, nlp, formulation, model, V_init):
    V_init = initial_guess_actuator_xd(options, nlp, formulation, model, V_init)
    V_init = initial_guess_actuator_xl(options, nlp, formulation, model, V_init)

    return V_init


def initial_guess_actuator_xd(options, nlp, formulation, model, V_init):
    level_siblings = model.architecture.get_all_level_siblings()

    tf_init = tools_init.guess_tf(options, model, formulation)
    tgrid_coll = nlp.time_grids['coll'](tf_init)

    ua_norm = options['ua_norm']
    l_t_temp = options['xd']['l_t']
    _, radius = tools_init.get_cone_height_and_radius(options, model, l_t_temp)
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
            V_init = tools_init.insert_dict(dict, var_type, name, name_stripped, V_init)

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

def set_psi_variables(V_init, name, model, nlp, tgrid_coll, level_siblings, omega_norm):
    kite_parent = name[3:]
    kite, parent = struct_op.split_kite_and_parent(kite_parent, model.architecture)

    idx = 0
    for nn in range(nlp.n_k):
        for dd in range(nlp.d):

            t = tgrid_coll[nn, dd]
            psi_scale = 2. * np.pi
            psi = tools_init.get_azimuthal_angle(t, level_siblings, kite, parent, omega_norm)

            init_val = psi / psi_scale
            V_init['coll_var', nn, dd, 'xd', name] = init_val

            if dd == 0:
                V_init['xd', nn, name] = init_val

            idx = idx + 1

    return V_init

def set_dpsi_variables(V_init, name, model, nlp, tgrid_coll, level_siblings, omega_norm):
    V_init = tools_init.insert_val(V_init, 'xd', name, omega_norm / (2. * np.pi))
    return V_init


def set_sinpsi_variables(V_init, name, model, nlp, tgrid_coll, level_siblings, omega_norm):
    kite_parent = name[6:]
    kite, parent = struct_op.split_kite_and_parent(kite_parent, model.architecture)

    idx = 0
    for nn in range(nlp.n_k):
        for dd in range(nlp.d):
            t = tgrid_coll[nn, dd]
            psi = tools_init.get_azimuthal_angle(t, level_siblings, kite, parent, omega_norm)

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
            psi = tools_init.get_azimuthal_angle(t, level_siblings, kite, parent, omega_norm)

            init_val = np.cos(psi)
            V_init['coll_var', nn, dd, 'xl', name] = init_val

            idx = idx + 1

    return V_init





def initial_guess_actuator_xl(options, nlp, formulation, model, V_init):
    level_siblings = model.architecture.get_all_level_siblings()

    tf_init = tools_init.guess_tf(options, model, formulation)
    tgrid_coll = nlp.time_grids['coll'](tf_init)

    ua_norm = options['ua_norm']
    l_t_temp = options['xd']['l_t']
    _, radius = tools_init.get_cone_height_and_radius(options, model, l_t_temp)
    omega_norm = ua_norm / radius


    u_hat, v_hat, w_hat = get_local_wind_reference_frame(options)
    uzero_matr = cas.horzcat(u_hat, v_hat, w_hat)
    uzero_matr_cols = cas.reshape(uzero_matr, (9, 1))

    n_rot_hat, y_rot_hat, z_rot_hat = tools_init.get_rotor_reference_frame(options)
    rot_matr = cas.horzcat(n_rot_hat, y_rot_hat, z_rot_hat)
    rot_matr_cols = cas.reshape(rot_matr, (9, 1))

    dict = {}
    dict['rot_matr'] = rot_matr_cols
    dict['area'] = cas.DM(1.)
    dict['cmy'] = cas.DM(0.)
    dict['cmz'] = cas.DM(0.)
    dict['uzero_matr'] = uzero_matr_cols
    dict['g_vec_length'] = cas.DM(1.)
    dict['n_vec_length'] = cas.DM(1.)
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
    dict['tanhalfchi'] = np.tan(dict['chi'] / 2.)
    dict['sechalfchi'] = 1. / np.cos(dict['chi'] / 2.)
    dict['LL'] = cas.DM([0.375, 0., 0., 0., -1., 0., 0., -1., 0.])
    dict['corr'] = 1. - options['xd']['a']

    var_type = 'xl'
    for name in struct_op.subkeys(model.variables, 'xl'):
        name_stripped = struct_op.get_node_variable_name(name)

        if 'psi' in name_stripped:
            V_init = set_azimuth_variables(V_init, name, model, nlp, tgrid_coll, level_siblings, omega_norm)

        elif name_stripped in dict.keys():
            V_init = tools_init.insert_dict(dict, var_type, name, name_stripped, V_init)

    return V_init





def get_gamma_angle(initialization_options):
    n_rot_hat = tools_init.get_ehat_tether(initialization_options)
    u_hat = vect_op.xhat_np()
    gamma = vect_op.angle_between(u_hat, n_rot_hat)
    return gamma

def get_chi_angle(options, var_type):
    a = options['xd']['a']
    gamma = get_gamma_angle(options)
    chi = (0.6 * a + 1.) * gamma
    return chi

def get_local_wind_reference_frame(initialization_options):
    u_hat = vect_op.xhat_np()
    n_rot_hat = tools_init.get_ehat_tether(initialization_options)
    w_hat = vect_op.normed_cross(u_hat, n_rot_hat)
    v_hat = vect_op.normed_cross(w_hat, u_hat)
    return u_hat, v_hat, w_hat
