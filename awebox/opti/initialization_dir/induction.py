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
initialization of induction variables
_python _version 2.7 / casadi-3.4.5
- _author: rachel leuthold, jochem de schutter, thilo bronnenmeyer (alu-fr, 2017 - 21)
'''



import numpy as np
import casadi.tools as cas
from awebox.logger.logger import Logger as awelogger

import awebox.tools.vector_operations as vect_op
import awebox.tools.struct_operations as struct_op
import awebox.tools.print_operations as print_op

import awebox.opti.initialization_dir.tools as tools_init

import awebox.mdl.aero.induction_dir.vortex_dir.filament_list as vortex_filament_list
import awebox.mdl.aero.induction_dir.vortex_dir.flow as vortex_flow

def initial_guess_induction(init_options, nlp, formulation, model, V_init):

    comparison_labels = init_options['model']['comparison_labels']

    if comparison_labels:
        V_init = initial_guess_general(V_init)

    any_act = any(label[:3] == 'act' for label in comparison_labels)
    if any_act:
        V_init = initial_guess_actuator(init_options, nlp, model, V_init)

    any_vor = any(label[:3] == 'vor' for label in comparison_labels)
    if any_vor:
        V_init = initial_guess_vortex(init_options, nlp, formulation, model, V_init)

    return V_init


def initial_guess_general(V_init):
    return V_init


def initial_guess_vortex(init_options, nlp, formulation, model, V_init):

    if not nlp.discretization == 'direct_collocation':
        message = 'vortex induction model is only defined for direct-collocation model, at this point'
        awelogger.logger.error(message)
        raise Exception(message)

    if init_options['induction']['vortex_representation'] == 'state':

        # create the dictionaries
        dict_xd, dict_coll = reserve_space_in_wake_node_position_dicts(init_options, nlp, model)

        # save values into dictionaries
        dict_xd, dict_coll = save_regular_wake_node_position_into_dicts(dict_xd, dict_coll, init_options, nlp, formulation,
                                                                            model, V_init)
        # set dictionary values into V_init
        V_init = set_wake_node_positions_from_dict(dict_xd, dict_coll, init_options, nlp, model, V_init)

    elif init_options['induction']['vortex_representation'] == 'alg':

        V_init = get_alg_repr_fixing_initialization(init_options, nlp, model, V_init)

    V_init = set_wake_strengths(init_options, nlp, model, V_init)

    return V_init


def set_wu_ind(init_options, nlp, model, V_init):

    n_k = nlp.n_k
    d = nlp.d
    wake_nodes = init_options['model']['vortex_wake_nodes']
    rings = wake_nodes - 1

    for kdx in range(n_k):

        for ddx in range(d+1):
            # remember that V_init is in si coords, ie. not yet scaled
            if ddx == 0:
                ddx_arg = None
            else:
                ddx_arg = ddx-1
            variables_si = struct_op.get_variables_at_time(init_options, V_init, None, model.variables, kdx, ddx=ddx_arg)
            filament_list = vortex_filament_list.get_list(init_options, variables_si, model.architecture)
            filaments = filament_list.shape[1]

            for kite_obs in model.architecture.kite_nodes:
                u_ind_kite = vortex_flow.get_induced_velocity_at_kite(init_options, filament_list, variables_si, model.architecture,
                                                               kite_obs)
                ind_name = 'wu_ind_' + str(kite_obs)

                if ddx == 0:
                    V_init['xl', kdx, ind_name] = u_ind_kite
                else:
                    V_init['coll_var', kdx, ddx, 'xl', ind_name] = u_ind_kite

                for fdx in range(filaments):
                    # biot-savart of filament induction
                    filament = filament_list[:, fdx]
                    u_ind_fil = vortex_flow.get_induced_velocity_at_kite(init_options, filament, variables_si, model.architecture,
                                                               kite_obs)

                    ind_name = 'wu_fil_' + str(fdx) + '_' + str(kite_obs)

                    if ddx == 0:
                        V_init['xl', kdx, ind_name] = u_ind_fil
                    else:
                        V_init['coll_var', kdx, ddx, 'xl', ind_name] = u_ind_fil


    return V_init

def reserve_space_in_wake_node_position_dicts(init_options, nlp, model):
    n_k = nlp.n_k
    d = nlp.d
    wingtips = ['ext', 'int']
    kite_nodes = model.architecture.kite_nodes
    wake_nodes = init_options['model']['vortex_wake_nodes']

    # create space for vortex nodes
    dict_coll = {}
    dict_xd = {}
    for kite in kite_nodes:
        dict_coll[kite] = {}
        dict_xd[kite] = {}
        for tip in wingtips:
            dict_coll[kite][tip] = {}
            dict_xd[kite][tip] = {}

            for wake_node in range(wake_nodes):
                dict_coll[kite][tip][wake_node] = {}
                dict_xd[kite][tip][wake_node] = {}

                for ndx in range(n_k+1):
                    dict_xd[kite][tip][wake_node][ndx] = np.zeros((3, 1))

                    dict_coll[kite][tip][wake_node][ndx] = {}
                    for ddx in range(d):
                        dict_coll[kite][tip][wake_node][ndx][ddx] = np.zeros((3, 1))

    return dict_xd, dict_coll


def set_wake_strengths(init_options, nlp, model, V_init):
    n_k = nlp.n_k
    d = nlp.d

    wake_gamma = guess_wake_gamma_val(init_options)

    kite_nodes = model.architecture.kite_nodes
    wake_nodes = init_options['model']['vortex_wake_nodes']
    rings = wake_nodes - 1

    for kite in kite_nodes:
        for ring in range(rings):
            var_name = 'wg' + '_' + str(kite) + '_' + str(ring)

            for ndx in range(n_k):
                V_init['xl',ndx,var_name] = wake_gamma
                for ddx in range(d):
                    V_init['coll_var', ndx, ddx, 'xl', var_name] = wake_gamma

    return V_init




def set_wake_node_positions_from_dict(dict_xd, dict_coll, init_options, nlp, model, V_init):
    n_k = nlp.n_k
    d = nlp.d
    wingtips = ['ext', 'int']

    kite_nodes = model.architecture.kite_nodes
    wake_nodes = init_options['model']['vortex_wake_nodes']

    for kite in kite_nodes:
        for tip in wingtips:
            for wake_node in range(wake_nodes):
                var_name = 'wx_' + str(kite) + '_' + tip + '_' + str(wake_node)

                for ndx in range(n_k + 1):
                    wx_local = dict_xd[kite][tip][wake_node][ndx]
                    V_init['xd', ndx, var_name] = wx_local

                for ndx in range(n_k):
                    for ddx in range(d):
                        wx_local = dict_coll[kite][tip][wake_node][ndx][ddx]
                        V_init['coll_var', ndx, ddx, 'xd', var_name] = wx_local

    return V_init


def save_regular_wake_node_position_into_dicts(dict_xd, dict_coll, init_options, nlp, formulation, model, V_init):
    vec_u_infty = tools_init.get_wind_speed(init_options, V_init['xd', 0, 'q10'][2]) * vect_op.xhat_np()

    b_ref = init_options['sys_params_num']['geometry']['b_ref']
    n_k = nlp.n_k
    d = nlp.d

    control_intervals = n_k + 1

    wingtips = ['ext', 'int']
    signs = [+1., -1.]
    kite_nodes = model.architecture.kite_nodes
    parent_map = model.architecture.parent_map
    wake_nodes = init_options['model']['vortex_wake_nodes']



    time_final = init_options['precompute']['time_final']
    tgrid_xd = nlp.time_grids['x'](time_final)
    tgrid_coll = nlp.time_grids['coll'](time_final)

    for kite in kite_nodes:
        parent = parent_map[kite]
        for sdx in range(len(wingtips)):
            sign = signs[sdx]
            tip = wingtips[sdx]

            for wake_node in range(wake_nodes):

                index = control_intervals - wake_node

                period = int(np.floor(index / (n_k)))
                leftover = np.mod(index, (n_k))

                q_kite = V_init['coll_var', leftover-1, -1, 'xd', 'q' + str(kite) + str(parent)]

                t_shed = period * time_final + tgrid_xd[leftover]

                for ndx in range(n_k + 1):
                    t_local_xd = tgrid_xd[ndx]
                    q_convected_xd = guess_vortex_node_position(t_shed, t_local_xd, q_kite, init_options, model,
                                                                kite, b_ref, sign, vec_u_infty)
                    dict_xd[kite][tip][wake_node][ndx] = q_convected_xd

                for ndx in range(n_k):
                    for ddx in range(d):
                        t_local_coll = tgrid_coll[ndx, ddx]
                        q_convected_coll = guess_vortex_node_position(t_shed, t_local_coll, q_kite, init_options,
                                                                      model, kite, b_ref, sign, vec_u_infty)
                        dict_coll[kite][tip][wake_node][ndx][ddx] = q_convected_coll

    return dict_xd, dict_coll


def guess_vortex_node_position(t_shed, t_local, q_kite, init_options, model, kite, b_ref, sign, vec_u_infty):
    ehat_radial = tools_init.get_ehat_radial(t_shed, init_options, model, kite)
    q_tip = q_kite + b_ref * sign * ehat_radial / 2.

    time_convected = t_local - t_shed
    q_convected = q_tip + vec_u_infty * time_convected

    return q_convected

def guess_wake_gamma_val(init_options):
    gamma = cas.DM(0.) #init_options['induction']['vortex_gamma_scale']

    return gamma


def initial_guess_actuator(init_options, nlp, model, V_init):
    V_init = initial_guess_actuator_xd(init_options, model, V_init)
    V_init = initial_guess_actuator_xl(init_options, model, V_init)
    V_init = set_azimuth_variables(V_init, init_options, model, nlp)

    return V_init


def initial_guess_actuator_xd(init_options, model, V_init):

    dict = {}
    dict['a'] = cas.DM(init_options['xd']['a'])
    dict['asin_uasym'] = cas.DM(0.)
    dict['acos_uasym'] = cas.DM(0.)
    dict['a_uaxi'] = dict['a']
    dict['a_uasym'] = dict['a']


    var_type = 'xd'
    for name in struct_op.subkeys(model.variables, var_type):
        name_stripped, _ = struct_op.split_name_and_node_identifier(name)

        if name_stripped in dict.keys():
            V_init = tools_init.insert_dict(dict, var_type, name, name_stripped, V_init)

    return V_init

def set_azimuth_variables(V_init, init_options, model, nlp):

    level_siblings = model.architecture.get_all_level_siblings()
    omega_norm = init_options['precompute']['angular_speed']

    for kite in model.architecture.kite_nodes:
        parent = model.architecture.parent_map[kite]
        kite_parent = str(kite) + str(parent)
        V_init = set_psi_variables(init_options, V_init, kite_parent, model, nlp, level_siblings, omega_norm)

    return V_init

def set_psi_variables(init_options, V_init, kite_parent, model, nlp, level_siblings, omega_norm):
    kite, parent = struct_op.split_kite_and_parent(kite_parent, model.architecture)

    time_final = init_options['precompute']['time_final']
    tgrid_xd = nlp.time_grids['x'](time_final)
    tgrid_coll = nlp.time_grids['coll'](time_final)

    for ndx in range(nlp.n_k):

        t = tgrid_xd[ndx]
        psi = tools_init.get_azimuthal_angle(t, init_options, level_siblings, kite, parent, omega_norm)
        V_init['xl', ndx, 'psi' + str(kite_parent)] = psi

        if '[xl,' + str(ndx) + ',cospsi' + str(kite_parent) + ',0]' in V_init.labels():
            V_init['xl', ndx, 'cospsi' + str(kite_parent)] = np.cos(psi)
        if '[xl,' + str(ndx) + ',sinpsi' + str(kite_parent) + ',0]' in V_init.labels():
            V_init['xl', ndx, 'sinpsi' + str(kite_parent)] = np.sin(psi)

        for ddx in range(nlp.d):
            t = tgrid_coll[ndx, ddx]
            psi = tools_init.get_azimuthal_angle(t, init_options, level_siblings, kite, parent, omega_norm)
            V_init['coll_var', ndx, ddx, 'xl', 'psi' + str(kite_parent)] = psi

            if '[coll_var,' + str(ndx) + ',' + str(ddx) + 'xl,cospsi' + str(kite_parent) + ',0]' in V_init.labels():
                V_init['collvar', ndx, ddx, 'xl', 'cospsi' + str(kite_parent)] = np.cos(psi)
            if '[coll_var,' + str(ndx) + ',' + str(ddx) + 'xl,sinpsi' + str(kite_parent) + ',0]' in V_init.labels():
                V_init['collvar', ndx, ddx, 'xl', 'sinpsi' + str(kite_parent)] = np.sin(psi)

    return V_init


def initial_guess_actuator_xl(init_options, model, V_init):

    u_hat, v_hat, w_hat = get_local_wind_reference_frame(init_options)
    wind_dcm = cas.horzcat(u_hat, v_hat, w_hat)
    wind_dcm_cols = cas.reshape(wind_dcm, (9, 1))

    n_rot_hat, y_rot_hat, z_rot_hat = tools_init.get_rotor_reference_frame(init_options)
    act_dcm = cas.horzcat(n_rot_hat, y_rot_hat, z_rot_hat)
    act_dcm_cols = cas.reshape(act_dcm, (9, 1))

    b_ref = init_options['sys_params_num']['geometry']['b_ref']

    dict = {}
    dict['a'] = cas.DM(init_options['xl']['a'])
    dict['a_qaxi'] = dict['a']
    dict['a_qasym'] = dict['a']
    dict['local_a'] = dict['a']
    dict['asin_qasym'] = cas.DM(0.)
    dict['acos_qasym'] = cas.DM(0.)
    dict['ui'] = cas.DM.zeros((3, 1)) #remember that induction homotopy has not yet begun.
    dict['varrho'] = cas.DM(init_options['precompute']['radius'] / b_ref)
    dict['bar_varrho'] = dict['varrho']
    dict['act_dcm'] = act_dcm_cols
    dict['n_vec_length'] = cas.DM(init_options['induction']['n_vec_length'])
    dict['wind_dcm'] = wind_dcm_cols
    dict['z_vec_length'] = cas.DM(init_options['induction']['z_vec_length'])
    dict['u_vec_length'] = vect_op.norm(tools_init.get_wind_speed(init_options, V_init['xd', 0, 'q10'][2]))
    dict['gamma'] = get_gamma_angle(init_options)
    dict['g_vec_length'] = cas.DM(init_options['induction']['g_vec_length'])
    dict['cosgamma'] = np.cos(dict['gamma'])
    dict['singamma'] = np.sin(dict['gamma'])

    var_type = 'xl'
    for name in struct_op.subkeys(model.variables, var_type):
        name_stripped, _ = struct_op.split_name_and_node_identifier(name)

        if name_stripped in dict.keys():
            V_init = tools_init.insert_dict(dict, var_type, name, name_stripped, V_init)

    return V_init





def get_gamma_angle(init_options):
    n_rot_hat = tools_init.get_ehat_tether(init_options)
    u_hat, _, _ = get_local_wind_reference_frame(init_options)
    gamma = vect_op.angle_between(u_hat, n_rot_hat)
    return gamma

def get_chi_angle(init_options, var_type):

    a = init_options[var_type]['a']
    gamma = get_gamma_angle(init_options)
    chi = (0.6 * a + 1.) * gamma
    return chi

def get_local_wind_reference_frame(init_options):
    u_hat = vect_op.xhat_np()
    n_rot_hat, y_rot_hat, z_rot_hat = tools_init.get_rotor_reference_frame(init_options)

    if vect_op.norm(u_hat - n_rot_hat) < 1.e-5:
        v_hat = y_rot_hat
        w_hat = z_rot_hat

    else:
        w_hat = vect_op.normed_cross(u_hat, n_rot_hat)
        v_hat = vect_op.normed_cross(w_hat, u_hat)
    return u_hat, v_hat, w_hat


def get_alg_repr_fixing_initialization(init_options, nlp, model, V_init):

    n_k = nlp.n_k
    d = nlp.d

    wake_nodes = init_options['model']['vortex_wake_nodes']
    kite_nodes = model.architecture.kite_nodes
    wingtips = ['ext', 'int']

    for kite in kite_nodes:
        for tip in wingtips:
            for wake_node in range(wake_nodes):

                for ndx in range(n_k):

                    for ddx in range(d):
                        V_init = get_local_alg_repr_fixing_initialization(init_options, V_init, nlp, model,
                                                                          kite, tip, wake_node, ndx, ddx)

                    if ndx > 0:
                        V_init = get_continuity_fixing_initialization(V_init, kite, tip, wake_node, ndx)

                    else:
                        V_init = get_alg_periodic_fixing_initialization(V_init, kite, tip, wake_node)


    return V_init


def get_local_alg_repr_fixing_initialization(init_options, V_init, nlp, model, kite, tip, wake_node, ndx, ddx):
    if tip == 'exit':
        sign = +1.
    else:
        sign = -1.

    parent = model.architecture.parent_map[kite]

    b_ref = init_options['sys_params_num']['geometry']['b_ref']

    time_final = init_options['precompute']['time_final']
    tgrid = nlp.time_grids['coll'](time_final)
    current_time = tgrid[ndx, ddx]

    n_k = nlp.n_k

    # # if wake_node = 0, then shed at ndx
    # # if wake_node = 1, then shed at (ndx - 1) ---- > corresponds to (ndx - 2), ddx = -1
    # # .... if shedding_ndx is 1, then shedding_ndx -> 1
    # # ....  if shedding_ndx is 0, then shedding_ndx -> n_k
    # # ....  if shedding_ndx is -1, then shedding_ndx -> n_k - 1
    # # .... so, shedding_ndx -> np.mod(ndx - wake_node, n_k) -----> np.mod(ndx - wake_node - 1, n_k), ddx=-1
    subtracted_ndx = ndx - wake_node
    shedding_ndx = np.mod(subtracted_ndx, n_k)
    periods_passed = np.floor(subtracted_ndx / n_k)

    if wake_node == 0:
        shedding_ddx = ddx
    else:
        shedding_ddx = -1

    shedding_time = time_final * periods_passed + tgrid[shedding_ndx, shedding_ddx]

    q_kite = V_init['coll_var', shedding_ndx, shedding_ddx, 'xd', 'q' + str(kite) + str(parent)]
    vec_u_infty = tools_init.get_wind_speed(init_options, q_kite[2]) * vect_op.xhat_np()

    wx_found = guess_vortex_node_position(shedding_time, current_time, q_kite, init_options, model, kite,
                                                     b_ref, sign, vec_u_infty)

    var_name = 'wx_' + str(kite) + '_' + tip + '_' + str(wake_node)
    V_init['coll_var', ndx, ddx, 'xl', var_name] = wx_found

    return V_init

def get_continuity_fixing_initialization(V_init, kite, tip, wake_node, ndx):
    var_name = 'wx_' + str(kite) + '_' + tip + '_' + str(wake_node)
    V_init['xl', ndx, var_name] = V_init['coll_var', ndx-1, -1, 'xl', var_name]
    return V_init

def get_alg_periodic_fixing_initialization(V_init, kite, tip, wake_node):
    var_name = 'wx_' + str(kite) + '_' + tip + '_' + str(wake_node)
    V_init['xl', 0, var_name] = V_init['coll_var', -1, -1, 'xl', var_name]
    return V_init
