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
initialization of induction variables
_python _version 2.7 / casadi-3.4.5
- _author: rachel leuthold, jochem de schutter, thilo bronnenmeyer (alu-fr, 2017 - 20)
'''



import numpy as np
import casadi.tools as cas
import awebox.tools.vector_operations as vect_op
from awebox.logger.logger import Logger as awelogger
import awebox.tools.struct_operations as struct_op
import awebox.opti.initialization_dir.tools as tools_init
import awebox.tools.print_operations as print_op
import awebox.mdl.aero.induction_dir.vortex_dir.tools as vortex_tools
import awebox.mdl.aero.induction_dir.vortex_dir.filament_list as vortex_filament_list
import awebox.mdl.aero.induction_dir.vortex_dir.flow as vortex_flow
import awebox.mdl.aero.induction_dir.vortex_dir.biot_savart as biot_savart

def initial_guess_induction(init_options, nlp, formulation, model, V_init):

    comparison_labels = init_options['model']['comparison_labels']

    if comparison_labels:
        V_init = initial_guess_general(init_options, nlp, formulation, model, V_init)

    any_act = any(label[:3] == 'act' for label in comparison_labels)
    if any_act:
        V_init = initial_guess_actuator(init_options, nlp, formulation, model, V_init)

    any_vor = any(label[:3] == 'vor' for label in comparison_labels)
    if any_vor:
        V_init = initial_guess_vortex(init_options, nlp, formulation, model, V_init)

    return V_init


def initial_guess_general(init_options, nlp, formulation, model, V_init):
    return V_init


def initial_guess_vortex(init_options, nlp, formulation, model, V_init):

    if not nlp.discretization == 'direct_collocation':
        message = 'vortex induction model is only defined for direct-collocation model, at this point'
        awelogger.logger.error(message)
        raise Exception(message)

    # create the dictionaries
    dict_xd, dict_coll = reserve_space_in_wake_node_position_dicts(init_options, nlp, model)

    # save values into dictionaries
    dict_xd, dict_coll = save_regular_wake_node_position_into_dicts(dict_xd, dict_coll, init_options, nlp, formulation,
                                                                        model, V_init)
    # set dictionary values into V_init
    V_init = set_wake_node_positions_from_dict(dict_xd, dict_coll, init_options, nlp, model, V_init)

    V_init = set_wake_strengths(init_options, nlp, model, V_init)

    return V_init


def set_wu_ind(init_options, nlp, model, V_init):

    n_k = nlp.n_k
    d = nlp.d
    wake_nodes = init_options['aero']['vortex']['wake_nodes']
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
    U_ref = guess_vortex_node_velocity(init_options)

    kite_nodes = model.architecture.kite_nodes
    wake_nodes = init_options['model']['vortex_wake_nodes']

    for kite in kite_nodes:
        for tip in wingtips:
            for wake_node in range(wake_nodes):
                var_name = 'wx_' + str(kite) + '_' + tip + '_' + str(wake_node)

                if init_options['induction']['vortex_representation'] == 'state':

                    for ndx in range(n_k + 1):
                        wx_local = dict_xd[kite][tip][wake_node][ndx]
                        V_init['xd', ndx, var_name] = wx_local
                        V_init['xd', ndx, 'd' + var_name] = U_ref

                    for ndx in range(n_k):
                        for ddx in range(d):
                            wx_local = dict_coll[kite][tip][wake_node][ndx][ddx]
                            V_init['coll_var', ndx, ddx, 'xd', var_name] = wx_local
                            V_init['coll_var', ndx, ddx, 'xd', 'd' + var_name] = U_ref

                else:
                    print_op.warn_about_temporary_funcationality_removal(location='init.induction')

    return V_init


def save_regular_wake_node_position_into_dicts(dict_xd, dict_coll, init_options, nlp, formulation, model, V_init):
    U_ref = init_options['sys_params_num']['wind']['u_ref'] * vect_op.xhat_np()
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
                                                                kite, b_ref, sign, U_ref)
                    dict_xd[kite][tip][wake_node][ndx] = q_convected_xd

                for ndx in range(n_k):
                    for ddx in range(d):
                        t_local_coll = tgrid_coll[ndx, ddx]
                        q_convected_coll = guess_vortex_node_position(t_shed, t_local_coll, q_kite, init_options,
                                                                      model, kite, b_ref, sign, U_ref)
                        dict_coll[kite][tip][wake_node][ndx][ddx] = q_convected_coll

    return dict_xd, dict_coll


def guess_vortex_node_position(t_shed, t_local, q_kite, init_options, model, kite, b_ref, sign, U_ref):
    ehat_radial = tools_init.get_ehat_radial(t_shed, init_options, model, kite)
    q_tip = q_kite + b_ref * sign * ehat_radial / 2.

    time_convected = t_local - t_shed
    q_convected = q_tip + U_ref * time_convected

    return q_convected

def guess_vortex_node_velocity(init_options):
    U_ref = init_options['sys_params_num']['wind']['u_ref'] * vect_op.xhat_np()

    return U_ref



def guess_wake_gamma_val(init_options):
    gamma = cas.DM(0.) #init_options['induction']['vortex_gamma_scale']

    return gamma


def initial_guess_actuator(init_options, nlp, formulation, model, V_init):
    V_init = initial_guess_actuator_xd(init_options, nlp, formulation, model, V_init)
    V_init = initial_guess_actuator_xl(init_options, nlp, formulation, model, V_init)

    return V_init


def initial_guess_actuator_xd(init_options, nlp, formulation, model, V_init):
    level_siblings = model.architecture.get_all_level_siblings()

    time_final = init_options['precompute']['time_final']
    omega_norm = init_options['precompute']['angular_speed']

    tgrid_coll = nlp.time_grids['coll'](time_final)

    dict = {}
    dict['a'] = cas.DM(init_options['xd']['a'])
    dict['asin'] = cas.DM(0.)
    dict['acos'] = cas.DM(0.)
    dict['ct'] = 4. * dict['a'] * (1. - dict['a'])
    dict['bar_varrho'] = cas.DM(1.)

    var_type = 'xd'
    for name in struct_op.subkeys(model.variables, var_type):
        name_stripped, _ = struct_op.split_name_and_node_identifier(name)

        if 'psi' in name_stripped:
            V_init = set_azimuth_variables(V_init, init_options, name, model, nlp, tgrid_coll, level_siblings, omega_norm)

        elif name_stripped in dict.keys():
            V_init = tools_init.insert_dict(dict, var_type, name, name_stripped, V_init)

    return V_init

def set_azimuth_variables(V_init, init_options, name, model, nlp, tgrid_coll_x, level_siblings, omega_norm):

    if 'cos' in name:
        V_init = set_cospsi_variables(init_options, V_init, name, model, nlp, tgrid_coll_x, level_siblings, omega_norm)
    elif 'sin' in name:
        V_init = set_sinpsi_variables(init_options, V_init, name, model, nlp, tgrid_coll_x, level_siblings, omega_norm)
    elif 'dpsi' in name:
        V_init = set_dpsi_variables(V_init, name, init_options)
    else:
        V_init = set_psi_variables(init_options, V_init, name, model, nlp, tgrid_coll_x, level_siblings, omega_norm)

    return V_init

def set_psi_variables(init_options, V_init, name, model, nlp, tgrid_coll, level_siblings, omega_norm):
    kite_parent = name[3:]
    kite, parent = struct_op.split_kite_and_parent(kite_parent, model.architecture)

    idx = 0
    for nn in range(nlp.n_k):
        for dd in range(nlp.d):

            t = tgrid_coll[nn, dd]
            psi_scale = get_psi_scale()
            psi = tools_init.get_azimuthal_angle(t, init_options, level_siblings, kite, parent, omega_norm)

            init_val = psi / psi_scale
            V_init['coll_var', nn, dd, 'xd', name] = init_val

            if dd == 0:
                V_init['xd', nn, name] = init_val

            idx = idx + 1

    return V_init

def get_psi_scale():
    psi_scale = 2. * np.pi
    return psi_scale

def set_dpsi_variables(V_init, name, init_options):

    dpsi = tools_init.get_dpsi(init_options)
    psi_scale = get_psi_scale()
    dpsi_scaled = dpsi / psi_scale
    V_init = tools_init.insert_val(V_init, 'xd', name, dpsi_scaled)
    return V_init


def set_sinpsi_variables(init_options, V_init, name, model, nlp, tgrid_coll_x, level_siblings, omega_norm):
    kite_parent = name[6:]
    kite, parent = struct_op.split_kite_and_parent(kite_parent, model.architecture)

    idx = 0
    for nn in range(nlp.n_k):
        for dd in range(nlp.d+1):
            t = tgrid_coll_x[nn*(nlp.d+1) + dd]
            psi = tools_init.get_azimuthal_angle(t, init_options, level_siblings, kite, parent, omega_norm)

            init_val = np.sin(psi)
            if dd == 0:
                V_init['xl', nn, name] = init_val
            else:
                V_init['coll_var', nn, dd-1, 'xl', name] = init_val

            idx = idx + 1

    return V_init

def set_cospsi_variables(init_options, V_init, name, model, nlp, tgrid_coll_x, level_siblings, omega_norm):
    kite_parent = name[6:]
    kite, parent = struct_op.split_kite_and_parent(kite_parent, model.architecture)

    idx = 0
    for nn in range(nlp.n_k):
        for dd in range(nlp.d+1):
            t = tgrid_coll_x[nn*(nlp.d+1) + dd]
            psi = tools_init.get_azimuthal_angle(t, init_options, level_siblings, kite, parent, omega_norm)

            init_val = np.cos(psi)
            if dd == 0:
                V_init['xl', nn, name] = init_val
            else:
                V_init['coll_var', nn, dd-1, 'xl', name] = init_val
            idx = idx + 1

    return V_init





def initial_guess_actuator_xl(init_options, nlp, formulation, model, V_init):
    level_siblings = model.architecture.get_all_level_siblings()

    time_final = init_options['precompute']['time_final']
    omega_norm = init_options['precompute']['angular_speed']

    tgrid_coll_x = nlp.time_grids['x_coll'](time_final)

    u_hat, v_hat, w_hat = get_local_wind_reference_frame(init_options)
    uzero_matr = cas.horzcat(u_hat, v_hat, w_hat)
    uzero_matr_cols = cas.reshape(uzero_matr, (9, 1))

    n_rot_hat, y_rot_hat, z_rot_hat = tools_init.get_rotor_reference_frame(init_options)
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
    dict['gamma'] = get_gamma_angle(init_options)
    dict['cosgamma'] = np.cos(dict['gamma'])
    dict['singamma'] = np.sin(dict['gamma'])
    dict['chi'] = get_chi_angle(init_options, 'xl')
    dict['coschi'] = np.cos(dict['chi'])
    dict['sinchi'] = np.sin(dict['chi'])
    dict['tanhalfchi'] = np.tan(dict['chi'] / 2.)
    dict['sechalfchi'] = 1. / np.cos(dict['chi'] / 2.)
    dict['LL'] = cas.DM([0.375, 0., 0., 0., -1., 0., 0., -1., 0.])
    dict['corr'] = 1. - init_options['xd']['a']

    var_type = 'xl'
    for name in struct_op.subkeys(model.variables, 'xl'):
        name_stripped, _ = struct_op.split_name_and_node_identifier(name)

        if 'psi' in name_stripped:
            V_init = set_azimuth_variables(V_init, init_options, name, model, nlp, tgrid_coll_x, level_siblings, omega_norm)

        elif name_stripped in dict.keys():
            V_init = tools_init.insert_dict(dict, var_type, name, name_stripped, V_init)

    return V_init





def get_gamma_angle(init_options):
    n_rot_hat = tools_init.get_ehat_tether(init_options)
    u_hat, _, _ = get_local_wind_reference_frame(init_options)
    gamma = vect_op.angle_between(u_hat, n_rot_hat)
    return gamma

def get_chi_angle(init_options, var_type):
    a = init_options['xd']['a']
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
