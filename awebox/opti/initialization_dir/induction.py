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
import copy
from awebox.logger.logger import Logger as awelogger

import awebox.tools.vector_operations as vect_op
import awebox.tools.struct_operations as struct_op
import awebox.tools.print_operations as print_op

import awebox.opti.initialization_dir.tools as tools_init

import awebox.mdl.aero.induction_dir.vortex_dir.far_wake as vortex_filament_list
import awebox.mdl.aero.induction_dir.vortex_dir.flow as vortex_flow
import awebox.mdl.aero.induction_dir.vortex_dir.strength as vortex_strength
import awebox.mdl.aero.induction_dir.vortex_dir.fixing as vortex_fixing

import pdb

def initial_guess_induction(init_options, nlp, formulation, model, V_init, p_fix_num):

    comparison_labels = init_options['model']['comparison_labels']

    if comparison_labels:
        V_init = initial_guess_general(V_init)

    any_act = any(label[:3] == 'act' for label in comparison_labels)
    if any_act:
        V_init = initial_guess_actuator(init_options, nlp, model, V_init)

    any_vor = any(label[:3] == 'vor' for label in comparison_labels)
    if any_vor:
        V_init = initial_guess_vortex(init_options, nlp, model, V_init, p_fix_num)

    return V_init


def initial_guess_general(V_init):
    return V_init


def initial_guess_vortex(init_options, nlp, model, V_init, p_fix_num):

    if not nlp.discretization == 'direct_collocation':
        message = 'vortex induction model is only defined for direct-collocation model, at this point'
        awelogger.logger.error(message)
        raise Exception(message)

    if init_options['induction']['vortex_far_wake_model'] == 'pathwise_filament':
        V_init = set_far_wake_convection_velocity_initialization(init_options, V_init, model)

    if init_options['induction']['vortex_representation'] == 'state':
        V_init = set_state_vortex_repr_strength_initialization(init_options, nlp, model, V_init, p_fix_num)
        V_init = set_state_vortex_repr_position_initialization(init_options, nlp, model, V_init, p_fix_num)

    elif init_options['induction']['vortex_representation'] == 'alg':
        V_init = set_algebraic_vortex_repr_position_initialization(init_options, nlp, model, V_init, p_fix_num)
        V_init = set_algebraic_vortex_repr_strength_initialization(init_options, nlp, model, V_init, p_fix_num)

    # induced velocities initialized to zero, since induction homotopy at beginning of path

    return V_init











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
    dict['u_vec_length'] = vect_op.norm(tools_init.get_wind_velocity(init_options, V_init['xd', 0, 'q10'][2]))
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







######################## vortex

def set_far_wake_convection_velocity_initialization(init_options, V_init, model):

    n_k = init_options['n_k']

    kite_nodes = model.architecture.kite_nodes
    wingtips = ['ext', 'int']

    V_init_si = copy.deepcopy(V_init)
    V_init_scaled = struct_op.si_to_scaled(V_init_si, model.scaling)

    for kite in kite_nodes:
        for tip in wingtips:
            var_name = 'wu_farwake_' + str(kite) + '_' + tip

            for ndx in range(n_k):

                velocity = vortex_fixing.get_far_wake_velocity_val(init_options, V_init_scaled, model, kite, ndx)
                V_init['xl', ndx, var_name] = velocity

                for ddx in range(init_options['collocation']['d']):
                    velocity = vortex_fixing.get_far_wake_velocity_val(init_options, V_init_scaled, model, kite, ndx, ddx)
                    V_init['coll_var', ndx, ddx, 'xl', var_name] = velocity

    return V_init


def set_algebraic_vortex_repr_position_initialization(init_options, nlp, model, V_init, p_fix_num):

    n_k = nlp.n_k
    d = nlp.d

    wake_nodes = init_options['induction']['vortex_wake_nodes']
    kite_nodes = model.architecture.kite_nodes
    wingtips = ['ext', 'int']

    V_init_si = copy.deepcopy(V_init)
    V_init_scaled = struct_op.si_to_scaled(V_init_si, model.scaling)
    Outputs_init = nlp.Outputs(nlp.Outputs_fun(V_init_scaled, p_fix_num))

    for kite in kite_nodes:
        for tip in wingtips:
            for wake_node in range(wake_nodes):

                var_name = 'wx_' + str(kite) + '_' + tip + '_' + str(wake_node)

                for ndx in range(n_k):
                    for ddx in range(d):

                        wx_val = vortex_fixing.get_local_algebraic_repr_collocation_position_value(init_options, V_init_scaled, Outputs_init, model, nlp.time_grids, kite, tip, wake_node, ndx, ddx)
                        V_init['coll_var', ndx, ddx, 'xl', var_name] = wx_val


    V_init_si = copy.deepcopy(V_init)
    V_init_scaled = struct_op.si_to_scaled(V_init_si, model.scaling)

    for kite in kite_nodes:
        for tip in wingtips:
            for wake_node in range(wake_nodes):

                var_name = 'wx_' + str(kite) + '_' + tip + '_' + str(wake_node)

                for ndx in range(n_k):

                    wx_val = vortex_fixing.get_local_algebraic_repr_shooting_position_value(V_init_scaled, model, kite, tip, wake_node, ndx)
                    V_init['xl', ndx, var_name] = wx_val

    return V_init



def set_state_vortex_repr_position_initialization(init_options, nlp, model, V_init, p_fix_num):
    n_k = nlp.n_k

    wake_nodes = init_options['induction']['vortex_wake_nodes']
    kite_nodes = model.architecture.kite_nodes
    wingtips = ['ext', 'int']

    time_final = init_options['precompute']['time_final']
    tgrid_xd = nlp.time_grids['x'](time_final)
    tgrid_coll = nlp.time_grids['coll'](time_final)

    V_init_si = copy.deepcopy(V_init)
    V_init_scaled = struct_op.si_to_scaled(V_init_si, model.scaling)
    Outputs_init = nlp.Outputs(nlp.Outputs_fun(V_init_scaled, p_fix_num))

    for kite in kite_nodes:
        for tip in wingtips:
            for wake_node in range(wake_nodes):

                if wake_node < n_k:
                    shooting_ndx = n_k - wake_node
                    collocation_ndx = shooting_ndx - 1

                    wx_fixing = Outputs_init[
                        'coll_outputs', collocation_ndx, -1, 'aerodynamics', 'wingtip_' + tip + str(kite)]
                    fixing_time = tgrid_coll[collocation_ndx, -1]

                    var_name_local = 'wx_' + str(kite) + '_' + str(tip) + '_' + str(wake_node)

                else:

                    wake_node_upstream = wake_node - n_k
                    var_name_upsteam = 'wx_' + str(kite) + '_' + str(tip) + '_' + str(wake_node_upstream)

                    wx_fixing = V_init['xd', -1, var_name_upsteam]
                    fixing_time = tgrid_xd[0]

                    var_name_local = 'wx_' + str(kite) + '_' + str(tip) + '_' + str(wake_node)

                for sdx in range(n_k):
                    for ddx in range(nlp.d):
                        local_time = tgrid_coll[sdx, ddx]
                        delta_t = local_time - fixing_time
                        vec_u = tools_init.get_wind_velocity(init_options, wx_fixing[2])
                        wx_local = wx_fixing + vec_u * delta_t
                        V_init['coll_var', sdx, ddx, 'xd', var_name_local] = wx_local

                for sdx in range(n_k + 1):
                    local_time = tgrid_xd[sdx]
                    delta_t = local_time - fixing_time
                    vec_u = tools_init.get_wind_velocity(init_options, wx_fixing[2])
                    wx_local = wx_fixing + vec_u * delta_t
                    V_init['xd', sdx, var_name_local] = wx_local

                    if sdx < n_k:
                        V_init['xddot', sdx, 'd' + var_name_local] = vec_u

    return V_init


def set_algebraic_vortex_repr_strength_initialization(init_options, nlp, model, V_init, p_fix_num):
    n_k = nlp.n_k
    d = nlp.d

    wake_nodes = init_options['induction']['vortex_wake_nodes']
    kite_nodes = model.architecture.kite_nodes
    rings = wake_nodes

    V_init_si = copy.deepcopy(V_init)
    V_init_scaled = struct_op.si_to_scaled(V_init_si, model.scaling)
    Outputs_init = nlp.Outputs(nlp.Outputs_fun(V_init_scaled, p_fix_num))

    for kite in kite_nodes:
        for ring in range(rings):

            var_name = 'wg_' + str(kite) + '_' + str(ring)

            for ndx in range(n_k):
                for ddx in range(d):
                    gamma_val = vortex_strength.get_local_algebraic_repr_collocation_strength_val(init_options,
                                                                                                  Outputs_init, kite,
                                                                                                  ring, ndx, ddx)
                    V_init['coll_var', ndx, ddx, 'xl', var_name] = gamma_val

    V_init_si = copy.deepcopy(V_init)
    V_init_scaled = struct_op.si_to_scaled(V_init_si, model.scaling)

    for kite in kite_nodes:
        for ring in range(rings):

            var_name = 'wg_' + str(kite) + '_' + str(ring)

            for ndx in range(n_k):
                gamma_val = vortex_strength.get_local_algebraic_repr_shooting_strength_val(V_init_scaled, model, kite,
                                                                                           ring, ndx)
                V_init['xl', ndx, var_name] = gamma_val

    return V_init


def set_state_vortex_repr_strength_initialization(init_options, nlp, model, V_init, p_fix_num):
    n_k = nlp.n_k
    d = nlp.d

    wake_nodes = init_options['induction']['vortex_wake_nodes']
    kite_nodes = model.architecture.kite_nodes
    rings = wake_nodes

    V_init_si = copy.deepcopy(V_init)
    V_init_scaled = struct_op.si_to_scaled(V_init_si, model.scaling)
    Outputs_init = nlp.Outputs(nlp.Outputs_fun(V_init_scaled, p_fix_num))

    for kite in kite_nodes:
        for ring in range(rings):

            var_name = 'wg_' + str(kite) + '_' + str(ring)

            for ndx in range(n_k):
                for ddx in range(d):
                    gamma_val = vortex_strength.get_local_state_repr_collocation_strength_val(init_options,
                                                                                              Outputs_init, kite, ring,
                                                                                              ndx, ddx)
                    V_init['coll_var', ndx, ddx, 'xl', var_name] = gamma_val

    V_init_si = copy.deepcopy(V_init)
    V_init_scaled = struct_op.si_to_scaled(V_init_si, model.scaling)
    Outputs_init = nlp.Outputs(nlp.Outputs_fun(V_init_scaled, p_fix_num))


    for kite in kite_nodes:
        for ring in range(rings):

            var_name = 'wg_' + str(kite) + '_' + str(ring)

            for ndx in range(n_k):
                if ndx == 0:
                    gamma_val = vortex_strength.get_local_state_repr_collocation_strength_val(init_options,
                                                                                              Outputs_init, kite, ring)
                    V_init['xl', ndx, var_name] = gamma_val
                else:
                    gamma_val = V_init['coll_var', ndx - 1, -1, 'xl', var_name]
                    V_init['xl', ndx, var_name] = gamma_val

    return V_init

