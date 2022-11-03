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

import awebox.mdl.aero.induction_dir.actuator_dir.actuator as actuator
import awebox.mdl.aero.induction_dir.vortex_dir.vortex as vortex

import pdb

def initial_guess_induction(init_options, nlp, model, V_init, p_fix_num):

    if actuator.model_is_included_in_comparison(init_options):
        V_init = initial_guess_actuator(init_options, nlp, model, V_init)

    if vortex.model_is_included_in_comparison(init_options):
        V_init = initial_guess_vortex(init_options, nlp, model, V_init, p_fix_num)

    return V_init


def initial_guess_vortex(init_options, nlp, model, V_init, p_fix_num):

    if not nlp.discretization == 'direct_collocation':
        message = 'vortex induction model is only defined for direct-collocation model, at this point'
        awelogger.logger.error(message)
        raise Exception(message)

    V_init = vortex.get_initialization(init_options, V_init, p_fix_num, nlp, model)

    return V_init











def initial_guess_actuator(init_options, nlp, model, V_init):
    V_init = initial_guess_actuator_x(init_options, model, V_init)
    V_init = initial_guess_actuator_z(init_options, model, V_init)
    V_init = set_azimuth_variables(V_init, init_options, model, nlp)

    return V_init


def initial_guess_actuator_x(init_options, model, V_init):

    dict = {}
    dict['a'] = cas.DM(init_options['x']['a'])
    dict['asin_uasym'] = cas.DM(0.)
    dict['acos_uasym'] = cas.DM(0.)
    dict['a_uaxi'] = dict['a']
    dict['a_uasym'] = dict['a']

    var_type = 'x'
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
    tgrid_x = nlp.time_grids['x'](time_final)
    tgrid_coll = nlp.time_grids['coll'](time_final)

    for ndx in range(nlp.n_k):

        t = tgrid_x[ndx]
        psi = tools_init.get_azimuthal_angle(t, init_options, level_siblings, kite, parent, omega_norm)

        if 'z' in list(V_init.keys()):
            V_init['z', ndx, 'psi' + str(kite_parent)] = psi

            if '[z,' + str(ndx) + ',cospsi' + str(kite_parent) + ',0]' in V_init.labels():
                V_init['z', ndx, 'cospsi' + str(kite_parent)] = np.cos(psi)
            if '[z,' + str(ndx) + ',sinpsi' + str(kite_parent) + ',0]' in V_init.labels():
                V_init['z', ndx, 'sinpsi' + str(kite_parent)] = np.sin(psi)

        for ddx in range(nlp.d):
            t = tgrid_coll[ndx, ddx]
            psi = tools_init.get_azimuthal_angle(t, init_options, level_siblings, kite, parent, omega_norm)
            V_init['coll_var', ndx, ddx, 'z', 'psi' + str(kite_parent)] = psi

            if '[coll_var,' + str(ndx) + ',' + str(ddx) + 'z,cospsi' + str(kite_parent) + ',0]' in V_init.labels():
                V_init['collvar', ndx, ddx, 'z', 'cospsi' + str(kite_parent)] = np.cos(psi)
            if '[coll_var,' + str(ndx) + ',' + str(ddx) + 'z,sinpsi' + str(kite_parent) + ',0]' in V_init.labels():
                V_init['collvar', ndx, ddx, 'z', 'sinpsi' + str(kite_parent)] = np.sin(psi)

    return V_init


def initial_guess_actuator_z(init_options, model, V_init):

    u_hat, v_hat, w_hat = get_local_wind_reference_frame(init_options)
    wind_dcm = cas.horzcat(u_hat, v_hat, w_hat)
    wind_dcm_cols = cas.reshape(wind_dcm, (9, 1))

    n_rot_hat, y_rot_hat, z_rot_hat = tools_init.get_rotor_reference_frame(init_options)
    act_dcm = cas.horzcat(n_rot_hat, y_rot_hat, z_rot_hat)
    act_dcm_cols = cas.reshape(act_dcm, (9, 1))

    b_ref = init_options['sys_params_num']['geometry']['b_ref']

    dict = {}
    dict['a'] = cas.DM(init_options['z']['a'])
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
    dict['u_vec_length'] = vect_op.norm(tools_init.get_wind_velocity(init_options))
    dict['gamma'] = get_gamma_angle(init_options)
    dict['g_vec_length'] = cas.DM(init_options['induction']['g_vec_length'])
    dict['cosgamma'] = np.cos(dict['gamma'])
    dict['singamma'] = np.sin(dict['gamma'])

    var_type = 'z'
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

