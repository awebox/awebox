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
import awebox.mdl.aero.geometry_dir.unit_normal as unit_normal
import awebox.mdl.aero.induction_dir.vortex_dir.vortex as vortex


import awebox.mdl.aero.induction_dir.vortex_dir.tools as vortex_tools

def initial_guess_induction(init_options, nlp, model, V_init_si, p_fix_num):

    V_init_si = initial_guess_general(model, V_init_si)

    if actuator.model_is_included_in_comparison(init_options):
        V_init_si = initial_guess_actuator(init_options, nlp, model, V_init_si)

    if vortex.model_is_included_in_comparison(init_options):
        V_init_si = initial_guess_vortex(init_options, nlp, model, V_init_si, p_fix_num)

    return V_init_si


def initial_guess_general(model, V_init):

    dict = {}
    dict['ui'] = cas.DM.zeros((3, 1))  # remember that induction homotopy has not yet begun.

    for var_type in ['x', 'z']:
        for name in struct_op.subkeys(model.variables, var_type):
            name_stripped, _ = struct_op.split_name_and_node_identifier(name)
            if name_stripped in dict.keys():
                V_init = tools_init.insert_dict(dict, var_type, name, name_stripped, V_init)

    return V_init


def initial_guess_vortex(init_options, nlp, model, V_init_si, p_fix_num):

    if not nlp.discretization == 'direct_collocation':
        message = 'vortex induction model is only defined for direct-collocation model, at this point'
        print_op.log_and_raise_error(message)

    print_op.base_print('Initializing vortex variables...')

    V_init_si = vortex.get_initialization(init_options, V_init_si, p_fix_num, nlp, model)

    return V_init_si


def initial_guess_actuator(init_options, nlp, model, V_init):
    V_init = initial_guess_actuator_a_values(init_options, model, V_init)
    V_init = initial_guess_actuator_support(init_options, model, V_init)
    V_init = set_azimuth_variables(V_init, init_options, model, nlp)

    sanity_check_actuator_variables(init_options, model, nlp, V_init, epsilon=1.e-5)

    return V_init


def sanity_check_actuator_variables(init_options, model, nlp, V_init, epsilon=1.e-5):
    Xdot = struct_op.construct_Xdot_struct(init_options, model.variables_dict)(0.)
    for ndx in [2, nlp.n_k-2]:  # spot check
        if ndx in range(nlp.n_k):
            variables = struct_op.get_variables_at_time(init_options, V_init, Xdot, model.variables, ndx, ddx=None)
            actuator.sanity_check(init_options, variables, model.wind, model.architecture, epsilon)

    return None


def initial_guess_actuator_a_values(init_options, model, V_init):

    a_ref = cas.DM(init_options['z']['a'])

    dict = {}

    dict['local_a'] = cas.DM(a_ref)
    for label in ['qaxi', 'qasym', 'uaxi', 'uasym']:
        dict['a_' + label] = cas.DM(a_ref)
        for a_name in ['acos', 'asin']:
            dict[a_name + '_' + label] = cas.DM(0.)

    for var_type in ['x', 'z']:
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

            label_start = '[z,' + str(ndx) + ','
            repdict = {'psi': psi, 'cospsi': np.cos(psi), 'sinpsi': np.sin(psi)}
            for name, val in repdict.items():
                if label_start + name + str(kite_parent) + ',0]' in V_init.labels():
                    V_init['z', ndx, name + str(kite_parent)] = val

        for ddx in range(nlp.d):
            t = tgrid_coll[ndx, ddx]
            psi = tools_init.get_azimuthal_angle(t, init_options, level_siblings, kite, parent, omega_norm)

            label_start = '[coll_var,' + str(ndx) + ',' + str(ddx) + ',z,'
            repdict = {'psi': psi, 'cospsi': np.cos(psi), 'sinpsi': np.sin(psi)}

            for name, val in repdict.items():
                if label_start + name + str(kite_parent) + ',0]' in V_init.labels():
                    V_init['coll_var', ndx, ddx, 'z', name + str(kite_parent)] = val

    return V_init


def initial_guess_actuator_support(init_options, model, V_init):

    u_hat, v_hat, w_hat = get_local_wind_reference_frame(init_options)
    wind_dcm = cas.horzcat(u_hat, v_hat, w_hat)
    wind_dcm_cols = cas.reshape(wind_dcm, (9, 1))

    b_ref = init_options['sys_params_num']['geometry']['b_ref']

    dict = {}

    for parent in model.architecture.layer_nodes:

        Xdot = struct_op.construct_Xdot_struct(init_options, model.variables_dict)(0.)
        variables = struct_op.get_variables_at_time(init_options, V_init, Xdot, model.variables, 0)
        n_hat_act = unit_normal.get_n_vec(init_options, parent, variables, model.architecture, model.scaling)
        z_hat_act = w_hat
        y_hat_act = vect_op.normed_cross(z_hat_act, n_hat_act)
        act_dcm = cas.horzcat(n_hat_act, y_hat_act, z_hat_act)
        act_dcm_cols = cas.reshape(act_dcm, (9, 1))

        dict['act_dcm'] = act_dcm_cols
        dict['wind_dcm'] = wind_dcm_cols
        dict['n_vec_length'] = cas.DM(init_options['induction']['n_vec_length'])

        if parent == 0:
            # TODO: rocking mode : define q1 of tether attachment node in the model, and choose between arm or fixed
            # Is arm length available here ? What about arm angle ?
            parent_position = np.zeros((3, 1))
        else:
            grandparent = model.architecture.parent_map[parent]
            parent_position = V_init['x', 0, 'q' + str(parent) + str(grandparent)]

        height = init_options['precompute']['height']
        n_rot_hat = tools_init.get_ehat_tether(init_options)
        center = parent_position + n_rot_hat * height

        dict['act_q' + str(parent)] = center
        dict['act_dq' + str(parent)] = cas.DM.zeros((3, 1))
        dict['u_vec_length' + str(parent)] = vect_op.norm(model.wind.get_velocity(center[2]))

        uzero_hat = model.wind.get_wind_direction()
        gamma = vect_op.angle_between(n_rot_hat, uzero_hat)
        u_comp = cas.mtimes(n_rot_hat.T, uzero_hat)
        g_vec_length = u_comp / cas.cos(gamma)

        dict['gamma' + str(parent)] = gamma
        dict['g_vec_length' + str(parent)] = g_vec_length
        dict['cosgamma' + str(parent)] = np.cos(gamma)
        dict['singamma' + str(parent)] = np.sin(gamma)

        dict['varrho' + str(parent)] = cas.DM(init_options['precompute']['radius'] / b_ref)
        dict['bar_varrho' + str(parent)] = dict['varrho' + str(parent)]
        dict['area' + str(parent)] = 2. * np.pi * init_options['precompute']['radius'] * b_ref

        q_infty = init_options['induction']['dynamic_pressure']
        a_ref = cas.DM(init_options['z']['a'])
        dict['thrust' + str(parent)] = 4. * a_ref * (1. - a_ref) * dict['area' + str(parent)] * q_infty

    var_type = 'z'
    for name in struct_op.subkeys(model.variables, var_type):
        name_stripped, _ = struct_op.split_name_and_node_identifier(name)

        if name in dict.keys():
            V_init = tools_init.insert_dict(dict, var_type, name, name, V_init)
        elif name_stripped in dict.keys():
            V_init = tools_init.insert_dict(dict, var_type, name, name_stripped, V_init)

    return V_init


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

