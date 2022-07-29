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
"""
flow functions for the vortex based model
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2020-21
"""
import pdb

from awebox.logger.logger import Logger as awelogger
import casadi.tools as cas

import awebox.mdl.aero.induction_dir.tools_dir.flow as general_flow

import awebox.tools.print_operations as print_op
import awebox.tools.vector_operations as vect_op
import awebox.tools.constraint_operations as cstr_op



def get_superposition_cstr(options, wind, variables_si, objects, architecture):

    cstr_list = cstr_op.ConstraintList()

    u_ref = wind.get_speed_ref()

    for kite_obs in architecture.kite_nodes:
        u_ind_kite = cas.DM.zeros((3, 1))

        for elem_list_name in objects.keys():
            elem_list = objects[elem_list_name]

            for fdx in range(elem_list.number_of_elements):
                ind_name = 'wu_' + elem_list_name + '_' + str(fdx) + '_' + str(kite_obs)
                local_var = variables_si['xl'][ind_name]
                u_ind_kite += local_var

        # superposition of filament induced velocities at kite
        ind_name = 'wu_ind_' + str(kite_obs)
        local_var = variables_si['xl'][ind_name]
        local_resi = (local_var - u_ind_kite) / u_ref

        local_cstr = cstr_op.Constraint(expr=local_resi,
                                        name='superposition_' + str(kite_obs),
                                        cstr_type='eq')
        cstr_list.append(local_cstr)

    return cstr_list

def get_induced_velocity_at_kite(variables_si, vortex_objects, kite_obs, selection='all_wake'):

    u_ind_kite = cas.DM.zeros((3, 1))

    if selection == 'all_wake':
        ind_name = 'wu_ind_' + str(kite_obs)
        u_ind_kite = variables_si['xl'][ind_name]

    elif selection == 'far_wake':
        for elem_list_name in vortex_objects.keys():

            if 'near' not in elem_list_name:
                elem_list = vortex_objects[elem_list_name]

                for fdx in range(elem_list.number_of_elements):
                    ind_name = 'wu_' + elem_list_name + '_' + str(fdx) + '_' + str(kite_obs)
                    local_var = variables_si['xl'][ind_name]
                    u_ind_kite += local_var

    elif selection == 'near_wake':
        for elem_list_name in vortex_objects.keys():

            if 'near' in elem_list_name:
                elem_list = vortex_objects[elem_list_name]

                for fdx in range(elem_list.number_of_elements):
                    ind_name = 'wu_' + elem_list_name + '_' + str(fdx) + '_' + str(kite_obs)
                    local_var = variables_si['xl'][ind_name]
                    u_ind_kite += local_var

    return u_ind_kite


def get_induced_velocity_at_observer(vortex_objects, x_obs, n_hat=None):

    u_ind = cas.DM.zeros((3, 1))
    for elem_list_name in vortex_objects.keys():
        elem_list = vortex_objects[elem_list_name]
        u_ind += cas.sum2(elem_list.evaluate_biot_savart_induction_for_all_elements(x_obs=x_obs, n_hat=n_hat))

    return u_ind

def get_induction_factor_at_kite(options, wind, variables_si, vortex_objects, architecture, kite_obs, n_hat=vect_op.xhat()):

    u_ind = get_induced_velocity_at_kite(variables_si, vortex_objects, kite_obs)
    u_projected = cas.mtimes(u_ind.T, n_hat)

    parent = architecture.parent_map[kite_obs]
    a_calc = compute_induction_factor(options, u_projected, wind, parent, variables_si, architecture)

    return a_calc

def compute_induction_factor(options, u_projected, wind, parent, variables, architecture):

    induction_factor_normalizing_speed = options['aero']['vortex']['induction_factor_normalizing_speed']
    if induction_factor_normalizing_speed == 'u_zero':
        u_vec = general_flow.get_uzero_vec(options, wind, parent, variables, architecture)
    elif induction_factor_normalizing_speed == 'u_inf':
        u_vec = general_flow.get_kite_uinfy_vec(variables, wind, kite, parent)
    elif induction_factor_normalizing_speed == 'u_ref':
        u_vec = wind.get_speed_ref()
    else:
        message = 'desired induction_factor_normalizing_speed (' + induction_factor_normalizing_speed + ') is not yet available'
        awelogger.logger.error(message)
        raise Exception(message)
    u_normalizing = vect_op.smooth_norm(u_vec)

    a_calc = -1. * u_projected / u_normalizing

    return a_calc

def get_induction_factor_at_observer(options, wind, parent, variables, architecture, vortex_objects, x_obs, u_zero, n_hat=vect_op.xhat()):
    u_projected = get_induced_velocity_at_observer(vortex_objects, x_obs, n_hat=None)
    a_calc = compute_induction_factor(options, u_projected, wind, parent, variables, architecture)
    return a_calc

def test(test_list):

    print_op.warn_about_temporary_funcationality_removal(location='vortex_flow.test')
    #
    # options = {}
    # options['induction'] = {}
    # options['induction']['vortex_core_radius'] = 0.
    #
    # x_obs = 0.5 * vect_op.xhat_np()
    #
    # u_ind = get_induced_velocity_at_observer(options, test_list, x_obs)
    #
    # xhat_component = cas.mtimes(u_ind.T, vect_op.xhat())
    # if not (xhat_component == 0):
    #     message = 'induced velocity at observer does not work as expected. ' \
    #               'test u_ind component in plane of QSVR (along xhat) is ' + str(xhat_component)
    #     awelogger.logger.error(message)
    #     raise Exception(message)
    #
    # yhat_component = cas.mtimes(u_ind.T, vect_op.yhat())
    # if not (yhat_component == 0):
    #     message = 'induced velocity at observer does not work as expected. ' \
    #               'test u_ind component in plane of QSVR (along yhat) is ' + str(yhat_component)
    #     awelogger.logger.error(message)
    #     raise Exception(message)
    #
    # zhat_component = cas.mtimes(u_ind.T, vect_op.zhat())
    # sign_along_zhat = vect_op.sign(zhat_component)
    # sign_comparison = (sign_along_zhat - (-1))**2.
    # if not (sign_comparison < 1.e-8):
    #     message = 'induced velocity at observer does not work as expected. ' \
    #               'sign of test u_ind component out-of-plane of QSVR (projected on zhat) is ' + str(sign_along_zhat)
    #     awelogger.logger.error(message)
    #     raise Exception(message)
    #
    # calculated_norm = vect_op.norm(u_ind)
    # expected_norm = 0.675237 #0.752133
    # norm_comparison = (calculated_norm - expected_norm)**2.
    # if not (norm_comparison < 1.e-8):
    #     message = 'induced velocity at observer does not work as expected. ' \
    #               'squared difference of norm of test u_ind vector is ' + str(norm_comparison)
    #     awelogger.logger.error(message)
    #     raise Exception(message)

    return None