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
vortex model of awebox aerodynamics
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2019-21
'''

import casadi.tools as cas

import awebox.mdl.aero.induction_dir.vortex_dir.convection as convection
import awebox.mdl.aero.induction_dir.vortex_dir.flow as flow
import awebox.mdl.aero.induction_dir.vortex_dir.tools as tools
import awebox.mdl.aero.induction_dir.tools_dir.unit_normal as unit_normal
import awebox.mdl.aero.induction_dir.vortex_dir.filament_list as vortex_filament_list
import awebox.mdl.aero.induction_dir.vortex_dir.biot_savart as biot_savart
import awebox.tools.vector_operations as vect_op
import awebox.tools.constraint_operations as cstr_op
import awebox.tools.print_operations as print_op


from awebox.logger.logger import Logger as awelogger
import numpy as np

def get_vortex_cstr(options, wind, variables_si, architecture):

    vortex_representation = options['aero']['vortex']['representation']
    cstr_list = cstr_op.ConstraintList()

    if vortex_representation == 'state':
        state_conv_cstr = convection.get_state_repr_convection_cstr(options, wind, variables_si, architecture)
        cstr_list.append(state_conv_cstr)

    superposition_cstr = flow.get_superposition_cstr(options, wind, variables_si, architecture)
    cstr_list.append(superposition_cstr)

    return cstr_list

def get_induction_trivial_residual(options, wind, variables_si, architecture):

    resi = []

    filaments = vortex_filament_list.expected_number_of_filaments(options, architecture)
    u_ref = wind.get_velocity_ref()

    for kite_obs in architecture.kite_nodes:

        for fdx in range(filaments):
            u_ind_fil = cas.DM.zeros((3, 1))

            ind_name = 'wu_fil_' + str(fdx) + '_' + str(kite_obs)
            local_var = variables_si['xl'][ind_name]
            local_resi = (local_var - u_ind_fil) / u_ref
            resi = cas.vertcat(resi, local_resi)

    return resi


def get_induction_final_residual(options, wind, variables_si, outputs, architecture):

    resi = []

    # induced velocity residuals
    columnized_list = outputs['vortex']['filament_list']
    filament_list = vortex_filament_list.decolumnize(options, architecture, columnized_list)
    number_of_filaments = filament_list.shape[1]

    expected_number_of_filaments = vortex_filament_list.expected_number_of_filaments(options, architecture)
    if int(number_of_filaments) != int(expected_number_of_filaments):
        message = 'construction of vortex induction residual finds a number of filaments (' + \
                  str(number_of_filaments) + ') that is not the same as the expected ' \
                  'number of filaments (' + str(expected_number_of_filaments) + ')'
        awelogger.logger.error(message)
        raise Exception(message)

    u_ref = wind.get_velocity_ref()

    for kite_obs in architecture.kite_nodes:

        for fdx in range(number_of_filaments):
            # biot-savart of filament induction
            filament = filament_list[:, fdx]

            u_ind_fil = flow.get_induced_velocity_at_kite(options, filament, variables_si, architecture, kite_obs)

            ind_name = 'wu_fil_' + str(fdx) + '_' + str(kite_obs)
            local_var = variables_si['xl'][ind_name]

            scaler = 10.

            local_resi = (local_var - u_ind_fil) / u_ref / scaler
            resi = cas.vertcat(resi, local_resi)

    return resi

def collect_vortex_outputs(model_options, atmos, wind, variables_si, outputs, parameters, architecture):

    biot_savart.test()
    test_list = vortex_filament_list.test()
    flow.test(test_list)

    if 'vortex' not in list(outputs.keys()):
        outputs['vortex'] = {}

    filament_list = vortex_filament_list.get_list(model_options, variables_si, architecture)

    columnized_list = vortex_filament_list.columnize(filament_list)
    outputs['vortex']['filament_list'] = columnized_list

    last_filament_list = vortex_filament_list.get_last_list(model_options, variables_si, architecture)

    kite_nodes = architecture.kite_nodes
    for kite in kite_nodes:
        parent = architecture.parent_map[kite]

        u_ind = flow.get_induced_velocity_at_kite(model_options, filament_list, variables_si, architecture, kite)

        n_hat = unit_normal.get_n_hat(model_options, parent, variables_si, parameters, architecture)
        local_a = flow.get_induction_factor_at_kite(model_options, filament_list, wind, variables_si, parameters, architecture, kite, n_hat=n_hat)

        last_u_ind = flow.get_induced_velocity_at_kite(model_options, last_filament_list, variables_si, architecture, kite)
        last_u_ind_norm = vect_op.norm(last_u_ind)
        last_u_ind_norm_over_ref = last_u_ind_norm / wind.get_velocity_ref()

        est_truncation_error = (last_u_ind_norm) / vect_op.norm(u_ind)

        outputs['vortex']['u_ind' + str(kite)] = u_ind
        outputs['vortex']['u_ind_norm' + str(kite)] = vect_op.norm(u_ind)
        outputs['vortex']['local_a' + str(kite)] = local_a

        outputs['vortex']['last_u_ind' + str(kite)] = last_u_ind
        outputs['vortex']['last_u_ind_norm_over_ref' + str(kite)] = last_u_ind_norm_over_ref

        outputs['vortex']['est_truncation_error' + str(kite)] = est_truncation_error

    return outputs

def compute_global_performance(power_and_performance, plot_dict):

    kite_nodes = plot_dict['architecture'].kite_nodes

    max_est_trunc_list = []
    max_est_discr_list = []
    last_u_ind_norm_over_ref_list = []

    for kite in kite_nodes:
        trunc_name = 'est_truncation_error' + str(kite)
        local_max_est_trunc = np.max(np.array(plot_dict['outputs']['vortex'][trunc_name][0]))
        max_est_trunc_list += [local_max_est_trunc]

        kite_local_a = np.array(plot_dict['outputs']['vortex']['local_a' + str(kite)][0])
        max_kite_local_a = np.max(kite_local_a)
        min_kite_local_a = np.min(kite_local_a)
        local_max_est_discr = (max_kite_local_a - min_kite_local_a) / max_kite_local_a
        max_est_discr_list += [local_max_est_discr]

        local_last_u_ind_norm_over_ref = np.max(np.array(plot_dict['outputs']['vortex']['last_u_ind_norm_over_ref' + str(kite)]))
        last_u_ind_norm_over_ref_list += [local_last_u_ind_norm_over_ref]

    max_last_u_ind_norm_over_ref = np.max(np.array(last_u_ind_norm_over_ref_list))
    power_and_performance['vortex_max_last_u_ind_norm_over_ref'] = max_last_u_ind_norm_over_ref

    max_est_trunc = np.max(np.array(max_est_trunc_list))
    power_and_performance['vortex_max_est_truncation_error'] = max_est_trunc

    max_est_discr = np.max(np.array(max_est_discr_list))
    power_and_performance['vortex_max_est_discretization_error'] = max_est_discr

    return power_and_performance
