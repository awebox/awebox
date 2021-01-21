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
vortex model of awebox aerodynamics
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2019
'''

import casadi.tools as cas

import awebox.mdl.aero.induction_dir.vortex_dir.convection as convection
import awebox.mdl.aero.induction_dir.vortex_dir.flow as flow
import awebox.mdl.aero.induction_dir.vortex_dir.tools as tools
import awebox.tools.print_operations as print_op
import awebox.mdl.aero.induction_dir.tools_dir.unit_normal as unit_normal
import awebox.mdl.aero.induction_dir.vortex_dir.filament_list as vortex_filament_list
import awebox.mdl.aero.induction_dir.vortex_dir.biot_savart as biot_savart
import awebox.tools.vector_operations as vect_op

def get_residual(options, wind, variables_si, outputs, architecture):

    vortex_representation = options['aero']['vortex']['representation']

    resi = []

    if vortex_representation == 'state':
        state_conv_resi = convection.get_state_repr_convection_residual(options, wind, variables_si, architecture)
        resi = cas.vertcat(resi, state_conv_resi)

    # induced velocity residuals
    columnized_list = outputs['vortex']['filament_list']
    filament_list = vortex_filament_list.decolumnize(options, architecture, columnized_list)
    filaments = filament_list.shape[1]
    u_ref = wind.get_velocity_ref()

    for kite_obs in architecture.kite_nodes:
        u_ind_kite = cas.DM.zeros((3, 1))
        for fdx in range(filaments):
            ind_name = 'wu_fil_' + str(fdx) + '_' + str(kite_obs)
            local_var = variables_si['xl'][ind_name]
            u_ind_kite += local_var

        # superposition of filament induced velocities at kite
        ind_name = 'wu_ind_' + str(kite_obs)
        local_var = variables_si['xl'][ind_name]
        local_resi = (local_var - u_ind_kite) / u_ref
        resi = cas.vertcat(resi, local_resi)

    return resi

def get_induction_trivial_residual(options, wind, variables_si, outputs, architecture):
    resi = []

    # induced velocity residuals
    columnized_list = outputs['vortex']['filament_list']
    filament_list = vortex_filament_list.decolumnize(options, architecture, columnized_list)
    filaments = filament_list.shape[1]
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
    filaments = filament_list.shape[1]
    u_ref = wind.get_velocity_ref()

    for kite_obs in architecture.kite_nodes:

        for fdx in range(filaments):
            # biot-savart of filament induction
            filament = filament_list[:, fdx]

            u_ind_fil = flow.get_induced_velocity_at_kite(options, filament, variables_si, architecture, kite_obs)

            ind_name = 'wu_fil_' + str(fdx) + '_' + str(kite_obs)
            local_var = variables_si['xl'][ind_name]

            local_resi = (local_var - u_ind_fil) / u_ref
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

        u_ind_vortex = flow.get_induced_velocity_at_kite(model_options, filament_list, variables_si, architecture, kite)

        n_hat = unit_normal.get_n_hat(model_options, parent, variables_si, parameters, architecture)
        local_a = flow.get_induction_factor_at_kite(model_options, filament_list, wind, variables_si, parameters, architecture, kite, n_hat=n_hat)

        last_ui_norm = vect_op.norm(flow.get_induced_velocity_at_kite(model_options, last_filament_list, variables_si, architecture, kite))
        last_ui_norm_over_ref = last_ui_norm / wind.get_velocity_ref()

        outputs['vortex']['u_ind_vortex' + str(kite)] = u_ind_vortex
        outputs['vortex']['local_a' + str(kite)] = local_a
        outputs['vortex']['last_ui_norm_over_ref' + str(kite)] = last_ui_norm_over_ref

    return outputs


