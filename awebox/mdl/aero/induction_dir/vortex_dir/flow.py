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
"""
flow functions for the vortex based model
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2017-20
- edit: jochem de schutter, alu-fr 2019
"""

import awebox.mdl.aero.induction_dir.general_dir.flow as general_flow
import awebox.mdl.aero.induction_dir.vortex_dir.biot_savart as biot_savart
import awebox.mdl.aero.induction_dir.vortex_dir.tools as vortex_tools

import awebox.mdl.aero.induction_dir.general_dir.geom as general_geom
import awebox.mdl.aero.induction_dir.actuator_dir.flow as actuator_flow

import awebox.tools.vector_operations as vect_op

import casadi.tools as cas

import pdb


def get_kite_effective_velocity(options, variables, wind, kite_obs, architecture):

    parent = architecture.parent_map[kite_obs]
    u_app_kite = general_flow.get_kite_apparent_velocity(variables, wind, kite_obs, parent)

    u_ind_kite = get_induced_velocity_at_kite(options, wind, variables, kite_obs, architecture)

    u_eff_kite = u_app_kite + u_ind_kite

    return u_eff_kite



def get_induced_velocity_at_kite(options, wind, variables, kite_obs, architecture):

    parent = architecture.parent_map[kite_obs]

    n_k = options['aero']['vortex']['n_k']
    d = options['aero']['vortex']['d']
    periods_tracked = options['aero']['vortex']['periods_tracked']
    n_rings_per_kite = vortex_tools.get_number_of_rings_per_kite(n_k, d, periods_tracked)

    u_ind_kite = cas.DM.zeros((3,1))
    for kite in architecture.kite_nodes:
        for rdx in range(1, n_rings_per_kite):
            u_ind_val = get_induced_velocity_at_kite_from_kite_and_ring(options, variables, wind, kite_obs, parent,
                                                                        kite, rdx)
            u_ind_kite = u_ind_kite + u_ind_val

    return u_ind_kite


def get_induction_factor_at_kite(options, wind, variables, kite_obs, architecture):

    u_ind_kite = get_induced_velocity_at_kite(options, wind, variables, kite_obs, architecture)

    parent = architecture.parent_map[kite_obs]
    n_hat = general_geom.get_n_hat_var(variables, parent)

    u_app_act = actuator_flow.get_uzero_vec(options, wind, parent, variables, architecture)
    u_mag = vect_op.smooth_norm(u_app_act)

    a_calc = -1. * cas.mtimes(u_ind_kite.T, n_hat) / u_mag

    return a_calc


def get_last_induced_velocity_at_kite(options, wind, variables, kite_obs, architecture):
    parent = architecture.parent_map[kite_obs]

    n_k = options['aero']['vortex']['n_k']
    d = options['aero']['vortex']['d']
    periods_tracked = options['aero']['vortex']['periods_tracked']
    n_rings_per_kite = vortex_tools.get_number_of_rings_per_kite(n_k, d, periods_tracked)

    rdx = n_rings_per_kite - 1

    u_ind_kite = cas.DM.zeros((3,1))
    for kite in architecture.kite_nodes:
        u_ind_val = get_induced_velocity_at_kite_from_kite_and_ring(options, variables, wind, kite_obs, parent,
                                                                    kite, rdx)
        u_ind_kite = u_ind_kite + u_ind_val

    return u_ind_kite


def get_last_induction_factor_at_kite(options, wind, variables, kite_obs, architecture):
    u_ind_kite = get_last_induced_velocity_at_kite(options, wind, variables, kite_obs, architecture)

    parent = architecture.parent_map[kite_obs]
    n_hat = general_geom.get_n_hat_var(variables, parent)

    u_app_act = actuator_flow.get_uzero_vec(options, wind, parent, variables, architecture)
    u_mag = vect_op.smooth_norm(u_app_act)

    a_calc = -1. * cas.mtimes(u_ind_kite.T, n_hat) / u_mag

    return a_calc


def get_residuals(options, variables, wind, architecture):

    resi = []
    for kite_obs in architecture.kite_nodes:
        parent = architecture.parent_map[kite_obs]

        n_k = options['aero']['vortex']['n_k']
        d = options['aero']['vortex']['d']
        periods_tracked = options['aero']['vortex']['periods_tracked']
        n_rings_per_kite = vortex_tools.get_number_of_rings_per_kite(n_k, d, periods_tracked)

        for kite in architecture.kite_nodes:
            for rdx in range(1, n_rings_per_kite):
                new_resi = get_residual_at_kite_from_kite_and_ring(options, variables, wind, kite_obs, parent, kite, rdx)
                resi = cas.vertcat(resi, new_resi)

    return resi


def get_residual_at_kite_from_kite_and_ring(options, variables, wind, kite_obs, parent, kite, rdx):
    u_ind_val = get_induced_velocity_at_kite_from_kite_and_ring(options, variables, wind, kite_obs, parent, kite, rdx)

    u_ind_var = variables['xl']['w_ind_' + str(kite_obs) + '_' + str(kite) + '_' + str(rdx)]

    resi_unscaled = u_ind_val - u_ind_var

    scale = wind.get_velocity_ref()
    resi = resi_unscaled / scale

    return resi


def get_induced_velocity_at_kite_from_kite_and_ring(options, variables, wind, kite_obs, parent, kite, rdx):
    filament_list = vortex_tools.get_list_of_filaments_by_kite_and_ring(options, variables, wind, kite, parent, rdx).T

    include_normal_info = False
    segment_list = biot_savart.get_biot_savart_segment_list(filament_list, options, variables, kite_obs, parent,
                                                include_normal_info)

    # define the symbolic function
    n_symbolics = segment_list.shape[0]
    seg_data_sym = cas.SX.sym('seg_data_sym', (n_symbolics, 1))
    filament_sym = biot_savart.filament(seg_data_sym)
    filament_fun = cas.Function('filament_fun', [seg_data_sym], [filament_sym])

    # evaluate the symbolic function
    total_u_vec_ind = vortex_tools.evaluate_symbolic_on_segments_and_sum(filament_fun, segment_list)

    return total_u_vec_ind

