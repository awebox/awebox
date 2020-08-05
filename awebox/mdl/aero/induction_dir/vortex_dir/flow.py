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
- author: rachel leuthold, alu-fr 2020
"""

import awebox.mdl.aero.induction_dir.general_dir.flow as general_flow
import awebox.mdl.aero.induction_dir.vortex_dir.biot_savart as biot_savart
import awebox.mdl.aero.induction_dir.vortex_dir.tools as vortex_tools

import awebox.mdl.aero.induction_dir.general_dir.geom as general_geom
import awebox.mdl.aero.induction_dir.actuator_dir.flow as actuator_flow

import awebox.tools.vector_operations as vect_op

import casadi.tools as cas
import awebox.mdl.aero.induction_dir.tools_dir.unit_normal as unit_normal
import pdb

def get_induced_velocity_at_kite(options, wind, variables, parameters, kite_obs, architecture, include_normal_info=False):

    n_k = options['aero']['vortex']['n_k']
    d = options['aero']['vortex']['d']
    periods_tracked = options['aero']['vortex']['periods_tracked']
    n_rings_per_kite = vortex_tools.get_number_of_rings_per_kite(n_k, d, periods_tracked)

    filament_list = []
    for kite in architecture.kite_nodes:
        for rdx in range(1, n_rings_per_kite):
            parent = architecture.parent_map[kite]
            local_list = vortex_tools.get_list_of_filaments_by_kite_and_ring(options, variables, wind, kite, parent, rdx).T
            filament_list = cas.horzcat(filament_list, local_list)

    segment_list = biot_savart.list_filaments_kiteobs_and_normal_info(filament_list, options, variables, parameters,
                                                                      kite_obs, architecture, include_normal_info)

    total_u_vec_ind = make_symbolic_filament_and_sum(segment_list, include_normal_info)
    return total_u_vec_ind



def get_induced_velocity_at_observer(point_obs, options, wind, variables, architecture, n_hat=None):

    n_k = options['aero']['vortex']['n_k']
    d = options['aero']['vortex']['d']
    periods_tracked = options['aero']['vortex']['periods_tracked']
    n_rings_per_kite = vortex_tools.get_number_of_rings_per_kite(n_k, d, periods_tracked)

    filament_list = []
    for kite in architecture.kite_nodes:
        for rdx in range(1, n_rings_per_kite):
            parent = architecture.parent_map[kite]
            local_list = vortex_tools.get_list_of_filaments_by_kite_and_ring(options, variables, wind, kite, parent, rdx).T
            filament_list = cas.horzcat(filament_list, local_list)

    segment_list = biot_savart.list_filament_observer_and_normal_info(point_obs, filament_list, options, n_hat=n_hat)

    if n_hat == None:
        include_normal_info = False
    else:
        include_normal_info = True

    total_u_vec_ind = make_symbolic_filament_and_sum(segment_list, include_normal_info=include_normal_info)

    return total_u_vec_ind



def make_symbolic_filament_and_sum(segment_list, include_normal_info=False):

    # define the symbolic function
    n_symbolics = segment_list.shape[0]
    seg_data_sym = cas.SX.sym('seg_data_sym', (n_symbolics, 1))

    if include_normal_info:
        filament_sym = biot_savart.filament_normal(seg_data_sym)
    else:
        filament_sym = biot_savart.filament(seg_data_sym)

    filament_fun = cas.Function('filament_fun', [seg_data_sym], [filament_sym])

    # evaluate the symbolic function
    total_u_vec_ind = vortex_tools.evaluate_symbolic_on_segments_and_sum(filament_fun, segment_list)

    return total_u_vec_ind


def get_induction_factor_at_kite(options, wind, variables, parameters, kite_obs, architecture):

    u_ind_kite_projected = get_induced_velocity_at_kite(options, wind, variables, parameters, kite_obs, architecture,
                                 include_normal_info=True)

    parent = architecture.parent_map[kite_obs]

    u_app_act = actuator_flow.get_uzero_vec(options, wind, parent, variables, architecture)
    u_mag = vect_op.smooth_norm(u_app_act)

    a_calc = -1. * u_ind_kite_projected / u_mag

    return a_calc


def get_induction_factor_at_observer(point_obs, options, wind, variables, parameters, parent_obs, architecture):

    n_vec_val = unit_normal.get_n_vec(options, parent_obs, variables, parameters, architecture)
    n_hat = vect_op.normalize(n_vec_val)

    u_ind_kite_projected = get_induced_velocity_at_observer(point_obs, options, wind, variables, architecture, n_hat=n_hat)

    u_app_act = actuator_flow.get_uzero_vec(options, wind, parent_obs, variables, architecture)
    u_mag = vect_op.smooth_norm(u_app_act)

    a_calc = -1. * u_ind_kite_projected / u_mag

    return a_calc


def get_last_induced_velocity_at_kite(options, wind, variables, parameters, kite_obs, architecture):

    n_k = options['aero']['vortex']['n_k']
    d = options['aero']['vortex']['d']
    periods_tracked = options['aero']['vortex']['periods_tracked']
    n_rings_per_kite = vortex_tools.get_number_of_rings_per_kite(n_k, d, periods_tracked)

    rdx = n_rings_per_kite - 1

    u_ind_kite = cas.DM.zeros((3,1))
    for kite in architecture.kite_nodes:
        u_ind_val = get_induced_velocity_at_kite_from_kite_and_ring(options, variables, parameters, wind,
                                                                    kite_obs, architecture,
                                                                    kite, rdx)
        u_ind_kite = u_ind_kite + u_ind_val

    return u_ind_kite


def get_last_induction_factor_at_kite(options, wind, variables, parameters, kite_obs, architecture):
    u_ind_kite = get_last_induced_velocity_at_kite(options, wind, variables, parameters, kite_obs, architecture)

    parent = architecture.parent_map[kite_obs]

    n_vec_val = unit_normal.get_n_vec(options, parent, variables, parameters, architecture)
    n_hat = vect_op.normalize(n_vec_val)

    u_app_act = actuator_flow.get_uzero_vec(options, wind, parent, variables, architecture)
    u_mag = vect_op.smooth_norm(u_app_act)

    a_calc = -1. * cas.mtimes(u_ind_kite.T, n_hat) / u_mag

    return a_calc



def get_induced_velocity_at_kite_from_kite_and_ring(options, variables, parameters, wind, kite_obs, architecture, kite, rdx):
    parent = architecture.parent_map[kite]
    filament_list = vortex_tools.get_list_of_filaments_by_kite_and_ring(options, variables, wind, kite, parent, rdx).T

    include_normal_info = False
    segment_list = biot_savart.list_filaments_kiteobs_and_normal_info(filament_list, options, variables, parameters,
                                                                      kite_obs, architecture, include_normal_info)

    total_u_vec_ind = make_symbolic_filament_and_sum(segment_list)
    return total_u_vec_ind


def get_induced_velocity_at_observer_from_kite_and_ring(point_obs, options, variables, wind, parent, kite, rdx):
    filament_list = vortex_tools.get_list_of_filaments_by_kite_and_ring(options, variables, wind, kite, parent, rdx).T
    segment_list = biot_savart.list_filament_observer_and_normal_info(point_obs, filament_list, options)
    total_u_vec_ind = make_symbolic_filament_and_sum(segment_list)
    return total_u_vec_ind
