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

import pdb

import casadi.tools as cas
import numpy as np
import matplotlib.pyplot as plt

import awebox.mdl.aero.induction_dir.vortex_dir.tools as vortex_tools

import awebox.mdl.aero.induction_dir.vortex_dir.state_repr_dir.bound_wake as alg_bound_wake
import awebox.mdl.aero.induction_dir.vortex_dir.state_repr_dir.near_wake as alg_near_wake
import awebox.mdl.aero.induction_dir.vortex_dir.state_repr_dir.far_wake as alg_far_wake
import awebox.mdl.aero.induction_dir.vortex_dir.state_repr_dir.fixing as alg_fixing

import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.wake as obj_wake

import awebox.tools.vector_operations as vect_op
import awebox.tools.constraint_operations as cstr_op
import awebox.ocp.ocp_constraint as ocp_constraint
import awebox.tools.print_operations as print_op

from awebox.logger.logger import Logger as awelogger


def get_wake(options, architecture, wind, variables_si, parameters):

    tools.check_positive_vortex_wake_nodes(options)

    wake_dict = None
    print_op.warn_about_temporary_functionality_removal(location='state_representation')
    #
    # variables_scaled = system_variables['scaled']
    # variables_si = system_variables['SI']
    #
    # near_wake = vortex_near_wake.get_list(options, variables_si, parameters, architecture, wind)
    # far_wake_filaments, tangential_cylinder, longitudinal_cylinder = vortex_far_wake.get_lists(options, variables_si, parameters, architecture, wind)
    #
    # vortex_lists = {}
    #
    # if near_wake.number_of_elements > 0:
    #     vortex_lists['near_fil'] = near_wake
    #
    # if far_wake_filaments.number_of_elements > 0:
    #     vortex_lists['far_fil'] = far_wake_filaments
    #
    # if tangential_cylinder.number_of_elements > 0:
    #     vortex_lists['tan_cyl'] = tangential_cylinder
    #
    # if longitudinal_cylinder.number_of_elements > 0:
    #     vortex_lists['long_cyl'] = longitudinal_cylinder
    #
    # for key in vortex_lists.keys():
    #     elem_list = vortex_lists[key]
    #     elem_list.confirm_list_has_expected_dimensions()
    #     elem_list.define_model_variables_to_info_function(variables_scaled, parameters)
    #     elem_list.define_biot_savart_induction_function()

    return wake_dict
#
# def precompute_vortex_functions(options, system_variables, parameters, elem_list):
#
#     variables_scaled = system_variables['scaled']
#     variables_si = system_variables['SI']
#
#     x_obs_sym = cas.SX.sym('x_obs_sym', (3, 1))
#     n_hat_sym = cas.SX.sym('n_hat_sym', (3, 1))
#
#     #     u_ind_unprojected = self.evaluate_total_biot_savart_induction(x_obs=x_obs_sym, n_hat=None)
#     #     model_induction_fun = cas.Function('model_induction_fun', [variables_scaled, parameters, x_obs_sym], [u_ind_unprojected])
#     #     self.__model_induction_fun = model_induction_fun
#     #
#     #     u_ind_projected = cas.mtimes(n_hat_sym.T, u_ind_unprojected)
#     #     model_projected_induction_fun = cas.Function('model_projected_induction_fun', [variables_scaled, parameters, x_obs_sym, n_hat_sym], [u_ind_projected])
#     #     self.__model_projected_induction_fun = model_projected_induction_fun
#     #
#     #     return None
#
#
#     if precompute_model_induction_fun:
#         u_ind = flow.get_induced_velocity_at_observer(vortex_objects, x_obs_sym, n_hat=None)
#         model_induction_fun = cas.Function('model_induction_fun', [variables_scaled, parameters, x_obs_sym], [u_ind])
#         elem_list.set_model_induction_fun(model_induction_fun)
#
#     if precompute_model_projected_induction_fun:
#         u_ind_proj = flow.get_induced_velocity_at_observer(vortex_objects, x_obs_sym, n_hat=n_hat_sym)
#         model_projected_induction_fun = cas.Function('model_projected_induction_fun', [variables_scaled, parameters, x_obs_sym, n_hat_sym], [u_ind_proj])
#         elem_list.set_model_projected_induction_fun(model_projected_induction_fun)
#
#     if precompute_model_induction_factor_fun:
#         a_calc = flow.get_induction_factor_at_observer()
#
# def get_vortex_cstr(options, wind, variables_si, parameters, objects, architecture):
#
#     vortex_representation = options['aero']['vortex']['representation']
#     cstr_list = cstr_op.ConstraintList()
#
#     if vortex_representation == 'state':
#         state_conv_cstr = state_convection.get_state_repr_convection_cstr(options, wind, variables_si, architecture)
#         cstr_list.append(state_conv_cstr)
#
#     superposition_cstr = flow.get_superposition_cstr(options, wind, variables_si, objects, architecture)
#     cstr_list.append(superposition_cstr)
#
#     vortex_far_wake_model = options['aero']['vortex']['far_wake_model']
#     if ('cylinder' in vortex_far_wake_model):
#         radius_cstr = vortex_far_wake.get_cylinder_radius_cstr(options, wind, variables_si, parameters, architecture)
#         cstr_list.append(radius_cstr)
#
#     return cstr_list
#
# def get_induction_trivial_residual(options, wind, variables_si, architecture, objects):
#
#     resi = []
#
#     u_ref = wind.get_speed_ref()
#
#     for kite_obs in architecture.kite_nodes:
#
#         for elem_list_name in objects.keys():
#             elem_list = objects[elem_list_name]
#             number_elements = elem_list.number_of_elements
#             if number_elements > 0:
#                 for fdx in range(number_elements):
#                     u_ind_fil = cas.DM.zeros((3, 1))
#
#                     ind_name = 'wu_' + elem_list_name + '_' + str(fdx) + '_' + str(kite_obs)
#                     local_var = variables_si['xl'][ind_name]
#                     local_resi = (local_var - u_ind_fil) / u_ref
#                     resi = cas.vertcat(resi, local_resi)
#
#     return resi
#
#
# def get_induction_final_residual(options, wind, variables_si, outputs, architecture, objects):
#
#     vortex_far_wake_model = options['aero']['vortex']['far_wake_model']
#     repetitions = options['aero']['vortex']['repetitions']
#
#     resi = []
#
#     u_ref = wind.get_speed_ref()
#
#     for kite_obs in architecture.kite_nodes:
#         parent_obs = architecture.parent_map[kite_obs]
#
#         for elem_list_name in objects.keys():
#             elem_list = objects[elem_list_name]
#
#             x_obs = variables_si['xd']['q' + str(kite_obs) + str(parent_obs)]
#
#             if vortex_far_wake_model == 'repetition':
#                 all_biot_savarts = cas.DM.zeros(3, elem_list.number_of_elements)
#                 for pdx in range(repetitions):
#                     all_biot_savarts += elem_list.evaluate_biot_savart_induction_for_all_elements(x_obs=x_obs,
#                                                                                                  n_hat=None,
#                                                                                                  period=pdx,
#                                                                                                  wind=wind,
#                                                                                                  optimization_period=optimization_period)
#             else:
#                 all_biot_savarts = elem_list.evaluate_biot_savart_induction_for_all_elements(x_obs=x_obs, n_hat=None)
#
#             for fdx in range(elem_list.number_of_elements):
#                 u_ind_fil = all_biot_savarts[:, fdx]
#
#                 ind_name = 'wu_' + elem_list_name + '_' + str(fdx) + '_' + str(kite_obs)
#                 local_var = variables_si['xl'][ind_name]
#                 local_resi = (local_var - u_ind_fil) / u_ref
#                 resi = cas.vertcat(resi, local_resi)
#
#     return resi


def test(test_includes_visualization):
    return None

# test()