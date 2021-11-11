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
from multiprocessing import Pool

import awebox.mdl.aero.induction_dir.vortex_dir.convection as convection
import awebox.mdl.aero.induction_dir.vortex_dir.flow as flow
import awebox.mdl.aero.induction_dir.vortex_dir.tools as tools
import awebox.mdl.aero.induction_dir.tools_dir.unit_normal as unit_normal
import awebox.mdl.aero.induction_dir.vortex_dir.element as vortex_element
import awebox.mdl.aero.induction_dir.vortex_dir.far_wake as vortex_far_wake
import awebox.mdl.aero.induction_dir.vortex_dir.cylinder_list as vortex_cylinder_list
import awebox.mdl.aero.induction_dir.vortex_dir.near_wake as vortex_near_wake
import awebox.mdl.aero.induction_dir.vortex_dir.biot_savart as biot_savart
import awebox.tools.vector_operations as vect_op
import awebox.tools.constraint_operations as cstr_op
import awebox.tools.print_operations as print_op


from awebox.logger.logger import Logger as awelogger
import numpy as np
import pdb

def construct_objects(options, variables_si, parameters, architecture, wind):

    near_wake = vortex_near_wake.get_list(options, variables_si, parameters, architecture, wind)
    far_wake_filaments, tangential_cylinder, longitudinal_cylinder = vortex_far_wake.get_lists(options, variables_si, parameters, architecture, wind)

    vortex_lists = {}

    if near_wake.number_of_elements > 0:
        vortex_lists['near_fil'] = near_wake

    if far_wake_filaments.number_of_elements > 0:
        vortex_lists['far_fil'] = far_wake_filaments

    if tangential_cylinder.number_of_elements > 0:
        vortex_lists['tan_cyl'] = tangential_cylinder

    if longitudinal_cylinder.number_of_elements > 0:
        vortex_lists['long_cyl'] = longitudinal_cylinder

    for key in vortex_lists.keys():
        elem_list = vortex_lists[key]
        elem_list.confirm_list_has_expected_dimensions()
        elem_list.make_symbolic_biot_savart_function()

    return vortex_lists

def get_vortex_cstr(options, wind, variables_si, objects, architecture):

    vortex_representation = options['aero']['vortex']['representation']
    cstr_list = cstr_op.ConstraintList()

    if vortex_representation == 'state':
        state_conv_cstr = convection.get_state_repr_convection_cstr(options, wind, variables_si, architecture)
        cstr_list.append(state_conv_cstr)

    superposition_cstr = flow.get_superposition_cstr(options, wind, variables_si, objects, architecture)
    cstr_list.append(superposition_cstr)

    return cstr_list

def get_induction_trivial_residual(options, wind, variables_si, architecture, objects):

    resi = []

    u_ref = wind.get_speed_ref()

    for kite_obs in architecture.kite_nodes:

        for elem_list_name in objects.keys():
            elem_list = objects[elem_list_name]
            number_elements = elem_list.number_of_elements
            if number_elements > 0:
                for fdx in range(number_elements):
                    u_ind_fil = cas.DM.zeros((3, 1))

                    ind_name = 'wu_' + elem_list_name + '_' + str(fdx) + '_' + str(kite_obs)
                    local_var = variables_si['xl'][ind_name]
                    local_resi = (local_var - u_ind_fil) / u_ref
                    resi = cas.vertcat(resi, local_resi)

    return resi


def get_induction_final_residual(options, wind, variables_si, outputs, architecture, objects):

    resi = []

    u_ref = wind.get_speed_ref()

    for kite_obs in architecture.kite_nodes:
        parent_obs = architecture.parent_map[kite_obs]

        for elem_list_name in objects.keys():
            elem_list = objects[elem_list_name]

            x_obs = variables_si['xd']['q' + str(kite_obs) + str(parent_obs)]
            all_biot_savarts = elem_list.evaluate_biot_savart_induction_for_all_elements(x_obs=x_obs, n_hat=None)

            for fdx in range(elem_list.number_of_elements):
                u_ind_fil = all_biot_savarts[:, fdx]

                ind_name = 'wu_' + elem_list_name + '_' + str(fdx) + '_' + str(kite_obs)
                local_var = variables_si['xl'][ind_name]
                local_resi = (local_var - u_ind_fil) / u_ref
                resi = cas.vertcat(resi, local_resi)

    return resi

def test():

    vect_op.test_altitude()
    biot_savart.test_filament()
    biot_savart.test_longtitudinal_cylinder()
    biot_savart.test_tangential_cylinder()

    print_op.warn_about_temporary_funcationality_removal(location='vortex.test')
    vortex_element.test_filament_type()

    # freestream_filament_far_wake_test_list = vortex_filament_list.test(far_wake_model = 'freestream_filament')
    # flow.test(freestream_filament_far_wake_test_list)
    # pathwise_filament_far_wake_test_list = vortex_filament_list.test(far_wake_model = 'pathwise_filament')
    # flow.test(pathwise_filament_far_wake_test_list)

    return None

def collect_vortex_outputs(model_options, atmos, wind, variables_si, outputs, vortex_objects, parameters, architecture):

    # break early and loud if there are problems
    test()

    if 'vortex' not in list(outputs.keys()):
        outputs['vortex'] = {}


    print_op.warn_about_temporary_funcationality_removal(location='vortex.outputs.columnize')
    # columnized_list = vortex_filament_list.columnize(filament_list)
    # outputs['vortex']['filament_list'] = columnized_list
    # far_wake_list = vortex_filament_list.get_far_wake_list(model_options, variables_si, architecture, wind)

    kite_nodes = architecture.kite_nodes
    for kite_obs in kite_nodes:

        parent_obs = architecture.parent_map[kite_obs]

        u_ind = flow.get_induced_velocity_at_kite(variables_si, vortex_objects, kite_obs)

        n_hat = unit_normal.get_n_hat(model_options, parent_obs, variables_si, parameters, architecture)
        local_a = flow.get_induction_factor_at_kite(model_options, wind, variables_si, vortex_objects, architecture, kite_obs, n_hat=n_hat)

        far_wake_u_ind = flow.get_induced_velocity_at_kite(variables_si, vortex_objects, kite_obs, selection='far_wake')
        far_wake_u_ind_norm = vect_op.norm(far_wake_u_ind)
        far_wake_u_ind_norm_over_ref = far_wake_u_ind_norm / wind.get_speed_ref()

        est_truncation_error = (far_wake_u_ind_norm) / vect_op.norm(u_ind)

        outputs['vortex']['u_ind' + str(kite_obs)] = u_ind
        outputs['vortex']['u_ind_norm' + str(kite_obs)] = vect_op.norm(u_ind)
        outputs['vortex']['local_a' + str(kite_obs)] = local_a

        outputs['vortex']['far_wake_u_ind' + str(kite_obs)] = far_wake_u_ind
        outputs['vortex']['far_wake_u_ind_norm_over_ref' + str(kite_obs)] = far_wake_u_ind_norm_over_ref

        outputs['vortex']['est_truncation_error' + str(kite_obs)] = est_truncation_error

    return outputs

def compute_global_performance(power_and_performance, plot_dict):


    kite_nodes = plot_dict['architecture'].kite_nodes

    max_est_trunc_list = []
    max_est_discr_list = []
    far_wake_u_ind_norm_over_ref_list = []

    all_local_a = None

    for kite in kite_nodes:

        trunc_name = 'est_truncation_error' + str(kite)
        local_max_est_trunc = np.max(np.array(plot_dict['outputs']['vortex'][trunc_name][0]))
        max_est_trunc_list += [local_max_est_trunc]

        kite_local_a = np.ndarray.flatten(np.array(plot_dict['outputs']['vortex']['local_a' + str(kite)][0]))
        if all_local_a is None:
            all_local_a = kite_local_a
        else:
            all_local_a = np.vstack([all_local_a, kite_local_a])

        max_kite_local_a = np.max(kite_local_a)
        min_kite_local_a = np.min(kite_local_a)
        local_max_est_discr = (max_kite_local_a - min_kite_local_a) / max_kite_local_a
        max_est_discr_list += [local_max_est_discr]

        local_far_wake_u_ind_norm_over_ref = np.max(np.array(plot_dict['outputs']['vortex']['far_wake_u_ind_norm_over_ref' + str(kite)]))
        far_wake_u_ind_norm_over_ref_list += [local_far_wake_u_ind_norm_over_ref]

    average_local_a = np.average(all_local_a)
    power_and_performance['vortex_average_local_a'] = average_local_a

    stdev_local_a = np.std(all_local_a)
    power_and_performance['vortex_stdev_local_a'] = stdev_local_a

    max_far_wake_u_ind_norm_over_ref = np.max(np.array(far_wake_u_ind_norm_over_ref_list))
    power_and_performance['vortex_max_far_wake_u_ind_norm_over_ref'] = max_far_wake_u_ind_norm_over_ref

    max_est_trunc = np.max(np.array(max_est_trunc_list))
    power_and_performance['vortex_max_est_truncation_error'] = max_est_trunc

    max_est_discr = np.max(np.array(max_est_discr_list))
    power_and_performance['vortex_max_est_discretization_error'] = max_est_discr

    return power_and_performance
