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
constraints to create "intermediate condition" fixing constraints on the positions of the wake nodes,
to be referenced/used from ocp.constraints
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2020-21
'''
import copy
import pdb

import numpy as np
import awebox.mdl.aero.induction_dir.vortex_dir.tools as vortex_tools
import awebox.mdl.aero.induction_dir.vortex_dir.alg_repr_dir.fixing as alg_fixing

import awebox.tools.struct_operations as struct_op
import awebox.tools.print_operations as print_op
import awebox.tools.vector_operations as vect_op
import awebox.tools.constraint_operations as cstr_op

import casadi.tools as cas
from awebox.logger.logger import Logger as awelogger


################# define the actual constraint

def get_initialization(init_options, V_init, p_fix_num, nlp, model):

    time_grids = nlp.time_grids

    V_init_si = copy.deepcopy(V_init)
    V_init_scaled = struct_op.si_to_scaled(V_init_si, model.scaling)

    Outputs_init = nlp.Outputs_struct(nlp.Outputs_structured_fun(V_init_scaled, p_fix_num))

    integral_output_components = nlp.integral_output_components

    Integral_outputs_struct = integral_output_components[0]
    Integral_outputs_fun = integral_output_components[1]
    Integral_outputs_scaled = Integral_outputs_struct(Integral_outputs_fun(V_init_scaled, p_fix_num))

    abbreviated_variables = vortex_tools.get_list_of_abbreviated_variables(init_options)
    for abbreviated_var_name in abbreviated_variables:
        V_init_scaled = append_specific_initialization(abbreviated_var_name, init_options, V_init_scaled, Outputs_init,
                                                       Integral_outputs_scaled, model, time_grids)

    V_init_si = struct_op.scaled_to_si(V_init_scaled, model.scaling)
    V_init_si = append_induced_velocities(init_options, V_init_si, p_fix_num, nlp, model)

    check_that_outputs_init_was_plausibly_constructed(init_options, Outputs_init, model.architecture)
    check_that_zeroth_ring_shedding_circulation_behaves_reasonably(V_init_si, p_fix_num, nlp, model)

    return V_init_si


def append_induced_velocities(init_options, V_init_si, p_fix_num, nlp, model):
    for ndx in range(nlp.n_k):
        for ddx in range(nlp.d):
            V_init_si = append_induced_velocities_at_time(init_options, V_init_si, p_fix_num, model, ndx, ddx)

    return V_init_si


def append_induced_velocities_at_time(init_options, V_init_si, p_fix_num, model, ndx, ddx):

    wake = model.wake
    architecture = model.architecture

    Xdot = struct_op.construct_Xdot_struct(init_options, model.variables_dict)(0.)
    variables_si = struct_op.get_variables_at_time(init_options, V_init_si, Xdot, model.variables, ndx,
                                                   ddx=ddx)
    variables_scaled = struct_op.variables_si_to_scaled(model.variables, variables_si, model.scaling)
    parameters = struct_op.get_parameters_at_time(V_init_si, p_fix_num, model.parameters)

    tentative_lifted_name = vortex_tools.get_element_biot_savart_numerator_name('bound',
                                                                              'finite_filament',
                                                                              0,
                                                                              architecture.kite_nodes[0])
    if tentative_lifted_name in model.variables_dict['z'].keys():
        use_lifted_biot_savart_residual_assembly = True
    else:
        use_lifted_biot_savart_residual_assembly = False

    for kite_obs in architecture.kite_nodes:
        parent_obs = architecture.parent_map[kite_obs]
        x_obs = variables_si['x', 'q' + str(kite_obs) + str(parent_obs)]

        total_u_ind = cas.DM.zeros((3, 1))

        for substructure_type in wake.get_initialized_substructure_types_with_at_least_one_element():
            substructure = wake.get_substructure(substructure_type)
            for element_type in substructure.get_initialized_element_types():
                element_list = substructure.get_list(element_type)
                element_number = 0
                for elem in element_list.list:
                    u_ind_elem_name = vortex_tools.get_element_induced_velocity_name(substructure_type,
                                                                                     element_type,
                                                                                     element_number,
                                                                                     kite_obs)

                    value, num, den = elem.calculate_biot_savart_induction(elem.info_dict, x_obs)
                    value_fun = cas.Function('value_fun', [model.variables, model.parameters], [value])

                    value_eval = value_fun(variables_scaled, parameters)
                    V_init_si['coll_var', ndx, ddx, 'z', u_ind_elem_name] = value_eval

                    if (ddx is None) and ('z' in list(V_init_si.keys())):
                        V_init_si['z', ndx-1, u_ind_elem_name] = value_eval

                    if use_lifted_biot_savart_residual_assembly:
                        u_ind_num_elem_name = vortex_tools.get_element_biot_savart_numerator_name(substructure_type,
                                                                                                  element_type,
                                                                                                  element_number,
                                                                                                  kite_obs)
                        u_ind_den_elem_name = vortex_tools.get_element_biot_savart_denominator_name(substructure_type,
                                                                                                    element_type,
                                                                                                    element_number,
                                                                                                    kite_obs)

                        num_fun = cas.Function('num_fun', [model.variables, model.parameters], [num])
                        den_fun = cas.Function('den_fun', [model.variables, model.parameters], [den])

                        num_eval = num_fun(variables_scaled, parameters)
                        den_eval = den_fun(variables_scaled, parameters)

                        V_init_si['coll_var', ndx, ddx, 'z', u_ind_num_elem_name] = num_eval
                        V_init_si['coll_var', ndx, ddx, 'z', u_ind_den_elem_name] = den_eval

                        if (ddx is None) and ('z' in list(V_init_si.keys())):
                            V_init_si['z', ndx - 1, u_ind_num_elem_name] = num_eval
                            V_init_si['z', ndx - 1, u_ind_den_elem_name] = den_eval

                    element_number += 1
                    total_u_ind += value_eval

                if not (element_number == element_list.number_of_elements):
                    message = 'something went wrong with the initialization of vortex induced velocities. the wrong number of elements'
                    print_op.log_and_raise_error(message)

        u_ind_name = vortex_tools.get_induced_velocity_at_kite_name(kite_obs)
        V_init_si['coll_var', ndx, ddx, 'z', u_ind_name] = total_u_ind

        if (ddx is None) and ('z' in list(V_init_si.keys())):
            V_init_si['z', ndx - 1, u_ind_name] = total_u_ind

    return V_init_si

def check_that_zeroth_ring_shedding_circulation_behaves_reasonably(V_init_si, p_fix_num, nlp, model, epsilon=1.e-4):

    V_init_scaled = struct_op.si_to_scaled(V_init_si, model.scaling)

    Outputs_init = nlp.Outputs_struct(nlp.Outputs_structured_fun(V_init_scaled, p_fix_num))

    [Integral_outputs, Integral_outputs_fun] = nlp.integral_output_components
    int_out = Integral_outputs(Integral_outputs_fun(V_init_scaled, p_fix_num))
    tgrid_x = nlp.time_grids['x'](V_init_si['theta', 't_f'])
    tgrid_coll = nlp.time_grids['coll'](V_init_si['theta', 't_f'])

    kite_test = model.architecture.kite_nodes[0]
    ndx_test = 0
    ddx_test = -1

    period = tgrid_coll[-1, -1]
    time_end = tgrid_coll[ndx_test, ddx_test]
    time_start = tgrid_coll[ndx_test - 1, ddx_test] - (ndx_test < 1) * period
    delta_t = time_end - time_start

    integrated_total = int_out['coll_int_out', -1, -1, 'integrated_circulation' + str(kite_test)]
    integrated_end = int_out['coll_int_out', ndx_test, ddx_test, 'integrated_circulation' + str(kite_test)]
    integrated_start = int_out['coll_int_out', ndx_test-1, ddx_test, 'integrated_circulation' + str(kite_test)] - (ndx_test < 1)*integrated_total
    definite_integrated_circulation = integrated_end - integrated_start

    average_circulation = definite_integrated_circulation / delta_t

    expected_strength = average_circulation
    found_strength = V_init_si['coll_var', ndx_test, ddx_test, 'z', 'wg_' + str(kite_test) + '_0']
    non_coll_found = V_init_si['z', ndx_test+1, 'wg_' + str(kite_test) + '_0']

    cond1 = (found_strength/expected_strength - 1.)**2. < epsilon**2.
    cond2 = (non_coll_found/expected_strength - 1.)**2. < epsilon**2.

    # check that the circulation averaging works as expected
    delta_t_x = np.array(tgrid_x[1:]) - np.array(tgrid_x[:-1])
    definite_integrated_circulation_x = np.array(int_out['int_out', 1:, 'integrated_circulation' + str(kite_test)]) - np.array(int_out['int_out', :-1, 'integrated_circulation' + str(kite_test)])
    average_circulation_x = np.array([definite_integrated_circulation_x[idx] / delta_t_x[idx] for idx in range(delta_t_x.shape[0])])

    cond3 = True
    for ndx in range(average_circulation.shape[0]):
        circulation_outputs = Outputs_init['coll_outputs', ndx, :, 'aerodynamics', 'circulation' + str(kite_test)]
        average_is_less_than_max = average_circulation_x[ndx] < np.max(np.array(circulation_outputs))
        average_is_more_than_min = average_circulation_x[ndx] > np.min(np.array(circulation_outputs))
        cond3 = cond3 and average_is_less_than_max and average_is_more_than_min

    criteria = cond1 and cond2 and cond3
    if not criteria:
        message = 'something went wrong when initializing the vortex ring strength variables. '
        print_op.log_and_raise_error(message)

    return None

def append_specific_initialization(abbreviated_var_name, init_options, V_init_scaled, Outputs_init, Integral_outputs_scaled, model, time_grids):

    n_k = init_options['n_k']
    d = init_options['collocation']['d']

    kite_shed_or_parent_shed_list, tip_list, wake_node_or_ring_list = vortex_tools.get_kite_or_parent_and_tip_and_node_or_ring_list_for_abbreviated_vars(abbreviated_var_name, init_options, model.architecture)

    for kite_shed_or_parent_shed in kite_shed_or_parent_shed_list:
        for tip in tip_list:
            for wake_node_or_ring in wake_node_or_ring_list:

                for ndx in range(n_k):
                    V_init_scaled = get_specific_local_initialization(abbreviated_var_name, init_options, V_init_scaled, Outputs_init,
                                                                      Integral_outputs_scaled, model, time_grids,
                                                                      kite_shed_or_parent_shed, tip, wake_node_or_ring, ndx)

                    for ddx in range(d):
                        V_init_scaled = get_specific_local_initialization(abbreviated_var_name, init_options, V_init_scaled, Outputs_init,
                                                                          Integral_outputs_scaled, model,
                                                                          time_grids, kite_shed_or_parent_shed, tip,
                                                                          wake_node_or_ring, ndx, ddx)

    return V_init_scaled


def check_that_outputs_init_was_plausibly_constructed(init_options, Outputs_init, architecture):
    expected_radius = init_options['precompute']['radius']
    expected_period = init_options['precompute']['winding_period']
    epsilon = 0.05

    for parent in architecture.layer_nodes:
        outputs_radius = Outputs_init['coll_outputs', :, :, 'geometry', 'average_radius' + str(parent)]
        outputs_period = Outputs_init['coll_outputs', :, :, 'geometry', 'average_period_of_rotation' + str(parent)]
        
        for ndx in range(len(outputs_radius)):
            for ddx in range(len(outputs_radius[ndx])):
                radius_error = (expected_radius - outputs_radius[ndx][ddx]) / expected_radius
                radius_error_greater_than_theshhold = (radius_error**2. > epsilon**2.)

                period_error = (expected_period - outputs_period[ndx][ddx]) / expected_period
                period_error_greater_than_theshhold = (period_error ** 2. > epsilon ** 2.)

                if radius_error_greater_than_theshhold or period_error_greater_than_theshhold:
                    message = 'something went wrong when computing the outputs used to initialize the vortex variables. is it possible that the si and scaled inputs have gotten confused?'
                    print_op.log_and_raise_error(message)

    return None

def get_specific_local_initialization(abbreviated_var_name, init_options, V_init_scaled, Outputs, Integral_outputs_scaled, model, time_grids, kite_shed_or_parent_shed, tip,
                                      wake_node_or_ring, ndx, ddx=None):

    var_name = vortex_tools.get_var_name(abbreviated_var_name, kite_shed_or_parent_shed=kite_shed_or_parent_shed, tip=tip, wake_node_or_ring=wake_node_or_ring)

    # V['coll_var', ndx-1, -1, 'z', var_name] = V['z', ndx, var_name]

    if (ddx is None):
        ndx_find = ndx - 1
        ddx_find = -1
        # notice that this implies/requires periodicity.
        # todo: add check to ensure that periodic operation is expected

    else:
        ndx_find = ndx
        ddx_find = ddx

    # look-up the actual value from the Outputs. Keep the computing here minimal.
    if abbreviated_var_name == 'wx':
        var_val_si = alg_fixing.get_local_convected_position_value(init_options, V_init_scaled, Outputs, model, time_grids, kite_shed_or_parent_shed, tip, wake_node_or_ring, ndx_find, ddx_find)
    elif abbreviated_var_name == 'wg':
        var_val_si = alg_fixing.get_local_average_circulation_value(init_options, V_init_scaled, Integral_outputs_scaled, model, time_grids, kite_shed_or_parent_shed, wake_node_or_ring, ndx_find, ddx_find)
    elif abbreviated_var_name == 'wh':
        var_val_si = alg_fixing.get_local_cylinder_pitch_value(init_options, Outputs, kite_shed_or_parent_shed, wake_node_or_ring, ndx_find, ddx_find)
    elif abbreviated_var_name == 'wx_center':
        var_val_si = alg_fixing.get_local_cylinder_center_value(init_options, Outputs, kite_shed_or_parent_shed, wake_node_or_ring, ndx_find, ddx_find)
    else:
        message = 'get_specific_local_constraint function is not set up for this abbreviation (' + abbreviated_var_name + ') yet.'
        print_op.log_and_raise_error(message)

    var_val_scaled = struct_op.var_si_to_scaled('z', var_name, var_val_si, model.scaling)

    if ddx is None:
        V_init_scaled['z', ndx, var_name] = var_val_scaled
    else:
        V_init_scaled['coll_var', ndx, ddx, 'z', var_name] = var_val_scaled

    return V_init_scaled
