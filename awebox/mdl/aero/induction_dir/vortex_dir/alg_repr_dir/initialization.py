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



def get_initialization(init_options, V_init_si, p_fix_num, nlp, model):

    time_grids = nlp.time_grids

    V_init_si_temp = copy.deepcopy(V_init_si)
    V_init_scaled = struct_op.si_to_scaled(V_init_si_temp, model.scaling)

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

    check_that_precomputed_radius_and_period_correspond_to_outputs(init_options, Outputs_init, model.architecture)
    check_that_zeroth_ring_shedding_circulation_behaves_reasonably(V_init_si, p_fix_num, nlp, model)

    if model.options['aero']['vortex']['double_check_wingtip_fixing']:
        vortex_tools.check_that_wake_node_0_always_lays_on_wingtips(init_options, p_fix_num, Outputs_init, model, V_si=V_init_si)

    return V_init_si


def append_induced_velocities(init_options, V_init_si, p_fix_num, nlp, model):

    degree_of_induced_velocity_lifting = get_degree_of_induced_velocity_lifting(model)

    if degree_of_induced_velocity_lifting == 1:
        for kite_obs in model.architecture.kite_nodes:
            u_ind_name = vortex_tools.get_induced_velocity_at_kite_name(kite_obs)
            ref_value = model.scaling['z', u_ind_name]

            control_length = count_z_length_on_controls(model, nlp)
            for ndx in range(control_length):
                V_init_si['z', ndx, u_ind_name] = ref_value

            for ndx in range(nlp.n_k):
                for ddx in range(nlp.d):
                    V_init_si['coll_var', ndx, ddx, 'z', u_ind_name] = ref_value

    if degree_of_induced_velocity_lifting >= 2:
        function_dict = make_induced_velocities_functions(model, nlp)
        V_init_si = append_induced_velocities_using_parallelization(init_options, function_dict, V_init_si, p_fix_num, nlp, model)

    return V_init_si


def get_degree_of_induced_velocity_lifting(model):
    return model.options['aero']['vortex']['degree_of_induced_velocity_lifting']


def make_induced_velocities_functions(model, nlp):

    print_op.base_print('computing induced velocity functions...')

    wake = model.wake
    degree_of_induced_velocity_lifting = get_degree_of_induced_velocity_lifting(model)

    x_obs_sym = cas.SX.sym('x_obs_sym', (3, 1))

    function_dict = {}
    for substructure_type in wake.get_initialized_substructure_types_with_at_least_one_element():
        if substructure_type not in function_dict.keys():
            function_dict[substructure_type] = {}

        substructure = wake.get_substructure(substructure_type)
        for element_type in substructure.get_initialized_element_types():

            if element_type not in function_dict[substructure_type].keys():
                function_dict[substructure_type][element_type] = {}

            element_list = substructure.get_list(element_type)
            element_number = 0
            for elem in element_list.list:

                if elem not in function_dict[substructure_type][element_type].keys():
                    function_dict[substructure_type][element_type][elem] = {}

                value, num, den = elem.calculate_biot_savart_induction(elem.info_dict, x_obs_sym)
                value_separate_fun = cas.Function('value_separate_fun', [x_obs_sym, model.variables, model.parameters], [value])
                num_separate_fun = cas.Function('num_separate_fun', [x_obs_sym, model.variables, model.parameters], [num])
                den_separate_fun = cas.Function('den_separate_fun', [x_obs_sym, model.variables, model.parameters], [den])

                value_sym = cas.SX.sym('value_sym', value.shape)
                num_sym = cas.SX.sym('num_sym', num.shape)
                den_sym = cas.SX.sym('den_sym', den.shape)

                if elem.biot_savart_residual_fun is None:
                    elem.define_biot_savart_induction_residual_function(degree_of_induced_velocity_lifting)
                residual_fun = elem.biot_savart_residual_fun

                if degree_of_induced_velocity_lifting == 1:
                    message = 'this method (make_induced_velocities_functions) should not have been called when initializing the induced velocity variables'
                    print_op.log_and_raise_error(message)
                elif degree_of_induced_velocity_lifting == 2:
                    residual = residual_fun(elem.info, x_obs_sym, value_sym)
                elif degree_of_induced_velocity_lifting == 3:
                    residual = residual_fun(elem.info, x_obs_sym, value_sym, num_sym, den_sym)
                else:
                    message = 'unexpected degree_of_induced_velocity_lifting (' + str(degree_of_induced_velocity_lifting) + ')'
                    print_op.log_and_raise_error(message)

                resi_separate_fun = cas.Function('value_separate_fun', [x_obs_sym, model.variables, model.parameters, value_sym, num_sym, den_sym], [residual])

                x_obs_len = x_obs_sym.shape[0]
                model_variables_len = model.variables.cat.shape[0]
                model_parameters_len = model.parameters.cat.shape[0]

                all_inputs_shape = (x_obs_len + model_variables_len + model_parameters_len, 1)
                inputs_sym = cas.SX.sym('inputs_sym', all_inputs_shape)
                inputs_x_obs = inputs_sym[0:x_obs_len]
                inputs_variables = inputs_sym[x_obs_len: x_obs_len + model_variables_len]
                inputs_parameters = inputs_sym[x_obs_len + model_variables_len: x_obs_len + model_variables_len + model_parameters_len]

                test_value_len = value_sym.shape[0]
                test_num_len = num_sym.shape[0]
                test_den_len = den_sym.shape[0]
                test_inputs_shape = (x_obs_len + model_variables_len + model_parameters_len + test_value_len + test_num_len + test_den_len, 1)
                test_inputs_sym = cas.SX.sym('test_inputs_sym', test_inputs_shape)
                test_inputs_x_obs = test_inputs_sym[0:x_obs_len]
                test_inputs_variables = test_inputs_sym[x_obs_len: x_obs_len + model_variables_len]
                test_inputs_parameters = test_inputs_sym[x_obs_len + model_variables_len: x_obs_len + model_variables_len + model_parameters_len]
                test_inputs_value = test_inputs_sym[x_obs_len + model_variables_len + model_parameters_len: x_obs_len + model_variables_len + model_parameters_len + test_value_len]
                test_inputs_num = test_inputs_sym[x_obs_len + model_variables_len + model_parameters_len + test_value_len: x_obs_len + model_variables_len + model_parameters_len + test_value_len + test_num_len]
                test_inputs_den = test_inputs_sym[x_obs_len + model_variables_len + model_parameters_len + test_value_len + test_num_len: x_obs_len + model_variables_len + model_parameters_len + test_value_len + test_num_len + test_den_len]

                if not (inputs_x_obs.shape == x_obs_sym.shape and test_inputs_x_obs.shape == x_obs_sym.shape):
                    message = 'mapping concatenation of x_obs does not have the necessary shape'
                    print_op.log_and_raise_error(message)
                if not (inputs_variables.shape == model.variables.shape and test_inputs_variables.shape == model.variables.shape):
                    message = 'mapping concatenation of model.variables does not have the necessary shape'
                    print_op.log_and_raise_error(message)
                if not (inputs_parameters.shape == model.parameters.shape and test_inputs_parameters.shape == model.parameters.shape):
                    message = 'mapping concatenation of model.parameters does not have the necessary shape'
                    print_op.log_and_raise_error(message)
                if not (test_inputs_value.shape == value.shape):
                    message = 'mapping concatenation of biot-savart value does not have the necessary shape'
                    print_op.log_and_raise_error(message)
                if not (test_inputs_num.shape == num.shape):
                    message = 'mapping concatenation of biot-savart numerator does not have the necessary shape'
                    print_op.log_and_raise_error(message)
                if not (test_inputs_den.shape == den.shape):
                    message = 'mapping concatenation of biot-savart denominator does not have the necessary shape'
                    print_op.log_and_raise_error(message)

                separate_function_types = ['value', 'num', 'den', 'resi']
                for separate_type in separate_function_types:

                    if separate_type == 'value':
                        local_concat = value_separate_fun(inputs_x_obs, inputs_variables, inputs_parameters)
                    elif separate_type == 'num':
                        local_concat = num_separate_fun(inputs_x_obs, inputs_variables, inputs_parameters)
                    elif separate_type == 'den':
                        local_concat = den_separate_fun(inputs_x_obs, inputs_variables, inputs_parameters)
                    elif separate_type == 'resi':
                        local_concat = resi_separate_fun(test_inputs_x_obs, test_inputs_variables,
                                                              test_inputs_parameters, test_inputs_value,
                                                              test_inputs_num, test_inputs_den)
                    else:
                        message = 'something went wrong when deciding between supposedly pre-defined options: ' + str(separate_type)
                        print_op.log_and_raise_error(message)

                    if separate_type == 'resi':
                        concat_fun = cas.Function(separate_type + '_fun', [test_inputs_sym], [local_concat])
                    else:
                        concat_fun = cas.Function(separate_type + '_fun', [inputs_sym], [local_concat])

                    parallelization_type = model.options['construction']['parallelization']['type']
                    if parallelization_type in ['serial', 'openmp', 'thread']:
                        map_on_control_nodes = concat_fun.map(count_z_length_on_controls(model, nlp),
                                                              parallelization_type)
                        map_on_collocation_nodes = concat_fun.map(nlp.n_k * nlp.d, parallelization_type)
                        function_dict[substructure_type][element_type][elem][
                            separate_type + '_map_on_control_nodes'] = map_on_control_nodes
                        function_dict[substructure_type][element_type][elem][
                            separate_type + '_map_on_collocation_nodes'] = map_on_collocation_nodes
                    elif parallelization_type == 'concurrent_futures':
                        function_dict[substructure_type][element_type][elem][
                            separate_type + '_concat_fun'] = concat_fun
                    else:
                        message = 'sorry, but the awebox has not yet set up ' + parallelization_type + ' parallelization'
                        print_op.log_and_raise_error(message)

                element_number += 1

            if not (element_number == element_list.number_of_elements):
                message = 'something went wrong with the initialization of vortex induced velocities. the wrong number of elements'
                print_op.log_and_raise_error(message)

    return function_dict


def count_z_length_on_controls(model, nlp):
    sample_z_label = model.variables_dict['z'].labels()[0]
    sample_z_name = sample_z_label[1:-3]
    if ('z' in nlp.V.keys()):
        test_name = '[z,' + str(nlp.n_k) + ',' + sample_z_name + ',0]'
        if test_name in nlp.V.labels():
            return nlp.n_k + 1
        else:
            return nlp.n_k
    else:
        message = 'vortex method is not yet setup for the case that V does not contain a z structure'
        print_op.log_and_raise_error(message)
    return None

def append_induced_velocities_using_parallelization(init_options, function_dict, V_init_si, p_fix_num, nlp, model):

    wake = model.wake
    architecture = model.architecture
    degree_of_induced_velocity_lifting = get_degree_of_induced_velocity_lifting(model)

    Xdot = None

    control_length = count_z_length_on_controls(model, nlp)
    stacked_inputs_on_control_nodes = []
    for ndx in range(control_length):
        ddx = None
        variables_si = (struct_op.get_variables_at_time(init_options, V_init_si, Xdot, model.variables, ndx, ddx=ddx))
        variables_scaled = struct_op.variables_si_to_scaled(model.variables, variables_si, model.scaling)
        parameters = struct_op.get_parameters_at_time(V_init_si, p_fix_num, model.parameters)
        stacked_inputs_on_control_nodes = cas.horzcat(stacked_inputs_on_control_nodes, cas.vertcat(variables_scaled, parameters))

    stacked_inputs_on_collocation_nodes = []
    for ndx in range(nlp.n_k):
        for ddx in range(nlp.d):
            variables_si = struct_op.get_variables_at_time(init_options, V_init_si, Xdot, model.variables, ndx,
                                                           ddx=ddx)
            variables_scaled = struct_op.variables_si_to_scaled(model.variables, variables_si, model.scaling)
            parameters = struct_op.get_parameters_at_time(V_init_si, p_fix_num, model.parameters)
            stacked_inputs_on_collocation_nodes = cas.horzcat(stacked_inputs_on_collocation_nodes,
                                                          cas.vertcat(variables_scaled, parameters))

    total_of_values_on_control = cas.DM.zeros((3, nlp.n_k))
    total_of_values_on_collocation = cas.DM.zeros((3, nlp.n_k * nlp.d))

    # prepare the progress bar. (does nothing except count items in following for-loop)
    print_op.base_print('appending induced velocity variables...')
    index_progress = 0
    total_progress = 0
    for kite_obs in architecture.kite_nodes:
        for substructure_type in wake.get_initialized_substructure_types_with_at_least_one_element():
            substructure = wake.get_substructure(substructure_type)
            for element_type in substructure.get_initialized_element_types():
                element_list = substructure.get_list(element_type)
                element_number = -1
                for elem in element_list.list:
                    element_number += 1
                    if vortex_tools.not_bound_and_shed_is_obs(init_options, substructure_type, element_type, element_number, kite_obs, architecture):
                        separate_function_types = ['value', 'num', 'den']
                        for separate_type in separate_function_types:
                            total_progress += 1

    # actually do the calculation
    for kite_obs in architecture.kite_nodes:
        parent_obs = architecture.parent_map[kite_obs]
        u_ind_name = vortex_tools.get_induced_velocity_at_kite_name(kite_obs)

        horizontally_concat_x_obs = []
        for ndx in range(control_length):
            ddx = None
            variables_si = struct_op.get_variables_at_time(init_options, V_init_si, Xdot, model.variables, ndx,
                                                           ddx=ddx)
            x_obs = variables_si['x', 'q' + str(kite_obs) + str(parent_obs)]
            horizontally_concat_x_obs = cas.horzcat(horizontally_concat_x_obs, x_obs)

        kite_stacked_inputs_on_control_nodes = cas.vertcat(horizontally_concat_x_obs, stacked_inputs_on_control_nodes)

        horizontally_concat_x_obs = []
        for ndx in range(nlp.n_k):
            for ddx in range (nlp.d):
                variables_si = struct_op.get_variables_at_time(init_options, V_init_si, Xdot, model.variables, ndx,
                                                               ddx=ddx)
                x_obs = variables_si['x', 'q' + str(kite_obs) + str(parent_obs)]
                horizontally_concat_x_obs = cas.horzcat(horizontally_concat_x_obs, x_obs)
        kite_stacked_inputs_on_collocation_nodes = cas.vertcat(horizontally_concat_x_obs,
                                                           stacked_inputs_on_collocation_nodes)

        for substructure_type in wake.get_initialized_substructure_types_with_at_least_one_element():
            substructure = wake.get_substructure(substructure_type)
            for element_type in substructure.get_initialized_element_types():
                element_list = substructure.get_list(element_type)
                element_number = -1
                for elem in element_list.list:
                    element_number += 1

                    if vortex_tools.not_bound_and_shed_is_obs(init_options, substructure_type, element_type, element_number, kite_obs, architecture):

                        test_kite_stacked_inputs_on_control_nodes = copy.deepcopy(kite_stacked_inputs_on_control_nodes)
                        test_kite_stacked_inputs_on_collocation_nodes = copy.deepcopy(
                            kite_stacked_inputs_on_collocation_nodes)

                        separate_function_types = ['value', 'num', 'den']
                        for separate_type in separate_function_types:

                            if separate_type == 'value':
                                var_name = vortex_tools.get_element_induced_velocity_name(substructure_type,
                                                                                               element_type,
                                                                                               element_number, kite_obs)
                            elif separate_type == 'num':
                                var_name = vortex_tools.get_element_biot_savart_numerator_name(substructure_type,
                                                                                               element_type,
                                                                                               element_number, kite_obs)
                            elif separate_type == 'den':
                                var_name = vortex_tools.get_element_biot_savart_denominator_name(substructure_type,
                                                                                               element_type,
                                                                                               element_number, kite_obs)
                            else:
                                message = 'reached mutually exclusive option'
                                print_op.log_and_raise_error(message)

                            index_progress += 1
                            print_op.print_progress(index_progress, total_progress)

                            map_on_control_nodes = function_dict[substructure_type][element_type][elem][
                                separate_type + '_map_on_control_nodes']
                            map_on_collocation_nodes = function_dict[substructure_type][element_type][elem][
                                separate_type + '_map_on_collocation_nodes']

                            outputs_on_control = map_on_control_nodes(kite_stacked_inputs_on_control_nodes)
                            outputs_on_collocation = map_on_collocation_nodes(
                                kite_stacked_inputs_on_collocation_nodes)

                            test_kite_stacked_inputs_on_control_nodes = cas.vertcat(test_kite_stacked_inputs_on_control_nodes, outputs_on_control)
                            test_kite_stacked_inputs_on_collocation_nodes = cas.vertcat(
                                test_kite_stacked_inputs_on_collocation_nodes, outputs_on_collocation)

                            if '[z,0,' + var_name + ',0]' in V_init_si.labels():
                                for ndx in range(control_length):
                                    V_init_si['z', ndx, var_name] = outputs_on_control[:, ndx]

                            if '[coll_var,0,0,z,' + var_name + ',0]' in V_init_si.labels():
                                cdx = 0
                                for ndx in range(nlp.n_k):
                                    for ddx in range(nlp.d):
                                        V_init_si['coll_var', ndx, ddx, 'z', var_name] = outputs_on_collocation[:, cdx]
                                        cdx += 1

                            if separate_type == 'value':
                                total_of_values_on_control = total_of_values_on_control + outputs_on_control
                                total_of_values_on_collocation = total_of_values_on_collocation + outputs_on_collocation

                        # test that the residual is satisfied
                        sanity_check_biot_savart_at_initialization(function_dict, nlp, substructure_type, element_type, elem,
                                                                   test_kite_stacked_inputs_on_control_nodes,
                                                                   test_kite_stacked_inputs_on_collocation_nodes)

    print_op.close_progress()

    for ndx in range(control_length):
        V_init_si['z', ndx, u_ind_name] = total_of_values_on_control[:, ndx]

    cdx = 0
    for ndx in range(nlp.n_k):
        for ddx in range(nlp.d):
            V_init_si['coll_var', ndx, ddx, 'z', u_ind_name] = total_of_values_on_collocation[:, cdx]
            cdx += 1

    return V_init_si


def sanity_check_biot_savart_at_initialization(function_dict, nlp, substructure_type, element_type, elem, test_kite_stacked_inputs_on_control_nodes, test_kite_stacked_inputs_on_collocation_nodes, threshold=1.e-4):

    if 'resi_map_on_control_nodes' in function_dict[substructure_type][element_type][elem].keys():
        resi_map_on_control_nodes = function_dict[substructure_type][element_type][elem]['resi_map_on_control_nodes']
        resi_map_on_collocation_nodes = function_dict[substructure_type][element_type][elem][
            'resi_map_on_collocation_nodes']
        resi_on_control = resi_map_on_control_nodes(test_kite_stacked_inputs_on_control_nodes)
        resi_on_collocation = resi_map_on_collocation_nodes(test_kite_stacked_inputs_on_collocation_nodes)
    elif 'resi_concat_fun' in function_dict[substructure_type][element_type][elem].keys():
        resi_concat_fun = function_dict[substructure_type][element_type][elem]['resi_concat_fun']
        resi_on_control = struct_op.concurrent_future_map(resi_concat_fun, test_kite_stacked_inputs_on_control_nodes)
        resi_on_collocation = struct_op.concurrent_future_map(resi_concat_fun, test_kite_stacked_inputs_on_collocation_nodes)

    norm_sq_resi_on_control = cas.mtimes(vect_op.columnize(resi_on_control).T, vect_op.columnize(resi_on_control)) / float(nlp.n_k)
    norm_sq_resi_on_collocation = cas.mtimes(vect_op.columnize(resi_on_collocation).T,
                                             vect_op.columnize(resi_on_collocation)) / float(nlp.n_k * nlp.d)

    mapping_failure = (norm_sq_resi_on_control > threshold) or (norm_sq_resi_on_collocation > threshold)
    if mapping_failure:
        message = 'something went wrong with the mapping of the induced velocity initialization.'
        message += ' norm_squared resi on control nodes = ' + print_op.repr_g(norm_sq_resi_on_control)
        message += ' and norm_squared resi on collocation nodes = ' + print_op.repr_g(norm_sq_resi_on_collocation)
        print_op.log_and_raise_error(message)

    return None


def get_collocation_overstepping_ndx(n_k, ndx):
    # E       A       B       C       D
    # (A, -1) (B, -1) (C, -1) (D, -1) (E, -1)

    if ndx < n_k - 1:
        ndx_on_collocation_overstepping = ndx + 1
    else:
        ndx_on_collocation_overstepping = -1
    return ndx_on_collocation_overstepping


def check_that_zeroth_ring_shedding_circulation_behaves_reasonably(V_init_si, p_fix_num, nlp, model, epsilon=1.e-2):

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
        average_is_less_than_or_equal_to_max = average_circulation_x[ndx] <= np.max(np.array(circulation_outputs))
        average_is_more_than_or_equal_to_min = average_circulation_x[ndx] >= np.min(np.array(circulation_outputs))
        cond3 = cond3 and average_is_less_than_or_equal_to_max and average_is_more_than_or_equal_to_min

    criteria = cond1 and cond2 and cond3
    if not criteria:
        message = 'something went wrong when initializing the vortex ring strength variables. '
        print_op.log_and_raise_error(message)

    return None

def check_that_wake_node_0_always_has_a_convection_time_of_zero(time_grids, V_init_scaled, init_options, n_k, d):

    tcoll = time_grids['coll'](V_init_scaled['theta', 't_f'])
    epsilon = 1.e-7

    all_convection_times_are_zero = True
    for ndx in range(n_k):
        conv_time_control = alg_fixing.get_the_convection_time_from_the_current_indices_and_wake_node(init_options,
                                                                                                          tcoll,
                                                                                                          wake_node=0,
                                                                                                          ndx=ndx)
        control_time_is_zero = (conv_time_control) ** 2. < epsilon ** 2.
        all_convection_times_are_zero = all_convection_times_are_zero and control_time_is_zero

    for ddx in range(d):
            conv_time_collocation = alg_fixing.get_the_convection_time_from_the_current_indices_and_wake_node(init_options, tcoll, wake_node=0, ndx=ndx, ddx=ddx)
            coll_time_is_zero = (conv_time_collocation)**2. < epsilon**2.
            all_convection_times_are_zero = all_convection_times_are_zero and coll_time_is_zero

    conv_time_final = alg_fixing.get_the_convection_time_from_the_current_indices_and_wake_node(init_options,
                                                                                                  tcoll,
                                                                                                  wake_node=0,
                                                                                                  ndx=-1)
    final_time_is_zero = (conv_time_final) ** 2. < epsilon ** 2.
    all_convection_times_are_zero = all_convection_times_are_zero and final_time_is_zero

    if not all_convection_times_are_zero:
        message = 'sometime went wrong when computing the convection time, because the convection time does not always evaluate as zero on wake node 0'
        print_op.log_and_raise_error(message)
    return None


def append_specific_initialization(abbreviated_var_name, init_options, V_init_scaled, Outputs_init, Integral_outputs_scaled, model, time_grids):

    print_op.base_print('appending ' + abbreviated_var_name + ' variables...')

    n_k = init_options['n_k']
    d = init_options['collocation']['d']

    check_that_wake_node_0_always_has_a_convection_time_of_zero(time_grids, V_init_scaled, init_options, n_k, d)

    kite_shed_or_parent_shed_list, tip_list, wake_node_or_ring_list = vortex_tools.get_kite_or_parent_and_tip_and_node_or_ring_list_for_abbreviated_vars(abbreviated_var_name, init_options, model.architecture)

    total_progress = len(kite_shed_or_parent_shed_list) * len(tip_list) * len(wake_node_or_ring_list) * n_k * d
    index_progress = 0
    for kite_shed_or_parent_shed in kite_shed_or_parent_shed_list:
        for tip in tip_list:
            for wake_node_or_ring in wake_node_or_ring_list:

                for ndx in range(n_k):
                    V_init_scaled = get_specific_local_initialization(abbreviated_var_name, init_options, V_init_scaled, Outputs_init,
                                                                      Integral_outputs_scaled, model, time_grids,
                                                                      kite_shed_or_parent_shed, tip, wake_node_or_ring, ndx)

                    for ddx in range(d):

                        print_op.print_progress(index_progress, total_progress)

                        V_init_scaled = get_specific_local_initialization(abbreviated_var_name, init_options, V_init_scaled, Outputs_init,
                                                                          Integral_outputs_scaled, model,
                                                                          time_grids, kite_shed_or_parent_shed, tip,
                                                                          wake_node_or_ring, ndx, ddx)

                        index_progress += 1

    print_op.close_progress()

    return V_init_scaled


def check_that_precomputed_radius_and_period_correspond_to_outputs(init_options, Outputs_init, architecture):

    # this check could not/will not work when warmstarting/tracking a non-circular path.
    # therefore, we're going to turn it off, if the output geometry values are not reasonably consistent
    initialization_based_on_circular_motion = True

    expected_radius = init_options['precompute']['radius']
    expected_period = init_options['precompute']['winding_period']
    dict_expected_columnized = {'radius': expected_radius, 'period': expected_period}

    if architecture.number_of_kites == 1:
        # todo: fix the underlying problem: estimate-radius routine works badly on single kite systems.
        # (radius of curvature should work for fine in circular trajectory, but get-period-of-rotation currently depends on radius
        # not radius-of-curvature)
        epsilon = 0.5
    else:
        epsilon = 0.05

    for parent in architecture.layer_nodes:
        outputs_radius = Outputs_init['coll_outputs', :, :, 'geometry', 'average_radius_of_curvature' + str(parent)]
        outputs_period = Outputs_init['coll_outputs', :, :, 'geometry', 'average_period_of_rotation' + str(parent)]

        outputs_radius_columnized = vect_op.columnize(outputs_radius)
        outputs_period_columnized = vect_op.columnize(outputs_period)
        dict_outputs_columnized = {'radius': outputs_radius_columnized, 'period': outputs_period_columnized}

        circular_thresh = 1.e-4
        for local_outputs_columnized in dict_outputs_columnized.values():
            for idx in range(local_outputs_columnized.shape[0]):
                local_diff = (local_outputs_columnized[idx] - local_outputs_columnized[0]) / local_outputs_columnized[0]
                if local_diff**2. > circular_thresh**2.:
                    initialization_based_on_circular_motion = False

        if initialization_based_on_circular_motion:
            for local_output_name, local_outputs_columnized in dict_outputs_columnized.items():
                expected_val = dict_expected_columnized[local_output_name]
                for idx in range(local_outputs_columnized.shape[0]):
                    local_error = (local_outputs_columnized[idx] - expected_val) / expected_val
                    error_greater_than_threshold = (local_error**2. > epsilon**2.)

                    if error_greater_than_threshold:
                        message = 'something went wrong when computing the output ' + local_output_name + ' used to initialize the vortex variables. you do seem to be trying to initialize from a circular orbit, so is it possible that the si and scaled inputs have gotten confused?'
                        print_op.log_and_raise_error(message)

    return None

def get_specific_local_initialization(abbreviated_var_name, init_options, V_init_scaled, Outputs, Integral_outputs_scaled, model, time_grids, kite_shed_or_parent_shed, tip,
                                      wake_node_or_ring, ndx, ddx=None):

    var_name = vortex_tools.get_var_name(abbreviated_var_name, kite_shed_or_parent_shed=kite_shed_or_parent_shed, tip=tip, wake_node_or_ring=wake_node_or_ring)

    # reminder: V['coll_var', ndx-1, -1, 'z', var_name] = V['z', ndx, var_name]

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
