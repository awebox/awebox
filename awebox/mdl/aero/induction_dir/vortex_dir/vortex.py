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

import awebox.mdl.aero.geometry_dir.unit_normal as unit_normal

import awebox.mdl.aero.induction_dir.general_dir.tools as general_tools
import awebox.mdl.aero.induction_dir.general_dir.flow as general_flow

import awebox.mdl.aero.induction_dir.vortex_dir.tools as vortex_tools
import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.element as obj_element
import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.element_list as obj_element_list
import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.finite_filament as obj_fin_fil
import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.semi_infinite_filament as obj_si_fil
import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.semi_infinite_right_cylinder as obj_si_cyl
import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.semi_infinite_tangential_right_cylinder as obj_si_tan_cyl
import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.semi_infinite_longitudinal_right_cylinder as obj_si_long_cyl
import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.wake_substructure as obj_wake_substructure
import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.wake as obj_wake

import awebox.mdl.aero.induction_dir.vortex_dir.alg_repr_dir.algebraic_representation as algebraic_representation
import awebox.mdl.aero.induction_dir.vortex_dir.alg_repr_dir.structure as alg_structure

import awebox.tools.vector_operations as vect_op
import awebox.tools.constraint_operations as cstr_op
import awebox.tools.print_operations as print_op
import awebox.tools.struct_operations as struct_op

from awebox.logger.logger import Logger as awelogger

import numpy as np
import casadi.tools as cas

def build(model_options, architecture, wind, variables_si, parameters):

    vortex_tools.check_positive_vortex_wake_nodes(model_options)

    vortex_representation = general_tools.get_option_from_possible_dicts(model_options, 'representation', 'vortex')
    if vortex_representation == 'alg':
        return algebraic_representation.build(model_options, architecture, wind, variables_si, parameters)
    else:
        vortex_tools.log_and_raise_unknown_representation_error(vortex_representation)

    return None


def get_model_constraints(model_options, wake, system_variables, parameters, architecture, scaling):

    cstr_list = cstr_op.ConstraintList()

    superposition_cstr = get_superposition_cstr(model_options, wake, system_variables, architecture, scaling)
    cstr_list.append(superposition_cstr)

    biot_savart_cstr = get_biot_savart_cstr(wake, model_options, system_variables, parameters, architecture, scaling)
    cstr_list.append(biot_savart_cstr)

    vortex_representation = general_tools.get_option_from_possible_dicts(model_options, 'representation', 'vortex')
    if vortex_representation == 'state':
        vortex_tools.log_and_raise_unknown_representation_error(vortex_representation)

    return cstr_list

def get_superposition_cstr(model_options, wake, system_variables, architecture, scaling):

    variables_si = system_variables['SI']
    # variables_scaled = system_variables['scaled']

    cstr_list = cstr_op.ConstraintList()

    for kite_obs in architecture.kite_nodes:
        vec_u_superposition = vortex_tools.superpose_induced_velocities_at_kite(model_options, wake, variables_si, kite_obs, architecture)

        vec_u_ind = get_induced_velocity_at_kite_si(variables_si, kite_obs)

        resi_si = vec_u_ind - vec_u_superposition

        scaling_list = []
        var_name = vortex_tools.get_induced_velocity_at_kite_name(kite_obs)
        for dim in range(resi_si.shape[0]):
            scaling_list += [float(scaling['z', var_name, dim])]

        for substructure_type in wake.get_initialized_substructure_types_with_at_least_one_element():
            substructure = wake.get_substructure(substructure_type)
            for element_type in substructure.get_initialized_element_types():
                element_list = substructure.get_list(element_type)
                number_of_elements = element_list.number_of_elements
                for element_number in range(number_of_elements):

                    if vortex_tools.not_bound_and_shed_is_obs(model_options, substructure_type, element_type, element_number, kite_obs,
                                              architecture):
                        elem_u_ind_name = vortex_tools.get_element_induced_velocity_name(substructure_type, element_type, element_number, kite_obs)
                        for dim in range(3):
                            scaling_list += [float(scaling['z', elem_u_ind_name, dim])]

        scaling_val = np.max(np.array(scaling_list))
        resi_scaled = resi_si / scaling_val

        local_cstr = cstr_op.Constraint(expr=resi_scaled,
                                        name='superposition_' + str(kite_obs),
                                        cstr_type='eq')
        cstr_list.append(local_cstr)

    return cstr_list


def get_local_biot_savart_constraint(local_resi_si, cstr_name, value_name, biot_savart_residual_assembly, scaling, num_name=None, den_name=None):

    cstr_expr = []
    if biot_savart_residual_assembly == 'lifted':
        num_scaling = scaling['z', num_name]
        den_scaling = scaling['z', den_name]
        constraint_scaling_extension = cas.vertcat(num_scaling, num_scaling, den_scaling)
    else:
        constraint_scaling_extension = scaling['z', value_name]

    for cdx in range(local_resi_si.shape[0]):
        local_scale = constraint_scaling_extension[cdx]

        local_expr = local_resi_si[cdx] / local_scale
        cstr_expr = cas.vertcat(cstr_expr, local_expr)

    local_cstr = cstr_op.Constraint(expr=cstr_expr,
                                    name=cstr_name,
                                    cstr_type='eq')
    return local_cstr


def get_biot_savart_cstr(wake, model_options, system_variables, parameters, architecture, scaling):
    biot_savart_residual_assembly = model_options['aero']['vortex']['biot_savart_residual_assembly']

    variables_si = system_variables['SI']

    cstr_list = cstr_op.ConstraintList()

    available_substructures = wake.get_initialized_substructure_types_with_at_least_one_element()

    for substructure_type in available_substructures:
        substructure = wake.get_substructure(substructure_type)

        for kite_obs in architecture.kite_nodes:
            resi_si = substructure.construct_biot_savart_residual_at_kite(model_options, variables_si, kite_obs, architecture)

            for element_type in substructure.get_initialized_element_types():

                element_list = substructure.get_list(element_type)

                number_of_elements = element_list.number_of_elements
                for element_number in range(number_of_elements):

                    if vortex_tools.not_bound_and_shed_is_obs(model_options, substructure_type, element_type, element_number,
                                                              kite_obs, architecture):

                        cstr_name = 'biot_savart_' + str(substructure_type) + '_' + str(element_type) + '_' + str(element_number) + '_' + str(kite_obs)

                        local_resi_si = resi_si[:, element_number]

                        value_name = vortex_tools.get_element_induced_velocity_name(substructure_type, element_type,
                                                                                    element_number, kite_obs)
                        num_name = vortex_tools.get_element_biot_savart_numerator_name(substructure_type, element_type, element_number, kite_obs)
                        den_name = vortex_tools.get_element_biot_savart_denominator_name(substructure_type, element_type, element_number, kite_obs)

                        local_cstr = get_local_biot_savart_constraint(local_resi_si, cstr_name, value_name,
                                                         biot_savart_residual_assembly, scaling, num_name=num_name,
                                                         den_name=den_name)
                        cstr_list.append(local_cstr)

    return cstr_list

def get_ocp_constraints(nlp_options, V, Outputs_structured, Integral_outputs, model, time_grids):

    ocp_cstr_list = cstr_op.OcpConstraintList()

    if model_is_included_in_comparison(nlp_options):
        vortex_representation = general_tools.get_option_from_possible_dicts(nlp_options, 'representation', 'vortex')
        if vortex_representation == 'alg':
            return algebraic_representation.get_ocp_constraints(nlp_options, V, Outputs_structured, Integral_outputs, model, time_grids)
        else:
            vortex_tools.log_and_raise_unknown_representation_error(vortex_representation)

    return ocp_cstr_list


def get_initialization(nlp_options, V_init_si, p_fix_num, nlp, model):

    if model_is_included_in_comparison(nlp_options):
        vortex_representation = general_tools.get_option_from_possible_dicts(nlp_options, 'representation', 'vortex')

        if vortex_representation == 'alg':
            return algebraic_representation.get_initialization(nlp_options, V_init_si, p_fix_num, nlp, model)
        else:
            vortex_tools.log_and_raise_unknown_representation_error(vortex_representation)

    return V_init_si


def get_induced_velocity_at_kite_si(variables_si, kite_obs):
    return vortex_tools.get_induced_velocity_at_kite_si(variables_si, kite_obs)


def model_is_included_in_comparison(options):
    return vortex_tools.model_is_included_in_comparison(options)


def collect_vortex_outputs(model_options, wind, wake, variables_si, outputs, architecture, scaling):

    # break early and loud if there are problems
    test_includes_visualization = model_options['aero']['vortex']['test_includes_visualization']
    test(test_includes_visualization)

    if 'vortex' not in list(outputs.keys()):
        outputs['vortex'] = {}

    kite_nodes = architecture.kite_nodes
    for kite_obs in kite_nodes:

        parent_obs = architecture.parent_map[kite_obs]

        vec_u_ind = vortex_tools.get_induced_velocity_at_kite_si(variables_si, kite_obs)
        n_hat = unit_normal.get_n_hat(model_options, parent_obs, variables_si, architecture, scaling)
        u_normalizing = vortex_tools.get_induction_factor_normalizing_speed(model_options, wind, kite_obs, parent_obs, variables_si, architecture)
        u_ind = vect_op.norm(vec_u_ind)

        local_a = general_flow.compute_induction_factor(vec_u_ind, n_hat, u_normalizing)

        vec_u_ind_from_far_wake = vortex_tools.superpose_induced_velocities_at_kite(model_options, wake, variables_si, kite_obs, architecture, substructure_types=['far'])
        u_ind_norm_from_far_wake = vect_op.norm(vec_u_ind_from_far_wake)
        u_ind_norm_from_far_wake_over_u_ref = u_ind_norm_from_far_wake / wind.get_speed_ref()

        est_truncation_error = u_ind_norm_from_far_wake / u_ind

        outputs['vortex']['u_ind' + str(kite_obs)] = u_ind
        outputs['vortex']['u_ind_norm' + str(kite_obs)] = vect_op.norm(u_ind)
        outputs['vortex']['local_a' + str(kite_obs)] = local_a

        outputs['vortex']['u_ind_from_far_wake' + str(kite_obs)] = u_ind_norm_from_far_wake
        outputs['vortex']['u_ind_from_far_wake_over_u_ref' + str(kite_obs)] = u_ind_norm_from_far_wake_over_u_ref

        outputs['vortex']['est_truncation_error' + str(kite_obs)] = est_truncation_error

    return outputs



def compute_global_performance(global_outputs, Outputs_structured, architecture):

    if 'vortex' not in global_outputs.keys():
        global_outputs['vortex'] = {}

    kite_nodes = architecture.kite_nodes

    max_est_discr_list = []
    max_u_ind_from_far_wake_over_u_ref_list = []

    all_local_a = None

    for kite in kite_nodes:

        trunc_name = 'est_truncation_error' + str(kite)
        local_a_name = 'local_a' + str(kite)
        local_normalized_far_u_ind_name = 'u_ind_from_far_wake_over_u_ref' + str(kite)

        local_est_trunc = []
        kite_local_a = []
        local_normalized_far_u_ind = []

        for ndx in range(len(Outputs_structured['coll_outputs'])):
            for ddx in range(len(Outputs_structured['coll_outputs', 0])):
                local_est_trunc = cas.vertcat(local_est_trunc, Outputs_structured['coll_outputs', ndx, ddx, 'vortex', trunc_name])
                kite_local_a = cas.vertcat(kite_local_a, Outputs_structured['coll_outputs', ndx, ddx, 'vortex', local_a_name])
                local_normalized_far_u_ind = cas.vertcat(local_normalized_far_u_ind, Outputs_structured['coll_outputs', ndx, ddx, 'vortex', local_normalized_far_u_ind_name])

        if all_local_a is None:
            all_local_a = kite_local_a
        else:
            all_local_a = cas.vertcat(all_local_a, kite_local_a)

        # todo: something here does not work correctly
        print_op.warn_about_temporary_functionality_alteration()

        max_kite_local_a = vect_op.smooth_max(kite_local_a)
        min_kite_local_a = vect_op.smooth_min(kite_local_a)
        local_max_est_discr = (max_kite_local_a - min_kite_local_a) / max_kite_local_a
        max_est_discr_list = cas.vertcat(max_est_discr_list, local_max_est_discr)

        local_max_normalized_far_u_ind = vect_op.smooth_max(local_normalized_far_u_ind)
        max_u_ind_from_far_wake_over_u_ref_list = cas.vertcat(max_u_ind_from_far_wake_over_u_ref_list, local_max_normalized_far_u_ind)

    average_local_a = vect_op.average(all_local_a)
    stdev_local_a = vect_op.stdev(all_local_a)
    global_outputs['vortex']['average_local_a'] = average_local_a
    global_outputs['vortex']['stdev_local_a'] = stdev_local_a

    max_u_ind_from_far_wake_over_u_ref = vect_op.smooth_max(max_u_ind_from_far_wake_over_u_ref_list)
    global_outputs['vortex']['max_u_ind_from_far_wake_over_u_ref'] = max_u_ind_from_far_wake_over_u_ref

    max_est_trunc = vect_op.smooth_max(local_est_trunc)
    global_outputs['vortex']['max_est_truncation_error'] = max_est_trunc

    max_est_discr = vect_op.smooth_max(max_est_discr_list)
    global_outputs['vortex']['max_est_discretization_error'] = max_est_discr

    return global_outputs


def get_dictionary_of_derivatives(outputs, architecture):
    derivative_dict = {}

    local_scaling = 1.
    for kite in architecture.kite_nodes:
        local_circulation = outputs['aerodynamics']['circulation' + str(kite)]
        derivative_dict['integrated_circulation' + str(kite)] = (local_circulation, local_scaling)

    return derivative_dict


def test_that_model_constraint_residuals_have_correct_shape(biot_savart_residual_assembly='split'):

    model_options, architecture, wind, var_struct, param_struct, variables_dict = alg_structure.construct_test_model_variable_structures(biot_savart_residual_assembly=biot_savart_residual_assembly)
    wake = build(model_options, architecture, wind, var_struct, param_struct)

    total_number_of_elements = vortex_tools.get_total_number_of_vortex_elements(model_options, architecture)
    number_of_observers = architecture.number_of_kites

    dimension_of_velocity = 3
    number_of_constraints_per_element_and_observer = dimension_of_velocity
    if biot_savart_residual_assembly == 'lifted':
        number_of_constraints_per_element_and_observer += 4

    variables_si = var_struct
    system_variables = {'SI': variables_si, 'scaled': variables_si}

    parameters = param_struct
    scaling = var_struct(1.)

    superposition_cstr = get_superposition_cstr(model_options, wake, system_variables, architecture, scaling)
    found_superposition_shape = superposition_cstr.get_expression_list('all').shape
    expected_superposition_shape = (number_of_observers * dimension_of_velocity, 1)
    cond1 = (found_superposition_shape == expected_superposition_shape)

    biot_savart_cstr = get_biot_savart_cstr(wake, model_options, system_variables, parameters, architecture, scaling)
    found_biot_savart_shape = biot_savart_cstr.get_expression_list('all').shape
    expected_biot_savart_len = total_number_of_elements * number_of_observers * number_of_constraints_per_element_and_observer
    if not vortex_tools.model_includes_induced_velocity_from_kite_bound_on_itself(model_options, variables_si, architecture):
        expected_biot_savart_len -= number_of_observers * number_of_constraints_per_element_and_observer
    expected_biot_savart_shape = (expected_biot_savart_len, 1)
    cond2 = (found_biot_savart_shape == expected_biot_savart_shape)

    criteria = cond1 and cond2
    if not criteria:
        message = 'an incorrect number of induction residuals have been defined for the algebraic-representation vortex wake'
        print_op.log_and_raise_error(message)


def test_that_model_doesnt_include_velocity_from_bound_kite_on_itsself():
    wake_nodes = 3
    model_options, architecture, wind, var_struct, param_struct, variables_dict = alg_structure.construct_test_model_variable_structures(wake_nodes=wake_nodes, number_of_kites=2)
    includes = vortex_tools.model_includes_induced_velocity_from_kite_bound_on_itself(model_options, var_struct, architecture)
    if includes:
        message = 'something went wrong: there seems to be a induced velocity variable defined that describes the induction from a kites bound vortex on itself'
        print_op.log_and_raise_error(message)
    return None


def test_that_get_shedding_kite_from_element_number_tool_works_correctly():
    wake_nodes = 3
    model_options, architecture, wind, var_struct, param_struct, variables_dict = alg_structure.construct_test_model_variable_structures(wake_nodes=wake_nodes, number_of_kites=2)
    element_type = 'finite_filament'

    near_rings = wake_nodes - 1

    from_element_number_to_expected_kite_shed_dict = {'bound':
                                                          {0: 2,
                                                           1: 3,
                                                           2: 'error',
                                                           -1: 3
                                                           },
                                                        'near':
                                                            {0: 2,
                                                             (near_rings * 3) - 1: 2,
                                                             (near_rings * 3): 3,
                                                             (near_rings * 3) * 2 - 1: 3,
                                                             (near_rings * 3) * 2: 'error',
                                                             -1: 3
                                                            }
                                                       }

    for wake_type in from_element_number_to_expected_kite_shed_dict.keys():
        for element_number, expected_output in from_element_number_to_expected_kite_shed_dict[wake_type].items():
            try:
                found_output = vortex_tools.get_shedding_kite_from_element_number(model_options, wake_type, element_type, element_number, architecture, suppress_error_logging=True)
            except:
                found_output = 'error'

            if not(expected_output == found_output):
                message = 'wrong shedding kite found (' + str(found_output) + ') for ' + wake_type + ' element number ' + str(element_number)
                print_op.log_and_raise_error(message)

    return None


def test(test_includes_visualization=False):

    message = 'checking vortex model functionality...'
    awelogger.logger.info(message)

    vect_op.test_altitude()
    vect_op.test_elliptic_k()
    vect_op.test_elliptic_e()
    vect_op.test_elliptic_pi()

    obj_element.test()
    obj_element_list.test()

    obj_fin_fil.test(test_includes_visualization)
    obj_si_fil.test(test_includes_visualization)
    obj_si_cyl.test()
    obj_si_tan_cyl.test(test_includes_visualization)
    obj_si_long_cyl.test(test_includes_visualization)

    obj_wake_substructure.test()
    obj_wake.test()

    test_that_model_doesnt_include_velocity_from_bound_kite_on_itsself()

    algebraic_representation.test(test_includes_visualization)

    for biot_savart_residual_assembly in ['division', 'split', 'lifted']:
        test_that_model_constraint_residuals_have_correct_shape(biot_savart_residual_assembly)

    test_that_get_shedding_kite_from_element_number_tool_works_correctly()

    message = 'Vortex model functionality checked.'
    awelogger.logger.info(message)

    return None

# test()
