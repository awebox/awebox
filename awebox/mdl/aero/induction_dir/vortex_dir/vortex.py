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

def build(model_options, architecture, wind, variables_si, variables_scaled, parameters):

    vortex_tools.check_positive_vortex_wake_nodes(model_options)

    vortex_representation = general_tools.get_option_from_possible_dicts(model_options, 'representation', 'vortex')
    if vortex_representation == 'alg':
        return algebraic_representation.build(model_options, architecture, wind, variables_si, variables_scaled, parameters)
    else:
        vortex_tools.log_and_raise_unknown_representation_error(vortex_representation)

    return None


def get_model_constraints(model_options, wake, system_variables, parameters, outputs, architecture, scaling):

    vortex_representation = general_tools.get_option_from_possible_dicts(model_options, 'representation', 'vortex')
    if vortex_representation == 'state':
        vortex_tools.log_and_raise_unknown_representation_error(vortex_representation)

    degree_of_induced_velocity_lifting = model_options['aero']['vortex']['degree_of_induced_velocity_lifting']

    cstr_list = cstr_op.ConstraintList()

    wingtip_position_cstr = get_wingtip_position_cstr(model_options, system_variables['SI'], outputs, architecture, scaling)
    cstr_list.append(wingtip_position_cstr)

    if degree_of_induced_velocity_lifting == 1:
        unlifted_cstr = get_unlifted_cstr(wake, system_variables, parameters, architecture, scaling)
        cstr_list.append(unlifted_cstr)

    elif degree_of_induced_velocity_lifting >= 2:
        superposition_cstr = get_superposition_cstr(model_options, wake, system_variables, architecture, scaling)
        cstr_list.append(superposition_cstr)

        biot_savart_cstr = get_biot_savart_cstr(wake, model_options, system_variables, parameters, architecture, scaling)
        cstr_list.append(biot_savart_cstr)


    return cstr_list

def get_wingtip_position_cstr(model_options, variables_si, outputs, architecture, scaling):

    message = 'adding shedding position constraints'
    print_op.base_print(message, level='info')

    resi_expr = []
    for kite_shed in architecture.kite_nodes:
        for tip in ['int', 'ext']:
            wake_node = 0

            fixing_name = vortex_tools.get_wake_node_position_name(kite_shed=kite_shed, tip=tip,
                                                                   wake_node=wake_node)

            node_position_si = vortex_tools.get_wake_node_position_si(model_options, kite_shed, tip, wake_node,
                                      variables_si=variables_si, scaling=scaling)
            wingtip_position_si = outputs['aerodynamics']['wingtip_' + tip + str(kite_shed)]
            local_resi_si = wingtip_position_si - node_position_si
            local_resi_scaled = struct_op.var_si_to_scaled('z', fixing_name, local_resi_si, scaling)
            resi_expr = cas.vertcat(resi_expr, local_resi_scaled)

    local_cstr = cstr_op.Constraint(expr=resi_expr,
                                    name='wx_wingtip_shed',
                                    cstr_type='eq')
    print_op.close_progress()

    return local_cstr

def try_getting_wingtip_position_si(kite_shed, tip, Outputs, ndx, ddx=None):
    if ddx is None:
        ndx = ndx - 1
        ddx = -1

    wingtip_position_si = Outputs['coll_outputs', ndx, ddx, 'aerodynamics', 'wingtip_' + tip + str(kite_shed)]
    return wingtip_position_si


def get_unlifted_cstr(wake, system_variables, parameters, architecture, scaling):

    variables_si = system_variables['SI']

    cstr_list = cstr_op.ConstraintList()

    for kite_obs in architecture.kite_nodes:
        parent_obs = architecture.parent_map[kite_obs]

        x_obs = variables_si['x']['q' + str(kite_obs) + str(parent_obs)]
        print_op.warn_about_temporary_functionality_alteration()
        # was calculate until may 30th.
        # vec_u_computed = wake.evaluate_total_biot_savart_induction(x_obs=x_obs)
        vec_u_computed = wake.calculate_total_biot_savart_at_x_obs(system_variables['scaled'], parameters, x_obs=x_obs)

        vec_u_ind = get_induced_velocity_at_kite_si(variables_si, kite_obs)
        resi_si = vec_u_ind - vec_u_computed

        var_name = vortex_tools.get_induced_velocity_at_kite_name(kite_obs)
        scaling_val = scaling['z', var_name]
        resi_scaled = resi_si / scaling_val

        local_cstr = cstr_op.Constraint(expr=resi_scaled,
                                        name='wu_ind_induced_velocity' + str(kite_obs),
                                        cstr_type='eq')
        cstr_list.append(local_cstr)

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
                                        name='wu_ind_superposition_' + str(kite_obs),
                                        cstr_type='eq')
        cstr_list.append(local_cstr)

    return cstr_list


def get_local_biot_savart_constraint(local_resi_si, cstr_name, value_name, degree_of_induced_velocity_lifting, scaling, num_name=None, den_name=None):

    cstr_expr = []

    if degree_of_induced_velocity_lifting == 1:
        message = 'this constraint (get_local_biot_savart_constraint) should not have been called in this circumstance, because the induced velocities are unlifted'
        print_op.log_and_raise_error(message)

    elif degree_of_induced_velocity_lifting == 2:
        constraint_scaling_extension = scaling['z', value_name]

    elif degree_of_induced_velocity_lifting == 3:
        num_scaling = scaling['z', num_name]
        den_scaling = scaling['z', den_name]
        value_scaling = scaling['z', value_name]
        top_scaling = num_scaling
        constraint_scaling_extension = cas.vertcat(top_scaling, num_scaling, den_scaling)

    else:
        message = 'unexpected degree_of_induced_velocity_lifting (' + str(degree_of_induced_velocity_lifting) + ')'
        print_op.log_and_raise_error(message)

    for cdx in range(local_resi_si.shape[0]):
        local_scale = constraint_scaling_extension[cdx]

        local_expr = local_resi_si[cdx] / local_scale
        cstr_expr = cas.vertcat(cstr_expr, local_expr)

    local_cstr = cstr_op.Constraint(expr=cstr_expr,
                                    name=cstr_name,
                                    cstr_type='eq')
    return local_cstr


def get_biot_savart_cstr(wake, model_options, system_variables, parameters, architecture, scaling):

    this_is_not_a_test = 'model_bounds' in model_options.keys()
    all_scaling_values_are_unity = vect_op.norm(scaling.cat - cas.DM.ones(scaling.shape)) < 1.e-8
    if this_is_not_a_test and all_scaling_values_are_unity:
        message = 'something went terribly wrong when passing scaling values into biot-savart constraint'
        print_op.log_and_raise_error(message)

    degree_of_induced_velocity_lifting = model_options['aero']['vortex']['degree_of_induced_velocity_lifting']

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

                        cstr_name = 'wu_ind_biot_savart_' + str(substructure_type) + '_' + str(element_type) + '_' + str(element_number) + '_' + str(kite_obs)

                        local_resi_si = resi_si[:, element_number]

                        value_name = vortex_tools.get_element_induced_velocity_name(substructure_type, element_type,
                                                                                    element_number, kite_obs)
                        num_name = vortex_tools.get_element_biot_savart_numerator_name(substructure_type, element_type, element_number, kite_obs)
                        den_name = vortex_tools.get_element_biot_savart_denominator_name(substructure_type, element_type, element_number, kite_obs)

                        # double check that we actually passed the scaling values in
                        if this_is_not_a_test:
                            biot_savart_scaling_values = scaling['z', value_name]
                            if degree_of_induced_velocity_lifting == 3:
                                biot_savart_scaling_values = cas.vertcat(scaling['z', value_name], scaling['z', num_name], scaling['z', den_name])
                            for idx in range(biot_savart_scaling_values.shape[0]):
                                if(biot_savart_scaling_values[idx] == 1):
                                    message = 'there is (at least one) unit value in the biot-savart scaling entries. therefore, something must have gone wrong when passing them into the biot-savart constraint.'
                                    print_op.log_and_raise_error(message)

                        local_cstr = get_local_biot_savart_constraint(local_resi_si, cstr_name, value_name,
                                                         degree_of_induced_velocity_lifting, scaling, num_name=num_name,
                                                         den_name=den_name)
                        cstr_list.append(local_cstr)

    return cstr_list

def get_ocp_constraints(nlp_options, V, P, Xdot, Outputs_structured, Integral_outputs, model, time_grids):

    ocp_cstr_list = cstr_op.OcpConstraintList()

    if model_is_included_in_comparison(nlp_options):
        vortex_representation = general_tools.get_option_from_possible_dicts(nlp_options, 'representation', 'vortex')
        if vortex_representation == 'alg':
            return algebraic_representation.get_ocp_constraints(nlp_options, V, P, Xdot, Outputs_structured, Integral_outputs, model, time_grids)
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
    return vortex_tools.get_induced_velocity_at_kite_si(kite_obs, variables_si=variables_si)


def model_is_included_in_comparison(options):
    return vortex_tools.model_is_included_in_comparison(options)


def collect_vortex_outputs(model_options, wind, wake, system_variables, parameters, outputs, architecture):

    # break early and loud if there are problems
    test_includes_visualization = model_options['aero']['vortex']['test_includes_visualization']
    test(test_includes_visualization)

    if 'vortex' not in list(outputs.keys()):
        outputs['vortex'] = {}

    variables_si = system_variables['SI']
    kite_nodes = architecture.kite_nodes
    for kite_obs in kite_nodes:

        parent_obs = architecture.parent_map[kite_obs]
        q_kite = variables_si['x']['q' + str(kite_obs) + str(parent_obs)]

        vec_u_ind = vortex_tools.get_induced_velocity_at_kite_si(kite_obs, variables_si=variables_si)
        n_hat = unit_normal.get_n_hat(model_options, parent_obs, variables_si, architecture)
        u_normalizing = vortex_tools.get_induction_factor_normalizing_speed(model_options, wind, kite_obs, parent_obs, variables_si, architecture)
        u_ind_norm = vect_op.norm(vec_u_ind)

        local_a = general_flow.compute_induction_factor(vec_u_ind, n_hat, u_normalizing)

        outputs['vortex']['vec_u_ind' + str(kite_obs)] = vec_u_ind
        outputs['vortex']['u_ind_norm' + str(kite_obs)] = u_ind_norm
        outputs['vortex']['local_a' + str(kite_obs)] = local_a

        x_obs = q_kite

        u_ref = wind.get_speed_ref()

        substructure_types = wake.get_initialized_substructure_types()
        for substructure in substructure_types:
            print_op.warn_about_temporary_functionality_alteration()
            # may 4th version is calculate
            # calculate is max_iterations_exceeded in power_problem predictor step
            # evaluate is restoration failed
            # vec_u_ind_from_substructure = wake.get_substructure(substructure).evaluate_total_biot_savart_induction(x_obs=x_obs)
            vec_u_ind_from_substructure = wake.get_substructure(substructure).calculate_total_biot_savart_at_x_obs(system_variables['scaled'], parameters, x_obs=x_obs)
            u_ind_norm_from_substructure = vect_op.norm(vec_u_ind_from_substructure)
            u_ind_norm_from_substructure_over_total = u_ind_norm_from_substructure / u_ind_norm
            u_ind_norm_from_substructure_over_ref = u_ind_norm_from_substructure / u_ref

            base_name = 'u_ind_norm_from_' + substructure + '_wake'
            outputs['vortex'][base_name + str(kite_obs)] = u_ind_norm_from_substructure
            outputs['vortex'][base_name + '_over_total' + str(kite_obs)] = u_ind_norm_from_substructure_over_total
            outputs['vortex'][base_name + '_over_ref' + str(kite_obs)] = u_ind_norm_from_substructure_over_ref

        if 'rotation' in outputs.keys():
            ehat_normal = outputs['rotation']['ehat_normal' + str(parent_obs)]
            ehat_tangential = outputs['rotation']['ehat_tangential' + str(kite_obs)]
            ehat_radial = outputs['rotation']['ehat_radial' + str(kite_obs)]

            ehat_chord = outputs['aerodynamics']['ehat_chord' + str(kite_obs)]
            ehat_span = outputs['aerodynamics']['ehat_span' + str(kite_obs)]
            ehat_up = outputs['aerodynamics']['ehat_up' + str(kite_obs)]

            xhat = vect_op.xhat_dm()
            yhat = vect_op.yhat_dm()
            zhat = vect_op.zhat_dm()

            rot_dir_dict = {'radial': ehat_radial, 'tangential': ehat_tangential, 'normal': ehat_normal, 'chord': ehat_chord, 'span': ehat_span, 'up': ehat_up, 'x': xhat, 'y': yhat, 'z': zhat}
            for rot_name, rot_ehat in rot_dir_dict.items():
                outputs['vortex']['u_ind_' + rot_name + str(kite_obs)] = cas.mtimes(vec_u_ind.T, rot_ehat)

            print_op.warn_about_temporary_functionality_alteration(reason='there is something really wrong here')
            outputs['vortex']['vec_u_ind' + str(kite_obs) + '_xi_00_neg_xhat' ] = -1. * cas.mtimes(vec_u_ind.T, xhat)

            b_ref = parameters['theta0', 'geometry', 'b_ref']
            for extra_xi_obs in model_options['aero']['vortex']['additional_induction_observation_points']:
                x_obs_extra = q_kite + extra_xi_obs * b_ref * ehat_span
                # extra_ui_calculate = wake.calculate_total_biot_savart_at_x_obs(system_variables['scaled'], parameters, x_obs=x_obs_extra)
                extra_ui_evaluate = wake.evaluate_total_biot_savart_induction(x_obs=x_obs_extra)
                # outputs['vortex']['vec_u_ind' + str(kite_obs) + '_xi_' + str(extra_xi_obs) + '_neg_xhat_calculate' ] = -1. * cas.mtimes(extra_ui_calculate.T, xhat)
                outputs['vortex']['vec_u_ind' + str(kite_obs) + '_xi_' + str(extra_xi_obs) + '_neg_xhat_evaluate' ] = -1. * cas.mtimes(extra_ui_evaluate.T, xhat)


    total_truncation_error_at_kite = cas.DM(0.)
    base_name = 'u_ind_norm_from_far_wake'
    for kite_obs in kite_nodes:
        total_truncation_error_at_kite += outputs['vortex'][base_name + '_over_ref' + str(kite_obs)]
    est_truncation_error = total_truncation_error_at_kite / float(len(kite_nodes))
    outputs['vortex']['est_truncation_error'] = est_truncation_error

    return outputs

def test_that_wake_related_ocp_variables_are_all_constrained_using_cstr_names(nlp_options, V, ocp_cstr_list):

    if nlp_options['collocation']['name_constraints']:

        message = 'double-checking that all vortex variables are constrained...'
        print_op.base_print(message, level='info')

        identifying_strings = ['wx', 'wg', 'wu']

        count_variable = 0
        for label in V.labels():
            for local_str in identifying_strings:
                if (',' + local_str + '_') in label:
                    count_variable += 1

        count_cstr = 0
        eq_name_list = ocp_cstr_list.get_name_list('eq')
        for eq_name in eq_name_list:
            for local_str in identifying_strings:
                if local_str in eq_name:
                    count_cstr += 1

        if count_variable != count_cstr:
            message = 'something went wrong with the wake portion of the ocp generation. the number of associated variables is ' + str(count_variable)
            message += ', while the number of associated constraints is ' + str(count_cstr)
            print_op.log_and_raise_error(message)

    return None



def get_dictionary_of_derivatives(outputs, architecture):
    derivative_dict = {}

    local_scaling = 1.
    for kite in architecture.kite_nodes:
        local_circulation = outputs['aerodynamics']['circulation' + str(kite)]
        derivative_dict['integrated_circulation' + str(kite)] = (local_circulation, local_scaling)

    return derivative_dict


def test_that_model_constraint_residuals_have_correct_shape(degree_of_induced_velocity_lifting=3):

    model_options, architecture, wind, var_struct, param_struct, variables_dict = alg_structure.construct_test_model_variable_structures(degree_of_induced_velocity_lifting=degree_of_induced_velocity_lifting)
    wake = build(model_options, architecture, wind, var_struct, var_struct, param_struct)

    total_number_of_elements = vortex_tools.get_total_number_of_vortex_elements(model_options, architecture)
    number_of_observers = architecture.number_of_kites

    if degree_of_induced_velocity_lifting == 1:
        message = 'this test (test_that_model_constraint_residuals_have_correct_shape) is not appropriate for the unlifted velocities case, as there should not *be* any constraints applied'
        print_op.log_and_raise_error(message)

    if degree_of_induced_velocity_lifting >= 2:
        dimension_of_velocity = 3
        number_of_constraints_per_element_and_observer = dimension_of_velocity

    if degree_of_induced_velocity_lifting == 3:
        len_bs_numerator = 3
        len_bs_denominator = 1
        number_of_constraints_per_element_and_observer += len_bs_numerator + len_bs_denominator

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

    return None

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

    for degree_of_induced_velocity_lifting in [2, 3]:
        test_that_model_constraint_residuals_have_correct_shape(degree_of_induced_velocity_lifting)

    test_that_get_shedding_kite_from_element_number_tool_works_correctly()

    message = 'Vortex model functionality checked.'
    awelogger.logger.info(message)

    return None

# test()
