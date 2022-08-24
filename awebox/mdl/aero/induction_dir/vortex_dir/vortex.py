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

import awebox.mdl.aero.induction_dir.general_dir.tools as general_tools
import awebox.mdl.aero.induction_dir.general_dir.unit_normal as unit_normal
import awebox.mdl.aero.induction_dir.general_dir.flow as general_flow

import awebox.mdl.aero.induction_dir.vortex_dir.tools as vortex_tools
import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.element as obj_element
import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.element_list as obj_element_list
import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.finite_filament as obj_fin_fil
import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.semi_infinite_filament as obj_si_fil
import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.semi_infinite_cylinder as obj_si_cyl
import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.semi_infinite_tangential_cylinder as obj_si_tan_cyl
import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.semi_infinite_longitudinal_cylinder as obj_si_long_cyl
import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.wake_substructure as obj_wake_substructure
import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.wake as obj_wake

import awebox.mdl.aero.induction_dir.vortex_dir.alg_repr_dir.algebraic_representation as algebraic_representation
import awebox.mdl.aero.induction_dir.vortex_dir.alg_repr_dir.structure as alg_structure
import awebox.mdl.aero.induction_dir.vortex_dir.state_repr_dir.state_representation as state_representation

import awebox.tools.vector_operations as vect_op
import awebox.tools.constraint_operations as cstr_op
import awebox.ocp.ocp_constraint as ocp_constraint
import awebox.tools.print_operations as print_op

from awebox.logger.logger import Logger as awelogger

import numpy as np


def build(model_options, architecture, wind, variables_si, parameters):

    vortex_tools.check_positive_vortex_wake_nodes(model_options)

    vortex_representation = general_tools.get_option_from_possible_dicts(model_options, 'representation', 'vortex')
    if vortex_representation == 'alg':
        return algebraic_representation.build(model_options, architecture, wind, variables_si, parameters)
    elif vortex_representation == 'state':
        print_op.warn_about_temporary_functionality_removal(location='vortex.state')
        return state_representation.build(model_options, architecture, wind, variables_si, parameters)
    else:
        vortex_tools.log_and_raise_unknown_representation_error(vortex_representation)

    return None


def get_model_constraints(model_options, wake, wind, variables_si, architecture):

    cstr_list = cstr_op.ConstraintList()

    superposition_cstr = get_superposition_cstr(wake, wind, variables_si, architecture)
    cstr_list.append(superposition_cstr)

    biot_savart_cstr = get_biot_savart_cstr(wake, wind, variables_si, architecture)
    cstr_list.append(biot_savart_cstr)

    vortex_representation = general_tools.get_option_from_possible_dicts(model_options, 'representation', 'vortex')
    if vortex_representation == 'state':
        print_op.warn_about_temporary_functionality_removal(location='vortex.model_constraints:state_convection_resi')

    return cstr_list

def get_superposition_cstr(wake, wind, variables_si, architecture):

    cstr_list = cstr_op.ConstraintList()

    u_ref = wind.get_speed_ref()

    for kite_obs in architecture.kite_nodes:
        vec_u_superposition = vortex_tools.superpose_induced_velocities_at_kite(wake, variables_si, kite_obs)

        vec_u_ind = get_induced_velocity_at_kite_si(variables_si, kite_obs)

        resi = (vec_u_ind - vec_u_superposition) / u_ref

        local_cstr = cstr_op.Constraint(expr=resi,
                                        name='superposition_' + str(kite_obs),
                                        cstr_type='eq')
        cstr_list.append(local_cstr)

    return cstr_list

def get_biot_savart_cstr(wake, wind, variables_si, architecture):

    cstr_list = cstr_op.ConstraintList()

    for substructure_type in wake.get_initialized_substructure_types_with_at_least_one_element():

        for kite_obs in architecture.kite_nodes:
            resi = wake.get_substructure(substructure_type).construct_biot_savart_at_kite_residuals(wind, variables_si, kite_obs,
                                                                                                    architecture.parent_map[kite_obs])

            local_cstr = cstr_op.Constraint(expr=resi,
                                            name='biot_savart_' + str(substructure_type) + '_' + str(kite_obs),
                                            cstr_type='eq')
            cstr_list.append(local_cstr)

    return cstr_list

def get_ocp_constraints(nlp_options, V, Outputs, Integral_outputs, model, time_grids):

    ocp_cstr_list = ocp_constraint.OcpConstraintList()

    if model_is_included_in_comparison(nlp_options):
        vortex_representation = general_tools.get_option_from_possible_dicts(nlp_options, 'representation', 'vortex')
        if vortex_representation == 'alg':
            return algebraic_representation.get_ocp_constraints(nlp_options, V, Outputs, Integral_outputs, model, time_grids)
        elif vortex_representation == 'state':
            print_op.warn_about_temporary_functionality_removal(location='vortex.state')
            return state_representation.get_ocp_constraints(nlp_options, V, Outputs, Integral_outputs, model, time_grids)
        else:
            vortex_tools.log_and_raise_unknown_representation_error(vortex_representation)

    return ocp_cstr_list


def get_initialization(nlp_options, V_init, p_fix_num, nlp, model):

    if model_is_included_in_comparison(nlp_options):
        vortex_representation = general_tools.get_option_from_possible_dicts(nlp_options, 'representation', 'vortex')

        if vortex_representation == 'alg':
            return algebraic_representation.get_initialization(nlp_options, V_init, p_fix_num, nlp, model)
        elif vortex_representation == 'state':
            print_op.warn_about_temporary_functionality_removal(location='vortex.state')
            return state_representation.get_initialization(nlp_options, V_init, p_fix_num, nlp, model)
        else:
            vortex_tools.log_and_raise_unknown_representation_error(vortex_representation)

    return V_init

def get_induced_velocity_at_kite_si(variables_si, kite_obs):
    return vortex_tools.get_induced_velocity_at_kite_si(variables_si, kite_obs)

def model_is_included_in_comparison(options):
    comparison_labels = general_tools.get_option_from_possible_dicts(options, 'comparison_labels', 'vortex')
    any_vor = any(label[:3] == 'vor' for label in comparison_labels)
    return any_vor

def collect_vortex_outputs(model_options, wind, wake, variables_si, outputs, parameters, architecture):

    # break early and loud if there are problems
    test_includes_visualization = model_options['aero']['vortex']['test_includes_visualization']
    test(test_includes_visualization)

    if 'vortex' not in list(outputs.keys()):
        outputs['vortex'] = {}

    kite_nodes = architecture.kite_nodes
    for kite_obs in kite_nodes:

        parent_obs = architecture.parent_map[kite_obs]

        vec_u_ind = vortex_tools.get_induced_velocity_at_kite_si(variables_si, kite_obs)
        n_hat = unit_normal.get_n_hat(model_options, parent_obs, variables_si, parameters, architecture)
        u_normalizing = vortex_tools.get_induction_factor_normalizing_speed(model_options, wind, kite_obs, parent_obs, variables_si, architecture)
        u_ind = vect_op.norm(vec_u_ind)

        local_a = general_flow.compute_induction_factor(vec_u_ind, n_hat, u_normalizing)

        vec_u_ind_from_far_wake = vortex_tools.superpose_induced_velocities_at_kite(wake, variables_si, kite_obs, substructure_types = ['far'])
        u_ind_from_far_wake = vect_op.norm(vec_u_ind_from_far_wake)
        u_ind_from_far_wake_over_u_ref = u_ind_from_far_wake / wind.get_speed_ref()

        est_truncation_error = u_ind_from_far_wake / u_ind

        outputs['vortex']['u_ind' + str(kite_obs)] = u_ind
        outputs['vortex']['u_ind_norm' + str(kite_obs)] = vect_op.norm(u_ind)
        outputs['vortex']['local_a' + str(kite_obs)] = local_a

        outputs['vortex']['u_ind_from_far_wake' + str(kite_obs)] = u_ind_from_far_wake
        outputs['vortex']['u_ind_from_far_wake_over_u_ref' + str(kite_obs)] = u_ind_from_far_wake_over_u_ref

        outputs['vortex']['est_truncation_error' + str(kite_obs)] = est_truncation_error

    return outputs

def compute_global_performance(power_and_performance, plot_dict):

    kite_nodes = plot_dict['architecture'].kite_nodes

    max_est_trunc_list = []
    max_est_discr_list = []
    max_u_ind_from_far_wake_over_u_ref_list = []

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

        local_max_u_ind_from_far_wake_over_u_ref = np.max(np.array(plot_dict['outputs']['vortex']['u_ind_from_far_wake_over_u_ref' + str(kite)]))
        max_u_ind_from_far_wake_over_u_ref_list += [local_max_u_ind_from_far_wake_over_u_ref]

    average_local_a = np.average(all_local_a)
    power_and_performance['vortex_average_local_a'] = average_local_a

    stdev_local_a = np.std(all_local_a)
    power_and_performance['vortex_stdev_local_a'] = stdev_local_a

    max_u_ind_from_far_wake_over_u_ref = np.max(np.array(max_u_ind_from_far_wake_over_u_ref_list))
    power_and_performance['vortex_max_u_ind_from_far_wake_over_u_ref'] = max_u_ind_from_far_wake_over_u_ref

    max_est_trunc = np.max(np.array(max_est_trunc_list))
    power_and_performance['vortex_max_est_truncation_error'] = max_est_trunc

    max_est_discr = np.max(np.array(max_est_discr_list))
    power_and_performance['vortex_max_est_discretization_error'] = max_est_discr

    return power_and_performance


def get_derivative_dict_for_alongside_integration(outputs, architecture):
    derivative_dict = {}

    for kite in architecture.kite_nodes:
        local_circulation = outputs['aerodynamics']['circulation' + str(kite)]
        derivative_dict['integrated_circulation' + str(kite)] = local_circulation

    return derivative_dict

def test_that_model_constraint_residuals_have_correct_shape():

    options, architecture, wind, var_struct, param_struct, variables_dict = alg_structure.construct_test_model_variable_structures()
    wake = build(options, architecture, wind, var_struct, param_struct)

    total_number_of_elements = vortex_tools.get_total_number_of_vortex_elements(options, architecture)
    number_of_observers = architecture.number_of_kites
    dimension_of_velocity = 3

    variables_si = var_struct

    superposition_cstr = get_superposition_cstr(wake, wind, variables_si, architecture)
    found_superposition_shape = superposition_cstr.get_expression_list('all').shape
    expected_superposition_shape = (number_of_observers * dimension_of_velocity, 1)
    cond1 = (found_superposition_shape == expected_superposition_shape)

    biot_savart_cstr = get_biot_savart_cstr(wake, wind, variables_si, architecture)
    found_biot_savart_shape = biot_savart_cstr.get_expression_list('all').shape
    expected_biot_savart_shape = (total_number_of_elements * number_of_observers * dimension_of_velocity, 1)
    cond2 = (found_biot_savart_shape == expected_biot_savart_shape)

    criteria = cond1 and cond2
    if not criteria:
        message = 'an incorrect number of induction residuals have been defined for the algebraic-representation vortex wake'
        awelogger.logger.error(message)
        raise Exception(message)


def test(test_includes_visualization=False):

    message = 'check vortex model functionality...'
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

    algebraic_representation.test(test_includes_visualization)
    state_representation.test(test_includes_visualization)

    test_that_model_constraint_residuals_have_correct_shape()

    message = 'Vortex model functionality checked.'
    awelogger.logger.info(message)

    return None

# test()
