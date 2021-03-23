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
actuator_disk model of awebox aerodynamics
sets up the axial-induction actuator disk equation
currently for untilted rotor with no tcf.
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2017-21
- edit: jochem de schutter, alu-fr 2019
'''

import casadi as cas
import numpy as np
from awebox.logger.logger import Logger as awelogger

import awebox.mdl.aero.induction_dir.actuator_dir.geom as actuator_geom
import awebox.mdl.aero.induction_dir.actuator_dir.flow as actuator_flow
import awebox.mdl.aero.induction_dir.actuator_dir.coeff as actuator_coeff
import awebox.mdl.aero.induction_dir.actuator_dir.force as actuator_force

import awebox.mdl.aero.induction_dir.tools_dir.flow as general_flow
import awebox.mdl.aero.induction_dir.tools_dir.geom as general_geom

import awebox.tools.print_operations as print_op
import awebox.tools.vector_operations as vect_op
import awebox.tools.constraint_operations as cstr_op

def get_actuator_cstr(model_options, atmos, wind, variables, parameters, outputs, architecture):

    cstr_list = cstr_op.ConstraintList()

    act_comp_labels = actuator_flow.get_actuator_comparison_labels(model_options)

    layer_parent_map = architecture.layer_nodes
    for parent in layer_parent_map:

        for label in act_comp_labels:
            induction_factor_cstr = get_induction_factor_cstr(model_options, atmos, wind, variables, outputs, parameters, parent, architecture, label)
            cstr_list.append(induction_factor_cstr)

        actuator_orientation_cstr = get_actuator_orientation_cstr(model_options, wind, parent, variables, parameters, architecture)
        cstr_list.append(actuator_orientation_cstr)

        gamma_cstr = actuator_flow.get_gamma_cstr(parent, variables)
        cstr_list.append(gamma_cstr)

        children = architecture.kites_map[parent]
        for kite in children:
            a_assignment_cstr = actuator_flow.get_induction_factor_assignment_cstr(model_options, variables, kite, parent)
            cstr_list.append(a_assignment_cstr)

            varrho_and_psi_cstr = actuator_geom.get_varrho_and_psi_cstr(model_options, kite, variables, parameters, architecture)
            cstr_list.append(varrho_and_psi_cstr)

        bar_varrho_cstr = actuator_geom.get_bar_varrho_cstr(model_options, parent, variables, architecture)
        cstr_list.append(bar_varrho_cstr)

    return cstr_list




def get_induction_factor_cstr(model_options, atmos, wind, variables, outputs, parameters, parent, architecture, label):

    if label == 'qaxi':
        resi = get_momentum_theory_residual(model_options, atmos, wind, variables, outputs, parameters, parent, architecture, label)

    elif label == 'qasym':
        resi = get_steady_asym_pitt_peters_residual(model_options, atmos, wind, variables, parameters, outputs, parent, architecture, label)

    elif label == 'uaxi':
        resi = get_unsteady_axi_pitt_peters_residual(model_options, atmos, wind, variables, outputs, parameters, parent, architecture, label)

    elif label == 'uasym':
        resi = get_unsteady_asym_pitt_peters_residual(model_options, atmos, wind, variables, parameters, outputs, parent,
                                               architecture, label)

    else:
        resi = []
        awelogger.logger.error('model not yet implemented.')

    name = 'actuator_induction_factor_' + label + '_' + str(parent)
    cstr = cstr_op.Constraint(expr=resi,
                              name=name,
                              cstr_type='eq')

    return cstr

def get_momentum_theory_residual(model_options, atmos, wind, variables, outputs, parameters, parent, architecture, label):

    a_var = actuator_flow.get_a_var(variables, parent, label)
    a_ref = actuator_flow.get_a_ref(model_options)

    thrust = actuator_force.get_actuator_thrust(model_options, variables, parameters, outputs, parent, architecture)

    area = actuator_geom.get_actuator_area(model_options, parent, variables, parameters)
    qzero = actuator_flow.get_actuator_dynamic_pressure(model_options, atmos, wind, variables, parent, architecture)

    thrust_den = qzero * area
    thrust_ref = actuator_coeff.get_thrust_ref(model_options, atmos, wind, parameters)

    corr_val = actuator_flow.get_corr_val(model_options, atmos, wind, variables, outputs, parameters, parent, architecture, label)
    LLinv11 = 4. * corr_val

    term_1 = 0.
    term_2 = LLinv11 * a_var * thrust_den
    term_3 = -1. * thrust

    resi_unscaled = term_1 + term_2 + term_3

    # term_2_ref = 4. * (1. - a_ref) * a_ref * thrust_ref
    term_3_ref = thrust_ref

    resi = resi_unscaled / term_3_ref

    return resi

def get_unsteady_axi_pitt_peters_residual(model_options, atmos, wind, variables, outputs, parameters, parent, architecture, label):

    a_var = actuator_flow.get_a_var(variables, parent, label)
    da_dt = actuator_flow.get_da_var(variables, parent, label)
    a_ref = actuator_flow.get_a_ref(model_options)

    thrust = actuator_force.get_actuator_thrust(model_options, variables, parameters, outputs, parent, architecture)
    area = actuator_geom.get_actuator_area(model_options, parent, variables, parameters)
    qzero = actuator_flow.get_actuator_dynamic_pressure(model_options, atmos, wind, variables, parent, architecture)

    corr_val = actuator_flow.get_corr_val(model_options, atmos, wind, variables, outputs, parameters, parent, architecture, label)
    LLinv11 = 4. * corr_val

    MM = actuator_coeff.get_MM_matrix()
    MM11 = MM[0, 0]

    t_star_num = actuator_coeff.get_t_star_numerator_val(variables, parameters, parent)
    t_star_den = actuator_coeff.get_t_star_denominator_val(variables, parent)
    t_star_num_ref = actuator_coeff.get_t_star_numerator_ref(model_options, parameters)
    t_star_den_ref = actuator_coeff.get_t_star_denominator_ref(wind)

    # tau = t / t_star
    # t = tau t_star
    # dt/dtau = t_star

    dt_dtau_num = t_star_num
    dt_dtau_den = t_star_den
    dt_dtau_num_ref = t_star_num_ref
    dt_dtau_den_ref = t_star_den_ref

    thrust_den = qzero * area
    thrust_ref = actuator_coeff.get_thrust_ref(model_options, atmos, wind, parameters)

    LLinv_ref = (4. * (1. - a_ref))
    da_dt_ref = a_ref

    term_1 = MM11 * da_dt * dt_dtau_num * thrust_den
    term_2 = LLinv11 * a_var * thrust_den * dt_dtau_den
    term_3 = -1. * thrust * dt_dtau_den

    resi_unscaled = term_1 + term_2 + term_3

    term_1_ref = MM11 * da_dt_ref * dt_dtau_num_ref * thrust_ref #ATP, 5m32s -- 2e9, ..e-8
    term_2_ref = LLinv_ref * a_ref * thrust_ref * dt_dtau_den_ref #ATP, 5m26 -- 3e10, 2e-8
    term_3_ref = thrust_ref * dt_dtau_den_ref

    resi = resi_unscaled / term_1_ref

    return resi


def get_unsteady_asym_pitt_peters_residual(model_options, atmos, wind, variables, parameters, outputs, parent, architecture, label):

    a_all = actuator_flow.get_a_all_var(variables, parent, label)
    da_dt = actuator_flow.get_da_all_var(variables, parent, label)
    a_ref = actuator_flow.get_a_ref(model_options)

    c_all, moment_den = actuator_coeff.get_c_all_components(model_options, atmos, wind, variables, parameters,
                                                              outputs, parent, architecture)

    LL = actuator_coeff.get_LL_matrix_val(model_options, atmos, wind, variables, outputs, parameters, parent, architecture, label)
    MM = actuator_coeff.get_MM_matrix()

    moment_ref = actuator_coeff.get_moment_ref(model_options, atmos, wind, parameters)

    t_star_num = actuator_coeff.get_t_star_numerator_val(variables, parameters, parent)
    t_star_den = actuator_coeff.get_t_star_denominator_val(variables, parent)
    t_star_num_ref = actuator_coeff.get_t_star_numerator_ref(model_options, parameters)
    t_star_den_ref = actuator_coeff.get_t_star_denominator_ref(wind)

    # tau = t / t_star
    # t = tau t_star
    # dt/dtau = t_star

    dt_dtau_num = t_star_num
    dt_dtau_den = t_star_den
    dt_dtau_num_ref = t_star_num_ref
    dt_dtau_den_ref = t_star_den_ref

    LL_ref = 1./ (4. * (1. - a_ref))
    da_dt_ref = a_ref

    term_1 = cas.mtimes(LL, cas.mtimes(MM, da_dt)) * dt_dtau_num * moment_den
    term_2 = a_all * moment_den * dt_dtau_den
    term_3 = -1. * cas.mtimes(LL, c_all) * dt_dtau_den

    resi_unscaled = term_1 + term_2 + term_3

    term_1_ref = LL_ref * MM[0, 0] * da_dt_ref * dt_dtau_num_ref * moment_ref #rest. failed. licq
    term_2_ref = a_ref * moment_ref * dt_dtau_den_ref # solve. 2e10, 2e-8. 5m?s
    term_3_ref = LL_ref * moment_ref * dt_dtau_den_ref # solve. licq

    resi = resi_unscaled / term_2_ref

    return resi

def get_steady_asym_pitt_peters_residual(model_options, atmos, wind, variables, parameters, outputs, parent, architecture, label):

    c_all, moment_denom = actuator_coeff.get_c_all_components(model_options, atmos, wind, variables, parameters, outputs, parent, architecture)

    LL_matr = actuator_coeff.get_LL_matrix_val(model_options, atmos, wind, variables, outputs, parameters, parent, architecture, label)

    a_all = actuator_flow.get_a_all_var(variables, parent, label)

    moment_ref = actuator_coeff.get_moment_ref(model_options, atmos, wind, parameters)
    a_ref = actuator_flow.get_a_ref(model_options)

    term_1 = 0.
    term_2 = a_all * moment_denom
    term_3 = -1. * cas.mtimes(LL_matr, c_all)

    resi_unscaled = term_1 + term_2 + term_3

    # term_2_ref = a_ref * moment_ref
    term_3_ref = 1./ (4. * a_ref * (1. - a_ref)) * moment_ref

    resi = resi_unscaled / term_3_ref

    return resi

def get_actuator_orientation_cstr(model_options, wind, parent, variables, parameters, architecture):

    #         system_lifted.extend([('act_dcm' + str(layer_node), (9, 1))])
    #         system_lifted.extend([('n_vec_length' + str(layer_node), (1, 1))])
    #
    #         system_lifted.extend([('wind_dcm' + str(layer_node), (9, 1))])
    #         system_lifted.extend([('u_vec_length' + str(layer_node), (1, 1))])
    #         system_lifted.extend([('z_vec_length' + str(layer_node), (1, 1))])
    # --------------------
    # 9 variables total
    # --------------------
    # 21 constraints total

    cstr_list = cstr_op.ConstraintList()

    act_dcm_cstr = actuator_geom.get_act_dcm_ortho_cstr(parent, variables)
    cstr_list.append(act_dcm_cstr) # 6 constraints

    nhat_cstr = actuator_geom.get_act_dcm_n_along_normal_cstr(model_options, parent, variables, parameters, architecture)
    cstr_list.append(nhat_cstr) # 3 constraints

    wind_dcm_cstr = actuator_flow.get_wind_dcm_ortho_cstr(parent, variables)
    cstr_list.append(wind_dcm_cstr) # 6 constraints

    uhat_cstr = actuator_flow.get_wind_dcm_u_along_uzero_cstr(model_options, wind, parent, variables, parameters, architecture)
    cstr_list.append(uhat_cstr) # 3 constraints

    align_cstr = actuator_flow.get_orientation_z_along_wzero_cstr(variables, parent)
    cstr_list.append(align_cstr) # 3 constraints

    return cstr_list



def collect_actuator_outputs(model_options, atmos, wind, variables, outputs, parameters, architecture):

    kite_nodes = architecture.kite_nodes
    act_comp_labels = actuator_flow.get_actuator_comparison_labels(model_options)

    if 'actuator' not in list(outputs.keys()):
        outputs['actuator'] = {}

    for kite in kite_nodes:

        parent = architecture.parent_map[kite]

        outputs['actuator']['radius_vec' + str(kite)] = actuator_geom.get_kite_radius_vector(model_options, kite, variables, architecture)
        outputs['actuator']['radius' + str(kite)] = actuator_geom.get_kite_radius(kite, variables, architecture, parameters)

        for label in act_comp_labels:
            outputs['actuator']['local_a_' + label + str(kite)] = actuator_flow.get_local_induction_factor(model_options, variables, kite, parent, label)

    layer_parents = architecture.layer_nodes
    for parent in layer_parents:

        for label in act_comp_labels:
            outputs['actuator']['a0_' + label + str(parent)] = actuator_flow.get_a_var(variables, parent, label)

        center = general_geom.get_center_point(model_options, parent, variables, architecture)
        velocity = general_geom.get_center_velocity(parent, variables, architecture)
        area = actuator_geom.get_actuator_area(model_options, parent, variables, parameters)
        avg_radius = actuator_geom.get_average_radius(model_options, variables, parent, architecture, parameters)
        nhat = actuator_geom.get_n_hat_var(variables, parent)

        outputs['actuator']['center' + str(parent)] = center
        outputs['actuator']['velocity' + str(parent)] = velocity
        outputs['actuator']['area' + str(parent)] = area
        outputs['actuator']['avg_radius' + str(parent)] = avg_radius
        outputs['actuator']['nhat' + str(parent)] = nhat
        outputs['actuator']['bar_varrho' + str(parent)] = actuator_geom.get_bar_varrho_var(variables, parent)

        u_a = general_flow.get_uzero_vec(model_options, wind, parent, variables, architecture)
        yaw_angle = actuator_flow.get_gamma_var(variables, parent)
        q_app = actuator_flow.get_actuator_dynamic_pressure(model_options, atmos, wind, variables, parent, architecture)

        outputs['actuator']['u_app' + str(parent)] = u_a
        outputs['actuator']['yaw' + str(parent)] = yaw_angle
        outputs['actuator']['yaw_deg' + str(parent)] = yaw_angle * 180. / np.pi
        outputs['actuator']['dyn_pressure' + str(parent)] = q_app

        thrust = actuator_force.get_actuator_thrust(model_options, variables, parameters, outputs, parent, architecture)
        ct = actuator_coeff.get_ct_val(model_options, atmos, wind, variables, outputs, parameters, parent, architecture)
        outputs['actuator']['thrust' + str(parent)] = thrust
        outputs['actuator']['ct' + str(parent)] = ct

        outputs['actuator']['w_z_check' + str(parent)] = actuator_flow.get_wzero_parallel_z_rotor_check(variables, parent)
        outputs['actuator']['gamma_check' + str(parent)] = actuator_flow.get_gamma_check(model_options, wind, parent, variables,
                                                                                         parameters, architecture)
        outputs['actuator']['gamma_comp' + str(parent)] = actuator_flow.get_gamma_val(model_options, wind, parent, variables, parameters, architecture)

    return outputs
