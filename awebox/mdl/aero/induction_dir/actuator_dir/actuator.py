#
#    This file is part of awebox.
#
#    awebox -- A modeling and optimization framework for multi-kite AWE systems.
#    Copyright (C) 2017-2019 Jochem De Schutter, Rachel Leuthold, Moritz Diehl,
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
'''
actuator_disk model of awebox aerodynamics
sets up the axial-induction actuator disk equation
currently for untilted rotor with no tcf.
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2017-20
- edit: jochem de schutter, alu-fr 2019
'''

import casadi as cas
import numpy as np
from awebox.logger.logger import Logger as awelogger

import awebox.mdl.aero.induction_dir.general_dir.geom as general_geom

import awebox.mdl.aero.induction_dir.actuator_dir.geom as actuator_geom
import awebox.mdl.aero.induction_dir.actuator_dir.flow as actuator_flow
import awebox.mdl.aero.induction_dir.actuator_dir.coeff as actuator_coeff
import awebox.tools.print_operations as print_op


def get_residual(model_options, atmos, wind, variables, parameters, outputs, architecture):

    all_residuals = []

    act_comp_labels = actuator_flow.get_actuator_comparison_labels(model_options)
    any_asym = any('asym' in label for label in act_comp_labels)
    any_unsteady = any(label[0] == 'u' for label in act_comp_labels)

    layer_parent_map = architecture.layer_nodes
    for parent in layer_parent_map:

        for label in act_comp_labels:
            induction_trivial = get_induction_residual(model_options, atmos, wind, variables, outputs, parameters, parent, architecture, label)
            all_residuals = cas.vertcat(all_residuals, induction_trivial)

            # corr_resi = actuator_flow.get_corr_residual(model_options, variables, parent, label)
            # all_residuals = cas.vertcat(all_residuals, corr_resi)

            chi_resi = actuator_flow.get_chi_residual(model_options, parent, variables, label)
            all_residuals = cas.vertcat(all_residuals, chi_resi)

            # if any_asym and label in ['qasym', 'uasym']:
            #     chi_trig_resi = actuator_flow.get_chi_trig_residual(model_options, parent, variables, label)
            #     all_residuals = cas.vertcat(all_residuals, chi_trig_resi)
            #
            #     LL_resi = actuator_coeff.get_LL_residual(model_options, variables, parent, label)
            #     all_residuals = cas.vertcat(all_residuals, LL_resi)
            #
            #     c_tilde_resi = actuator_coeff.get_c_tilde_residual(model_options, variables, parent, label)
            #     all_residuals = cas.vertcat(all_residuals, c_tilde_resi)

        # if any_asym:
        #     moments_final = actuator_coeff.get_moments_residual(model_options, atmos, wind, variables, parameters, outputs, parent, architecture)
        #     all_residuals = cas.vertcat(all_residuals, moments_final)

        rot_matr_residual = general_geom.get_rot_matr_residual(model_options, parent, variables, parameters, architecture)
        all_residuals = cas.vertcat(all_residuals, rot_matr_residual)

        # thrust_final = actuator_coeff.get_thrust_residual(model_options, atmos, wind, variables, parameters, outputs, parent, architecture)
        # all_residuals = cas.vertcat(all_residuals, thrust_final)

        # dt_resi = actuator_coeff.get_t_star_residual(model_options, atmos, wind, variables, parameters, outputs, parent, architecture)
        # all_residuals = cas.vertcat(all_residuals, dt_resi)

        uzero_matr_resi = actuator_flow.get_uzero_matr_residual(model_options, wind, parent, variables, parameters, architecture)
        all_residuals = cas.vertcat(all_residuals, uzero_matr_resi)

        # qzero_resi = actuator_flow.get_qzero_residual(model_options, parent, atmos, wind, variables, architecture)
        # all_residuals = cas.vertcat(all_residuals, qzero_resi)

        gamma_resi = actuator_flow.get_gamma_residual(model_options, wind, parent, variables, architecture)
        all_residuals = cas.vertcat(all_residuals, gamma_resi)

        # area_resi = actuator_geom.get_area_residual(model_options, parent, variables, parameters)
        # all_residuals = cas.vertcat(all_residuals, area_resi)

        children = architecture.kites_map[parent]
        for kite in children:
            local_a_resi = actuator_flow.get_local_a_residual(model_options, variables, kite, parent)
            all_residuals = cas.vertcat(all_residuals, local_a_resi)

            varrho_resi = actuator_geom.get_varrho_residual(model_options, kite, variables, parameters, architecture)
            all_residuals = cas.vertcat(all_residuals, varrho_resi)

        bar_varrho_resi = actuator_geom.get_bar_varrho_residual(model_options, parent, variables, architecture)
        all_residuals = cas.vertcat(all_residuals, bar_varrho_resi)

    return all_residuals



def get_induction_residual(model_options, atmos, wind, variables, outputs, parameters, parent, architecture, label):

    if label == 'qaxi':
        induction_final = get_momentum_theory_residual(model_options, atmos, wind, variables, outputs, parameters, parent, architecture, label)

    elif label == 'qasym':
        print_op.warn_about_temporary_funcationality_removal(editor='rachel', location='actuator.get_induction_residual')
        # induction_final = get_steady_asym_pitt_peters_residual(model_options, variables, parent, label)

    elif label == 'uaxi':
        print_op.warn_about_temporary_funcationality_removal(editor='rachel', location='actuator.get_induction_residual')
        induction_final = get_unsteady_axi_pitt_peters_residual(model_options, variables, parent, label)

    elif label == 'uasym':
        print_op.warn_about_temporary_funcationality_removal(editor='rachel', location='actuator.get_induction_residual')
        induction_final = get_unsteady_asym_pitt_peters_residual(model_options, variables, parent, label)

    else:
        induction_final = []
        awelogger.logger.error('model not yet implemented.')

    return induction_final

def get_momentum_theory_residual(model_options, atmos, wind, variables, outputs, parameters, parent, architecture, label):
    # a_all = actuator_flow.get_a_all_var(model_options, variables, parent, label)
    # c_all = actuator_coeff.get_c_all_var(model_options, variables, parent, label)
    # corr = actuator_flow.get_corr_var(variables, parent, label)
    #
    # LLinv11 = 4. * corr
    #
    # resi = ( LLinv11 * a_all - c_all )

    a_var = actuator_flow.get_a_var(model_options, variables, parent, label)

    thrust = actuator_coeff.get_actuator_thrust(model_options, variables, outputs, parent, architecture)
    area = actuator_geom.get_actuator_area(model_options, parent, variables, parameters)
    qzero = actuator_flow.get_actuator_dynamic_pressure(model_options, atmos, wind, variables, parent, architecture)

    corr_val = actuator_flow.get_corr_val(model_options, variables, parent, label)

    resi_unscaled = 4. * a_var * corr_val * area * qzero - thrust

    thrust_ref = actuator_coeff.get_thrust_ref(model_options, atmos, wind, parameters)
    resi = resi_unscaled / thrust_ref

    return resi

def get_unsteady_axi_pitt_peters_residual(model_options, variables, parent, label):

    a_all = actuator_flow.get_a_all_var(model_options, variables, parent, label)
    da_all = actuator_flow.get_da_all_var(model_options, variables, parent, label)
    c_all = actuator_coeff.get_c_all_var(model_options, variables, parent, label)

    corr = actuator_flow.get_corr_var(variables, parent, label)
    LLinv11 = 4. * corr

    MM = actuator_coeff.get_MM_matrix()
    MM11 = MM[0, 0]

    t_star = actuator_coeff.get_t_star_var(variables, parent)

    f_a0 = (MM11 * da_all * t_star + LLinv11 * a_all - c_all)
    f_induction = cas.vertcat(f_a0)
    return f_induction

def get_unsteady_asym_pitt_peters_residual(model_options, variables, parent, label):

    a_all = actuator_flow.get_a_all_var(model_options, variables, parent, label)
    da_all = actuator_flow.get_da_all_var(model_options, variables, parent, label)
    c_all = actuator_coeff.get_c_all_var(model_options, variables, parent, label)

    c_tilde = actuator_coeff.get_c_tilde_var(variables, parent, label)
    MM = actuator_coeff.get_MM_matrix()

    t_star = actuator_coeff.get_t_star_var(variables, parent)

    resi = ( cas.mtimes(MM, da_all) * t_star + c_tilde - c_all )
    return resi

def get_steady_asym_pitt_peters_residual(model_options, variables, parent, label):

    c_all = actuator_coeff.get_c_all_var(model_options, variables, parent, label)
    c_tilde = actuator_coeff.get_c_tilde_var(variables, parent, label)

    resi = (c_tilde - c_all )
    return resi







def collect_actuator_outputs(model_options, atmos, wind, variables, outputs, parameters, architecture):

    kite_nodes = architecture.kite_nodes
    act_comp_labels = actuator_flow.get_actuator_comparison_labels(model_options)

    if 'actuator' not in list(outputs.keys()):
        outputs['actuator'] = {}

    outputs['actuator']['f1'] = actuator_flow.get_f_val(model_options, wind, 1, variables, architecture)

    for kite in kite_nodes:

        parent = architecture.parent_map[kite]

        outputs['actuator']['radius_vec' + str(kite)] = actuator_geom.get_kite_radius_vector(model_options, kite, variables, architecture)
        outputs['actuator']['radius' + str(kite)] = actuator_geom.get_kite_radius(model_options, kite, variables, architecture, parameters)

        for label in act_comp_labels:
            outputs['actuator']['local_a_' + label + str(kite)] = actuator_flow.get_local_induction_factor(model_options, variables, kite, parent, label)

    layer_parents = architecture.layer_nodes
    for parent in layer_parents:

        for label in act_comp_labels:
            outputs['actuator']['a0_' + label + str(parent)] = actuator_flow.get_a_var(model_options, variables, parent, label)

        center = actuator_geom.get_center_point(model_options, parent, variables, architecture)
        velocity = actuator_geom.get_center_velocity(model_options, parent, variables, architecture)
        area = actuator_geom.get_actuator_area(model_options, parent, variables, parameters)
        avg_radius = actuator_geom.get_average_radius(model_options, variables, parent, architecture, parameters)
        nhat = general_geom.get_n_hat_var(variables, parent)

        outputs['actuator']['center' + str(parent)] = center
        outputs['actuator']['velocity' + str(parent)] = velocity
        outputs['actuator']['area' + str(parent)] = area
        outputs['actuator']['avg_radius' + str(parent)] = avg_radius
        outputs['actuator']['nhat' + str(parent)] = nhat
        outputs['actuator']['bar_varrho' + str(parent)] = actuator_geom.get_bar_varrho_var(model_options, variables, parent)

        u_a = actuator_flow.get_uzero_vec(model_options, wind, parent, variables, architecture)
        yaw_angle = actuator_flow.get_gamma_var(variables, parent)
        q_app = actuator_flow.get_actuator_dynamic_pressure(model_options, atmos, wind, variables, parent, architecture)

        outputs['actuator']['u_app' + str(parent)] = u_a
        outputs['actuator']['yaw' + str(parent)] = yaw_angle
        outputs['actuator']['yaw_deg' + str(parent)] = yaw_angle * 180. / np.pi
        outputs['actuator']['dyn_pressure' + str(parent)] = q_app
        outputs['actuator']['df' + str(parent)] = actuator_flow.get_df_val(model_options, wind, parent, variables, architecture)

        thrust = actuator_coeff.get_actuator_thrust(model_options, variables, outputs, parent, architecture)
        ct = actuator_coeff.get_ct_val(model_options, atmos, wind, variables, outputs, parameters, parent, architecture)
        outputs['actuator']['thrust' + str(parent)] = thrust
        outputs['actuator']['ct' + str(parent)] = ct

        outputs['actuator']['w_z_check' + str(parent)] = actuator_flow.get_wzero_parallel_z_rotor_check(variables, parent)
        outputs['actuator']['gamma_check' + str(parent)] = actuator_flow.get_gamma_check(model_options, wind, parent, variables,
                                                                                         parameters, architecture)
        outputs['actuator']['gamma_comp' + str(parent)] = actuator_flow.get_gamma_val(model_options, wind, parent, variables, parameters, architecture)

    return outputs
