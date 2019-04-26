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
- author: rachel leuthold, alu-fr 2017-18
- edit: jochem de schutter, alu-fr 2019
'''

import casadi as cas
import numpy as np
import logging

from . import geom as geom
from . import flow as flow
from . import coeff as coeff

import awebox.tools.vector_operations as vect_op

def get_trivial_residual(model_options, atmos, wind, variables, parameters, architecture):

    layer_parent_map = architecture.layer_nodes

    correct_tilt = model_options['aero']['actuator']['correct_tilt']

    a_ref = flow.get_a_ref(model_options)
    thrust_ref = coeff.get_thrust_ref(model_options, atmos, wind, parameters)
    area_ref = geom.get_area_ref(model_options, parameters)
    varrho_ref = geom.get_varrho_ref(model_options)
    qapp_ref = flow.get_qapp_ref(atmos, wind)

    all_residuals = []
    for parent in layer_parent_map:

        a_var = flow.get_a_var(model_options, variables, parent)
        ct_var = coeff.get_ct_var(model_options, variables, parent)
        area_var = geom.get_area_var(model_options, variables, parent, parameters)
        bar_varrho_var = geom.get_bar_varrho_var(model_options, variables, parent)
        qapp_var = flow.get_qapp_var(atmos, wind, variables, parent)


        induction_trivial = (a_var - a_ref) / a_ref
        all_residuals = cas.vertcat(all_residuals, induction_trivial)

        thrust_trivial = (thrust_ref - ct_var * area_ref * qapp_ref) / thrust_ref
        all_residuals = cas.vertcat(all_residuals, thrust_trivial)

        f_resi = flow.get_f_residual(model_options, wind, parent, variables, architecture)
        all_residuals = cas.vertcat(all_residuals, f_resi)

        nhat_resi = geom.get_nhat_residual(model_options, parent, variables, parameters, architecture)
        all_residuals = cas.vertcat(all_residuals, nhat_resi)

        area_trivial = (area_var - area_ref) / area_ref
        all_residuals = cas.vertcat(all_residuals, area_trivial)

        children = architecture.kites_map[parent]
        for kite in children:
            varrho_var = geom.get_varrho_var(variables, kite, architecture)
            varrho_trivial = (varrho_var - varrho_ref) / varrho_ref
            all_residuals = cas.vertcat(all_residuals, varrho_trivial)

        bar_varrho_trivial = (bar_varrho_var - varrho_ref) / varrho_ref
        all_residuals = cas.vertcat(all_residuals, bar_varrho_trivial)

        qapp_trivial = (qapp_var - qapp_ref) / qapp_ref
        all_residuals = cas.vertcat(all_residuals, qapp_trivial)

        if correct_tilt:
            cosgamma_resi = flow.get_cosgamma_residual(model_options, wind, parent, variables, architecture)
            all_residuals = cas.vertcat(all_residuals, cosgamma_resi)

    return all_residuals


def get_final_residual(model_options, atmos, wind, variables, parameters, outputs, architecture):

    steadyness = model_options['aero']['actuator']['steadyness']
    unsteady_model = model_options['aero']['actuator']['unsteady_model']
    correct_tilt =  model_options['aero']['actuator']['correct_tilt']
    layer_parent_map = architecture.layer_nodes

    all_residuals = []
    for parent in layer_parent_map:

        if steadyness == 'steady':
            induction_final = get_momentum_theory_residual(model_options, wind, variables, parent, architecture)

        else:
            if unsteady_model == 'axi_pitt_peters':
                induction_final = get_axi_pitt_peters_residual(model_options, variables, parent, parameters, wind)

            elif unsteady_model == 'axi_new':
                induction_final = get_axi_new_residual(model_options, variables, parent, parameters, wind)

            else:
                induction_final = []
                logging.warning('unsteady model not yet implemented.')

        all_residuals = cas.vertcat(all_residuals, induction_final)

        thrust_final = coeff.get_thrust_residual(model_options, atmos, wind, variables, parameters, outputs, parent, architecture)
        all_residuals = cas.vertcat(all_residuals, thrust_final)

        f_resi = flow.get_f_residual(model_options, wind, parent, variables, architecture)
        all_residuals = cas.vertcat(all_residuals, f_resi)

        nhat_resi = geom.get_nhat_residual(model_options, parent, variables, parameters, architecture)
        all_residuals = cas.vertcat(all_residuals, nhat_resi)

        area_resi = geom.get_area_residual(model_options, parent, variables, parameters)
        all_residuals = cas.vertcat(all_residuals, area_resi)

        children = architecture.kites_map[parent]
        for kite in children:
            varrho_resi = geom.get_varrho_residual(model_options, kite, variables, parameters, architecture)
            all_residuals = cas.vertcat(all_residuals, varrho_resi)

        bar_varrho_resi = geom.get_bar_varrho_residual(model_options, parent, variables, architecture)
        all_residuals = cas.vertcat(all_residuals, bar_varrho_resi)

        qapp_resi = flow.get_qapp_residual(model_options, parent, atmos, wind, variables, architecture)
        all_residuals = cas.vertcat(all_residuals, qapp_resi)

        if correct_tilt:
            cosgamma_resi = flow.get_cosgamma_residual(model_options, wind, parent, variables, architecture)
            all_residuals = cas.vertcat(all_residuals, cosgamma_resi)

    return all_residuals



def get_momentum_theory_residual(model_options, wind, variables, parent, architecture):

    a_var = flow.get_a_var(model_options, variables, parent)
    ct_var = coeff.get_ct_var(model_options, variables, parent)

    nonlin_correction = flow.get_nonlin_induction_correction(model_options, wind, variables, parent, architecture)

    f_induction = 4. * a_var * nonlin_correction - ct_var

    return f_induction

def get_axi_pitt_peters_residual(model_options, variables, parent, parameters, wind):

    a_var = flow.get_a_var(model_options, variables, parent)
    da_var = flow.get_da_var(model_options, variables, parent)
    ct_var = coeff.get_ct_var(model_options, variables, parent)

    t_star = geom.get_tstar_ref(parameters, wind)
    bar_varrho_var = geom.get_bar_varrho_var(model_options, variables, parent)
    da_timescale = t_star * (bar_varrho_var + 0.5)

    f_induction = 16./(3. * np.pi) * da_var * da_timescale + 4. * a_var * (1. - a_var) - ct_var

    return f_induction


def get_axi_new_residual(model_options, variables, parent, parameters, wind):

    a_var = flow.get_a_var(model_options, variables, parent)
    ct_var = coeff.get_ct_var(model_options, variables, parent)

    dbar_varrho_var = geom.get_dbar_varrho_var(variables, parent)
    df_var = flow.get_df_var(variables, parent)
    dct_var = coeff.get_dct_var(variables, parent)
    abs_dvarrho = vect_op.smooth_abs(dbar_varrho_var)

    t_star = geom.get_tstar_ref(parameters, wind)

    c1 = 4.837e-2
    c2 = 5.582e-2
    c3 = 8.730e-2
    c4 = 7.255e-1
    c5 = 1.065
    c6 = 1.404e-1
    c7 = 1.821e-1

    p2 = df_var * t_star * c1 \
         + abs_dvarrho * t_star * c2 \
         + abs_dvarrho * df_var * t_star**2. * c3 \
         + dct_var * t_star * c4 \
         + dct_var * df_var * t_star**2. * c5\
         + dct_var * abs_dvarrho * t_star**2. * c6 \
         + dct_var * abs_dvarrho * df_var * t_star**3. * c7

    resi = (2. * a_var + 1. + 2.* p2)^2 - 1. + ct_var

    return resi


def collect_actuator_outputs(model_options, atmos, wind, variables, outputs, parameters,architecture):

    kite_nodes = architecture.kite_nodes

    if 'actuator' not in list(outputs.keys()):
        outputs['actuator'] = {}

    for kite in kite_nodes:
        outputs['actuator']['radius_vec' + str(kite)] = geom.get_kite_radius_vector(model_options, kite, variables, architecture)
        outputs['actuator']['radius' + str(kite)] = geom.get_kite_radius(model_options, kite, variables, architecture)

    layer_parents = architecture.layer_nodes
    for parent in layer_parents:

        center = geom.get_center_point(model_options, parent, variables, architecture)
        velocity = geom.get_center_velocity(model_options, parent, variables, architecture)
        area = geom.get_actuator_area(model_options, parent, variables, parameters)
        avg_radius = geom.get_average_radius(model_options, variables, parent, architecture)
        nhat = geom.get_nhat_var(variables, parent)

        outputs['actuator']['center' + str(parent)] = center
        outputs['actuator']['velocity' + str(parent)] = velocity
        outputs['actuator']['area' + str(parent)] = area
        outputs['actuator']['avg_radius' + str(parent)] = avg_radius
        outputs['actuator']['nhat' + str(parent)] = nhat

        u_a = flow.get_rotor_apparent_velocity(model_options, wind, parent, variables, architecture)
        yaw_angle = flow.get_actuator_yaw_angle(model_options, wind, parent, variables, architecture)
        q_app = flow.get_actuator_dynamic_pressure(model_options, atmos, wind, variables, parent, architecture)

        outputs['actuator']['u_app' + str(parent)] = u_a
        outputs['actuator']['yaw' + str(parent)] = yaw_angle
        outputs['actuator']['yaw_deg' + str(parent)] = yaw_angle * 180. / np.pi
        outputs['actuator']['dyn_pressure' + str(parent)] = q_app

        thrust = coeff.get_actuator_thrust(model_options, variables, outputs, parent, architecture)
        thrust_coeff = coeff.get_ct_var(model_options, variables, parent)
        outputs['actuator']['thrust' + str(parent)] = thrust
        outputs['actuator']['thrust1_coeff' + str(parent)] = thrust / q_app / area
        outputs['actuator']['thrust2_area_coeff' + str(parent)] = thrust / q_app
        outputs['actuator']['thrust3_coeff' + str(parent)] = thrust_coeff

    return outputs
