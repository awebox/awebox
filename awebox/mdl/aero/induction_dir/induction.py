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
"""
induction and local flow manager
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2020-21
"""

import awebox.mdl.aero.induction_dir.actuator_dir.flow as actuator_flow
import awebox.mdl.aero.induction_dir.actuator_dir.actuator as actuator
import awebox.mdl.aero.induction_dir.vortex_dir.vortex as vortex
import awebox.mdl.aero.induction_dir.vortex_dir.flow as vortex_flow
import awebox.mdl.aero.induction_dir.vortex_dir.filament_list as vortex_filament_list
import awebox.mdl.aero.induction_dir.vortex_dir.linearization as vortex_linearization
import awebox.mdl.aero.induction_dir.tools_dir.flow as general_flow
import awebox.tools.print_operations as print_op
import awebox.tools.constraint_operations as cstr_op
from awebox.logger.logger import Logger as awelogger
import casadi.tools as cas

### residuals

def get_induction_cstr(options, atmos, wind, variables_si, parameters, outputs, architecture):

    cstr_list = cstr_op.ConstraintList()

    general_trivial = get_general_trivial_residual(options, wind, variables_si, architecture)
    general_final = get_general_final_residual(options, wind, variables_si, parameters, outputs, architecture)
    general_homotopy = parameters['phi', 'iota'] * general_trivial + (1. - parameters['phi', 'iota']) * general_final
    general_cstr = cstr_op.Constraint(expr=general_homotopy,
                                      name='induction_general',
                                      cstr_type='eq')
    cstr_list.append(general_cstr)

    specific_cstr = get_specific_cstr(options, atmos, wind, variables_si, parameters, outputs, architecture)
    cstr_list.append(specific_cstr)

    return cstr_list


def get_general_trivial_residual(options, wind, variables_si, architecture):
    resi = []

    for kite in architecture.kite_nodes:
        ind_val = cas.DM.zeros((3, 1))
        ind_var = get_kite_induced_velocity_var(variables_si, wind, kite)
        ind_resi = (ind_val - ind_var) / wind.get_velocity_ref()
        resi = cas.vertcat(resi, ind_resi)

    comparison_labels = options['aero']['induction']['comparison_labels']
    any_vor = any(label[:3] == 'vor' for label in comparison_labels)
    if any_vor:
        vortex_resi = vortex.get_induction_trivial_residual(options, wind, variables_si, architecture)
        resi = cas.vertcat(resi, vortex_resi)

    return resi

def get_general_final_residual(options, wind, variables_si, parameters, outputs, architecture):
    resi = []

    for kite in architecture.kite_nodes:
        ind_val = get_kite_induced_velocity_val(options, wind, variables_si, kite, architecture, parameters, outputs)
        ind_var = get_kite_induced_velocity_var(variables_si, wind, kite)
        ind_resi = (ind_val - ind_var) / wind.get_velocity_ref()
        resi = cas.vertcat(resi, ind_resi)

    comparison_labels = options['aero']['induction']['comparison_labels']
    any_vor = any(label[:3] == 'vor' for label in comparison_labels)
    if any_vor:
        vortex_resi = vortex.get_induction_final_residual(options, wind, variables_si, outputs, architecture)
        resi = cas.vertcat(resi, vortex_resi)

    return resi


def get_specific_cstr(options, atmos, wind, variables_si, parameters, outputs, architecture):

    cstr_list = cstr_op.ConstraintList()

    comparison_labels = options['aero']['induction']['comparison_labels']

    any_act = any(label[:3] == 'act' for label in comparison_labels)
    if any_act:
        actuator_cstr = actuator.get_actuator_cstr(options, atmos, wind, variables_si, parameters, outputs,
                                                   architecture)
        cstr_list.append(actuator_cstr)

    any_vor = any(label[:3] == 'vor' for label in comparison_labels)
    if any_vor:
        vortex_cstr = vortex.get_vortex_cstr(options, wind, variables_si, architecture)
        cstr_list.append(vortex_cstr)

    return cstr_list


## velocities

def get_kite_induced_velocity_var(variables, wind, kite):
    ind_var = variables['xl']['ui' + str(kite)] * wind.get_velocity_ref()
    return ind_var

def get_kite_induced_velocity_val(model_options, wind, variables, kite, architecture, parameters, outputs):
    induction_model = model_options['induction_model']
    parent = architecture.parent_map[kite]

    use_vortex_linearization = model_options['aero']['vortex']['use_linearization']
    force_zero = model_options['aero']['vortex']['force_zero']

    if induction_model == 'actuator':
        u_ind_kite = actuator_flow.get_kite_induced_velocity(model_options, variables, parameters, architecture, wind, kite, parent)
    elif induction_model == 'vortex' and not use_vortex_linearization and not force_zero:
        u_ind_kite = variables['xl']['wu_ind_' + str(kite)]
    elif induction_model == 'vortex' and use_vortex_linearization and not force_zero:
        u_ind_kite = vortex_linearization.get_induced_velocity_at_kite(model_options, variables, parameters, architecture, kite, outputs)
    elif induction_model == 'vortex' and force_zero:
        u_ind_kite = cas.DM.zeros((3, 1))
    elif induction_model == 'not_in_use':
        u_ind_kite = cas.DM.zeros((3, 1))
    else:
        message = 'specified induction model (' + induction_model + ') is not supported. continuing with ' \
                                                                    'zero induced velocity.'
        awelogger.logger.warning(message)
        u_ind_kite = cas.DM.zeros((3, 1))


    return u_ind_kite


def get_kite_effective_velocity(model_options, variables, wind, kite, architecture):

    parent = architecture.parent_map[kite]

    u_app_kite = general_flow.get_kite_apparent_velocity(variables, wind, kite, parent)
    u_ind_kite = get_kite_induced_velocity_var(variables, wind, kite)
    u_eff_kite = u_app_kite + u_ind_kite

    return u_eff_kite


#### outputs

def collect_outputs(options, atmos, wind, variables_si, outputs, parameters, architecture):

    comparison_labels = options['aero']['induction']['comparison_labels']

    any_act = any(label[:3] == 'act' for label in comparison_labels)
    if any_act:
        outputs = actuator.collect_actuator_outputs(options, atmos, wind, variables_si, outputs, parameters, architecture)

    any_vor = any(label[:3] == 'vor' for label in comparison_labels)
    if any_vor:
        outputs = vortex.collect_vortex_outputs(options, atmos, wind, variables_si, outputs, parameters, architecture)

    return outputs
