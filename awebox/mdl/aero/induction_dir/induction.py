#
#    This file is part of awebox.
#
#    awebox -- A modeling and optimization framework for multi-kite AWE systems.
#    Copyright (C) 2017-2020 Jochem De Schutter, Rachel Leuthold, Moritz Diehl,
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
- author: rachel leuthold, alu-fr 2020
"""

import awebox.mdl.aero.induction_dir.actuator_dir.flow as actuator_flow
import awebox.mdl.aero.induction_dir.actuator_dir.actuator as actuator
import awebox.mdl.aero.induction_dir.vortex_dir.vortex as vortex
import awebox.mdl.aero.induction_dir.vortex_dir.flow as vortex_flow
import awebox.mdl.aero.induction_dir.vortex_dir.filament_list as vortex_filament_list
import awebox.mdl.aero.induction_dir.vortex_dir.linearization as vortex_linearization
import awebox.mdl.aero.induction_dir.general_dir.flow as general_flow
import awebox.mdl.aero.induction_dir.general_dir.geom as general_geom
import awebox.tools.print_operations as print_op
from awebox.logger.logger import Logger as awelogger
import casadi.tools as cas

### residuals

def get_trivial_residual(options, atmos, wind, variables_si, parameters, outputs, architecture):
    resi = []

    ind_resi = get_induction_trivial_residual(options, atmos, wind, variables_si, parameters, outputs, architecture)
    resi = cas.vertcat(resi, ind_resi)

    spec_resi = get_specific_residuals(options, atmos, wind, variables_si, parameters, outputs, architecture)
    resi = cas.vertcat(resi, spec_resi)

    return resi


def get_final_residual(options, atmos, wind, variables_si, parameters, outputs, architecture):
    resi = []

    ind_resi = get_induction_final_residual(options, atmos, wind, variables_si, parameters, outputs, architecture)
    resi = cas.vertcat(resi, ind_resi)

    spec_resi = get_specific_residuals(options, atmos, wind, variables_si, parameters, outputs, architecture)
    resi = cas.vertcat(resi, spec_resi)

    return resi

def get_induction_trivial_residual(options, atmos, wind, variables_si, parameters, outputs, architecture):
    resi = []

    for kite in architecture.kite_nodes:
        ind_val = cas.DM.zeros((3, 1))
        ind_var = get_kite_induced_velocity_var(variables_si, wind, kite)
        ind_resi = (ind_val - ind_var) / wind.get_velocity_ref()
        resi = cas.vertcat(resi, ind_resi)

    comparison_labels = options['aero']['induction']['comparison_labels']
    any_vor = any(label[:3] == 'vor' for label in comparison_labels)
    if any_vor:
        vortex_resi = vortex.get_induction_trivial_residual(options, wind, variables_si, outputs, architecture)
        resi = cas.vertcat(resi, vortex_resi)


    return resi

def get_induction_final_residual(options, atmos, wind, variables_si, parameters, outputs, architecture):
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

def get_specific_residuals(options, atmos, wind, variables_si, parameters, outputs, architecture):
    resi = []

    comparison_labels = options['aero']['induction']['comparison_labels']

    any_act = any(label[:3] == 'act' for label in comparison_labels)
    if any_act:
        actuator_resi = actuator.get_residual(options, atmos, wind, variables_si, parameters, outputs,
                                              architecture)
        resi = cas.vertcat(resi, actuator_resi)

    any_vor = any(label[:3] == 'vor' for label in comparison_labels)
    if any_vor:
        vortex_resi = vortex.get_residual(options, wind, variables_si, outputs, architecture)
        resi = cas.vertcat(resi, vortex_resi)


    return resi


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
        awelogger.logger.warning('Specified induction model is not supported. Consider checking spelling.')


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
        if architecture.number_of_kites > 1:
            outputs['vortex']['f1'] = actuator_flow.get_f_val(options, wind, 1, variables_si, architecture)
        else:
            outputs['vortex']['f1'] = actuator_flow.get_f_val(options, wind, 0, variables_si, architecture)

    return outputs
