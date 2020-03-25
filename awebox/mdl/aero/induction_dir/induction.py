#
#    This file is part of awebox.
#
#    awebox -- A modeling and optimization framework for multi-kite AWE systems.
#    Copyright (C) 2017-2020 Jochem De Schutter, Rachel Leuthold, Moritz Diehl,
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
"""
induction and local flow manager
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2020
"""

import awebox.mdl.aero.induction_dir.actuator_dir.flow as actuator_flow
import awebox.mdl.aero.induction_dir.actuator_dir.actuator as actuator
import awebox.mdl.aero.induction_dir.vortex_dir.vortex as vortex
import awebox.mdl.aero.induction_dir.vortex_dir.flow as vortex_flow
import awebox.mdl.aero.induction_dir.general_dir.general as general
import awebox.mdl.aero.induction_dir.general_dir.flow as general_flow
import casadi.tools as cas

def get_trivial_residual(options, atmos, wind, variables, parameters, outputs, architecture):
    resi = []

    comparison_labels = options['aero']['induction']['comparison_labels']
    if comparison_labels:
        general_resi = general.get_trivial_residual(options, atmos, wind, variables, parameters, outputs, architecture)
        resi = cas.vertcat(resi, general_resi)

    any_act = any(label[:3] == 'act' for label in comparison_labels)
    if any_act:
        actuator_resi = actuator.get_trivial_residual(options, atmos, wind, variables, parameters, outputs,
                                                      architecture)
        resi = cas.vertcat(resi, actuator_resi)

    any_vor = any(label[:3] == 'vor' for label in comparison_labels)
    if any_vor:
        vortex_resi = vortex.get_trivial_residual(options, atmos, wind, variables, parameters, outputs, architecture)
        resi = cas.vertcat(resi, vortex_resi)

    return resi


def get_final_residual(options, atmos, wind, variables, parameters, outputs, architecture):
    resi = []

    comparison_labels = options['aero']['induction']['comparison_labels']
    if comparison_labels:
        general_resi = general.get_final_residual(options, atmos, wind, variables, parameters, outputs, architecture)
        resi = cas.vertcat(resi, general_resi)

    any_act = any(label[:3] == 'act' for label in comparison_labels)
    if any_act:
        actuator_resi = actuator.get_final_residual(options, atmos, wind, variables, parameters, outputs,
                                                      architecture)
        resi = cas.vertcat(resi, actuator_resi)

    any_vor = any(label[:3] == 'vor' for label in comparison_labels)
    if any_vor:
        vortex_resi = vortex.get_final_residual(options, atmos, wind, variables, parameters, outputs, architecture)
        resi = cas.vertcat(resi, vortex_resi)

    return resi


def collect_outputs(options, atmos, wind, variables, outputs, parameters, architecture):

    comparison_labels = options['aero']['induction']['comparison_labels']

    any_act = any(label[:3] == 'act' for label in comparison_labels)
    if any_act:
        outputs = actuator.collect_actuator_outputs(options, atmos, wind, variables, outputs, parameters, architecture)

    any_vor = any(label[:3] == 'vor' for label in comparison_labels)
    if any_vor:
        outputs = vortex.collect_vortex_outputs(options, atmos, wind, variables, outputs, parameters, architecture)
        outputs['vortex']['f1'] = actuator_flow.get_f_val(options, wind, 1, variables, architecture)

    return outputs

def get_kite_effective_velocity(model_options, variables, wind, kite, architecture):
    induction_model = model_options['induction_model']
    parent = architecture.parent_map[kite]

    u_app_kite = general_flow.get_kite_apparent_velocity(variables, wind, kite, parent)

    u_eff_kite = u_app_kite
    if induction_model == 'actuator':
        u_eff_kite = actuator_flow.get_kite_effective_velocity(model_options, variables, wind, kite, parent)
    elif induction_model == 'vortex':
        u_eff_kite = vortex_flow.get_kite_effective_velocity(model_options, variables, wind, kite, architecture)

    return u_eff_kite
