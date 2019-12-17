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
"""
induction and local flow manager
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2019
"""

import casadi.tools as cas

import awebox.tools.vector_operations as vect_op
import awebox.mdl.aero.actuator_disk_dir.flow as act_flow
import awebox.mdl.aero.actuator_disk_dir.actuator as actuator
import awebox.mdl.aero.vortex_dir.vortex as vortex
import numpy as np
import pdb
from awebox.logger.logger import Logger as awelogger

def get_trivial_residual(options, atmos, wind, variables, parameters, outputs, architecture):
    resi = []
    if options['induction_model'] == 'actuator':
        resi = actuator.get_trivial_residual(options, atmos, wind, variables, parameters, outputs, architecture)
    if options['induction_model'] == 'vortex':
        resi = vortex.get_trivial_residual(options, atmos, wind, variables, parameters, outputs, architecture)
    return resi

def get_final_residual(options, atmos, wind, variables, parameters, outputs, architecture):
    resi = []
    if options['induction_model'] == 'actuator':
        resi = actuator.get_final_residual(options, atmos, wind, variables, parameters, outputs, architecture)
    if options['induction_model'] == 'vortex':
        resi = vortex.get_final_residual(options, atmos, wind, variables, parameters, outputs, architecture)
    return resi


def collect_outputs(options, atmos, wind, variables, outputs, parameters, architecture):
    if options['induction_model'] == 'actuator':
        outputs = actuator.collect_actuator_outputs(options, atmos, wind, variables, outputs, parameters, architecture)
    if options['induction_model'] == 'vortex':
        32.0

    return outputs

def get_kite_effective_velocity(model_options, variables, wind, kite, parent):
    induction_model = model_options['induction_model']

    u_app_kite = get_kite_apparent_velocity(variables, wind, kite, parent)

    u_eff_kite = u_app_kite
    if induction_model == 'actuator':
        u_eff_kite = act_flow.get_kite_effective_velocity(model_options, variables, wind, kite, parent)
    if induction_model == 'vortex':
        32.0

    return u_eff_kite


def get_kite_apparent_velocity(variables, wind, kite, parent):
    q_kite = variables['xd']['q' + str(kite) + str(parent)]
    u_infty = wind.get_velocity(q_kite[2])
    u_kite = variables['xd']['dq' + str(kite) + str(parent)]
    u_app_kite = u_infty - u_kite

    return u_app_kite