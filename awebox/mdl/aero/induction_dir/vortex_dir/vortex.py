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
vortex model of awebox aerodynamics
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2019
'''

import casadi.tools as cas

import awebox.mdl.aero.induction_dir.vortex_dir.convection as convection
import awebox.mdl.aero.induction_dir.vortex_dir.biot_savart as biot_savart
import awebox.mdl.aero.induction_dir.vortex_dir.tools as vortex_tools


def get_trivial_residual(options, atmos, wind, variables, parameters, outputs, architecture):
    resi = convection.get_convection_residual(options, wind, variables, architecture)
    return resi

def get_final_residual(options, atmos, wind, variables, parameters, outputs, architecture):
    resi = get_trivial_residual(options, atmos, wind, variables, parameters, outputs, architecture)
    return resi

def collect_vortex_outputs(model_options, atmos, wind, variables, outputs, parameters, architecture):

    if 'vortex' not in list(outputs.keys()):
        outputs['vortex'] = {}

    filament_list = vortex_tools.get_filament_list(model_options, wind, variables, architecture)
    last_filament_list = vortex_tools.get_last_filament_list(model_options, wind, variables, architecture)

    dims = filament_list.shape
    reshaped_list = cas.reshape(filament_list, (dims[0] * dims[1], 1))
    outputs['vortex']['filament_list'] = reshaped_list

    kite_nodes = architecture.kite_nodes
    for kite in kite_nodes:

        parent = architecture.parent_map[kite]

        outputs['vortex']['u_ind_vortex' + str(kite)] = biot_savart.get_induced_velocity_at_kite(filament_list, model_options, variables, kite, parent)
        outputs['vortex']['local_a' + str(kite)] = biot_savart.get_induction_factor_at_kite(filament_list, model_options, wind, variables, kite, architecture)

        outputs['vortex']['last_a' + str(kite)] = biot_savart.get_induction_factor_at_kite(last_filament_list, model_options, wind, variables, kite, architecture)

    return outputs

