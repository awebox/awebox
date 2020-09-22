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
import awebox.mdl.aero.induction_dir.vortex_dir.flow as flow
import awebox.mdl.aero.induction_dir.vortex_dir.tools as tools
import awebox.tools.print_operations as print_op
import awebox.mdl.aero.induction_dir.tools_dir.unit_normal as unit_normal
import awebox.mdl.aero.induction_dir.vortex_dir.filament_list as vortex_filament_list
import awebox.mdl.aero.induction_dir.vortex_dir.biot_savart as biot_savart


def get_residual(options, atmos, wind, variables, parameters, outputs, architecture):
    # no self-induction! rigid wake convection only!
    resi = convection.get_convection_residual(options, wind, variables, architecture)
    return resi

def collect_vortex_outputs(model_options, atmos, wind, variables, outputs, parameters, architecture):

    biot_savart.test()
    test_list = vortex_filament_list.test(gamma_scale=5.)
    flow.test(test_list)

    if 'vortex' not in list(outputs.keys()):
        outputs['vortex'] = {}

    filament_list = vortex_filament_list.get_list(model_options, variables, architecture)

    columnized_list = vortex_filament_list.columnize(filament_list)
    outputs['vortex']['filament_list'] = columnized_list

    last_filament_list = vortex_filament_list.get_last_list(model_options, variables, architecture)

    kite_nodes = architecture.kite_nodes
    for kite in kite_nodes:
        parent = architecture.parent_map[kite]

        u_ind_vortex = flow.get_induced_velocity_at_kite(model_options, filament_list, variables, architecture, kite)

        n_hat = unit_normal.get_n_hat(model_options, parent, variables, parameters, architecture)
        local_a = flow.get_induction_factor_at_kite(model_options, filament_list, wind, variables, parameters, architecture, kite, n_hat=n_hat)
        last_a = flow.get_induction_factor_at_kite(model_options, last_filament_list, wind, variables, parameters, architecture, kite, n_hat=n_hat)

        outputs['vortex']['u_ind_vortex' + str(kite)] = u_ind_vortex
        outputs['vortex']['local_a' + str(kite)] = local_a
        outputs['vortex']['last_a' + str(kite)] = last_a

    return outputs


