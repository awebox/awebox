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
the linearized Biot-Savart version, for iterative solution.
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2020-
"""

import awebox.mdl.aero.induction_dir.vortex_dir.flow as vortex_flow
import awebox.mdl.aero.induction_dir.vortex_dir.filament_list as vortex_filament_list
import casadi.tools as cas

def get_induced_velocity_at_kite(model_options, variables, parameters, architecture, kite, outputs):

    lin_params = parameters['lin']

    var_sym = {}
    var_sym_cat = []
    var_actual_cat = []
    for var_type in variables.keys():
        var_sym[var_type] = variables[var_type](cas.SX.sym(var_type, (variables[var_type].cat.shape)))
        var_sym_cat = cas.vertcat(var_sym_cat, var_sym[var_type].cat)
        var_actual_cat = cas.vertcat(var_actual_cat, variables[var_type].cat)

    columnized_list = outputs['vortex']['filament_list']
    filament_list = vortex_filament_list.decolumnize(model_options, architecture, columnized_list)
    uind_sym = vortex_flow.get_induced_velocity_at_kite(model_options, filament_list, variables, architecture, kite)
    jac_sym = cas.jacobian(uind_sym, var_sym_cat)

    uind_fun = cas.Function('uind_fun', [var_sym_cat], [uind_sym])
    jac_fun = cas.Function('jac_fun', [var_sym_cat], [jac_sym])

    slope = jac_fun(lin_params)
    const = uind_fun(lin_params)

    uind_lin = cas.mtimes(slope, var_actual_cat) + const

    return uind_lin