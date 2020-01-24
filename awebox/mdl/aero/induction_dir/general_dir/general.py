
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
'''
general induction modelling
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2017-20
- edit: jochem de schutter, alu-fr 2019
'''

import casadi.tools as cas
import awebox.mdl.aero.induction_dir.general_dir.geom as general_geom

def get_trivial_residual(options, atmos, wind, variables, parameters, outputs, architecture):
    resi = []
    layer_nodes = architecture.layer_nodes

    for layer in layer_nodes:
        rot_matr_residual = general_geom.get_rot_matr_trivial(options, layer, variables, parameters, architecture)
        resi = cas.vertcat(resi, rot_matr_residual)

    return resi

def get_final_residual(options, atmos, wind, variables, parameters, outputs, architecture):
    resi = []
    layer_nodes = architecture.layer_nodes

    for layer in layer_nodes:
        rot_matr_residual = general_geom.get_rot_matr_residual(options, layer, variables, parameters, architecture)
        resi = cas.vertcat(resi, rot_matr_residual)

    return resi
