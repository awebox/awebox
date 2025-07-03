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
'''
actuator_disk model of awebox aerodynamics
sets up the axial-induction actuator disk equation
currently for untilted rotor with no tcf.
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2017-19
- edit: jochem de schutter, alu-fr 2019
'''

import casadi.tools as cas
import numpy as np

import awebox.mdl.aero.geometry_dir.geometry as geom
import awebox.mdl.aero.induction_dir.actuator_dir.geom as actuator_geom

import awebox.tools.vector_operations as vect_op
import awebox.tools.print_operations as print_op
import awebox.tools.struct_operations as struct_op
import awebox.tools.constraint_operations as cstr_op


def get_actuator_force(outputs, parent, architecture):

    children = architecture.kites_map[parent]

    total_force_aero = cas.DM.zeros((3, 1))
    for kite in children:
        aero_force = outputs['aerodynamics']['f_aero_earth' + str(kite)]
        total_force_aero = total_force_aero + aero_force

    return total_force_aero

def get_actuator_moment(model_options, variables, outputs, parent, architecture):

    children = architecture.kites_map[parent]

    total_moment_aero = np.zeros((3, 1))
    for kite in children:
        aero_force = outputs['aerodynamics']['f_aero_earth' + str(kite)]
        kite_radius = geom.geom.get_vector_from_center_to_kite(model_options, variables, architecture, kite)
        aero_moment = vect_op.cross(kite_radius, aero_force)

        total_moment_aero = total_moment_aero + aero_moment

    return total_moment_aero


def get_actuator_thrust_val(variables, outputs, parent, architecture):

    total_force_aero = get_actuator_force(outputs, parent, architecture)
    nhat = actuator_geom.get_n_hat_var(variables, parent)
    thrust = cas.mtimes(total_force_aero.T, nhat)

    return thrust


def get_thrust_var(variables_si, parent):
    var_type = 'z'
    var_name = 'thrust' + str(parent)
    var = struct_op.get_variable_from_model_or_reconstruction(variables_si, var_type, var_name)
    return var


def get_thrust_constraint(variables, outputs, parent, architecture, scaling):
    thrust_var = get_thrust_var(variables, parent)
    thrust_val = get_actuator_thrust_val(variables, outputs, parent, architecture)

    resi_si = thrust_val - thrust_var

    var_type = 'z'
    var_name = 'thrust' + str(parent)
    resi_scaled = struct_op.var_si_to_scaled(var_type, var_name, resi_si, scaling)

    name = 'actuator_thrust_' + str(parent)
    cstr = cstr_op.Constraint(expr=resi_scaled,
                              name=name,
                              cstr_type='eq')
    return cstr