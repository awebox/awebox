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
'''
vortex model of awebox aerodynamics
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2019-21
'''
import pdb

import casadi.tools as cas
import numpy as np
import matplotlib.pyplot as plt

import awebox.mdl.aero.induction_dir.vortex_dir.tools as vortex_tools

import awebox.mdl.aero.induction_dir.vortex_dir.alg_repr_dir.structure as alg_structure
import awebox.mdl.aero.induction_dir.vortex_dir.alg_repr_dir.bound_wake as alg_bound_wake
import awebox.mdl.aero.induction_dir.vortex_dir.alg_repr_dir.near_wake as alg_near_wake
import awebox.mdl.aero.induction_dir.vortex_dir.alg_repr_dir.far_wake as alg_far_wake
import awebox.mdl.aero.induction_dir.vortex_dir.alg_repr_dir.fixing as alg_fixing
import awebox.mdl.aero.induction_dir.vortex_dir.alg_repr_dir.strength as alg_strength

import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.wake as obj_wake

import awebox.tools.vector_operations as vect_op
import awebox.tools.constraint_operations as cstr_op
import awebox.ocp.ocp_constraint as ocp_constraint
import awebox.tools.print_operations as print_op

from awebox.logger.logger import Logger as awelogger

import numpy as np


def build(options, architecture, wind, variables_si, parameters):

    vortex_tools.check_positive_vortex_wake_nodes(options)

    bound_wake = alg_bound_wake.build(options, architecture, variables_si, parameters)
    near_wake = alg_near_wake.build(options, architecture, variables_si, parameters)
    far_wake = alg_far_wake.build(options, architecture, wind, variables_si, parameters)

    wake = obj_wake.Wake()
    wake.set_substructure(bound_wake)
    wake.set_substructure(near_wake)
    wake.set_substructure(far_wake)

    wake.define_biot_savart_induction_functions()

    return wake


def get_model_constraints(wake, wind, variables_si, architecture):

    cstr_list = cstr_op.ConstraintList()

    superposition_cstr = get_superposition_cstr(wake, wind, variables_si, architecture)
    cstr_list.append(superposition_cstr)

    induction_cstr = get_induction_cstr(wake, wind, variables_si, architecture)
    cstr_list.append(induction_cstr)

    return cstr_list

def get_superposition_cstr(wake, wind, variables_si, architecture):

    cstr_list = cstr_op.ConstraintList()

    u_ref = wind.get_speed_ref()

    for kite_obs in architecture.kite_nodes:
        vec_u_superposition = cas.DM.zeros((3, 1))

        for substructure_type in wake.get_initialized_substructure_types():
            initialized_elements = wake.get_substructure(substructure_type).get_initialized_element_types()
            for element_type in initialized_elements:
                number_of_elements = wake.get_substructure(substructure_type).get_list(element_type).number_of_elements
                for edx in range(number_of_elements):
                    elem_u_ind_si = alg_structure.get_element_induced_velocity_si(variables_si, substructure_type, element_type, edx, kite_obs)
                    vec_u_superposition += elem_u_ind_si

        vec_u_ind = alg_structure.get_induced_velocity_at_kite_si(variables_si, kite_obs)

        resi = (vec_u_ind - vec_u_superposition) / u_ref

        local_cstr = cstr_op.Constraint(expr=resi,
                                        name='superposition_' + str(kite_obs),
                                        cstr_type='eq')
        cstr_list.append(local_cstr)

    return cstr_list

def get_induction_cstr(wake, wind, variables_si, architecture):

    cstr_list = cstr_op.ConstraintList()

    for substructure_type in wake.get_initialized_substructure_types():
        for kite_obs in architecture.kite_nodes:
            resi = wake.get_substructure(substructure_type).construct_induced_velocity_at_kite_residuals(wind, variables_si, kite_obs,
                                                                           architecture.parent_map[kite_obs])

            local_cstr = cstr_op.Constraint(expr=resi,
                                            name='induction_' + str(substructure_type) + '_' + str(kite_obs),
                                            cstr_type='eq')
            cstr_list.append(local_cstr)

    return cstr_list

def get_ocp_constraints(nlp_options, V, Outputs, model, time_grids):
    ocp_cstr_list = ocp_constraint.OcpConstraintList()

    vortex_fixing_cstr = alg_fixing.get_constraint(nlp_options, V, Outputs, model, time_grids)
    ocp_cstr_list.append(vortex_fixing_cstr)

    vortex_strength_cstr = alg_strength.get_constraint(nlp_options, V, Outputs, model)
    ocp_cstr_list.append(vortex_strength_cstr)

    return ocp_cstr_list

def test_that_model_constraint_residuals_have_correct_shape():

    options, architecture, wind, var_struct, param_struct, variables_dict = alg_structure.construct_test_model_variable_structures()
    wake = build(options, architecture, wind, var_struct, param_struct)

    total_number_of_elements = alg_structure.get_total_number_of_vortex_elements(options, architecture)
    number_of_observers = architecture.number_of_kites
    dimension_of_velocity = 3

    variables_si = var_struct

    superposition_cstr = get_superposition_cstr(wake, wind, variables_si, architecture)
    found_superposition_shape = superposition_cstr.get_expression_list('all').shape
    expected_superposition_shape = (number_of_observers * dimension_of_velocity, 1)
    cond1 = (found_superposition_shape == expected_superposition_shape)

    induction_cstr = get_induction_cstr(wake, wind, variables_si, architecture)
    found_induction_shape = induction_cstr.get_expression_list('all').shape
    expected_induction_shape = (total_number_of_elements * number_of_observers * dimension_of_velocity, 1)
    cond2 = (found_induction_shape == expected_induction_shape)

    criteria = cond1 and cond2
    if not criteria:
        message = 'an incorrect number of induction residuals have been defined for the algebraic-representation vortex wake'
        awelogger.logger.error(message)
        raise Exception(message)

def test_drawing():
    plt.close('all')

    options, architecture, wind, var_struct, param_struct, variables_dict, variables_si, parameters = alg_structure.construct_vortex_ring_test_object()
    wake = build(options, architecture, wind, variables_si, parameters)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    wake.draw(ax, 'isometric')

    options, architecture, wind, var_struct, param_struct, variables_dict, variables_si, parameters = alg_structure.construct_straight_flight_test_object()
    wake = build(options, architecture, wind, variables_si, parameters)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    wake.draw(ax, 'isometric')

    # plt.show()

    return None

def test():
    alg_bound_wake.test()
    alg_near_wake.test()
    alg_far_wake.test()
    test_that_model_constraint_residuals_have_correct_shape()
    test_drawing()

    return None

test()