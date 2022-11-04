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
constructs the far-wake filament list
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2022
'''
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


import copy
import pdb

import casadi.tools as cas
import numpy as np

import awebox.mdl.aero.induction_dir.general_dir.tools as general_tools

import awebox.mdl.aero.induction_dir.vortex_dir.tools as vortex_tools
import awebox.mdl.aero.induction_dir.vortex_dir.alg_repr_dir.structure as alg_structure

import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.element_list as obj_element_list
import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.finite_filament as obj_finite_filament
import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.semi_infinite_filament as obj_semi_infinite_filament
import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.semi_infinite_right_cylinder as obj_semi_infinite_cylinder
import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.semi_infinite_tangential_right_cylinder as obj_semi_infinite_tangential_cylinder
import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.semi_infinite_longitudinal_right_cylinder as obj_semi_infinite_longitudinal_cylinder
import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.wake_substructure as obj_wake_substructure

import awebox.tools.vector_operations as vect_op
import awebox.tools.print_operations as print_op
from awebox.logger.logger import Logger as awelogger


def build(model_options, architecture, variables_si, parameters):

    near_wake = obj_wake_substructure.WakeSubstructure(substructure_type='near')

    if there_are_enough_wake_nodes_to_require_a_near_wake(model_options):
        for kite in architecture.kite_nodes:
            filament_list = build_per_kite(model_options, kite, variables_si, parameters)
            near_wake.append(filament_list)

    dict_of_expected_number_of_elements = vortex_tools.get_expected_number_of_near_wake_elements_dict(model_options, architecture)
    near_wake.set_expected_number_of_elements_from_dict(dict_of_expected_number_of_elements)
    near_wake.confirm_all_lists_have_expected_dimensions(dict_of_expected_number_of_elements.keys())

    return near_wake

def there_are_enough_wake_nodes_to_require_a_near_wake(model_options):
    wake_nodes = general_tools.get_option_from_possible_dicts(model_options, 'wake_nodes', 'vortex')
    return (wake_nodes > 1)

def build_per_kite(model_options, kite, variables_si, parameters):

    wake_nodes = general_tools.get_option_from_possible_dicts(model_options, 'wake_nodes', 'vortex')

    filament_list = obj_element_list.ElementList(expected_number_of_elements= 3 * (wake_nodes-1))

    for ring in range(wake_nodes-1):
        ring_filaments = build_per_kite_per_ring(model_options, kite, ring, variables_si, parameters)
        filament_list.append(ring_filaments)

    filament_list.confirm_list_has_expected_dimensions()

    return filament_list


def build_per_kite_per_ring(options, kite, ring, variables_si, parameters):
    filament_ordering = vortex_tools.ordering_of_near_wake_filaments_in_vortex_horseshoe()
    expected_number_of_elements = len(filament_ordering.keys())
    filament_list = obj_element_list.ElementList(expected_number_of_elements=expected_number_of_elements)

    for pdx in range(expected_number_of_elements):
        local_filament_position = filament_ordering[pdx]

        if (local_filament_position == 'ext') or (local_filament_position == 'int'):
            trailing_filaments = build_single_trailing_per_kite_per_ring(options, kite, ring, variables_si, parameters, local_filament_position)
            filament_list.append(trailing_filaments)

        elif local_filament_position == 'closing':
            closing_filament = build_closing_per_kite_per_ring(options, kite, ring, variables_si, parameters)
            filament_list.append(closing_filament)

        else:
            message = 'unexpected type of near wake vortex filament (' + local_filament_position + ')'
            print_op.error(message)

    filament_list.confirm_list_has_expected_dimensions()

    return filament_list


def build_closing_per_kite_per_ring(options, kite, ring, variables_si, parameters):

    wake_node = ring

    NE_wingtip = vortex_tools.get_NE_wingtip_name()
    PE_wingtip = vortex_tools.get_PE_wingtip_name()

    LENE = vortex_tools.get_wake_node_position_si(options, variables_si, kite, NE_wingtip, wake_node+1)
    LEPE = vortex_tools.get_wake_node_position_si(options, variables_si, kite, PE_wingtip, wake_node+1)

    strength = vortex_tools.get_vortex_ring_strength_si(variables_si, kite, ring+1)
    strength_prev = vortex_tools.get_vortex_ring_strength_si(variables_si, kite, ring)

    r_core = vortex_tools.get_r_core(options, parameters)

    dict_info = {'x_start': LENE,
                    'x_end': LEPE,
                    'r_core': r_core,
                    'strength': strength - strength_prev
                    }
    fil = obj_finite_filament.FiniteFilament(dict_info)

    return fil


def build_single_trailing_per_kite_per_ring(options, kite, ring, variables_si, parameters, tip):

    wake_node = ring

    strength = vortex_tools.get_vortex_ring_strength_si(variables_si, kite, ring)
    r_core = vortex_tools.get_r_core(options, parameters)

    filament_list = obj_element_list.ElementList(expected_number_of_elements=1)

    wingtips_and_strength_directions = vortex_tools.get_wingtip_name_and_strength_direction_dict()
    tip_directionality = wingtips_and_strength_directions[tip]
    x_start = vortex_tools.get_wake_node_position_si(options, variables_si, kite, tip, wake_node)
    x_end = vortex_tools.get_wake_node_position_si(options, variables_si, kite, tip, wake_node + 1)

    dict_info = {'x_start': x_start,
                 'x_end': x_end,
                 'r_core': r_core,
                 'strength': tip_directionality * strength
                 }
    fil = obj_finite_filament.FiniteFilament(dict_info)
    filament_list.append(fil)

    filament_list.confirm_list_has_expected_dimensions()

    return filament_list


def test_correct_finite_filaments_defined():

    options, architecture, wind, var_struct, param_struct, variables_dict, variables_si, parameters = alg_structure.construct_straight_flight_test_object()
    near_wake = build(options, architecture, variables_si, parameters)

    element_type = 'finite_filament'
    near_wake.confirm_all_lists_have_expected_dimensions([element_type])
    fil_list = near_wake.get_list(element_type)

    half_b_span_vec = 0.5 * vect_op.yhat_dm()
    x_kite = cas.DM.zeros((3, 1))

    x_NE_0 = x_kite - half_b_span_vec
    x_PE_0 = x_kite + half_b_span_vec
    x_NE_1 = x_NE_0 + vect_op.xhat()
    x_PE_1 = x_PE_0 + vect_op.xhat()
    strength_0 = 4.
    strength_1 = 1.

    PE_dict = {'x_start': x_PE_0,
                 'x_end': x_PE_1,
                 'r_core': 0.01,
                 'strength': strength_0
                 }
    filPE = obj_finite_filament.FiniteFilament(PE_dict)
    condition1 = (fil_list.is_element_in_list(filPE) == True)

    NE_dict = {'x_start': x_NE_0,
                 'x_end': x_NE_1,
                 'r_core': 0.01,
                 'strength': -1. * strength_0
                 }
    filNE = obj_finite_filament.FiniteFilament(NE_dict)
    condition2 = (fil_list.is_element_in_list(filNE) == True)

    closing_dict = {'x_start': x_NE_1,
               'x_end': x_PE_1,
               'r_core': 0.01,
               'strength': (strength_1 - strength_0)
               }
    fil_closing = obj_finite_filament.FiniteFilament(closing_dict)
    condition3 = (fil_list.is_element_in_list(fil_closing) == True)

    other_dict = {'x_start': 3. * vect_op.yhat_np() + vect_op.xhat_np(),
               'x_end': 0.5 * vect_op.yhat_np() + 2. * vect_op.xhat_np(),
               'r_core': 0.01,
               'strength': 1.
               }
    fil_other = obj_finite_filament.FiniteFilament(other_dict)
    condition4 = (fil_list.is_element_in_list(fil_other) == False)

    criteria = condition1 and condition2 and condition3 and condition4
    if not criteria:
        message = 'near_wake does not contain the expected vortex elements'
        print_op.error(message)

    return None

def test_finite_filament_drawing(test_includes_visualization=False):
    if test_includes_visualization:

        options, architecture, wind, var_struct, param_struct, variables_dict, variables_si, parameters = alg_structure.construct_straight_flight_test_object()
        near_wake = build(options, architecture, variables_si, parameters)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        near_wake.draw(ax, 'isometric')

        plt.show()

    return None

def test(test_includes_visualization=False):
    test_finite_filament_drawing(test_includes_visualization)
    test_correct_finite_filaments_defined()

# test()