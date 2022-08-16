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
constructs the bound filament list
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2022
'''
import pdb

import casadi.tools as cas
import numpy as np
import matplotlib.pyplot as plt

import awebox.mdl.aero.induction_dir.vortex_dir.tools as vortex_tools
import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.finite_filament as obj_finite_filament
import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.element_list as obj_element_list
import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.wake_substructure as obj_wake_substructure

import awebox.mdl.aero.induction_dir.vortex_dir.alg_repr_dir.structure as alg_structure

import awebox.tools.vector_operations as vect_op
import awebox.tools.print_operations as print_op
from awebox.logger.logger import Logger as awelogger

def build(options, architecture, variables_si, parameters):

    bound_wake = obj_wake_substructure.WakeSubstructure(substructure_type='bound')
    for kite in architecture.kite_nodes:
        filament_list = build_per_kite(options, kite, variables_si, parameters)
        bound_wake.append(filament_list)

    number_of_elements_dict = {'finite_filament':architecture.number_of_kites}
    bound_wake.set_expected_number_of_elements_from_dict(number_of_elements_dict)
    bound_wake.confirm_all_lists_have_expected_dimensions(['finite_filament'])

    return bound_wake

def build_per_kite(options, kite, variables_si, parameters):

    wake_node = 0
    ring = wake_node

    NE_wingtip = vortex_tools.get_NE_wingtip_name()
    PE_wingtip = vortex_tools.get_PE_wingtip_name()

    LENE = vortex_tools.get_wake_node_position_si(options, variables_si, kite, NE_wingtip, wake_node)
    LEPE = vortex_tools.get_wake_node_position_si(options, variables_si, kite, PE_wingtip, wake_node)

    strength = vortex_tools.get_vortex_ring_strength_si(variables_si, kite, ring)
    strength_prev = cas.DM.zeros((1, 1))

    r_core = vortex_tools.get_r_core(options, parameters)

    dict_info_LE = {'x_start': LENE,
                    'x_end': LEPE,
                    'r_core': r_core,
                    'strength': strength - strength_prev
                    }
    fil_LE = obj_finite_filament.FiniteFilament(dict_info_LE)
    filament_list = obj_element_list.ElementList(expected_number_of_elements=1)
    filament_list.append(fil_LE)

    filament_list.confirm_list_has_expected_dimensions()

    return filament_list

def test_correct_filaments_defined():

    options, architecture, wind, var_struct, param_struct, variables_dict, variables_si, parameters = alg_structure.construct_straight_flight_test_object()
    bound_wake = build(options, architecture, variables_si, parameters)

    element_type = 'finite_filament'
    bound_wake.confirm_all_lists_have_expected_dimensions([element_type])
    fil_list = bound_wake.get_list(element_type)

    half_b_span_vec = 0.5 * vect_op.yhat_dm()
    x_kite = cas.DM.zeros((3, 1))

    x_NE = x_kite - half_b_span_vec
    x_PE = x_kite + half_b_span_vec

    fil_dict = {'x_start': x_NE,
                 'x_end': x_PE,
                 'r_core': 0.01,
                 'strength': 4.
                 }
    fil = obj_finite_filament.FiniteFilament(fil_dict)
    condition1 = (fil_list.is_element_in_list(fil) == True)

    other_dict = {'x_start': 3. * vect_op.yhat_np() + vect_op.xhat_np(),
               'x_end': 0.5 * vect_op.yhat_np() + 2. * vect_op.xhat_np(),
               'r_core': 0.01,
               'strength': 1.
               }
    fil_other = obj_finite_filament.FiniteFilament(other_dict)
    condition2 = (fil_list.is_element_in_list(fil_other) == False)

    criteria = condition1 and condition2
    if not criteria:
        message = 'bound_wake does not contain the expected vortex elements'
        awelogger.logger.error(message)
        raise Exception(message)

    return None

def test_drawing():
    plt.close('all')

    options, architecture, wind, var_struct, param_struct, variables_dict, variables_si, parameters = alg_structure.construct_straight_flight_test_object()
    bound_wake = build(options, architecture, variables_si, parameters)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    bound_wake.draw(ax, 'isometric')

    # plt.show()

    return None


def test():
    test_correct_filaments_defined()
    test_drawing()
    return None

# test()