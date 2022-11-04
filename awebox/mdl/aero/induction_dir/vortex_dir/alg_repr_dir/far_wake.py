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
import awebox.mdl.aero.geometry_dir.geometry as geom
import awebox.mdl.aero.induction_dir.vortex_dir.alg_repr_dir.structure as alg_structure

import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.element_list as obj_element_list
import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.finite_filament as obj_finite_filament
import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.semi_infinite_filament as obj_semi_infinite_filament
import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.semi_infinite_right_cylinder as obj_semi_infinite_right_cylinder
import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.semi_infinite_tangential_right_cylinder as obj_semi_infinite_tangential_right_cylinder
import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.semi_infinite_longitudinal_right_cylinder as obj_semi_infinite_longitudinal_right_cylinder
import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.wake_substructure as obj_wake_substructure

import awebox.tools.vector_operations as vect_op
import awebox.tools.print_operations as print_op

from awebox.logger.logger import Logger as awelogger

def build(options, architecture, wind, variables_si, parameters):

    far_wake = obj_wake_substructure.WakeSubstructure(substructure_type='far')
    for kite in architecture.kite_nodes:
        filament_list = build_per_kite(options, kite, wind, variables_si, parameters, architecture)
        far_wake.append(filament_list)

    dict_of_expected_number_of_elements = vortex_tools.get_expected_number_of_far_wake_elements_dict(options, architecture)
    far_wake.set_expected_number_of_elements_from_dict(dict_of_expected_number_of_elements)
    far_wake.confirm_all_lists_have_expected_dimensions(dict_of_expected_number_of_elements.keys())

    return far_wake

def build_per_kite(options, kite, wind, variables_si, parameters, architecture):
    far_wake_element_type = general_tools.get_option_from_possible_dicts(options, 'far_wake_element_type', 'vortex')

    if far_wake_element_type == 'finite_filament':
        list = build_finite_filament_per_kite(options, kite, wind, variables_si, parameters)
    elif far_wake_element_type == 'semi_infinite_filament':
        list = build_semi_infinite_filament_per_kite(options, kite, wind, variables_si, parameters)
    elif far_wake_element_type == 'semi_infinite_right_cylinder':
        list = build_semi_infinite_right_cylinders_per_kite(options, kite, wind, variables_si, architecture)
    elif far_wake_element_type == 'not_in_use':
        list = obj_element_list.ElementList(expected_number_of_elements=0)
    else:
        message = 'unexpected type of far-wake vortex element (' + far_wake_element_type + '). maybe, check your spelling?'
        print_op.error(message)

    return list

def get_convection_direction(wind):
    l_hat = wind.get_wind_direction()
    return l_hat

def build_finite_filament_per_kite(options, kite, wind, variables_si, parameters):

    ring = get_far_wake_ring_number(options)
    wake_node = ring

    strength = vortex_tools.get_vortex_ring_strength_si(variables_si, kite, ring)
    r_core = vortex_tools.get_r_core(options, parameters)

    l_hat = get_convection_direction(wind)

    far_wake_convection_time = general_tools.get_option_from_possible_dicts(options, 'far_wake_convection_time', 'vortex')
    u_ref = options['wind']['u_ref'] #note, this value will not change in sweep.
    convection_distance = far_wake_convection_time * u_ref

    filament_list = obj_element_list.ElementList(expected_number_of_elements=2)

    wingtips_and_strength_directions = vortex_tools.get_wingtip_name_and_strength_direction_dict()
    for tip, tip_directionality in wingtips_and_strength_directions.items():
        x_start = vortex_tools.get_wake_node_position_si(options, variables_si, kite, tip, wake_node)
        vec_s = l_hat * convection_distance
        x_end = x_start + vec_s

        dict_info = {'x_start': x_start,
                     'x_end': x_end,
                     'r_core': r_core,
                     'strength': tip_directionality * strength
                     }
        fil = obj_finite_filament.FiniteFilament(dict_info)
        filament_list.append(fil)

    filament_list.confirm_list_has_expected_dimensions()

    return filament_list


def build_semi_infinite_filament_per_kite(options, kite, wind, variables_si, parameters):
    ring = get_far_wake_ring_number(options)
    wake_node = ring

    strength = vortex_tools.get_vortex_ring_strength_si(variables_si, kite, ring)
    r_core = vortex_tools.get_r_core(options, parameters)

    l_hat = get_convection_direction(wind)

    filament_list = obj_element_list.ElementList(expected_number_of_elements=2)

    wingtips_and_strength_directions = vortex_tools.get_wingtip_name_and_strength_direction_dict()
    for tip, tip_directionality in wingtips_and_strength_directions.items():
        x_start = vortex_tools.get_wake_node_position_si(options, variables_si, kite, tip, wake_node)

        dict_info = {'x_start': x_start,
                     'l_hat': l_hat,
                     'r_core': r_core,
                     'strength': tip_directionality * strength
                     }

        fil = obj_semi_infinite_filament.SemiInfiniteFilament(dict_info)
        filament_list.append(fil)

    filament_list.confirm_list_has_expected_dimensions()

    return filament_list

def build_semi_infinite_right_cylinders_per_kite(model_options, kite, wind, variables_si, architecture):

    parent = architecture.parent_map[kite]

    ring = get_far_wake_ring_number(model_options)
    wake_node = ring

    wingtips_and_strength_directions = vortex_tools.get_wingtip_name_and_strength_direction_dict()
    l_hat = get_convection_direction(wind)

    circulation_total = vortex_tools.get_vortex_ring_strength_si(variables_si, kite, ring)

    x_center = vortex_tools.get_far_wake_cylinder_center_position_si(variables_si, parent)
    pitch = vortex_tools.get_far_wake_cylinder_pitch_si(variables_si, parent)

    # the direction of the tangential vorticity must be opposite to the direction of the kite rotation
    kite_motion_is_right_hand_rule_positive_around_wind_direction = geom.kite_motion_is_right_hand_rule_positive_around_wind_direction(model_options, variables_si, kite, architecture, wind)
    # if y(x = 0) = -1 and y(x = 1) = +1 -> y = 2 x - 1
    kite_motion_directionality = 2. * kite_motion_is_right_hand_rule_positive_around_wind_direction - 1.

    epsilon_m = general_tools.get_option_from_possible_dicts(model_options, 'vortex_epsilon_m', 'vortex')
    epsilon_r = general_tools.get_option_from_possible_dicts(model_options, 'vortex_epsilon_r', 'vortex')

    approximation_order_for_elliptic_integrals = general_tools.get_option_from_possible_dicts(model_options, 'approximation_order_for_elliptic_integrals', 'vortex')


    tan_cyl_list = obj_element_list.ElementList(expected_number_of_elements=2)
    long_cyl_list = obj_element_list.ElementList(expected_number_of_elements=2)

    for tip, tip_directionality in wingtips_and_strength_directions.items():
        x_start = vortex_tools.get_wake_node_position_si(model_options, variables_si, kite, tip, wake_node)
        radius, l_start = obj_semi_infinite_right_cylinder.calculate_radius_and_l_start(x_start, x_center, l_hat)

        strength_tan = -1. * circulation_total / pitch * kite_motion_directionality * tip_directionality
        strength_long = circulation_total / (2. * np.pi * radius) * tip_directionality

        order_tan = {'x_center': x_center,
                      'l_hat': l_hat,
                      'radius': radius,
                      'l_start': l_start,
                      'epsilon_m': epsilon_m,
                      'epsilon_r': epsilon_r,
                      'strength': strength_tan
                      }
        tan_cyl = obj_semi_infinite_tangential_right_cylinder.SemiInfiniteTangentialRightCylinder(order_tan,
                                                                                                  approximation_order_for_elliptic_integrals)
        tan_cyl_list.append(tan_cyl)

        order_long = {'x_center': x_center,
                      'l_hat': l_hat,
                      'radius': radius,
                      'l_start': l_start,
                      'epsilon_m': epsilon_m,
                      'epsilon_r': epsilon_r,
                      'strength': strength_long
                      }
        long_cyl = obj_semi_infinite_longitudinal_right_cylinder.SemiInfiniteLongitudinalRightCylinder(order_long,
                                                                                                       approximation_order_for_elliptic_integrals)
        long_cyl_list.append(long_cyl)

    return [tan_cyl_list, long_cyl_list]

def get_far_wake_ring_number(options):
    wake_nodes = general_tools.get_option_from_possible_dicts(options, 'vortex_wake_nodes', 'vortex')
    return (wake_nodes - 1)

def test_correct_finite_filaments_defined():

    test_type = 'finite_filament'
    options, architecture, wind, var_struct, param_struct, variables_dict, variables_si, parameters = alg_structure.construct_straight_flight_test_object(test_type)
    far_wake = build(options, architecture, wind, variables_si, parameters)

    far_wake.confirm_all_lists_have_expected_dimensions([test_type])
    fil_list = far_wake.get_list(test_type)

    PE_dict = {'x_start': 0.5 * vect_op.yhat_np() + vect_op.xhat_np(),
                 'x_end': 0.5 * vect_op.yhat_np() + 2. * vect_op.xhat_np(),
                 'r_core': 0.01,
                 'strength': 1.
                 }
    filPE = obj_finite_filament.FiniteFilament(PE_dict)
    condition1 = (fil_list.is_element_in_list(filPE) == True)

    NE_dict = {'x_start': -0.5 * vect_op.yhat_np() + vect_op.xhat_np(),
               'x_end': -0.5 * vect_op.yhat_np() + 2. * vect_op.xhat_np(),
               'r_core': 0.01,
               'strength': -1.
               }
    filNE = obj_finite_filament.FiniteFilament(NE_dict)
    condition2 = (fil_list.is_element_in_list(filNE) == True)

    other_dict = {'x_start': 3. * vect_op.yhat_np() + vect_op.xhat_np(),
               'x_end': 0.5 * vect_op.yhat_np() + 2. * vect_op.xhat_np(),
               'r_core': 0.01,
               'strength': 1.
               }
    fil_other = obj_finite_filament.FiniteFilament(other_dict)
    condition3 = (fil_list.is_element_in_list(fil_other) == False)

    criteria = condition1 and condition2 and condition3
    if not criteria:
        message = 'far_wake (finite filament) does not contain the expected vortex elements'
        print_op.error(message)

    return None

def test_finite_filament_drawing(test_includes_visualization=False):
    if test_includes_visualization:

        test_type = 'finite_filament'
        options, architecture, wind, var_struct, param_struct, variables_dict, variables_si, parameters = alg_structure.construct_straight_flight_test_object(test_type)
        far_wake = build(options, architecture, wind, variables_si, parameters)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        far_wake.draw(ax, 'isometric')

        plt.show()

    return None

def test_correct_semi_infinite_filaments_defined():
    test_type = 'semi_infinite_filament'
    options, architecture, wind, var_struct, param_struct, variables_dict, variables_si, parameters = alg_structure.construct_straight_flight_test_object(test_type)
    far_wake = build(options, architecture, wind, variables_si, parameters)

    far_wake.confirm_all_lists_have_expected_dimensions([test_type])
    fil_list = far_wake.get_list(test_type)

    PE_dict = {'x_start': 0.5 * vect_op.yhat_np() + vect_op.xhat_np(),
                 'l_hat': vect_op.xhat_np(),
                 'r_core': 0.01,
                 'strength': 1.
                 }
    filPE = obj_semi_infinite_filament.SemiInfiniteFilament(PE_dict)
    condition1 = (fil_list.is_element_in_list(filPE) == True)

    NE_dict = {'x_start': -0.5 * vect_op.yhat_np() + vect_op.xhat_np(),
               'l_hat': vect_op.xhat_np(),
               'r_core': 0.01,
               'strength': -1.
               }
    filNE = obj_semi_infinite_filament.SemiInfiniteFilament(NE_dict)
    condition2 = (fil_list.is_element_in_list(filNE) == True)

    other_dict = {'x_start': 3. * vect_op.yhat_np() + vect_op.xhat_np(),
               'l_hat': vect_op.xhat_np(),
               'r_core': 0.01,
               'strength': 1.
               }
    fil_other = obj_semi_infinite_filament.SemiInfiniteFilament(other_dict)
    condition3 = (fil_list.is_element_in_list(fil_other) == False)

    criteria = condition1 and condition2 and condition3
    if not criteria:
        message = 'far_wake (semi-infinite filament) does not contain the expected vortex elements'
        print_op.error(message)

    return None


def test_semi_infinite_filament_drawing(test_includes_visualization=False):
    if test_includes_visualization:
        test_type = 'semi_infinite_filament'
        options, architecture, wind, var_struct, param_struct, variables_dict, variables_si, parameters = alg_structure.construct_straight_flight_test_object(test_type)
        far_wake = build(options, architecture, wind, variables_si, parameters)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        far_wake.draw(ax, 'isometric')

        plt.show()

    return None

def test_semi_infinite_right_cylinder_drawing(test_includes_visualization=False):
    if test_includes_visualization:

        test_type = 'semi_infinite_right_cylinder'
        options, architecture, wind, var_struct, param_struct, variables_dict, variables_si, parameters = alg_structure.construct_circular_flight_test_object(test_type)
        far_wake = build(options, architecture, wind, variables_si, parameters)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        far_wake.draw(ax, 'isometric')

        plt.show()

    return None

def test_correct_semi_infinite_right_cylinders_defined():
    test_type = 'semi_infinite_right_cylinder'
    options, architecture, wind, var_struct, param_struct, variables_dict, variables_si, parameters = alg_structure.construct_circular_flight_test_object(test_type)
    far_wake = build(options, architecture, wind, variables_si, parameters)

    far_wake.confirm_all_lists_have_expected_dimensions(['semi_infinite_tangential_right_cylinder', 'semi_infinite_longitudinal_right_cylinder'])
    tan_cyl_list = far_wake.get_list('semi_infinite_tangential_right_cylinder')
    long_cyl_list = far_wake.get_list('semi_infinite_longitudinal_right_cylinder')

    x_center = cas.DM.zeros((3, 1))
    l_hat = vect_op.xhat()
    radius_ext = 2.5
    radius_int = 1.5

    radius = {'ext': radius_ext,
              'int': radius_int}

    l_start = 1.
    epsilon_m = 1.0e-8
    epsilon_r = 1.0e-8

    strength = 1.

    strength_tan_ext = strength * (-1.) * 1.e8
    strength_tan_int = strength * (+1.) * 1.e8
    strength_long_ext = strength/(2. * np.pi * radius_ext)
    strength_long_int = -1. * strength/(2. * np.pi * radius_int)

    strength = {'ext':
                    {'tan': strength_tan_ext,
                     'long': strength_long_ext},
                'int':
                    {'tan': strength_tan_int,
                     'long': strength_long_int}
                }

    order_base = {'x_center': x_center,
                 'l_hat': l_hat,
                 'l_start': l_start,
                 'epsilon_m': epsilon_m,
                 'epsilon_r': epsilon_r
                  }

    conditions = {'ext':{},
                  'int':{}}
    total_conditions = 0.

    for tip in strength.keys():
        for dir in strength[tip].keys():
            order = copy.deepcopy(order_base)
            order['radius'] = radius[tip]
            order['strength'] = strength[tip][dir]

            if dir == 'tan':
                tan_cyl = obj_semi_infinite_tangential_right_cylinder.SemiInfiniteTangentialRightCylinder(order)
                conditions[tip][dir] = (tan_cyl_list.is_element_in_list(tan_cyl) == True)
                total_conditions += conditions[tip][dir]

            elif dir == 'long':
                long_cyl = obj_semi_infinite_longitudinal_right_cylinder.SemiInfiniteLongitudinalRightCylinder(order)
                conditions[tip][dir] = (long_cyl_list.is_element_in_list(long_cyl) == True)
                total_conditions += conditions[tip][dir]

    criteria = (total_conditions == 4)

    if not criteria:

        message = 'far_wake (semi-infinite right cylinder) does not contain the expected vortex elements'
        print_op.error(message)

    return None

def test(test_includes_visualization=False):
    test_finite_filament_drawing(test_includes_visualization)
    test_correct_finite_filaments_defined()

    test_semi_infinite_filament_drawing(test_includes_visualization)
    test_correct_semi_infinite_filaments_defined()

    test_semi_infinite_right_cylinder_drawing(test_includes_visualization)
    test_correct_semi_infinite_right_cylinders_defined()

    return None

# test()