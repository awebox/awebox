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
- author: rachel leuthold, alu-fr 2020
'''
import copy

import numpy as np
import awebox.mdl.aero.induction_dir.vortex_dir.tools as vortex_tools
import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.element_list as vortex_element_list
import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.element as vortex_element
import awebox.tools.struct_operations as struct_op
import awebox.tools.constraint_operations as cstr_op
import casadi.tools as cas
from awebox.logger.logger import Logger as awelogger

#
# def get_lists(options, variables_si, parameters, architecture, wind):
#
#     far_wake_model = options['induction']['vortex_far_wake_model']
#     kite_nodes = architecture.kite_nodes
#
#     expected_number_filaments = expected_number_of_filaments(options, architecture)
#     expected_number_directional_cylinders = expected_number_of_directional_cylinders(options, architecture)
#
#     filament_list = vortex_element_list.ElementList(expected_number_filaments)
#     tangential_cylinder_list = vortex_element_list.ElementList(expected_number_directional_cylinders)
#     longitudinal_cylinder_list = vortex_element_list.ElementList(expected_number_directional_cylinders)
#
#     if 'filament' in far_wake_model:
#         for kite in kite_nodes:
#             last_ring_fil_list = get_filament_list_from_kite(options, variables_si, parameters, wind, kite)
#             filament_list.append(last_ring_fil_list)
#
#     elif 'cylinder' in far_wake_model:
#         for kite in kite_nodes:
#             tan_list, long_list = get_cylinder_list_from_kite(options, variables_si, parameters, wind, kite, architecture)
#             tangential_cylinder_list.append(tan_list)
#             longitudinal_cylinder_list.append(long_list)
#
#     elif far_wake_model not in ['not_in_use', 'repetition']:
#         message = 'unknown vortex far-wake model selected.'
#         awelogger.logger.error(message)
#         raise Exception(message)
#
#     filament_list.confirm_list_has_expected_dimensions()
#
#     return filament_list, tangential_cylinder_list, longitudinal_cylinder_list
#
#
# def get_filament_list_from_kite(options, variables_si, parameters, wind, kite):
#
#     filament_list = vortex_element_list.ElementList()
#
#     wake_nodes = options['induction']['vortex_wake_nodes']
#     rings = wake_nodes
#
#     last_tracked_wake_node = wake_nodes - 1
#     ring = rings - 1 # remember: indexing starts at 0.
#
#     far_convection_time = options['induction']['vortex_far_convection_time']
#     far_wake_model = options['induction']['vortex_far_wake_model']
#
#     NE_wingtip = vortex_tools.get_NE_wingtip_name()
#     PE_wingtip = vortex_tools.get_PE_wingtip_name()
#
#     LENE = vortex_tools.get_wake_node_position_si(options, variables_si, kite, NE_wingtip, last_tracked_wake_node)
#     LEPE = vortex_tools.get_wake_node_position_si(options, variables_si, kite, PE_wingtip, last_tracked_wake_node)
#
#     if far_wake_model == 'pathwise_filament':
#         farwake_name = 'wu_farwake_' + str(kite) + '_'
#
#         if isinstance(variables_si, cas.structure3.DMStruct):
#             velocity_PE = variables_si['xl', farwake_name + PE_wingtip]
#             velocity_NE = variables_si['xl', farwake_name + NE_wingtip]
#         else:
#             velocity_PE = variables_si['xl'][farwake_name + PE_wingtip]
#             velocity_NE = variables_si['xl'][farwake_name + NE_wingtip]
#
#     elif far_wake_model == 'freestream_filament':
#         velocity_PE = wind.get_velocity(LEPE[[2]])
#         velocity_NE = wind.get_velocity(LENE[[2]])
#
#     TENE = LENE + far_convection_time * velocity_NE
#     TEPE = LEPE + far_convection_time * velocity_PE
#
#     strength = vortex_tools.get_ring_strength_si(variables_si, kite, ring)
#
#     r_core = vortex_tools.get_r_core(options, parameters)
#
#     dict_info_PE = {'x_start': LEPE,
#                     'x_end': TEPE,
#                     'r_core': r_core,
#                     'strength': strength
#                     }
#     fil_PE = vortex_element.Filament(dict_info_PE)
#     filament_list.append(fil_PE)
#
#     dict_info_NE = {'x_start': TENE,
#                     'x_end': LENE,
#                     'r_core': r_core,
#                     'strength': strength
#                     }
#     fil_NE = vortex_element.Filament(dict_info_NE)
#     filament_list.append(fil_NE)
#
#     return filament_list
#
# def get_cylinder_strength_signs(clockwise_rotation_about_xhat=True):
#
#     if clockwise_rotation_about_xhat:
#         clockwise_sign = +1.
#     else:
#         clockwise_sign = -1.
#
#     tan_PE_sign = clockwise_sign * -1.
#     long_PE_sign = +1.
#
#     tan_NE_sign = -1. * tan_PE_sign
#     long_NE_sign = -1. * long_PE_sign
#
#     NE_wingtip = vortex_tools.get_NE_wingtip_name()
#     PE_wingtip = vortex_tools.get_PE_wingtip_name()
#
#     signs = {'tan_' + PE_wingtip: tan_PE_sign,
#              'long_' + PE_wingtip: long_PE_sign,
#              'tan_' + NE_wingtip: tan_NE_sign,
#              'long_' + NE_wingtip: long_NE_sign
#              }
#
#     return signs
#
# def get_cylinder_list_from_kite(options, variables_si, parameters, wind, kite, architecture):
#
#     tangential_cylinder_list = vortex_element_list.ElementList()
#     longitudinal_cylinder_list = vortex_element_list.ElementList()
#
#     wake_nodes = options['induction']['vortex_wake_nodes']
#     last_tracked_wake_node = wake_nodes - 1
#
#     NE_wingtip = vortex_tools.get_NE_wingtip_name()
#     PE_wingtip = vortex_tools.get_PE_wingtip_name()
#
#     wr_name_base = 'wr_' + str(kite) + '_'
#     radius_NE = struct_op.get_variable_from_model_or_reconstruction(variables_si, 'xl', wr_name_base + NE_wingtip)
#     radius_PE = struct_op.get_variable_from_model_or_reconstruction(variables_si, 'xl', wr_name_base + PE_wingtip)
#
#     x_kite_NE = vortex_tools.get_wake_node_position_si(options, variables_si, kite, NE_wingtip, last_tracked_wake_node)
#     x_kite_PE = vortex_tools.get_wake_node_position_si(options, variables_si, kite, PE_wingtip, last_tracked_wake_node)
#
#     wx_center_name = 'wx_center_' + str(kite)
#     x_center = struct_op.get_variable_from_model_or_reconstruction(variables_si, 'xl', wx_center_name)
#
#     l_hat = wind.get_wind_direction()
#     l_start_NE = cas.mtimes((x_kite_NE - x_center).T, l_hat)
#     l_start_PE = cas.mtimes((x_kite_PE - x_center).T, l_hat)
#
#     strength = vortex_tools.get_ring_strength_si(variables_si, kite, last_tracked_wake_node)
#
#     epsilon = vortex_tools.get_epsilon(options, parameters)
#
#     NE_dict = {'x_center': x_center,
#                'radius': radius_NE,
#                'l_start': l_start_NE,
#                'l_hat': l_hat,
#                'epsilon': epsilon
#                }
#     strength_tan_NE, strength_long_NE = get_cylinder_strength(options, strength, variables_si, kite, NE_wingtip)
#
#     tan_NE_dict = copy.deepcopy(NE_dict)
#     tan_NE_dict['strength'] = strength_tan_NE
#     tan_NE = vortex_element.TangentialCylinder(tan_NE_dict)
#     tangential_cylinder_list.append(tan_NE)
#
#     long_NE_dict = copy.deepcopy(NE_dict)
#     long_NE_dict['strength'] = strength_long_NE
#     long_NE = vortex_element.LongitudinalCylinder(long_NE_dict)
#     longitudinal_cylinder_list.append(long_NE)
#
#     PE_dict = {'x_center': x_center,
#                'radius': radius_PE,
#                'l_start': l_start_PE,
#                'l_hat': l_hat,
#                'epsilon': epsilon
#                }
#     strength_tan_PE, strength_long_PE = get_cylinder_strength(options, strength, variables_si, kite, PE_wingtip)
#
#     tan_PE_dict = copy.deepcopy(PE_dict)
#     tan_PE_dict['strength'] = strength_tan_PE
#     tan_PE = vortex_element.TangentialCylinder(tan_PE_dict)
#     tangential_cylinder_list.append(tan_PE)
#
#     long_PE_dict = copy.deepcopy(PE_dict)
#     long_PE_dict['strength'] = strength_long_PE
#     long_PE = vortex_element.LongitudinalCylinder(long_PE_dict)
#     longitudinal_cylinder_list.append(long_PE)
#
#     return tangential_cylinder_list, longitudinal_cylinder_list
#
# def get_cylinder_strength(options, strength, variables_si, kite, tip):
#
#     # see pages 217 and 268 of Branlard
#
#     clockwise_rotation_about_xhat = options['aero']['vortex']['clockwise_rotation_about_xhat']
#     signs = get_cylinder_strength_signs(clockwise_rotation_about_xhat)
#
#     w_radius_name = 'wr_' + str(kite) + '_' + tip
#     w_pitch_name = 'wh_' + str(kite) + '_' + tip
#
#     radius = variables_si['xl'][w_radius_name]
#     pitch = variables_si['xl'][w_pitch_name]
#
#     strength_tan = signs['tan_' + tip] * strength / pitch
#     strength_long = signs['long_' + tip] * strength / (2. * np.pi * radius)
#
#     return strength_tan, strength_long
#
#
# def expected_number_of_filaments(options, architecture):
#
#     number_kites = architecture.number_of_kites
#
#     use = 1
#     filaments = use * 3 * number_kites * (wake_nodes - 1)
#
#     return filaments
#
# def expected_number_of_directional_cylinders(options, architecture):
#
#     far_wake_model = vortex_tools.get_option_from_possible_dicts(options, 'far_wake_model')
#     number_kites = architecture.number_of_kites
#
#     if 'cylinder' in far_wake_model:
#         use = 1
#     else:
#         use = 0
#
#     cylinders = use * 2 * number_kites
#
#     return cylinders
#
# def get_cylinder_radius_cstr(options, wind, variables_si, parameters, architecture):
#     wingtips = ['ext', 'int']
#     wake_nodes = options['aero']['vortex']['wake_nodes']
#     vortex_representation = options['aero']['vortex']['representation']
#
#     wake_node = wake_nodes - 1
#     l_hat = wind.get_wind_direction()
#     b_ref = parameters['theta0', 'geometry', 'b_ref']
#
#     cstr_list = cstr_op.ConstraintList()
#
#     for kite in architecture.kite_nodes:
#         for tip in wingtips:
#             coord_name = 'wx_' + str(kite) + '_' + tip + '_' + str(wake_node)
#             if vortex_representation == 'state':
#                 wx_node = variables_si['xd'][coord_name]
#             elif vortex_representation == 'alg':
#                 wx_node = variables_si['xl'][coord_name]
#             else:
#                 message = 'specified vortex representation ' + vortex_representation + ' is not supported'
#                 awelogger.logger.error(message)
#                 raise Exception(message)
#
#             wx_center_name = 'wx_center_' + str(kite)
#             wx_center = variables_si['xl'][wx_center_name]
#
#             radial_vec = wx_node - wx_center
#             radius_vec = radial_vec - cas.mtimes(radial_vec.T, l_hat) * l_hat
#
#             w_radius_name = 'wr_' + str(kite) + '_' + tip
#             wr = variables_si['xl'][w_radius_name]
#
#             resi_unscaled = wr**2. - cas.mtimes(radius_vec.T, radius_vec)
#             resi = resi_unscaled / b_ref**2.
#
#             name = 'far_wake_cylinder_radius_' + str(kite) + '_' + tip
#             cstr = cstr_op.Constraint(expr = resi, cstr_type='eq', name=name)
#             cstr_list.append(cstr)
#
#     return cstr_list
