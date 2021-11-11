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
constructs the filament list
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2020
'''
import pdb

import numpy as np
import awebox.mdl.aero.induction_dir.vortex_dir.tools as vortex_tools
import awebox.mdl.aero.induction_dir.vortex_dir.element_list as vortex_element_list
import awebox.mdl.aero.induction_dir.vortex_dir.element as vortex_element
import awebox.mdl.aero.induction_dir.tools_dir.geom as tools_geom
import awebox.tools.struct_operations as struct_op
import awebox.tools.vector_operations as vect_op
import awebox.mdl.wind as wind_module
import casadi.tools as cas
from awebox.logger.logger import Logger as awelogger
import awebox.tools.print_operations as print_op

import awebox.mdl.architecture as archi

def get_lists(options, variables_si, parameters, architecture, wind):

    far_wake_model = options['induction']['vortex_far_wake_model']
    kite_nodes = architecture.kite_nodes

    check_positive_vortex_wake_nodes(options)

    expected_number_filaments = expected_number_of_filaments(options, architecture)
    expected_number_directional_cylinders = expected_number_of_directional_cylinders(options, architecture)

    filament_list = vortex_element_list.ElementList(expected_number_filaments)
    tangential_cylinder_list = vortex_element_list.ElementList(expected_number_directional_cylinders)
    longitudinal_cylinder_list = vortex_element_list.ElementList(expected_number_directional_cylinders)

    for kite in kite_nodes:
        bound_list = get_bound_filament_from_kite_if_appropriate(options, variables_si, parameters, kite)
        filament_list.append(bound_list)

    if 'filament' in far_wake_model:
        for kite in kite_nodes:
            last_ring_fil_list = get_filament_list_from_kite(options, variables_si, parameters, wind, kite)
            filament_list.append(last_ring_fil_list)

    elif 'cylinder' in far_wake_model:
        for kite in kite_nodes:
            tan_list, long_list = get_cylinder_list_from_kite(options, variables_si, parameters, wind, kite, architecture)
            tangential_cylinder_list.append(tan_list)
            longitudinal_cylinder_list.append(long_list)

    else:
        message = 'unknown vortex far-wake model selected.'
        awelogger.logger.error(message)
        raise Exception(message)

    return filament_list, tangential_cylinder_list, longitudinal_cylinder_list


def check_positive_vortex_wake_nodes(options):
    wake_nodes = options['induction']['vortex_wake_nodes']
    if wake_nodes < 0:
        message = 'insufficient wake nodes for creating a filament list: wake_nodes = ' + str(wake_nodes)
        awelogger.logger.error(message)
        raise Exception(message)
    return None

def get_bound_filament_from_kite_if_appropriate(options, variables_si, parameters, kite):

    filament_list = vortex_element_list.ElementList()

    wake_nodes = options['induction']['vortex_wake_nodes']

    if wake_nodes == 1:
        last_tracked_wake_node = 0

        NE_wingtip = vortex_tools.get_NE_wingtip_name()
        PE_wingtip = vortex_tools.get_PE_wingtip_name()

        LENE = vortex_tools.get_wake_node_position_si(options, variables_si, kite, NE_wingtip, last_tracked_wake_node)
        LEPE = vortex_tools.get_wake_node_position_si(options, variables_si, kite, PE_wingtip, last_tracked_wake_node)

        strength = vortex_tools.get_ring_strength_si(variables_si, kite, last_tracked_wake_node)

        r_core = vortex_tools.get_r_core(options, parameters)

        dict_info_bound = {'x_start': LENE,
                           'x_end': LEPE,
                           'r_core': r_core,
                           'strength': strength
                           }
        fil_bound = vortex_element.Filament(dict_info_bound)
        filament_list.append(fil_bound)

    return filament_list


def get_filament_list_from_kite(options, variables_si, parameters, wind, kite):

    filament_list = vortex_element_list.ElementList()

    wake_nodes = options['induction']['vortex_wake_nodes']
    rings = wake_nodes

    last_tracked_wake_node = wake_nodes - 1
    ring = rings - 1 # remember: indexing starts at 0.

    far_convection_time = options['induction']['vortex_far_convection_time']
    far_wake_model = options['induction']['vortex_far_wake_model']

    NE_wingtip = vortex_tools.get_NE_wingtip_name()
    PE_wingtip = vortex_tools.get_PE_wingtip_name()

    LENE = vortex_tools.get_wake_node_position_si(options, variables_si, kite, NE_wingtip, last_tracked_wake_node)
    LEPE = vortex_tools.get_wake_node_position_si(options, variables_si, kite, PE_wingtip, last_tracked_wake_node)

    if far_wake_model == 'pathwise_filament':
        farwake_name = 'wu_farwake_' + str(kite) + '_'

        if isinstance(variables_si, cas.structure3.DMStruct):
            velocity_PE = variables_si['xl', farwake_name + PE_wingtip]
            velocity_NE = variables_si['xl', farwake_name + NE_wingtip]
        else:
            velocity_PE = variables_si['xl'][farwake_name + PE_wingtip]
            velocity_NE = variables_si['xl'][farwake_name + NE_wingtip]

    elif far_wake_model == 'freestream_filament':
        velocity_PE = wind.get_velocity(LEPE[[2]])
        velocity_NE = wind.get_velocity(LENE[[2]])

    TENE = LENE + far_convection_time * velocity_NE
    TEPE = LEPE + far_convection_time * velocity_PE

    strength = vortex_tools.get_ring_strength_si(variables_si, kite, ring)

    r_core = vortex_tools.get_r_core(options, parameters)

    dict_info_PE = {'x_start': LEPE,
                    'x_end': TEPE,
                    'r_core': r_core,
                    'strength': strength
                    }
    fil_PE = vortex_element.Filament(dict_info_PE)
    filament_list.append(fil_PE)

    dict_info_NE = {'x_start': TENE,
                    'x_end': LENE,
                    'r_core': r_core,
                    'strength': strength
                    }
    fil_NE = vortex_element.Filament(dict_info_NE)
    filament_list.append(fil_NE)

    return filament_list

def get_cylinder_list_from_kite(options, variables_si, parameters, wind, kite, architecture):

    tangential_cylinder_list = vortex_element_list.ElementList()
    longitudinal_cylinder_list = vortex_element_list.ElementList()

    wake_nodes = options['induction']['vortex_wake_nodes']
    last_tracked_wake_node = wake_nodes - 1

    NE_wingtip = vortex_tools.get_NE_wingtip_name()
    PE_wingtip = vortex_tools.get_PE_wingtip_name()

    x_kite_NE = vortex_tools.get_wake_node_position_si(options, variables_si, kite, NE_wingtip, last_tracked_wake_node)
    x_kite_PE = vortex_tools.get_wake_node_position_si(options, variables_si, kite, PE_wingtip, last_tracked_wake_node)

    wx_center_name = 'wx_center_' + str(kite)
    x_center = struct_op.get_variable_from_model_or_reconstruction(variables_si, 'xl', wx_center_name)

    strength = vortex_tools.get_ring_strength_si(variables_si, kite, last_tracked_wake_node)

    print_op.warn_about_temporary_funcationality_removal(location='far_wake.cylinder_list')
    strength_tan = strength
    strength_long = 0

    l_hat = wind.get_wind_direction()
    epsilon = vortex_tools.get_epsilon(options, parameters)

    tan_NE_dict = {'x_center': x_center,
                   'x_kite': x_kite_NE,
                   'l_hat': l_hat,
                   'epsilon': epsilon,
                   'strength': -1 * strength_tan
                   }
    tan_NE = vortex_element.TangentialCylinder(tan_NE_dict)
    tangential_cylinder_list.append(tan_NE)

    long_NE_dict = {'x_center': x_center,
                   'x_kite': x_kite_NE,
                   'l_hat': l_hat,
                   'epsilon': epsilon,
                   'strength': -1 * strength_long
                   }
    long_NE = vortex_element.LongitudinalCylinder(long_NE_dict)
    longitudinal_cylinder_list.append(long_NE)

    tan_PE_dict = {'x_center': x_center,
                   'x_kite': x_kite_PE,
                   'l_hat': l_hat,
                   'epsilon': epsilon,
                   'strength': strength_tan
                   }
    tan_PE = vortex_element.TangentialCylinder(tan_PE_dict)
    tangential_cylinder_list.append(tan_PE)

    long_PE_dict = {'x_center': x_center,
                   'x_kite': x_kite_PE,
                   'l_hat': l_hat,
                   'epsilon': epsilon,
                   'strength': strength_long
                   }
    long_PE = vortex_element.LongitudinalCylinder(long_PE_dict)
    longitudinal_cylinder_list.append(long_PE)

    return tangential_cylinder_list, longitudinal_cylinder_list

def expected_number_of_filaments(options, architecture):

    far_wake_model = options['induction']['vortex_far_wake_model']
    wake_nodes = options['induction']['vortex_wake_nodes']
    number_kites = architecture.number_of_kites

    if 'filament' in far_wake_model:
        use = 1
    else:
        use = 0

    filaments = use * 2 * number_kites

    # need to add bound filaments
    if wake_nodes == 1:
        filaments += number_kites

    return filaments

def expected_number_of_directional_cylinders(options, architecture):

    far_wake_model = options['induction']['vortex_far_wake_model']
    number_kites = architecture.number_of_kites

    if 'cylinder' in far_wake_model:
        use = 1
    else:
        use = 0

    cylinders = use * 2 * number_kites

    return cylinders

#

# def test(far_wake_model = 'freestream_filament'):
#
#     architecture = archi.Architecture({1:0})
#
#     options = {}
#
#     options['wind'] = {}
#     options['wind']['u_ref'] = 1.
#     options['wind']['model'] = 'uniform'
#     options['wind']['z_ref'] = -999.
#     options['wind']['log_wind'] = {'z0_air': -999}
#     options['wind']['power_wind'] = {'exp_ref': -999}
#
#     options['induction'] = {}
#     options['induction']['vortex_wake_nodes'] = 1
#     options['induction']['vortex_far_convection_time'] = 1.
#     options['induction']['vortex_far_wake_model'] = far_wake_model
#     options['induction']['vortex_u_ref'] = options['wind']['u_ref']
#     options['induction']['vortex_position_scale'] = 1.
#     options['induction']['vortex_representation'] = 'state'
#
#     wind = wind_module.Wind(options['wind'], options['wind'])
#     kite = architecture.kite_nodes[0]
#
#     xd_struct = cas.struct([
#         cas.entry("wx_" + str(kite) + "_ext_0", shape=(3, 1)),
#         cas.entry("wx_" + str(kite) + "_ext_1", shape=(3, 1)),
#         cas.entry("wx_" + str(kite) + "_int_0", shape=(3, 1)),
#         cas.entry("wx_" + str(kite) + "_int_1", shape=(3, 1))
#     ])
#     xl_struct = cas.struct([
#         cas.entry("wg_" + str(kite) + "_0"),
#         cas.entry('wu_farwake_' + str(kite) + '_int', shape=(3, 1)),
#         cas.entry('wu_farwake_' + str(kite) + '_ext', shape=(3, 1))
#     ])
#     var_struct = cas.struct_symSX([
#         cas.entry('xd', struct=xd_struct),
#         cas.entry('xl', struct=xl_struct)
#     ])
#
#     variables_si = var_struct(0.)
#     variables_si['xd', 'wx_' + str(kite) + '_ext_0'] = 0.5 * vect_op.yhat_np()
#     variables_si['xd', 'wx_' + str(kite) + '_int_0'] = -0.5 * vect_op.yhat_np()
#     variables_si['xl', 'wg_' + str(kite) + '_0'] = 1.
#     variables_si['xl', 'wu_farwake_' + str(kite) + '_ext'] = vect_op.xhat_np()
#     variables_si['xl', 'wu_farwake_' + str(kite) + '_int'] = vect_op.xhat_np()
#
#     test_list = get_list(options, variables_si, architecture, wind)
#
#     filaments = test_list.shape[1]
#
#     filament_count_test = filaments - 3
#     if not (filament_count_test == 0):
#         message = 'filament list does not work as expected. difference in number of filaments in test_list = ' + str(filament_count_test)
#         awelogger.logger.error(message)
#         raise Exception(message)
#
#     LE_expected = cas.DM(np.array([0., -0.5, 0., 0., 0.5, 0., 1.]))
#     PE_expected = cas.DM(np.array([0., 0.5, 0., 1., 0.5, 0., 1.]))
#     TE_expected = cas.DM(np.array([1., -0.5, 0., 0., -0.5, 0., 1.]))
#
#     expected_filaments = {'leading edge': LE_expected,
#                           'positive edge': PE_expected,
#                           'trailing edge': TE_expected}
#
#
#     for type in expected_filaments.keys():
#         expected_filament = expected_filaments[type]
#         expected_in_list = expected_filament_in_list(test_list, expected_filament)
#         if not expected_in_list:
#             message = 'filament list does not work as expected. ' + type + \
#                       ' test filament not in test_list.'
#             awelogger.logger.error(message)
#
#             with np.printoptions(precision=3, suppress=True):
#                 print('test_list:')
#                 print(np.array(test_list))
#
#             raise Exception(message)
#
#
#     NE_not_expected = cas.DM(np.array([1., -0.5, 0., 1., 0.5, 0., -1.]))
#     not_expected_filaments = {'negative edge': NE_not_expected}
#
#     for type in not_expected_filaments.keys():
#         not_expected_filament = not_expected_filaments[type]
#         is_reasonable = not (expected_filament_in_list(test_list, not_expected_filament))
#         if not is_reasonable:
#             message = 'filament list does not work as expected. ' + type + \
#                       ' test filament in test_list.'
#             awelogger.logger.error(message)
#
#             with np.printoptions(precision=3, suppress=True):
#                 print('test_list:')
#                 print(np.array(test_list))
#
#             raise Exception(message)
#
#
#     return test_list
#
# def expected_filament_in_list(test_list, expected_filament):
#
#     filaments = test_list.shape[1]
#
#     thresh = 1.e-8
#
#     for filament in range(filaments):
#         local = test_list[:, filament]
#         comparison = vect_op.norm(local - expected_filament)
#         if comparison < thresh:
#             return True
#
#     return False