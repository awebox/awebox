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
import casadi.tools as cas

import awebox.mdl.aero.induction_dir.vortex_dir.tools as tools
import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.element_list as vortex_element_list
import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.element as vortex_element
import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.wake as vortex_wake

import awebox.mdl.architecture as archi

import awebox.tools.vector_operations as vect_op
import awebox.tools.print_operations as print_op

#
# def get_list(options, variables_si, parameters, architecture):
#
#     expected_number_filaments = expected_number_of_filaments(options, architecture)
#     filament_list = vortex_element_list.ElementList(expected_number_filaments)
#
#     kite_nodes = architecture.kite_nodes
#     for kite in kite_nodes:
#         kite_fil = get_bound_filament_by_kite(options, variables_si, parameters, kite)
#         filament_list.append(kite_fil)
#
#     filament_list.confirm_list_has_expected_dimensions()
#
#     return filament_list
#
# def get_bound_filament_by_kite(options, variables_si, parameters, kite):
#
#     ring = 0
#     wake_node = ring
#
#     NE_wingtip = tools.get_NE_wingtip_name()
#     PE_wingtip = tools.get_PE_wingtip_name()
#
#     LENE = tools.get_wake_node_position_si(options, variables_si, kite, NE_wingtip, wake_node)
#     LEPE = tools.get_wake_node_position_si(options, variables_si, kite, PE_wingtip, wake_node)
#
#     strength = tools.get_ring_strength_si(variables_si, kite, ring)
#     strength_prev = cas.DM.zeros((1, 1))
#
#     r_core = tools.get_r_core(options, parameters)
#
#     dict_info_LE = {'x_start': LENE,
#                     'x_end': LEPE,
#                     'r_core': r_core,
#                     'strength': strength - strength_prev
#                     }
#     fil_LE = vortex_element.Filament(dict_info_LE)
#
#     return fil_LE
#
# def expected_number_of_filaments(options, architecture):
#     number_kites = architecture.number_of_kites
#     filaments = number_kites
#
#     return filaments
#
# def get_test_filament_list():
#
#     # basic test assumptions
#     b_ref = 1.
#     q_kite = cas.DM.zeros((3, 1))
#     dcm_kite = cas.DM.eye(3)
#
#     architecture = archi.Architecture({1: 0})
#     kite = architecture.kite_nodes[0]
#
#     # define options
#     options = {}
#
#     options['wind'] = {}
#     options['wind']['u_ref'] = 1.
#     options['wind']['model'] = 'uniform'
#     options['wind']['z_ref'] = -999.
#     options['wind']['log_wind'] = {'z0_air': -999}
#     options['wind']['power_wind'] = {'exp_ref': -999}
#
#     options = {'induction':{}, 'aero':{'vortex': {}}}
#     options['induction']['vortex_wake_nodes'] = 1
#     options['induction']['vortex_representation'] = 'alg'
#     options['aero']['vortex']['core_to_chord_ratio'] = 0.
#
#     # build the variables
#     xd_struct = cas.struct([
#         cas.entry('r10', shape=(9, 1)),
#         cas.entry('q10', shape=(3, 1))
#     ])
#     xl_struct = cas.struct([
#         cas.entry("wx_" + str(kite) + "_ext_0", shape=(3, 1)),
#         cas.entry("wx_" + str(kite) + "_int_0", shape=(3, 1)),
#         cas.entry("wg_" + str(kite) + "_0")
#     ])
#     var_struct = cas.struct_symSX([
#         cas.entry('xd', struct=xd_struct),
#         cas.entry('xl', struct=xl_struct)
#     ])
#
#     variables_si = var_struct(0.)
#     variables_si['xd', 'q10'] = q_kite
#     variables_si['xd', 'r10'] = cas.reshape(dcm_kite, (9, 1))
#     variables_si['xl', 'wg_' + str(kite) + '_0'] = 1.
#     variables_si['xl', 'wx_' + str(kite) + '_ext_0'] = 0.5 * b_ref * vect_op.yhat_dm()
#     variables_si['xl', 'wx_' + str(kite) + '_int_0'] = -0.5 * b_ref * vect_op.yhat_dm()
#
#     # build parameters
#     parameters = {}
#
#     # make filament list
#     filament_list = get_list(options, variables_si, parameters, architecture)
#
#     return filament_list
#
# def test():
#     filament_list = get_test_filament_list()
#     test_expected_number_of_filaments(filament_list)
#     return None
#
# def test_expected_number_of_filaments(filament_list):
#     test_passes = (filament_list.number_of_elements == 1)
#     message = 'incorrect number of filaments in bound_filament test'
#     print_op.print_test_outcome(test_passes, message)
#     return None