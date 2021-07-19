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
import awebox.mdl.aero.induction_dir.vortex_dir.tools as tools
import awebox.tools.struct_operations as struct_op
import awebox.tools.vector_operations as vect_op
import awebox.mdl.wind as wind_module
import casadi.tools as cas
from awebox.logger.logger import Logger as awelogger
import awebox.tools.print_operations as print_op

import awebox.mdl.architecture as archi

def get_list(options, variables_si, architecture, wind):

    kite_nodes = architecture.kite_nodes

    filament_list = []
    for kite in kite_nodes:
        kite_fil_list = get_list_by_kite(options, variables_si, architecture, wind, kite)
        filament_list = cas.horzcat(filament_list, kite_fil_list)

    return filament_list

def get_list_by_kite(options, variables_si, architecture, wind, kite):

    wake_nodes = options['induction']['vortex_wake_nodes']
    tracked_rings = wake_nodes - 1

    filament_list = []
    for ring in range(tracked_rings):
        ring_fil_list = get_list_by_ring(options, variables_si, kite, ring)
        filament_list = cas.horzcat(filament_list, ring_fil_list)

    far_wake_model = options['induction']['vortex_far_wake_model']
    if 'filament' in far_wake_model:
        last_ring_fil_list = get_far_wake_list_from_kite(options, variables_si, wind, kite)
        filament_list = cas.horzcat(filament_list, last_ring_fil_list)

    return filament_list

def get_far_wake_list(options, variables_si, architecture, wind):

    far_wake_model = options['induction']['vortex_far_wake_model']
    kite_nodes = architecture.kite_nodes

    filament_list = []
    if 'filament' in far_wake_model:
        for kite in kite_nodes:
            last_ring_fil_list = get_far_wake_list_from_kite(options, variables_si, wind, kite)
            filament_list = cas.horzcat(filament_list, last_ring_fil_list)

    return filament_list

def get_list_by_ring(options, variables_si, kite, ring):

    wake_node = ring

    NE_wingtip = get_NE_wingtip_name()
    PE_wingtip = get_PE_wingtip_name()

    TENE = tools.get_wake_node_position_si(options, variables_si, kite, NE_wingtip, wake_node + 1)
    LENE = tools.get_wake_node_position_si(options, variables_si, kite, NE_wingtip, wake_node)
    LEPE = tools.get_wake_node_position_si(options, variables_si, kite, PE_wingtip, wake_node)
    TEPE = tools.get_wake_node_position_si(options, variables_si, kite, PE_wingtip, wake_node + 1)

    strength = tools.get_ring_strength_si(variables_si, kite, ring)

    if ring == 0:
        strength_prev = cas.DM.zeros((1, 1))
    else:
        strength_prev = tools.get_ring_strength_si(variables_si, kite, ring - 1)

    filament_list = make_horseshoe_list(LENE, LEPE, TEPE, TENE, strength, strength_prev)
    return filament_list

def make_horseshoe_list(LENE, LEPE, TEPE, TENE, strength, strength_prev):
    PE_filament = cas.vertcat(LEPE, TEPE, strength)
    LE_filament = cas.vertcat(LENE, LEPE, strength - strength_prev)
    NE_filament = cas.vertcat(TENE, LENE, strength)
    filament_list = cas.horzcat(PE_filament, LE_filament, NE_filament)
    return filament_list


def get_far_wake_list_from_kite(options, variables_si, wind, kite):

    wake_nodes = options['induction']['vortex_wake_nodes']
    rings = wake_nodes

    if rings < 0:
        message = 'insufficient wake nodes for creating a filament list: wake_nodes = ' + str(wake_nodes)
        awelogger.logger.error(message)
        raise Exception(message)

    far_wake_model = options['induction']['vortex_far_wake_model']
    if 'filament' not in far_wake_model:
        message = 'far-wake filament list mistakenly requested, when desired far_wake_model is: ' + far_wake_model
        awelogger.logger.error(message)
        return Exception(message)

    last_tracked_wake_node = wake_nodes - 1
    ring = rings - 1 # remember: indexing starts at 0.

    far_convection_time = options['induction']['vortex_far_convection_time']
    far_wake_model = options['induction']['vortex_far_wake_model']

    NE_wingtip = get_NE_wingtip_name()
    PE_wingtip = get_PE_wingtip_name()

    LENE = tools.get_wake_node_position_si(options, variables_si, kite, NE_wingtip, last_tracked_wake_node)
    LEPE = tools.get_wake_node_position_si(options, variables_si, kite, PE_wingtip, last_tracked_wake_node)

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

    strength = tools.get_ring_strength_si(variables_si, kite, ring)

    if ring == 0:
        strength_prev = cas.DM.zeros((1, 1))
    else:
        strength_prev = tools.get_ring_strength_si(variables_si, kite, ring - 1)

    filament_list = make_horseshoe_list(LENE, LEPE, TEPE, TENE, strength, strength_prev)
    return filament_list

def append_normal_to_list(filament_list, n_hat):

    width = filament_list.shape[1]

    appended_list = cas.vertcat(
        filament_list,
        cas.DM.ones((1, width)) * n_hat[0],
        cas.DM.ones((1, width)) * n_hat[1],
        cas.DM.ones((1, width)) * n_hat[2]
    )

    return appended_list

def append_observer_to_list(filament_list, x_obs):
    width = filament_list.shape[1]

    appended_list = cas.vertcat(
        cas.DM.ones((1, width)) * x_obs[0],
        cas.DM.ones((1, width)) * x_obs[1],
        cas.DM.ones((1, width)) * x_obs[2],
        filament_list
    )

    return appended_list

def columnize(filament_list):
    dims = filament_list.shape
    columnized_list = cas.reshape(filament_list, (dims[0] * dims[1], 1))
    return columnized_list

def decolumnize(options, architecture, columnized_list):
    entries = columnized_list.shape[0]
    filaments = expected_number_of_filaments(options, architecture)
    arguments = int(float(entries) / float(filaments))

    filament_list = cas.reshape(columnized_list, (arguments, filaments))

    return filament_list

def expected_number_of_filaments(options, architecture):
    far_wake_model = options['induction']['vortex_far_wake_model']
    wake_nodes = options['induction']['vortex_wake_nodes']
    number_kites = architecture.number_of_kites

    if 'filament' in far_wake_model:
        rings = wake_nodes
    else:
        rings = wake_nodes - 1

    filaments = 3 * (rings) * number_kites

    return filaments

def gatekeeper_list_has_expected_number_of_filaments(options, architecture, columnized_filament_list):
    filament_list = decolumnize(options, architecture, columnized_filament_list)
    number_of_filaments = filament_list.shape[1]

    expected = expected_number_of_filaments(options, architecture)
    if int(number_of_filaments) != int(expected):
        message = 'construction of vortex induction residual finds a number of filaments (' + \
                  str(number_of_filaments) + ') that is not the same as the expected ' \
                  'number of filaments (' + str(expected) + ')'
        awelogger.logger.error(message)
        raise Exception(message)

    return True



def get_PE_wingtip_name():
    return 'ext'

def get_NE_wingtip_name():
    return 'int'


def test(far_wake_model = 'freestream_filament'):

    architecture = archi.Architecture({1:0})

    options = {}

    options['wind'] = {}
    options['wind']['u_ref'] = 1.
    options['wind']['model'] = 'uniform'
    options['wind']['z_ref'] = -999.
    options['wind']['log_wind'] = {'z0_air': -999}
    options['wind']['power_wind'] = {'exp_ref': -999}

    options['induction'] = {}
    options['induction']['vortex_wake_nodes'] = 1
    options['induction']['vortex_far_convection_time'] = 1.
    options['induction']['vortex_far_wake_model'] = far_wake_model
    options['induction']['vortex_u_ref'] = options['wind']['u_ref']
    options['induction']['vortex_position_scale'] = 1.
    options['induction']['vortex_representation'] = 'state'

    wind = wind_module.Wind(options['wind'], options['wind'])
    kite = architecture.kite_nodes[0]

    xd_struct = cas.struct([
        cas.entry("wx_" + str(kite) + "_ext_0", shape=(3, 1)),
        cas.entry("wx_" + str(kite) + "_ext_1", shape=(3, 1)),
        cas.entry("wx_" + str(kite) + "_int_0", shape=(3, 1)),
        cas.entry("wx_" + str(kite) + "_int_1", shape=(3, 1))
    ])
    xl_struct = cas.struct([
        cas.entry("wg_" + str(kite) + "_0"),
        cas.entry('wu_farwake_' + str(kite) + '_int', shape=(3, 1)),
        cas.entry('wu_farwake_' + str(kite) + '_ext', shape=(3, 1))
    ])
    var_struct = cas.struct_symSX([
        cas.entry('xd', struct=xd_struct),
        cas.entry('xl', struct=xl_struct)
    ])

    variables_si = var_struct(0.)
    variables_si['xd', 'wx_' + str(kite) + '_ext_0'] = 0.5 * vect_op.yhat_np()
    variables_si['xd', 'wx_' + str(kite) + '_int_0'] = -0.5 * vect_op.yhat_np()
    variables_si['xl', 'wg_' + str(kite) + '_0'] = 1.
    variables_si['xl', 'wu_farwake_' + str(kite) + '_ext'] = vect_op.xhat_np()
    variables_si['xl', 'wu_farwake_' + str(kite) + '_int'] = vect_op.xhat_np()

    test_list = get_list(options, variables_si, architecture, wind)

    filaments = test_list.shape[1]

    filament_count_test = filaments - 3
    if not (filament_count_test == 0):
        message = 'filament list does not work as expected. difference in number of filaments in test_list = ' + str(filament_count_test)
        awelogger.logger.error(message)
        raise Exception(message)

    LE_expected = cas.DM(np.array([0., -0.5, 0., 0., 0.5, 0., 1.]))
    PE_expected = cas.DM(np.array([0., 0.5, 0., 1., 0.5, 0., 1.]))
    TE_expected = cas.DM(np.array([1., -0.5, 0., 0., -0.5, 0., 1.]))

    expected_filaments = {'leading edge': LE_expected,
                          'positive edge': PE_expected,
                          'trailing edge': TE_expected}


    for type in expected_filaments.keys():
        expected_filament = expected_filaments[type]
        expected_in_list = expected_filament_in_list(test_list, expected_filament)
        if not expected_in_list:
            message = 'filament list does not work as expected. ' + type + \
                      ' test filament not in test_list.'
            awelogger.logger.error(message)

            with np.printoptions(precision=3, suppress=True):
                print('test_list:')
                print(np.array(test_list))

            raise Exception(message)


    NE_not_expected = cas.DM(np.array([1., -0.5, 0., 1., 0.5, 0., -1.]))
    not_expected_filaments = {'negative edge': NE_not_expected}

    for type in not_expected_filaments.keys():
        not_expected_filament = not_expected_filaments[type]
        is_reasonable = not (expected_filament_in_list(test_list, not_expected_filament))
        if not is_reasonable:
            message = 'filament list does not work as expected. ' + type + \
                      ' test filament in test_list.'
            awelogger.logger.error(message)

            with np.printoptions(precision=3, suppress=True):
                print('test_list:')
                print(np.array(test_list))

            raise Exception(message)


    return test_list

def expected_filament_in_list(test_list, expected_filament):

    filaments = test_list.shape[1]

    thresh = 1.e-8

    for filament in range(filaments):
        local = test_list[:, filament]
        comparison = vect_op.norm(local - expected_filament)
        if comparison < thresh:
            return True

    return False