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
######################################
# This file stores all quality tests
# Author: Thilo Bronnenmeyer, Kiteswarms, 2018
# edit: Rachel Leuthold, ALU-FR, 2019-20

######################################

import numpy as np
from awebox.logger.logger import Logger as awelogger
import casadi.tools as cas
import awebox.tools.struct_operations as struct_op
import awebox.tools.print_operations as print_op
import awebox.tools.vector_operations as vect_op

def test_opti_success(trial, test_param_dict, results):
    """
    Test whether optimization was successful
    :return: results
    """

    results['solve_succeeded'] = trial.optimization.solve_succeeded

    return results

def test_numerics(trial, test_param_dict, results):
    """
    Test whether optimal parameters are chosen in a reasonable way
    :return: results
    """

    # test if t_f makes sense
    t_f_min = test_param_dict['t_f_min']
    t_f = np.array(trial.optimization.V_final_si['theta', 't_f']).sum()  # compute sum in case of phase fix
    if t_f < t_f_min:
        awelogger.logger.warning('Final time < ' + str(t_f_min) + ' s for trial ' + trial.name)
        results['t_f_min'] = False
    else:
        results['t_f_min'] = True

    # test if t_f/k_k ratio makes sense
    max_control_interval = test_param_dict['max_control_interval']
    n_k = trial.nlp.n_k
    if t_f / float(n_k) > max_control_interval:
        awelogger.logger.warning('t_f/n_k ratio is > ' + str(max_control_interval) + ' s for trial ' + trial.name)
        results['max_control_interval'] = False
    else:
        results['max_control_interval'] = True

    return results

def test_invariants(trial, test_param_dict, results, input_values):
    """
    Test whether invariants reasonably sized
    :return: test results
    """

    # set test parameters from dictionary
    c_max = test_param_dict['c_max']
    dc_max = test_param_dict['dc_max']
    r_max = test_param_dict['r_max']

    # get architecture
    architecture = trial.model.architecture
    number_of_nodes = architecture.number_of_nodes
    parent_map = architecture.parent_map

    # DOF
    DOF6 = trial.options['user_options']['system_model']['kite_dof'] == 6

    # model outputs
    outputs = trial.model.outputs

    # loop over nodes
    for node in range(1,number_of_nodes):
        parent = parent_map[node]

        c_search = input_values['outputs']['invariants']['c' + str(node) + str(parent)][0]
        dc_search = input_values['outputs']['invariants']['dc' + str(node) + str(parent)][0]

        c_sol = np.max(np.abs(np.array(c_search)))
        dc_sol = np.max(np.abs(np.array(dc_search)))

        if DOF6 and node in architecture.kite_nodes:
            r_list = []
            for jdx in range(9):
                r_search = input_values['outputs']['invariants']['orthonormality' + str(node) + str(parent)][jdx]
                r_list.append(np.max(np.abs(np.array(r_search))))

            r_sol = max(r_list)

            # test whether invariants are small enough
            results = include_result_of_allowed_invariant_test(results, 'c', node, parent, c_sol, c_max, trial.name)
            results = include_result_of_allowed_invariant_test(results, 'dc', node, parent, dc_sol, dc_max, trial.name)

            if DOF6 and node in architecture.kite_nodes:
                results = include_result_of_allowed_invariant_test(results, 'r', node, parent, r_sol, r_max, trial.name)

    return results

def include_result_of_allowed_invariant_test(results, name, node, parent, sol_value, max_value, trial_name):
    combined_name = name + str(node) + str(parent)
    if sol_value > max_value:
        message = 'Invariant ' + combined_name + ' has value ' + str(sol_value) + ' > ' + str(max_value) + ' of V for trial ' + trial_name
        awelogger.logger.warning(message)
        results[combined_name] = False
    else:
        results[combined_name] = True

    return results


def test_node_altitude(trial, test_param_dict, results):
    """
    Test whether variables are of reasonable size and have correct signs
    :return: test results
    """

    # get discretization
    discretization = trial.options['nlp']['discretization']

    # get trial solution
    V_final_si = trial.optimization.V_final_si

    # extract system architecture
    architecture = trial.model.architecture
    number_of_nodes = architecture.number_of_nodes
    parent_map = architecture.parent_map

    results['min_node_height'] = True
    z_min = test_param_dict['z_min']

    # test if height of all nodes is positive
    for node in range(1, number_of_nodes):

        parent = parent_map[node]
        node_str = 'q' + str(node) + str(parent)
        error_message = 'Node ' + node_str + ' has negative height for trial ' + trial.name

        heights_x = np.array(V_final_si['x', :, node_str, 2])
        if np.min(heights_x) < z_min:
            results['min_node_height'] = False

        if discretization == 'direct_collocation':
            heights_coll_var = np.array(V_final_si['coll_var', :, :, 'x', node_str, 2])
            if np.min(heights_coll_var) < z_min:
                results['min_node_height'] = False

        if not results['min_node_height']:
            print_op.log_and_raise_error(error_message)

    return results

def test_power_balance(trial, test_param_dict, results, input_values):
    """Test whether conservation of energy holds at all nodes and for the entire system.
    this test is only going to be meaningful, if there are no fictitious forces.
    :return: test results
    """

    contains_no_fictitious_forces = (trial.optimization.V_opt['phi', 'gamma'] < 1.e-10)
    if contains_no_fictitious_forces:

        # extract info
        tgrid = input_values['time_grids']['ip']
        power_balance = input_values['outputs']['power_balance']

        check_energy_summation = test_param_dict['check_energy_summation']
        if check_energy_summation:
            results = summation_check_on_potential_and_kinetic_power(trial, test_param_dict['energy_summation_thresh'], results, input_values)

        balance = {}
        max_abs_system_power = 1.e-15
        system_net_power_timeseries = np.zeros(tgrid.shape)

        nodes_above_ground = range(1, trial.model.architecture.number_of_nodes)
        for node in nodes_above_ground:

            originating_power_timeseries = np.zeros(tgrid.shape)
            max_abs_node_power = 1.e-15  # preclude any div-by-zero errors

            if node in trial.model.architecture.children_map.keys():
                list_of_keys_where_power_is_transferred = ['P_tether' + str(child) for child in trial.model.architecture.children_map[node]]
            else:
                list_of_keys_where_power_is_transferred = []

            list_of_keys_where_power_arrives_at_node = []
            for keyname in list(power_balance.keys()):
                base_name, numeric_name = struct_op.split_name_and_node_identifier(keyname)
                if str(numeric_name) == str(node) or str(numeric_name) == trial.model.architecture.node_label(node):
                    list_of_keys_where_power_arrives_at_node += [keyname]

            for keyname in list_of_keys_where_power_arrives_at_node:
                timeseries = power_balance[keyname][0]
                originating_power_timeseries += timeseries
                max_abs_node_power = np.max([np.max(np.abs(timeseries)), max_abs_node_power])

            for keyname in list_of_keys_where_power_is_transferred:
                timeseries = power_balance[keyname][0]
                originating_power_timeseries -= timeseries
                max_abs_node_power = np.max([np.max(np.abs(timeseries)), max_abs_node_power])

            scaled_norm_net_power = np.linalg.norm(originating_power_timeseries) / max_abs_node_power
            balance[node] = scaled_norm_net_power

            # add node net power into system net power
            max_abs_system_power = np.max([max_abs_node_power, max_abs_system_power])
            system_net_power_timeseries += originating_power_timeseries

        if 'arm_angle' in input_values['x']:
            balance, system_net_power_timeseries, max_abs_system_power = \
                power_balance_arm(tgrid, power_balance, balance, system_net_power_timeseries, max_abs_system_power)

        scaled_norm_system_net_power = np.linalg.norm(system_net_power_timeseries) / max_abs_system_power
        balance['total'] = scaled_norm_system_net_power

        for node in balance.keys():
            if balance[node] > test_param_dict['power_balance_thresh']:
                message = 'energy balance of node ' + str(node) + ' of trial ' + trial.name + ' not consistent. ' \
                          + str(balance[node]) + ' > ' + str(test_param_dict['power_balance_thresh'])
                awelogger.logger.warning(message)
                results['energy_balance_' + str(node)] = False
            else:
                results['energy_balance_' + str(node)] = True

    return results

def power_balance_arm(tgrid, power_balance, balance, system_net_power_timeseries, max_abs_system_power):
    # Test conservation of energy for the arm
    originating_power_timeseries = np.zeros(tgrid.shape)
    max_abs_arm_power = 1.e-15
    for keyname in ['P_kin_arm_rot', 'P_tether_arm', 'P_gen_arm']:
        timeseries = power_balance[keyname][0]
        originating_power_timeseries += timeseries
        max_abs_arm_power = np.max([np.max(np.abs(timeseries)), max_abs_arm_power])

    scaled_norm_net_power = np.linalg.norm(originating_power_timeseries) / max_abs_arm_power
    balance['arm'] = scaled_norm_net_power

    # Test that no energy is lost in the tether
    originating_power_timeseries = np.zeros(tgrid.shape)
    max_abs_tether_power = 1.e-15
    for keyname in ['P_tether1', 'P_tether_arm']:
        timeseries = power_balance[keyname][0]
        originating_power_timeseries += timeseries
        max_abs_tether_power = np.max([np.max(np.abs(timeseries)), max_abs_tether_power])

    scaled_norm_net_power = np.linalg.norm(originating_power_timeseries) / max_abs_tether_power
    balance['arm_main_tether'] = scaled_norm_net_power

    max_abs_system_power = np.max([max_abs_arm_power, max_abs_tether_power, max_abs_system_power])
    system_net_power_timeseries += originating_power_timeseries

    return balance, system_net_power_timeseries, max_abs_system_power

def summation_check_on_potential_and_kinetic_power(trial, thresh, results, input_values):

    abbreviated_energy_names = ['pot', 'kin']

    kin_comp = np.array(input_values['outputs']['power_balance_comparison']['kinetic'][0])
    pot_comp = np.array(input_values['outputs']['power_balance_comparison']['potential'][0])
    comp_timeseries = {'pot': pot_comp, 'kin': kin_comp}

    # extract info
    tgrid = input_values['time_grids']['quality']
    power_balance = input_values['outputs']['power_balance']

    for abbreviated_name in abbreviated_energy_names:
        sum_timeseries = np.zeros(tgrid.shape)
        for energy_name in list(power_balance.keys()):
            if abbreviated_name in energy_name:
                timeseries = power_balance[energy_name][0]
                sum_timeseries += timeseries

        difference = cas.DM(sum_timeseries - comp_timeseries[abbreviated_name])

        error = float(cas.mtimes(difference.T, difference))

        if error > thresh:
            awelogger.logger.warning('some of the power based on ' + abbreviated_name + '. energy must have gotten lost, since a summation check fails. Considering trial ' + trial.name +  ' with ' + str(error) + ' > ' + str(thresh))
            results['power_summation_check_' + abbreviated_name] = False
        else:
            results['power_summation_check_' + abbreviated_name] = True

    return results

def power_balance_key_belongs_to_node(keyname, node):
    keyname_includes_nodenumber = (keyname[-len(str(node)):] == str(node))
    keyname_is_nonnumeric_before_nodenumber = not (keyname[-len(str(node)) - 1].isnumeric())
    key_belongs_to_node = keyname_includes_nodenumber and keyname_is_nonnumeric_before_nodenumber
    return key_belongs_to_node


def test_tracked_vortex_periods(trial, test_param_dict, results, input_values, global_input_values):

    if 'vortex' in input_values['outputs']:

        vortex_truncation_error_thresh = test_param_dict['vortex_truncation_error_thresh']

        local_max = []
        for keyname in input_values['outputs']['vortex'].keys():
            if 'est_truncation_error' in keyname:
                local_max += [np.max(np.array(input_values['outputs']['vortex'][keyname]))]
        max_est_truncation_error = np.max(np.array(local_max))
        if max_est_truncation_error > vortex_truncation_error_thresh:
            message = 'Vortex model estimates a large truncation error' \
                      + str(max_est_truncation_error) + ' > ' + str(vortex_truncation_error_thresh) \
                      + '. We recommend increasing the number of wake nodes.'
            awelogger.logger.warning(message)
            results['vortex_truncation_error'] = False
        else:
            results['vortex_truncation_error'] = True

    return results


def generate_test_param_dict(options):
    """
    Set parameters relevant for testing
    :return: dictionary with test parameters
    """

    test_param_dict = {}
    test_param_dict['c_max'] = options['test_param']['c_max']
    test_param_dict['dc_max'] = options['test_param']['dc_max']
    # test_param_dict['ddc_max'] = options['test_param']['ddc_max']
    test_param_dict['z_min'] = options['test_param']['z_min']
    test_param_dict['r_max'] = options['test_param']['r_max']
    test_param_dict['max_loyd_factor'] = options['test_param']['max_loyd_factor']
    test_param_dict['max_power_harvesting_factor'] = options['test_param']['max_power_harvesting_factor']
    test_param_dict['max_tension'] = options['test_param']['max_tension']
    test_param_dict['max_velocity'] = options['test_param']['max_velocity']
    test_param_dict['t_f_min'] = options['test_param']['t_f_min']
    test_param_dict['max_control_interval'] = options['test_param']['max_control_interval']
    test_param_dict['power_balance_thresh'] = options['test_param']['power_balance_thresh']
    test_param_dict['vortex_truncation_error_thresh'] = options['test_param']['vortex_truncation_error_thresh']
    test_param_dict['check_energy_summation'] = options['test_param']['check_energy_summation']
    test_param_dict['energy_summation_thresh'] = options['test_param']['energy_summation_thresh']
    test_param_dict['non_power_fraction_of_objective_thresh'] = options['test_param']['non_power_fraction_of_objective_thresh']

    return test_param_dict
