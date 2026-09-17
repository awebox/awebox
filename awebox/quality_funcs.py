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
from awebox.mdl.architecture import Architecture
from awebox.mdl.model import Model


def test_opti_success(trial, test_param_dict, results, printable_summary_dict={}, test_units_dict=None):
    """
    Test whether optimization was successful
    :return: results
    """

    results['solve_succeeded'] = trial.optimization.solve_succeeded

    printable_summary_dict['solver reports successful solution'] = {'found': results['solve_succeeded'], 'threshhold': True, 'units': None,
                                                 'test passed': results['solve_succeeded']}
    return results, printable_summary_dict

def add_test_to_printable_summary(print_name, name, found, printable_summary_dict, results, test_param_dict, test_units_dict):
    printable_summary_dict[print_name] = {'found': found,
                                    'threshhold': test_param_dict[name]}
    if test_units_dict is not None:
        printable_summary_dict[print_name]['units'] = test_units_dict[name]

    printable_summary_dict[print_name]['test passed'] = results[name]
    return printable_summary_dict

def test_numerics(trial, test_param_dict, results, printable_summary_dict, test_units_dict=None):
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
    printable_summary_dict = add_test_to_printable_summary('optimal period above recommended minimum', 't_f_min', t_f, printable_summary_dict, results, test_param_dict, test_units_dict)

    # test if t_f/k_k ratio makes sense
    max_control_interval = test_param_dict['max_control_interval']
    n_k = trial.nlp.n_k
    if t_f / float(n_k) > max_control_interval:
        awelogger.logger.warning('t_f/n_k ratio is > ' + str(max_control_interval) + ' s for trial ' + trial.name)
        results['max_control_interval'] = False
    else:
        results['max_control_interval'] = True
    printable_summary_dict = add_test_to_printable_summary('control interval shorter than recommended maximum', 'max_control_interval', t_f / float(n_k), printable_summary_dict, results,
                                                               test_param_dict, test_units_dict)

    return results, printable_summary_dict

def test_invariants(trial, test_param_dict, results, input_values, printable_summary_dict, test_units_dict=None):
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
            results, printable_summary_dict = include_result_of_allowed_invariant_test(results, 'c', node, parent, c_sol, c_max, trial.name, printable_summary_dict, test_units_dict=test_units_dict)
            results, printable_summary_dict = include_result_of_allowed_invariant_test(results, 'dc', node, parent, dc_sol, dc_max, trial.name, printable_summary_dict, test_units_dict=test_units_dict)

            if DOF6 and node in architecture.kite_nodes:
                results, printable_summary_dict = include_result_of_allowed_invariant_test(results, 'r', node, parent, r_sol, r_max, trial.name, printable_summary_dict, test_units_dict=test_units_dict)

    return results, printable_summary_dict

def include_result_of_allowed_invariant_test(results, name, node, parent, sol_value, max_value, trial_name, printable_summary_dict, test_units_dict=None):
    combined_name = name + str(node) + str(parent)
    if sol_value > max_value:
        message = 'Invariant ' + combined_name + ' has value ' + str(sol_value) + ' > ' + str(max_value) + ' of V for trial ' + trial_name
        awelogger.logger.warning(message)
        results[combined_name] = False
    else:
        results[combined_name] = True

    print_name = 'physical invariant ' + combined_name + ' remains invariant'
    printable_summary_dict[print_name] = {'found': sol_value, 'threshhold': max_value, 'test passed': results[combined_name]}
    if test_units_dict is not None:
        printable_summary_dict[print_name]['units'] = test_units_dict[name + "_max"]

    return results, printable_summary_dict


def test_node_altitude(trial, test_param_dict, results, printable_summary_dict, test_units_dict=None):
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
        found_height = np.min(heights_x)
        if found_height < z_min:
            results['min_node_height'] = False

        if discretization == 'direct_collocation':
            heights_coll_var = np.array(V_final_si['coll_var', :, :, 'x', node_str, 2])
            found_height = np.min(heights_coll_var)
            if found_height < z_min:
                results['min_node_height'] = False

        printable_summary_dict['node heights above recommended minimum'] = {'found': found_height, 'threshhold': z_min,
                                                 'test passed': results['min_node_height']}
        if test_units_dict is not None:
            printable_summary_dict['node heights above recommended minimum']['units'] = test_units_dict['z_min']

        if not results['min_node_height']:
            print_op.log_and_raise_error(error_message)

    return results, printable_summary_dict

def test_power_balance(trial, test_param_dict, results, input_values, printable_summary_dict, test_units_dict=None):
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
            results = summation_check_on_potential_and_kinetic_power(trial, test_param_dict['energy_summation_thresh'], results, input_values, test_units_dict['energy_summation_thresh'])

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

        scaled_norm_system_net_power = np.linalg.norm(system_net_power_timeseries) / max_abs_system_power
        balance['total'] = scaled_norm_system_net_power

        if balance['total'] > test_param_dict['power_balance_thresh']:
            message = 'total energy balance of trial ' + trial.name + ' not consistent. ' \
                      + str(balance['total']) + ' > ' + str(test_param_dict['power_balance_thresh'])
            awelogger.logger.warning(message)
            results['energy_balance' + 'total'] = False
        else:
            results['energy_balance' + 'total'] = True

        printable_summary_dict['energy balance remains consistent'] = {'found': balance['total'],
                                                              'threshhold': test_param_dict['power_balance_thresh'],
                                                              'test passed': results['energy_balance' + 'total']}
        if test_units_dict is not None:
            printable_summary_dict['energy balance remains consistent']['units'] = test_units_dict['power_balance_thresh']

    return results, printable_summary_dict

def summation_check_on_potential_and_kinetic_power(trial, thresh, results, input_values, printable_summary_dict, units=None):

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

        printable_summary_dict['power based on ' + abbreviated_name + '. is collected'] = {'found': error,
                                                              'threshhold': thresh,
                                                              'test passed': results['energy_balance' + abbreviated_name]}
        if units is not None:
            printable_summary_dict['power based on ' + abbreviated_name + '. is collected']['units'] = units


    return results, printable_summary_dict

def power_balance_key_belongs_to_node(keyname, node):
    keyname_includes_nodenumber = (keyname[-len(str(node)):] == str(node))
    keyname_is_nonnumeric_before_nodenumber = not (keyname[-len(str(node)) - 1].isnumeric())
    key_belongs_to_node = keyname_includes_nodenumber and keyname_is_nonnumeric_before_nodenumber
    return key_belongs_to_node


def test_tracked_vortex_periods(trial, test_param_dict, results, input_values, global_input_values, printable_summary_dict, test_units_dict=None):

    if 'vortex' in input_values['outputs']:
        results['vortex_truncation_error'] = True

        vortex_truncation_error_thresh = test_param_dict['vortex_truncation_error_thresh']

        est_truncation_error = trial.visualization.plot_dict['interpolation_si']['outputs']['vortex']['est_truncation_error']
        est_truncation_error = np.array(est_truncation_error)

        max_trunc_error = np.max(est_truncation_error)
        if max_trunc_error > vortex_truncation_error_thresh:
            message = 'Vortex model estimates a large truncation error' \
                      + str(max_trunc_error) + ' > ' + str(vortex_truncation_error_thresh) \
                      + '. We recommend increasing the number of wake nodes.'
            awelogger.logger.warning(message)
            results['vortex_truncation_error'] = False

        print_name = 'vortex model trunc. error below recommended maximum'
        printable_summary_dict[print_name] = {'found': max_trunc_error,
                                                              'threshhold': vortex_truncation_error_thresh,
                                                              'test passed': results['vortex_truncation_error']}
        if test_units_dict is not None:
            printable_summary_dict[print_name]['units'] = test_units_dict['vortex_truncation_error_thresh']

    return results, printable_summary_dict


def generate_test_param_dict(options, options_help_dict=None):
    """
    Set parameters relevant for testing
    :return: dictionary with test parameters
    """

    test_param_dict = {}
    test_units_dict = {}

    for name, val in options['test_param'].items():
        test_param_dict[name] = val

        if options_help_dict is not None:
            test_units_dict[name] = options_help_dict['quality']['test_param'][name][0][2]

    return test_param_dict, test_units_dict
