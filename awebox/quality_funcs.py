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
    t_f = np.array(trial.optimization.V_final['theta','t_f']).sum() # compute sum in case of phase fix
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

def test_invariants(trial, test_param_dict, results):
    """
    Test whether invariants reasonably sized
    :return: test results
    """
    # get discretization
    discretization = trial.options['nlp']['discretization']

    # set test parameters from dictionary
    c_max = test_param_dict['c_max']
    dc_max = test_param_dict['dc_max']
    ddc_max = test_param_dict['ddc_max']

    # get architecture
    architecture = trial.model.architecture
    number_of_nodes = architecture.number_of_nodes
    parent_map = architecture.parent_map

    # loop over nodes
    for node in range(1,number_of_nodes):
        for i in [0, 1]:
            parent = parent_map[node]
            out_local = trial.visualization.plot_dict['output_vals'][i]

            if discretization == 'direct_collocation':
                c_list = out_local['coll_outputs', :, :, 'tether_length', 'c' + str(node) + str(parent)]
                dc_list = out_local['coll_outputs', :, :, 'tether_length', 'dc' + str(node) + str(parent)]
                ddc_list = out_local['coll_outputs', :, :, 'tether_length','ddc' + str(node) + str(parent)]

            elif discretization == 'multiple_shooting':
                c_list = out_local['outputs', :, 'tether_length', 'c' + str(node) + str(parent)]
                dc_list = out_local['outputs', :, 'tether_length', 'dc' + str(node) + str(parent)]
                ddc_list = out_local['outputs', :, 'tether_length', 'ddc' + str(node) + str(parent)]

            c_avg = np.average(abs(np.array(c_list)))
            dc_avg = np.average(abs(np.array(dc_list)))
            ddc_avg = np.average(abs(np.array(ddc_list)))

            # test whether invariants are small enough
            if i == 0:
                suffix = 'init'
            elif i == 1:
                suffix = ''
                if c_avg > c_max:
                    awelogger.logger.warning('Invariant c' + str(node) + str(parent) + ' > ' + str(c_max) + ' of V' + suffix + ' for trial ' + trial.name)
                    results['c' + str(node) + str(parent)] = False
                else:
                    results['c' + str(node) + str(parent)] = True

                if dc_avg > dc_max:
                    awelogger.logger.warning('Invariant dc' + str(node) + str(parent) + ' > ' + str(dc_max) + ' of V' + suffix + '  for trial ' + trial.name)
                    results['dc' + str(node) + str(parent)] = False
                else:
                    results['dc' + str(node) + str(parent)] = True

                if ddc_avg > ddc_max:
                    awelogger.logger.warning('Invariant ddc' + str(node) + str(parent) + ' > ' + str(ddc_max) + ' of V' + suffix + ' for trial ' + trial.name)
                    results['ddc' + str(node) + str(parent)] = False
                else:
                    results['ddc' + str(node) + str(parent)] = True

    return results

def test_outputs(trial, test_param_dict, results):
    """
    Test whether outputs are of reasonable size/have correct signs
    :return: test results
    """

    # get discretization
    discretization = trial.options['nlp']['discretization']

    # check if loyd factor is sensible
    max_loyd_factor = test_param_dict['max_loyd_factor']
    if discretization == 'direct_collocation':
        loyd_factor = np.array(trial.optimization.output_vals[1]['coll_outputs', :, :, 'performance', 'loyd_factor'])
    elif discretization == 'multiple_shooting':
        loyd_factor = np.array(trial.optimization.output_vals[1]['outputs', :, 'performance', 'loyd_factor'])
    avg_loyd_factor = np.average(loyd_factor)
    if avg_loyd_factor > max_loyd_factor:
        awelogger.logger.warning('Average Loyd factor > ' + str(max_loyd_factor) + ' for trial ' + trial.name + '. Average Loyd factor is ' + str(avg_loyd_factor))
        results['loyd_factor'] = False
    else:
        results['loyd_factor'] = True

    # check if loyd factor is sensible
    max_power_harvesting_factor = test_param_dict['max_power_harvesting_factor']
    if discretization == 'direct_collocation':
        power_harvesting_factor = np.array(trial.optimization.output_vals[1]['coll_outputs', :, :, 'performance', 'phf'])
    elif discretization == 'multiple_shooting':
        power_harvesting_factor = np.array(trial.optimization.output_vals[1]['outputs', :, 'performance', 'phf'])
    avg_power_harvesting_factor = np.average(power_harvesting_factor)
    if avg_power_harvesting_factor > max_power_harvesting_factor:
        awelogger.logger.warning('Average power harvesting factor > ' + str(max_loyd_factor) + ' for trial ' + trial.name)
        results['power_harvesting_factor'] = False
    else:
        results['power_harvesting_factor'] = True

    # check if maximum tether stress is sensible
    max_tension = test_param_dict['max_tension']
    l_t = trial.visualization.plot_dict['xd']['l_t']
    lambda10 = trial.visualization.plot_dict['xa']['lambda10']
    main_tension = l_t[0] * lambda10[0]
    tension = np.max(main_tension)
    if tension > max_tension:
        awelogger.logger.warning('Max main tether tension > ' + str(max_tension*1e-6) + ' MN for trial ' + trial.name)
        results['tau_max'] = False
    else:
        results['tau_max'] = True

    return results

def test_variables(trial, test_param_dict, results):
    """
    Test whether variables are of reasonable size and have correct signs
    :return: test results
    """

    # get discretization
    discretization = trial.options['nlp']['discretization']

    # get trial solution
    V_final = trial.optimization.V_final

    # extract system architecture
    architecture = trial.model.architecture
    number_of_nodes = architecture.number_of_nodes
    parent_map = architecture.parent_map

    # test if height of all nodes is positive
    for node in range(1, number_of_nodes):
        parent = parent_map[node]
        node_str = 'q' + str(node) + str(parent)
        heights_xd = np.array(V_final['xd',:,node_str,2])
        if discretization == 'direct_collocation':
            heights_coll_var = np.array(V_final['coll_var',:,:,'xd',node_str,2])
            if np.min(heights_coll_var) < 0.:
                coll_height_flag = True
        if np.min(heights_xd) < 0.:
            awelogger.logger.warning('Node ' + node_str + ' has negative height for trial ' + trial.name)
            results['min_node_height'] = False
        if discretization == 'direct_collocation':
            if np.min(heights_coll_var) < 0:
                awelogger.logger.warning('Node ' + node_str + ' has negative height for trial ' + trial.name)
                results['min_node_height'] = False
        else:
            results['min_node_height'] = True

    return results

def test_power_balance(trial, test_param_dict, results):
    """Test whether conservation of energy holds at all nodes and for the entire system.
    :return: test results
    """

    # extract info
    tgrid = trial.visualization.plot_dict['time_grids']['ip']
    power_balance = trial.visualization.plot_dict['outputs']['power_balance']

    check_energy_summation = test_param_dict['check_energy_summation']
    if check_energy_summation:
        results = summation_check_on_potential_and_kinetic_power(trial, test_param_dict['energy_summation_thresh'], results)

    balance = {}
    max_abs_system_power = 1.e-15
    system_net_power_timeseries = np.zeros(tgrid.shape)

    nodes_above_ground = range(1, trial.model.architecture.number_of_nodes)
    for node in nodes_above_ground:

        node_power_timeseries = np.zeros(tgrid.shape)
        nodes_childrens_power_timeseries = np.zeros(tgrid.shape)
        max_abs_node_power = 1.e-15  # preclude any div-by-zero errors

        # how much power originates with the node itself
        for keyname in list(power_balance.keys()):
            if power_balance_key_belongs_to_node(keyname, node):
                timeseries = power_balance[keyname][0]
                node_power_timeseries += timeseries
                max_abs_node_power = np.max([np.max(np.abs(timeseries)), max_abs_node_power])

        # how much power is just being transferred from the node's children
        node_has_children = node in list(trial.model.architecture.children_map.keys())
        if node_has_children:
            children = trial.model.architecture.children_map[node]
            for child in children:
                timeseries = power_balance['P_tether'+str(child)][0]
                nodes_childrens_power_timeseries += timeseries
                max_abs_node_power = np.max([np.max(np.abs(timeseries)), max_abs_node_power])

        # avoid double-counting power that is just being transferred; only count power at point-of-origin
        net_power_timeseries = node_power_timeseries - nodes_childrens_power_timeseries

        scaled_norm_net_power = np.linalg.norm(net_power_timeseries) / max_abs_node_power
        balance[node] = scaled_norm_net_power

        # add node net power into system net power
        max_abs_system_power = np.max([max_abs_node_power, max_abs_system_power])
        system_net_power_timeseries += net_power_timeseries

    scaled_norm_system_net_power = np.linalg.norm(system_net_power_timeseries) / max_abs_system_power
    balance['total'] = scaled_norm_system_net_power

    for node in list(balance.keys()):
        if node == 'total' and balance[node] > test_param_dict['power_balance_thresh']:
            message = 'energy balance for node ' + str(node) + ' of trial ' + trial.name +  ' not consistent. ' \
                      + str(balance[node]) + ' > ' + str(test_param_dict['power_balance_thresh'])
            awelogger.logger.warning(message)
            results['energy_balance' + str(node)] = False
        else:
            results['energy_balance' + str(node)] = True

    return results

def summation_check_on_potential_and_kinetic_power(trial, thresh, results):

    types = ['pot', 'kin']

    kin_comp = np.array(trial.visualization.plot_dict['outputs']['power_balance_comparison']['kinetic'][0])
    pot_comp = np.array(trial.visualization.plot_dict['outputs']['power_balance_comparison']['potential'][0])
    comp_timeseries = {'pot': pot_comp, 'kin': kin_comp}

    # extract info
    tgrid = trial.visualization.plot_dict['time_grids']['ip']
    power_balance = trial.visualization.plot_dict['outputs']['power_balance']

    for type in types:
        sum_timeseries = np.zeros(tgrid.shape)
        for keyname in list(power_balance.keys()):
            if type in keyname:
                timeseries = power_balance[keyname][0]
                sum_timeseries += timeseries

        difference = cas.DM(sum_timeseries - comp_timeseries[type])

        error = float(cas.mtimes(difference.T, difference))

        if error > thresh:
            awelogger.logger.warning('some of the power based on ' + type + '. energy must have gotten lost, since a summation check fails. Considering trial ' + trial.name +  ' with ' + str(error) + ' > ' + str(thresh))
            results['power_summation_check_' + type] = False
        else:
            results['power_summation_check_' + type] = True

    return results

def power_balance_key_belongs_to_node(keyname, node):
    keyname_includes_nodenumber = (keyname[-len(str(node)):] == str(node))
    keyname_is_nonnumeric_before_nodenumber = not (keyname[-len(str(node)) - 1].isnumeric())
    key_belongs_to_node = keyname_includes_nodenumber and keyname_is_nonnumeric_before_nodenumber
    return key_belongs_to_node

def test_slack_equalities(trial, test_param_dict, results):

    if 'xl' in trial.model.variables.keys():

        V_final = trial.optimization.V_final
        xl_vars = trial.model.variables['xl']
        epsilon = test_param_dict['slacks_thresh']

        discretization = trial.options['nlp']['discretization']
        if discretization == 'direct_collocation':

            for idx in range(xl_vars.shape[0]):
                var_name = str(xl_vars[idx])

                if 'slack' in var_name:
                    slack_name = var_name[3:-2]

                    max_val = 0.

                    for ndx in range(trial.nlp.n_k):
                        for ddx in range(trial.nlp.d):

                            data = np.array(V_final['coll_var', ndx, ddx, 'xl', slack_name])
                            max_val = np.max([np.max(data), max_val])

                    if max_val < epsilon:
                        # assume that slack equalities are satisfied
                        results['slacks_' + var_name] = True
                    else:
                        awelogger.logger.warning('slacked equality did not find a feasible solution. ' + var_name + ' > ' + str(test_param_dict['slacks_thresh']))
                        # slack equalities are not satisfied
                        results['slacks_' + var_name] = False

        else:
            awelogger.logger.warning('slack test not yet implemented for multiple-shooting solution')

    return results

def test_tracked_vortex_periods(trial, test_param_dict, results):

    plot_dict = trial.visualization.plot_dict

    if 'vortex' in plot_dict['outputs']:
        vortex_truncation_error_thresh = test_param_dict['vortex_truncation_error_thresh']

        max_est_truncation_error = plot_dict['power_and_performance']['vortex_max_est_truncation_error']
        if max_est_truncation_error > vortex_truncation_error_thresh:
            message = 'Vortex model estimates a large truncation error' \
                      + str(max_est_truncation_error) + ' > ' + str(vortex_truncation_error_thresh) \
                      + '. We recommend increasing the number of tracked periods.'
            awelogger.logger.warning(message)
            results['vortex_truncation_error'] = False

    return results


def generate_test_param_dict(options):
    """
    Set parameters relevant for testing
    :return: dictionary with test parameters
    """

    test_param_dict = {}
    test_param_dict['c_max'] = options['test_param']['c_max']
    test_param_dict['dc_max'] = options['test_param']['dc_max']
    test_param_dict['ddc_max'] = options['test_param']['ddc_max']
    test_param_dict['max_loyd_factor'] = options['test_param']['max_loyd_factor']
    test_param_dict['max_power_harvesting_factor'] = options['test_param']['max_power_harvesting_factor']
    test_param_dict['max_tension'] = options['test_param']['max_tension']
    test_param_dict['max_velocity'] = options['test_param']['max_velocity']
    test_param_dict['t_f_min'] = options['test_param']['t_f_min']
    test_param_dict['max_control_interval'] = options['test_param']['max_control_interval']
    test_param_dict['power_balance_thresh'] = options['test_param']['power_balance_thresh']
    test_param_dict['slacks_thresh'] = options['test_param']['slacks_thresh']
    test_param_dict['vortex_truncation_error_thresh'] = options['test_param']['vortex_truncation_error_thresh']
    test_param_dict['check_energy_summation'] = options['test_param']['check_energy_summation']
    test_param_dict['energy_summation_thresh'] = options['test_param']['energy_summation_thresh']

    return test_param_dict
