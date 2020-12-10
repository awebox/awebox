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
various diagnostics for the optimization object

python-3.5 / casadi-3.4.5
- authors: rachel leuthold, thilo bronnenmeyer, jochem de schutter alu-fr 2018
'''

import awebox.tools.vector_operations as vect_op
import awebox.tools.struct_operations as struct_op
import awebox.tools.debug_operations as debug_op
import numpy as np

from awebox.logger.logger import Logger as awelogger
import casadi as cas


def print_homotopy_values(nlp, solution, p_fix_num):
    V = nlp.V

    # print the phi values:
    awelogger.logger.debug("{0:.<30}:".format('homotopy parameter values'))
    for phi_val in struct_op.subkeys(V, 'phi'):
        awelogger.logger.debug(" {0:>20} = {1:5}".format(phi_val, str(V(solution['x'])['phi', phi_val])))
    awelogger.logger.debug('')

    # print the cost components
    [cost_fun, cost_struct] = nlp.cost_components
    awelogger.logger.debug("{0:.<30}:".format('objective components'))
    for name in list(cost_fun.keys()):
        if 'problem' not in name and 'objective' not in name:
            awelogger.logger.debug(" {0:>20} = {1:5}".format(name[0:-9], str(cost_fun[name](V(solution['x']), p_fix_num))))
    awelogger.logger.debug('')

def print_runtime_values(stats):
    awelogger.logger.debug('')

    awelogger.logger.info("{0:.<30}: {1:<30}".format('solver return status', stats['return_status']))
    awelogger.logger.info("{0:.<30}: {1:<30}".format('number of iterations', stats['iter_count']))
    awelogger.logger.info("{0:.<30}: {1:<30}".format('total wall time', stats['t_wall_total']))

    awelogger.logger.info('')

    return None

def health_check(step_name, final_homotopy_step, nlp, solution, arg, options, solve_succeeded, stats, iterations):
    should_make_autorun_check = (options['health_check']['when']['autorun'])
    should_make_failure_check = (not solve_succeeded) and (options['health_check']['when']['failure'])
    should_make_final_check = (options['health_check']['when']['final']) and (step_name == final_homotopy_step)

    should_make_check = should_make_autorun_check or should_make_failure_check or should_make_final_check

    if should_make_check:
        debug_op.health_check(options['health_check'], nlp, solution, arg, stats, iterations)

    return None

def print_constraint_violations(nlp, V_vals, p_fix_num):
    g_fun = nlp.g_fun
    g = nlp.g
    g_opt = g(g_fun(V_vals, p_fix_num))
    for name in list(g_opt.keys()):
        awelogger.logger.debug(name)
        awelogger.logger.debug(g_opt[name])

    return None

def compute_power_indicators(power_and_performance, plot_dict):

    # geometric stuff
    kite_geometry = plot_dict['options']['solver']['initialization']['sys_params_num']['geometry']
    s_ref = kite_geometry['s_ref']

    # the actual power indicators
    if 'e' in plot_dict['integral_variables']:
        e_final = plot_dict['integral_outputs_final']['int_out',-1,'e']
    else:
        e_final = plot_dict['xd']['e'][0][-1]

    time_period = plot_dict['output_vals'][1]['final', 'time_period', 'val']
    avg_power = e_final / time_period
    surface_area = float(len(plot_dict['architecture'].kite_nodes)) * s_ref
    power_per_surface_area = avg_power / surface_area

    zeta = np.mean(plot_dict['outputs']['performance']['phf'][0])

    power_and_performance['e_final'] = e_final
    power_and_performance['time_period'] = time_period
    power_and_performance['avg_power'] = avg_power
    power_and_performance['zeta'] = zeta
    power_and_performance['power_per_surface_area'] = power_per_surface_area

    l_t_max = np.amax(plot_dict['xd']['l_t'][0])
    z_av = np.mean(plot_dict['xd']['q10'][2])

    power_and_performance['l_t_max'] = l_t_max
    power_and_performance['z_av'] = z_av

    return power_and_performance

def compute_efficiency_measures(power_and_performance, plot_dict):

    power_outputs = plot_dict['outputs']['power_balance']
    N = plot_dict['time_grids']['ip'].shape[0]

    # sum different power types over all system nodes
    P_lift_total = np.zeros((N))
    P_tetherdrag_total = np.zeros((N))
    P_drag_total = np.zeros((N))
    P_side_total = np.zeros((N))
    P_moment_total = np.zeros((N))
    P_gen_total = np.zeros((N))

    for name in list(power_outputs.keys()):

        if name[:6] == 'P_lift':
            P_lift_total += power_outputs[name][0]

        elif name[:12] == 'P_tetherdrag':
            P_tetherdrag_total += power_outputs[name][0]

        elif name[:6] == 'P_drag':
            P_drag_total += power_outputs[name][0]

        elif name[:6] == 'P_side':
            P_side_total += power_outputs[name][0]

        elif name[:] == 'P_moment':
            P_moment_total += power_outputs[name][0]

        elif name[:5] == 'P_gen':
            P_gen_total += power_outputs[name][0]

    epsilon = 1.e-6 # use this to decrease chance of div-by-zero errors at start of optimization
    if np.mean(P_side_total) > 0.0:
        P_in = np.mean(P_lift_total) + np.mean(P_side_total) + epsilon
    else:
        P_in = np.mean(P_lift_total) + epsilon
        power_and_performance['eff_sideforce_loss'] = -np.mean(P_side_total)/ P_in

    power_and_performance['eff_overall'] = - np.mean((power_outputs['P_tether1'][0]+P_gen_total))/P_in
    power_and_performance['eff_tether_drag_loss'] = -np.mean(P_tetherdrag_total)/P_in
    power_and_performance['eff_drag_loss'] =  -np.mean(P_drag_total)/P_in

    return power_and_performance

def compute_position_indicators(power_and_performance, plot_dict):

    # elevation angle
    q10 = plot_dict['xd']['q10']
    elevation = []
    for i in range(q10[0].shape[0]):
        elevation += [np.arccos(np.linalg.norm(np.array([q10[0][i], q10[1][i], 0.0])) / np.linalg.norm(np.array([q10[0][i], q10[1][i], q10[2][i]])))]
    elevation = np.mean(elevation) * 180 / np.pi
    power_and_performance['elevation'] = elevation

    # average velocity of kites (maybe?)
    parent_map = plot_dict['architecture'].parent_map
    number_of_nodes =  plot_dict['architecture'].number_of_nodes
    dq_final = 0.
    for node in range(1, number_of_nodes):
        parent = parent_map[node]
        dq = plot_dict['xd']['dq' + str(node) + str(parent)]
        parent = parent_map[node]
        dq_array = cas.vertcat(dq[0][-1], dq[1][-1], dq[2][-1])
        dq_norm_float = float(vect_op.norm(dq_array))
        dq_final += dq_norm_float
        dq_final = np.array(dq_final)
    dq_final /= float(number_of_nodes - 1.)
    power_and_performance['dq_final'] = dq_final

    # average connex-point velocity
    dq10 = plot_dict['xd']['dq10']
    dq10hat = []
    for i in range(dq10[0].shape[0]):
        dq = np.array([dq10[0][i], dq10[1][i], dq10[2][i]])
        q = np.array([q10[0][i], q10[1][i], q10[2][i]])
        dq10hat += [np.linalg.norm(
            dq - np.matmul(dq.T, q / np.linalg.norm(q)) * q / np.linalg.norm(q))]

    dq10_av = np.mean(np.array(dq10hat))
    power_and_performance['dq10_av'] = dq10_av

    return power_and_performance

def compute_tether_constraint_dissatisfaction(power_and_performance, plot_dict):

    cmax = 0.0
    for constraint in list(plot_dict['outputs']['tether_length'].keys()):
        if constraint[0] == 'c':
            cmax = np.amax([cmax, np.amax(np.abs(plot_dict['outputs']['tether_length'][constraint]))])
    power_and_performance['cmax'] = cmax

    return power_and_performance

def compute_tether_tension_indicators(power_and_performance, plot_dict):

    max_tension = np.max(plot_dict['outputs']['local_performance']['tether_force10'])
    power_and_performance['tension_max'] = max_tension

    # tension average over time
    avg_tension = np.average(plot_dict['outputs']['local_performance']['tether_force10'])
    power_and_performance['tension_avg'] = avg_tension

    return power_and_performance

def compute_power_and_performance(plot_dict):
    power_and_performance = {}

    power_and_performance = compute_power_indicators(power_and_performance, plot_dict)

    power_and_performance = compute_position_indicators(power_and_performance, plot_dict)

    power_and_performance = compute_tether_constraint_dissatisfaction(power_and_performance, plot_dict)

    power_and_performance = compute_tether_tension_indicators(power_and_performance, plot_dict)

    power_and_performance = compute_efficiency_measures(power_and_performance, plot_dict)

    return power_and_performance
