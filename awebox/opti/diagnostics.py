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
import awebox.tools.performance_operations as perf_op
import awebox.tools.print_operations as print_op

import awebox.mdl.aero.induction_dir.vortex_dir.vortex as vortex

from awebox.logger.logger import Logger as awelogger
import casadi as cas
import numpy as np


def print_homotopy_values(nlp, solution, p_fix_num):

    # todo: use print_operations here, to avoid the manual spacing

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

    awelogger.logger.info("{0:.<30}: {1:<30}".format('Solver return status', stats['return_status']))
    awelogger.logger.info("{0:.<30}: {1:<30}".format('Number of iterations', stats['iter_count']))
    awelogger.logger.info("{0:.<30}: {1:<30}".format('Total wall time', str(stats['t_wall_total'])+'s'))

    awelogger.logger.info('')

    return None

def health_check(trial_name, step_name, final_homotopy_step, nlp, model, solution, arg, solver_options, stats, iterations, cumulative_max_memory):

    solve_succeeded = stats['success']

    when = solver_options['health_check']['when']
    allowed_whens = ['never', 'failure', 'success', 'final', 'always', 'failure_or_final']

    if when not in allowed_whens:
        message = 'health check specified to run ' + when + ', but this is not among the expected frequency names: ' + repr(allowed_whens)
        print_op.log_and_raise_error(message)

    should_make_autorun_check = (when == 'always')
    should_make_success_check = (solve_succeeded) and (when == 'success')
    should_make_failure_check = (not solve_succeeded) and (when in ['failure', 'failure_or_final'])
    should_make_final_check = (when in ['final', 'failure_or_final']) and (step_name == final_homotopy_step)
    do_not_make_check = (when == 'never')

    should_make_check = (not do_not_make_check) and (should_make_autorun_check or should_make_success_check or should_make_failure_check or should_make_final_check)

    if should_make_check:
        return debug_op.health_check(trial_name, solver_options, nlp, model, solution, arg, stats, iterations, step_name, cumulative_max_memory)
    else:
        return True


def print_constraint_violations(nlp, V_vals, p_fix_num):
    g_fun = nlp.g_fun
    g = nlp.g
    g_opt = g(g_fun(V_vals, p_fix_num))
    for name in list(g_opt.keys()):
        awelogger.logger.debug(name)
        awelogger.logger.debug(g_opt[name])

    return None

def compute_power_indicators(power_and_performance, plot_dict):

    # todo: this probably shouldn't depend on plot_dict specifically, since it's living in the optimization directory

    # geometric stuff
    kite_geometry = plot_dict['options']['solver']['initialization']['sys_params_num']['geometry']
    s_ref = kite_geometry['s_ref']

    # the actual power indicators
    e_final_joules = plot_dict['global_output_vals']['e_final_joules'].full()[0][0]
    time_period = plot_dict['global_output_vals']['time_period'].full()[0][0]
    avg_power_watts = plot_dict['global_output_vals']['avg_power_watts'].full()[0][0]
    surface_area = float(len(plot_dict['architecture'].kite_nodes)) * s_ref
    power_per_surface_area = avg_power_watts / surface_area

    e_final_joules_without_fictitious = plot_dict['global_output_vals']['e_final_joules_without_fictitious'].full()[0][0]
    avg_power_watts_without_fictitious = plot_dict['global_output_vals']['avg_power_watts_without_fictitious'].full()[0][0]

    zeta = np.mean(plot_dict['interpolation_si']['outputs']['performance']['phf'][0])

    power_and_performance['e_final'] = e_final_joules
    power_and_performance['time_period'] = time_period
    power_and_performance['avg_power'] = avg_power_watts
    power_and_performance['zeta'] = zeta
    power_and_performance['power_per_surface_area'] = power_per_surface_area

    power_and_performance['e_final_without_fictitious'] = e_final_joules_without_fictitious
    power_and_performance['avg_power_without_fictitious'] = avg_power_watts_without_fictitious

    if 'l_t' in plot_dict['interpolation_si']['x'].keys():
        power_and_performance['l_t_max'] = np.max(np.abs(np.array(plot_dict['interpolation_si']['x']['l_t'][0])))
    elif 'l_t' in plot_dict['interpolation_si']['theta'].keys():
        power_and_performance['l_t_max'] = plot_dict['interpolation_si']['theta']['l_t']
    else:
        message = 'unable to find main tether length in plot_dict'
        print_op.log_and_raise_error(message)

    z_av = np.mean(plot_dict['interpolation_si']['x']['q10'][2])

    power_and_performance['z_av'] = z_av

    return power_and_performance


def compute_efficiency_measures(power_and_performance, plot_dict):

    power_outputs = plot_dict['interpolation_si']['outputs']['power_balance']
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
    power_and_performance['eff_drag_loss'] = -np.mean(P_drag_total)/P_in

    return power_and_performance

def compute_position_indicators(power_and_performance, plot_dict):

    interpolated_x_si = plot_dict['interpolation_si']['x']

    # elevation angle
    architecture = plot_dict['architecture']
    q10 = interpolated_x_si['q10']

    n_interpolation = q10[0].shape[0]

    elevation = []
    azimuth = []
    average_cone_angle = []

    for idx in range(n_interpolation):
        q10_0 = q10[0][idx]
        q10_1 = q10[1][idx]
        q10_2 = q10[2][idx]
        position_parent = np.array([q10_0, q10_1, q10_2])

        local_elevation = perf_op.get_elevation_angle(position_parent)
        elevation += [local_elevation]

        azimuth += [np.arctan2(q10_1, q10_0)]

        lowest_layer_node = architecture.parent_map[np.min(np.array(architecture.kite_nodes))]

        local_average_cone_angle = 0.
        set_of_kites_in_lowest_layer = architecture.get_kite_children(lowest_layer_node)
        number_of_kites_in_lowest_layer = len(set_of_kites_in_lowest_layer)
        for kite in set_of_kites_in_lowest_layer:

            parent = architecture.parent_map[kite]

            label = 'q' + str(kite) + str(parent)
            position_kite = []
            for dim in range(3):
                local_position = interpolated_x_si[label][dim][idx]
                position_kite = cas.vertcat(position_kite, local_position)
            position_kite = np.array(position_kite)

            if parent == 0:
                position_parent = np.zeros((3, 1))
                average_tether_vector = []
                for dim in range(3):
                    local_position = np.mean(np.array(interpolated_x_si['q10'][dim]))
                    average_tether_vector = cas.vertcat(average_tether_vector, local_position)
                average_tether_vector = np.array(average_tether_vector)
                position_grandparent = position_parent - average_tether_vector
            else:
                grandparent = architecture.parent_map[parent]

                label_parent = 'q' + str(parent) + str(grandparent)

                position_parent = []
                for dim in range(3):
                    local_position = interpolated_x_si[label_parent][dim][idx]
                    position_parent = cas.vertcat(position_parent, local_position)
                position_parent = np.array(position_parent)

                if grandparent == 0:
                    position_grandparent = np.zeros((3, 1))
                else:
                    great_grandparent = architecture.parent_map[grandparent]

                    label_grandparent = 'q' + str(grandparent) + str(great_grandparent)
                    position_grandparent = []
                    for dim in range(3):
                        local_position = interpolated_x_si[label_grandparent][dim][idx]
                        position_grandparent = cas.vertcat(position_grandparent, local_position)
                    position_grandparent = np.array(position_grandparent)

            local_kite_cone_angle = perf_op.get_cone_angle(position_kite, position_parent, position_grandparent)
            local_average_cone_angle += local_kite_cone_angle / float(number_of_kites_in_lowest_layer)
        average_cone_angle += [float(local_average_cone_angle)]

    # average, then convert to degrees
    elevation_deg = np.rad2deg(np.mean(elevation))
    azimuth_deg = np.rad2deg(np.mean(azimuth))
    average_cone_angle_deg = np.rad2deg(np.mean(np.array(average_cone_angle)))

    power_and_performance['elevation'] = elevation_deg
    power_and_performance['azimuth'] = azimuth_deg
    power_and_performance['cone_angle'] = average_cone_angle_deg

    # average flight speed
    # power_and_performance['flight_speed'] = ...

    # average velocity of kites (maybe?)
    parent_map = plot_dict['architecture'].parent_map
    number_of_nodes =  plot_dict['architecture'].number_of_nodes
    dq_final = 0.
    for node in range(1, number_of_nodes):
        parent = parent_map[node]
        dq = interpolated_x_si['dq' + str(node) + str(parent)]
        parent = parent_map[node]
        dq_array = cas.vertcat(dq[0][-1], dq[1][-1], dq[2][-1])
        dq_norm_float = float(vect_op.norm(dq_array))
        dq_final += dq_norm_float
        dq_final = np.array(dq_final)
    dq_final /= float(number_of_nodes - 1.)
    power_and_performance['dq_final'] = dq_final

    # average connex-point velocity
    dq10 = interpolated_x_si['dq10']
    dq10hat = []
    for i in range(dq10[0].shape[0]):
        dq = np.array([dq10[0][i], dq10[1][i], dq10[2][i]])
        q = np.array([q10[0][i], q10[1][i], q10[2][i]])
        dq10hat += [np.linalg.norm(
            dq.squeeze() - np.dot(dq.squeeze().T, q.squeeze() / np.linalg.norm(q)) * q.squeeze() / np.linalg.norm(q))]

    dq10_av = np.mean(np.array(dq10hat))
    power_and_performance['dq10_av'] = dq10_av

    return power_and_performance

def compute_tether_constraint_dissatisfaction(power_and_performance, plot_dict):

    cmax = 0.0
    for constraint in list(plot_dict['interpolation_si']['outputs']['invariants'].keys()):
        if constraint[0] == 'c':
            cmax = np.amax([cmax, np.amax(np.abs(plot_dict['interpolation_si']['outputs']['invariants'][constraint]))])
    power_and_performance['cmax'] = cmax

    return power_and_performance

def compute_tether_tension_indicators(power_and_performance, plot_dict):

    max_tension = np.max(plot_dict['interpolation_si']['outputs']['local_performance']['tether_force10'])
    power_and_performance['tension_max'] = max_tension

    # tension average over time
    avg_tension = np.average(plot_dict['interpolation_si']['outputs']['local_performance']['tether_force10'])
    power_and_performance['tension_avg'] = avg_tension

    return power_and_performance



def compute_control_frequency(power_and_performance, plot_dict):

    kite_nodes = plot_dict['architecture'].kite_nodes

    first_kite = kite_nodes[0]
    parent = plot_dict['architecture'].parent_map[first_kite]
    tentative_var_name = 'delta' + str(first_kite) + str(parent)

    if tentative_var_name in plot_dict['interpolation_si']['x'].keys():
        var_type = 'x'
    elif tentative_var_name in plot_dict['interpolation_si']['u'].keys():
        var_type = 'u'
    else:
        var_type = None

    if (var_type is not None):

        time_period = power_and_performance['time_period']

        number_control_surfaces = len(plot_dict['interpolation_si'][var_type][tentative_var_name])

        for kite in kite_nodes:
            parent = plot_dict['architecture'].parent_map[kite]

            frequency_stack = []
            for cs in range(number_control_surfaces):
                var_name = 'delta' + str(kite) + str(parent)
                vals = plot_dict['interpolation_si'][var_type][var_name][cs]
                steps = vals.shape[0] - 1
                dt = time_period / steps

                frequency = vect_op.estimate_1d_frequency(vals, dt=dt)
                frequency_stack = cas.vertcat(frequency_stack, frequency)

            power_and_performance['control_frequency' + str(kite) + str(parent)] = frequency_stack

    return power_and_performance


def compute_windings(power_and_performance, plot_dict):

    interpolated_x_si = plot_dict['interpolation_si']['x']

    n_interpolation = interpolated_x_si['q10'][0].shape[0]
    total_steps = float(n_interpolation)

    parent_map = plot_dict['architecture'].parent_map
    kite_nodes = plot_dict['architecture'].kite_nodes

    ehat_tether_x = 0.
    ehat_tether_y = 0.
    ehat_tether_z = 0.

    for idx in range(n_interpolation):
        q10 = cas.vertcat(interpolated_x_si['q10'][0][idx], interpolated_x_si['q10'][1][idx], interpolated_x_si['q10'][2][idx])
        local_ehat = vect_op.normalize(q10)
        ehat_tether_x += local_ehat[0] / total_steps
        ehat_tether_y += local_ehat[1] / total_steps
        ehat_tether_z += local_ehat[2] / total_steps

    ehat_tether = vect_op.normalize(cas.vertcat(ehat_tether_x, ehat_tether_y, ehat_tether_z))

    power_and_performance['winding_axis'] = ehat_tether

    ehat_side_a = vect_op.normed_cross(vect_op.yhat_np(), ehat_tether)
    # right handed coordinate system -> x/re: _a, y/im: _b, z/out: _tether
    ehat_side_b = vect_op.normed_cross(ehat_tether, ehat_side_a)

    # now project the path onto this plane
    for n in kite_nodes:
        parent = parent_map[n]

        theta_start = 0.
        theta_end = 0.

        # find the origin of the plane
        origin = np.zeros((3, 1))
        for idx in range(n_interpolation):
            name = 'q' + str(n) + str(parent)
            q = cas.vertcat(interpolated_x_si[name][0][idx], interpolated_x_si[name][1][idx],
                              interpolated_x_si[name][2][idx])
            q_in_plane = q - vect_op.dot(q, ehat_tether) * ehat_tether

            origin = origin + q_in_plane / total_steps

        # recenter the plane about origin
        for idx in range(n_interpolation-1):
            name = 'q' + str(n) + str(parent)
            q = cas.vertcat(interpolated_x_si[name][0][idx], interpolated_x_si[name][1][idx],
                              interpolated_x_si[name][2][idx])

            q_in_plane = q - vect_op.dot(q, ehat_tether) * ehat_tether
            q_recentered = q_in_plane - origin

            q_next = cas.vertcat(interpolated_x_si[name][0][idx+1], interpolated_x_si[name][1][idx+1],
                              interpolated_x_si[name][2][idx+1])
            q_next_in_plane = q_next - vect_op.dot(q_next, ehat_tether) * ehat_tether
            q_next_recentered = q_next_in_plane - origin

            delta_q = q_next_recentered - q_recentered

            x = vect_op.dot(q_recentered, ehat_side_a)
            y = vect_op.dot(q_recentered, ehat_side_b)
            r_squared = x**2. + y**2.

            dx = vect_op.dot(delta_q, ehat_side_a)
            dy = vect_op.dot(delta_q, ehat_side_b)

                # dx = vect_op.dot(dq_in_plane, ehat_side_a)
                # dy = vect_op.dot(dq_in_plane, ehat_side_b)

            dtheta = (x * dy - y * dx) / (r_squared + 1.0e-4)
            theta_end += dtheta

        winding = (theta_end - theta_start) / 2. / np.pi

        power_and_performance['winding' + str(n)] = winding

    return power_and_performance








def compute_power_and_performance(plot_dict):
    power_and_performance = {}

    power_and_performance = compute_power_indicators(power_and_performance, plot_dict)

    power_and_performance = compute_position_indicators(power_and_performance, plot_dict)

    power_and_performance = compute_tether_constraint_dissatisfaction(power_and_performance, plot_dict)

    power_and_performance = compute_tether_tension_indicators(power_and_performance, plot_dict)

    power_and_performance = compute_efficiency_measures(power_and_performance, plot_dict)

    power_and_performance = compute_control_frequency(power_and_performance, plot_dict)

    power_and_performance = compute_windings(power_and_performance, plot_dict)

    return power_and_performance
