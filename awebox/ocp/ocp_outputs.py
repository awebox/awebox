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
performance helping file,
finds various total-trajectory performance metrics requiring knowlege of V
_python-3.5 / casadi-3.4.5
- authors: rachel leuthold, jochem de schutter alu-fr 2017-18
'''
import pdb

import casadi.tools as cas
import numpy as np

import awebox.tools.vector_operations as vect_op
import awebox.tools.struct_operations as struct_op
import awebox.tools.print_operations as print_op
import awebox.mdl.aero.induction_dir.vortex_dir.vortex as vortex


def collect_global_outputs(nlp_options, Outputs, Outputs_structured, Integral_outputs, Integral_outputs_fun, model, V, P):

    global_outputs = {}
    global_outputs = include_time_period(nlp_options, V, global_outputs)

    # include_power_and_performance(nlp_options, Outputs, global_outputs, Integral_outputs, Integral_outputs_fun,
    #                               V, P, model)

    if 'Outputs_structured' is not None and ('vortex' in model.outputs.keys()):
        global_outputs = vortex.compute_global_performance(global_outputs, Outputs_structured, model.architecture)

    [outputs_struct, outputs_dict] = make_output_structure(global_outputs)

    return outputs_struct, outputs_dict


def include_time_period(nlp_options, V, outputs):

    if 'time_period' not in list(outputs.keys()):
        outputs['time_period'] = {}

    time_period = find_time_period(nlp_options, V)

    outputs['time_period']['val'] = time_period

    return outputs


# def include_power_and_performance(nlp_options, Outputs, global_outputs, Integral_outputs, Integral_outputs_fun, V, P, model):
#
#     p_and_p_string = 'power_and_performance'
#
#     power_and_performance = compute_power_and_performance(nlp_options, Outputs, global_outputs, Integral_outputs, Integral_outputs_fun, V, P, model)
#
#     if p_and_p_string not in list(global_outputs.keys()):
#         global_outputs[p_and_p_string] = power_and_performance
#     else:
#         for name in power_and_performance.keys():
#             global_outputs[p_and_p_string][name] = power_and_performance[name]
#
#     return global_outputs


# def compute_power_and_performance(nlp_options, Outputs, global_outputs, Integral_outputs, Integral_outputs_fun, V, P, model):
#     power_and_performance = {}
#
#     power_and_performance = compute_power_indicators(power_and_performance, nlp_options, Outputs, global_outputs, Integral_outputs, Integral_outputs_fun, V, P, model)
#
#     power_and_performance = compute_position_indicators(power_and_performance, plot_dict)
#
#     power_and_performance = compute_tether_constraint_dissatisfaction(power_and_performance, plot_dict)
#
#     power_and_performance = compute_tether_tension_indicators(power_and_performance, plot_dict)
#
#     power_and_performance = compute_efficiency_measures(power_and_performance, plot_dict)
#
#     power_and_performance = compute_windings(power_and_performance, nlp_options, V, model)
#
#     return power_and_performance
#
#
# def compute_efficiency_measures(power_and_performance, plot_dict):
#
#     power_outputs = plot_dict['outputs']['power_balance']
#     N = plot_dict['time_grids']['ip'].shape[0]
#
#     # sum different power types over all system nodes
#     P_lift_total = np.zeros((N))
#     P_tetherdrag_total = np.zeros((N))
#     P_drag_total = np.zeros((N))
#     P_side_total = np.zeros((N))
#     P_moment_total = np.zeros((N))
#     P_gen_total = np.zeros((N))
#
#     for name in list(power_outputs.keys()):
#
#         if name[:6] == 'P_lift':
#             P_lift_total += power_outputs[name][0]
#
#         elif name[:12] == 'P_tetherdrag':
#             P_tetherdrag_total += power_outputs[name][0]
#
#         elif name[:6] == 'P_drag':
#             P_drag_total += power_outputs[name][0]
#
#         elif name[:6] == 'P_side':
#             P_side_total += power_outputs[name][0]
#
#         elif name[:] == 'P_moment':
#             P_moment_total += power_outputs[name][0]
#
#         elif name[:5] == 'P_gen':
#             P_gen_total += power_outputs[name][0]
#
#     epsilon = 1.e-6 # use this to decrease chance of div-by-zero errors at start of optimization
#     if np.mean(P_side_total) > 0.0:
#         P_in = vect_op.average(P_lift_total) + vect_op.average(P_side_total) + epsilon
#     else:
#         P_in = vect_op.average(P_lift_total) + epsilon
#         power_and_performance['eff_sideforce_loss'] = -1. * vect_op.average(P_side_total)/ P_in
#         power_and_performance['eff_sideforce_loss'] = -1. * vect_op.average(P_side_total) / P_in
#
#     power_and_performance['eff_overall'] = -1. * vect_op.average((power_outputs['P_tether1'][0] + P_gen_total)) / P_in
#     power_and_performance['eff_tether_drag_loss'] = -1. * vect_op.average(P_tetherdrag_total) / P_in
#     power_and_performance['eff_drag_loss'] = -1. * vect_op.average(P_drag_total) / P_in
#
#     return power_and_performance
#
#
# def compute_windings(power_and_performance, nlp_options, V, model):
#
#     parent_map = model.architecture.parent_map
#     kite_nodes = model.architecture.kite_nodes
#
#     all_tether_dict = {}
#     for dim in range(3):
#         all_tether_dict[dim] = []
#
#     for ndx in range(nlp_options['n_k']):
#         if nlp_options['discretization'] == 'direct_collocation':
#             for ddx in range(nlp_options['collocation']['d']):
#                 current_ehat = vect_op.normalize(V['coll_var', ndx, ddx, 'x', 'q10'])
#                 for dim in range(3):
#                     all_tether_dict[dim] = cas.vertcat(all_tether_dict[dim], current_ehat[dim])
#         else:
#             current_ehat = vect_op.normalize(V['x', ndx, 'q10'])
#             for dim in range(3):
#                 all_tether_dict[dim] = cas.vertcat(all_tether_dict[dim], current_ehat[dim])
#
#     avg_tether = []
#     for dim in range(3):
#         avg_tether = cas.vertcat(avg_tether, vect_op.average(all_tether_dict[dim]))
#     ehat_tether = vect_op.normalize(avg_tether)
#
#     power_and_performance['winding_axis'] = ehat_tether
#
#     ehat_side_a = vect_op.normed_cross(vect_op.yhat_np(), ehat_tether)
#     # right handed coordinate system -> x/re: _a, y/im: _b, z/out: _tether
#     ehat_side_b = vect_op.normed_cross(ehat_tether, ehat_side_a)
#
#     # now project the path onto this plane
#     for kite in kite_nodes:
#         parent = parent_map[kite]
#
#         theta_start = 0.
#         theta_end = 0.
#
#         # find the origin of the plane
#
#         all_q_dict = {}
#         for dim in range(3):
#             all_q_dict[dim] = []
#
#         for ndx in range(nlp_options['n_k']):
#             if nlp_options['discretization'] == 'direct_collocation':
#                 for ddx in range(nlp_options['collocation']['d']):
#                     current_q = vect_op.normalize(V['coll_var', ndx, ddx, 'x', 'q' + str(kite) + str(parent)])
#                     for dim in range(3):
#                         all_q_dict[dim] = cas.vertcat(all_q_dict[dim], current_q[dim])
#             else:
#                 current_q = vect_op.normalize(V['x', ndx, 'q' + str(kite) + str(parent)])
#                 for dim in range(3):
#                     all_q_dict[dim] = cas.vertcat(all_q_dict[dim], current_q[dim])
#
#         avg_q = []
#         for dim in range(3):
#             avg_q = cas.vertcat(avg_q, vect_op.average(all_q_dict[dim]))
#
#         origin = avg_q - vect_op.dot(avg_q, ehat_tether) * ehat_tether
#
#         # recenter the plane about origin
#         n_interpolation = all_q_dict[0].shape[0]
#         for idx in range(n_interpolation-1):
#             q = []
#             q_next = []
#             for dim in range(3):
#                 q = cas.vertcat(q, all_q_dict[dim][idx])
#                 q_next = cas.vertcat(q_next, all_q_dict[dim][idx+1])
#
#             q_in_plane = q - vect_op.dot(q, ehat_tether) * ehat_tether
#             q_recentered = q_in_plane - origin
#
#             q_next_in_plane = q_next - vect_op.dot(q_next, ehat_tether) * ehat_tether
#             q_next_recentered = q_next_in_plane - origin
#
#             delta_q = q_next_recentered - q_recentered
#
#             x = vect_op.dot(q_recentered, ehat_side_a)
#             y = vect_op.dot(q_recentered, ehat_side_b)
#             r_squared = x**2. + y**2.
#
#             dx = vect_op.dot(delta_q, ehat_side_a)
#             dy = vect_op.dot(delta_q, ehat_side_b)
#
#             # dx = vect_op.dot(dq_in_plane, ehat_side_a)
#             # dy = vect_op.dot(dq_in_plane, ehat_side_b)
#
#             dtheta = (x * dy - y * dx) / (r_squared + 1.0e-4)
#             theta_end += dtheta
#
#         winding = (theta_end - theta_start) / 2. / np.pi
#
#         power_and_performance['winding' + str(kite)] = winding
#
#     return power_and_performance
#
#
# def compute_power_indicators(power_and_performance, nlp_options, Outputs, global_outputs, Integral_outputs, Integral_outputs_fun, V, P, model):
#
#     # geometric stuff
#     s_ref = P['theta0', 'geometry', 's_ref']
#
#     # the actual power indicators
#     if 'e' in model.integral_outputs.keys():
#         e_final = Integral_outputs(Integral_outputs_fun(V, P))['int_out', -1, 'e']
#     else:
#         e_final = V['x', -1, 'e']
#
#     time_period = global_outputs['time_period']['val']
#     avg_power = e_final / time_period
#     surface_area = float(len(model.architecture.kite_nodes)) * s_ref
#     power_per_surface_area = avg_power / surface_area
#
#     idx = struct_op.find_output_idx(model.outputs, 'performance', 'phf')
#
#     all_zeta = Outputs[idx, :]
#     zeta = vect_op.average(all_zeta)
#
#     power_and_performance['e_final'] = e_final
#     power_and_performance['time_period'] = time_period
#     power_and_performance['avg_power'] = avg_power
#     power_and_performance['zeta'] = zeta
#     power_and_performance['power_per_surface_area'] = power_per_surface_area
#
#     if 'l_t' in model.variables_dict['x'].keys():
#
#         all_l_t = []
#         for ndx in range(nlp_options['n_k']):
#             if nlp_options['discretization'] == 'direct_collocation':
#                 for ddx in range(nlp_options['collocation']['d']):
#                     all_l_t = cas.vertcat(all_l_t, V['coll_var', ndx, ddx, 'x', 'l_t'])
#             else:
#                 all_l_t = cas.vertcat(all_l_t, V['x', ndx, 'l_t'])
#
#         power_and_performance['l_t_max'] = vect_op.smooth_max(all_l_t)
#
#     elif 'l_t' in model.variables_dict['theta'].keys():
#         power_and_performance['l_t_max'] = V['theta', 'l_t']
#     else:
#         message = 'unable to find main tether length in plot_dict'
#         print_op.log_and_raise_error(message)
#
#     all_altitude = []
#     for ndx in range(nlp_options['n_k']):
#         if nlp_options['discretization'] == 'direct_collocation':
#             for ddx in range(nlp_options['collocation']['d']):
#                 all_altitude = cas.vertcat(all_altitude, V['coll_var', ndx, ddx, 'x', 'q10'][2])
#         else:
#             all_altitude = cas.vertcat(all_altitude, V['x', ndx, 'q10'][2])
#     z_av = vect_op.average(all_altitude)
#
#     power_and_performance['z_av'] = z_av
#
#     return power_and_performance
#


def make_output_structure(outputs):
    outputs_vec = []
    full_list = []

    outputs_dict = {}

    for output_type in list(outputs.keys()):

        local_list = []
        for name in list(outputs[output_type].keys()):
            # prepare empty entry list to generate substruct
            local_list += [cas.entry(name, shape=outputs[output_type][name].shape)]

            # generate vector with outputs - SX expressions
            outputs_vec = cas.vertcat(outputs_vec, outputs[output_type][name])

        # generate dict with sub-structs
        outputs_dict[output_type] = cas.struct_symMX(local_list)
        # prepare list with sub-structs to generate struct
        full_list += [cas.entry(output_type, struct=outputs_dict[output_type])]

        # generate "empty" structure
    out_struct = cas.struct_symMX(full_list)
    # generate structure with SX expressions
    outputs_struct = out_struct(outputs_vec)

    return [outputs_struct, outputs_dict]


def find_time_spent_in_reelout(nlp_numerics_options, V):

    nk = nlp_numerics_options['n_k']
    phase_fix_reel_out = nlp_numerics_options['phase_fix_reelout']
    time_period_zeroth = V['theta', 't_f', 0] * round(nk * phase_fix_reel_out) / nk
    return time_period_zeroth

def find_time_spent_in_reelin(nlp_numerics_options, V):
    nk = nlp_numerics_options['n_k']
    phase_fix_reel_out = nlp_numerics_options['phase_fix_reelout']
    time_period_first = V['theta', 't_f', 1] * (nk - round(nk * phase_fix_reel_out)) / nk
    return time_period_first

def find_time_period(nlp_numerics_options, V):

    if nlp_numerics_options['phase_fix'] == 'single_reelout':
        reelout_time = find_time_spent_in_reelout(nlp_numerics_options, V)
        reelin_time = find_time_spent_in_reelin(nlp_numerics_options, V)

        time_period = (reelout_time + reelin_time)
    else:
        time_period = V['theta', 't_f']

    return time_period

