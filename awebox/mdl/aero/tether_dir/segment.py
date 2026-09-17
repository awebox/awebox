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
drag on a tether segment (between two nodes)
_python-3.5 / casadi-3.4.5
- edited: rachel leuthold, jochem de schutter alu-fr 2017-20
'''

import os
import matplotlib
if "MPLBACKEND" not in os.environ:
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import awebox.tools.vector_operations as vect_op
import awebox.tools.print_operations as print_op
import casadi.tools as cas
import numpy as np

import awebox.mdl.aero.tether_dir.element as tether_element
import awebox.mdl.aero.tether_dir.reynolds as tether_reynolds
import awebox.mdl.aero.kite_dir.frames as kite_frames

def columnize_node_forces(force_upper=[], force_lower=[]):
    return cas.vertcat(force_upper, force_lower)

def unpack_node_force_column(columnized):
    return {'upper': columnized[:3], 'lower': columnized[3:]}

def plot_upper_node_force(num_fig, model_options, parameters, parameters_at_time, element_drag_fun, atmos, wind, projection=vect_op.xhat_np()):

    n_elements = 10

    trivial_segment_drag_fun, _ = get_trivial_segment_drag_fun(atmos, wind, parameters)
    trivial_drag_distribution_fun, _ = get_half_half_drag_force_distribution_fun(parameters, trivial_segment_drag_fun)
    split_segment_drag_fun, _ = get_multielement_segment_drag_fun(1, parameters, element_drag_fun)
    split_drag_distribution_fun, _ = get_half_half_drag_force_distribution_fun(parameters, split_segment_drag_fun)
    multi_segment_drag_fun, _ = get_multielement_segment_drag_fun(n_elements, parameters, element_drag_fun)
    multi_drag_distribution_fun, _ = get_multielement_drag_force_distribution_fun(n_elements, parameters, multi_segment_drag_fun)
    equi_segment_drag_fun, _ = get_equivalent_segment_drag_fun(n_elements, parameters, element_drag_fun)
    equi_drag_distribution_fun, _ = get_equivalent_drag_force_distribution_fun(parameters, equi_segment_drag_fun)
    buggy_segment_drag_fun, _ = get_equivalent_segment_drag_fun(n_elements, parameters, element_drag_fun, use_buggy_version_for_verification_purposes=True)
    buggy_drag_distribution_fun, _ = get_equivalent_drag_force_distribution_fun(parameters, buggy_segment_drag_fun)

    model_dict = {'trivial': {'dist_fun': trivial_drag_distribution_fun, 'list': [], 'color': 'm'},
            'split': {'dist_fun': split_drag_distribution_fun, 'list': [], 'color': 'r'},
            'multi': {'dist_fun': multi_drag_distribution_fun, 'list': [], 'color': 'y'},
            'equi': {'dist_fun': equi_drag_distribution_fun, 'list': [], 'color': 'g'},
            'buggy': {'dist_fun': buggy_drag_distribution_fun, 'list': [], 'color': 'b'}
            }

    inc_rad = 45. * np.pi / 180.
    q_upper = 500. * (vect_op.xhat_np() * np.cos(inc_rad) + vect_op.zhat_np() * np.sin(inc_rad))
    q_lower = 0. * vect_op.zhat_np()
    dq_lower = 0. * vect_op.zhat_np()
    diam = 0.01

    speed_list = np.arange(-50., 50., 10.)

    for speed in speed_list:
        dq_upper = speed * vect_op.yhat_np()
        segment_info_dict = {'q_upper': q_upper,
                             'q_lower': q_lower,
                             'dq_upper': dq_upper,
                             'dq_lower': dq_lower,
                             'diameter': diam}
        segment_info_columnized = tether_element.columnize_element_info(unpacked_as_dict=segment_info_dict)
        for model, local_dict in model_dict.items():
            columnized_node_forces = local_dict['dist_fun'](segment_info_columnized, parameters_at_time)
            node_forces_dict = unpack_node_force_column(columnized_node_forces)
            new_entry = cas.mtimes(node_forces_dict['upper'].T, projection)
            local_dict['list'] += [float(new_entry)]

    plt.figure(num_fig)
    for model, local_dict in model_dict.items():
        plt.plot(speed_list, local_dict['list'], local_dict['color'], label=model)
    plt.legend()
    plt.show()

def get_segment_drag_fun(model_options, parameters, atmos, wind, element_drag_fun, reynolds_fun, cd_tether_fun):
    info_to_add_to_applied_params_dict = {}
    local_info_to_add_to_applied_params_dict = {}

    tether_drag_model = model_options['tether']['tether_drag']['model_type']
    info_to_add_to_applied_params_dict['user_options.tether_drag_model'] = tether_drag_model

    if tether_drag_model == 'trivial':
        segment_drag_fun, local_info_to_add_to_applied_params_dict = get_trivial_segment_drag_fun(atmos, wind, parameters)
    elif tether_drag_model == 'kite_only':
        segment_drag_fun, local_info_to_add_to_applied_params_dict = get_kite_only_segment_drag_fun(atmos, parameters, reynolds_fun=reynolds_fun, cd_tether_fun=cd_tether_fun)
    elif tether_drag_model == 'multi':
        n_elements = model_options['tether']['aero_elements']
        info_to_add_to_applied_params_dict['model.tether.aero_elements'] = n_elements
        segment_drag_fun, local_info_to_add_to_applied_params_dict = get_multielement_segment_drag_fun(n_elements, parameters, element_drag_fun)
    elif tether_drag_model == 'split':
        segment_drag_fun, local_info_to_add_to_applied_params_dict = get_multielement_segment_drag_fun(1, parameters, element_drag_fun)
    elif 'equivalent' in tether_drag_model:
        n_elements = model_options['tether']['aero_elements']
        info_to_add_to_applied_params_dict['model.tether.aero_elements'] = n_elements
        use_buggy_version_for_verification_purposes = (tether_drag_model == 'equivalent_buggy')
        segment_drag_fun, local_info_to_add_to_applied_params_dict = get_equivalent_segment_drag_fun(n_elements, parameters, element_drag_fun,
                                        use_buggy_version_for_verification_purposes)
    elif tether_drag_model == 'not_in_use':
        info_sym = cas.SX.sym('info_sym', tether_element.get_size_of_element_info_column())
        segment_drag_fun = cas.Function('segment_drag_fun', [info_sym, parameters], [cas.DM.zeros((3, 1))])
    else:
        raise ValueError('tether drag model not supported.')

    for local_address, local_val in local_info_to_add_to_applied_params_dict.items():
        info_to_add_to_applied_params_dict[local_address] = local_val

    return segment_drag_fun, info_to_add_to_applied_params_dict


def get_drag_force_distribution_fun(model_options, parameters, segment_drag_fun):
    info_to_add_to_applied_params_dict = {}
    local_info_to_add_to_applied_params_dict = {}

    tether_drag_model = model_options['tether']['tether_drag']['model_type']
    info_to_add_to_applied_params_dict['user_options.tether_drag_model'] = tether_drag_model

    if tether_drag_model in ['trivial', 'split']:
        drag_distribution_fun, local_info_to_add_to_applied_params_dict = get_half_half_drag_force_distribution_fun(parameters, segment_drag_fun)
    elif tether_drag_model == 'kite_only':
        drag_distribution_fun, local_info_to_add_to_applied_params_dict = get_kite_only_drag_force_distribution_fun(parameters, segment_drag_fun)
    elif tether_drag_model == 'multi':
        n_elements = model_options['tether']['aero_elements']
        info_to_add_to_applied_params_dict['model.tether.aero_elements'] = n_elements
        drag_distribution_fun, local_info_to_add_to_applied_params_dict = get_multielement_drag_force_distribution_fun(n_elements, parameters, segment_drag_fun)
    elif 'equivalent' in tether_drag_model:
        drag_distribution_fun, local_info_to_add_to_applied_params_dict = get_equivalent_drag_force_distribution_fun(parameters, segment_drag_fun)
    elif tether_drag_model == 'not_in_use':
        info_sym = cas.SX.sym('info_sym', tether_element.get_size_of_element_info_column())
        drag_distribution_fun = cas.Function('drag_distribution_fun', [info_sym, parameters], [cas.DM.zeros((6, 1))])
    else:
        raise ValueError('tether drag model not supported.')

    for local_address, local_val in local_info_to_add_to_applied_params_dict.items():
        info_to_add_to_applied_params_dict[local_address] = local_val

    return drag_distribution_fun, info_to_add_to_applied_params_dict


def get_equivalent_segment_drag_fun(n_elements, parameters, element_drag_fun, use_buggy_version_for_verification_purposes=False):
    info_to_add_to_applied_params_dict = {}
    if use_buggy_version_for_verification_purposes:
        message = 'You have deliberately selected a tether model that includes substantial integration errors, and is not recommended. Please be absolutely sure this model is what you would like to use.'
        print_op.base_print(message, level='warning')

    segment_info_sym = cas.SX.sym('info_sym', tether_element.get_size_of_element_info_column())
    segment_info_dict = tether_element.unpack_element_info_column(segment_info_sym)

    q_center_segment = (segment_info_dict['q_upper'] + segment_info_dict['q_lower']) / 2.

    segment_force = cas.DM.zeros((3, 1))
    segment_moment = cas.DM.zeros((3, 1))
    for element in range(n_elements):
        elem_info = tether_element.get_info_column_linear_division_of_tether_segment(segment_info_dict, element, n_elements)

        if use_buggy_version_for_verification_purposes:
            local_force = element_drag_fun(segment_info_sym, parameters)
        else:
            local_force = element_drag_fun(elem_info, parameters)

        segment_force += local_force

        elem_info_dict = tether_element.unpack_element_info_column(elem_info)
        q_center_element = (elem_info_dict['q_upper'] + elem_info_dict['q_lower']) / 2.
        moment_arm = q_center_segment - q_center_element
        local_moment = vect_op.cross(moment_arm, local_force)
        segment_moment += local_moment

    force_and_moment = cas.vertcat(segment_force, segment_moment)
    segment_drag_fun = cas.Function('segment_drag_fun', [segment_info_sym, parameters], [force_and_moment])
    return segment_drag_fun, info_to_add_to_applied_params_dict

def get_equivalent_drag_force_distribution_fun(parameters, segment_drag_fun):
    info_to_add_to_applied_params_dict = {}
    info_sym = cas.SX.sym('info_sym', tether_element.get_size_of_element_info_column())
    drag_and_moment_earthfixed = segment_drag_fun(info_sym, parameters)
    total_force_earthfixed = drag_and_moment_earthfixed[0:3]
    total_moment_earthfixed = drag_and_moment_earthfixed[3:6]

    info_column = tether_element.unpack_element_info_column(info_sym)
    q_upper = info_column['q_upper']
    q_lower = info_column['q_lower']

    total_force_body = from_earthfixed_to_body(total_force_earthfixed, q_upper, q_lower)
    total_moment_body = from_earthfixed_to_body(total_moment_earthfixed, q_upper, q_lower)

    total_moment_body[2] = 0.

    total_vect = cas.vertcat(total_force_body, total_moment_body)

    tether = q_upper - q_lower
    Ainv = get_inverse_equivalence_matrix(vect_op.norm(tether))

    equiv_vect = cas.mtimes(Ainv, total_vect)

    equiv_force_upper_body = equiv_vect[0:3]
    equiv_force_lower_body = equiv_vect[3:6]

    equiv_force_upper_earthfixed = from_body_to_earthfixed(equiv_force_upper_body, q_upper, q_lower)
    equiv_force_lower_earthfixed = from_body_to_earthfixed(equiv_force_lower_body, q_upper, q_lower)

    node_forces = columnize_node_forces(force_upper=equiv_force_upper_earthfixed, force_lower=equiv_force_lower_earthfixed)
    drag_distribution_fun = cas.Function('drag_distribution_fun', [info_sym, parameters], [node_forces])
    return drag_distribution_fun, info_to_add_to_applied_params_dict

def get_body_axes(q_upper, q_lower):
    # todo: remove this when Rachel is done with verification testing.
    tether = q_upper - q_lower
    # xhat = vect_op.xhat()
    yhat = vect_op.yhat()
    ehat_z = vect_op.normalize(tether)
    ehat_x = vect_op.normed_cross(yhat, tether)
    ehat_y = vect_op.normed_cross(ehat_z, ehat_x)

    return ehat_x, ehat_y, ehat_z

def from_earthfixed_to_body(earthfixed_vector, q_upper, q_lower):
    # todo: remove this when Rachel is done with verification testing.
    [ehat_x, ehat_y, ehat_z] = get_body_axes(q_upper, q_lower)
    DCM = cas.horzcat(ehat_x, ehat_y, ehat_z)
    body_vector = kite_frames.from_earth_to_body(DCM, earthfixed_vector)
    return body_vector

def from_body_to_earthfixed(body_vector, q_upper, q_lower):
    # todo: remove this when Rachel is done with verification testing.
    [ehat_x, ehat_y, ehat_z] = get_body_axes(q_upper, q_lower)
    DCM = cas.horzcat(ehat_x, ehat_y, ehat_z)
    earthfixed_vector = kite_frames.from_body_to_earth(DCM, body_vector)
    return earthfixed_vector

def get_inverse_equivalence_matrix(tether_length):
    # todo: remove this when Rachel is done with verification testing.
    # equivalent forces at upper node = [a, b, c]
    # equivalent forces at lower node = [d, e, f]
    # total forces = [Fx, Fy, Fz]
    # total moment = [Mx, My, 0]

    # a + d = Fx
    # b + e = Fy
    # c + f = Fz
    # (L/2) (b - e) = Mx
    # (L/2) (a - d) = My
    # c - f = 0

    # A [a, b, c, d, e, f].T = [Fx, Fy, Fz, Mx, My, 0].T
    # [a, b, c, d, e, f].T = Ainv [Fx, Fy, Fz, Mx, My, 0].T

    L = tether_length

    argument_stack = [[0.5, 0., 0., 0., 1. / L, 0.],
                      [0., 0.5, 0., 1. / L, 0., 0.],
                      [0., 0., 0.5, 0., 0., 0.5],
                      [0.5, 0., 0., 0., -1. / L, 0.],
                      [0., 0.5, 0., -1. / L, 0., 0.],
                      [0., 0., 0.5, 0., 0., -0.5]]
    if isinstance(L, (int, float)):
        Ainv = np.matrix(argument_stack)
    elif isinstance(L, (cas.DM, cas.SX, cas.MX)):
        Ainv = cas.vertcat(*[
            cas.horzcat(*row)
            for row in argument_stack
        ])
    else:
        message = 'unfamiliar type of tether length input (' + str(type(L)) + ')'
        print_op.log_and_raise_error(message)

    return Ainv



def get_multielement_segment_drag_fun(n_elements, parameters, element_drag_fun):
    info_to_add_to_applied_params_dict = {}
    segment_info_sym = cas.SX.sym('info_sym', tether_element.get_size_of_element_info_column())
    segment_info_dict = tether_element.unpack_element_info_column(segment_info_sym)

    combined_info = []
    for element in range(n_elements):
        elem_info = tether_element.get_info_column_linear_division_of_tether_segment(segment_info_dict, element, n_elements)
        combined_info = cas.horzcat(combined_info, elem_info)

    drag_map = element_drag_fun.map(n_elements, 'serial')
    combined_drag = drag_map(combined_info, parameters)
    segment_drag_fun = cas.Function('segment_drag_fun', [segment_info_sym, parameters], [combined_drag])
    return segment_drag_fun, info_to_add_to_applied_params_dict


def get_segment_drag(n_elements, upper_node, variables, parameters, architecture, element_drag_fun=None, combined_drag_fun=None):

    q_top, q_bottom, dq_top, dq_bottom = tether_element.get_upper_and_lower_pos_and_vel(variables, upper_node, architecture)
    diam = tether_element.get_element_diameter(variables, upper_node, architecture)
    segment_info_dict = {'q_upper': q_top,
                         'q_lower': q_bottom,
                         'dq_upper': dq_top,
                         'dq_lower': dq_bottom,
                         'diameter': diam}
    segment_info_columnized = tether_element.columnize_element_info(unpacked_as_dict=segment_info_dict)
    if combined_drag_fun is None:
        combined_drag_fun, _ = get_multielement_segment_drag_fun(n_elements, parameters, element_drag_fun=element_drag_fun)

    return combined_drag_fun(segment_info_columnized, parameters)

def get_multielement_drag_force_distribution_fun(n_elements, parameters, segment_drag_fun):
    info_to_add_to_applied_params_dict = {}
    info_sym = cas.SX.sym('info_sym', tether_element.get_size_of_element_info_column())
    combined_drag = segment_drag_fun(info_sym, parameters)

    # integration step size
    ds = 1.0/n_elements

    # integration grid (midpoint rule)
    s_grid = np.linspace(0.5*ds, 1 - 0.5*ds, n_elements)

    # numerical evaluation of analytic drag force expressions
    # distributes the force according to the distance of each element from the segment endpoints
    force_upper = sum([s_grid[k]*combined_drag[:, k] for k in range(n_elements)])
    force_lower = sum([(1-s_grid[k])*combined_drag[:, k] for k in range(n_elements)])

    node_forces = columnize_node_forces(force_upper=force_upper, force_lower=force_lower)
    drag_distribution_fun = cas.Function('drag_distribution_fun', [info_sym, parameters], [node_forces])
    return drag_distribution_fun, info_to_add_to_applied_params_dict


def get_distributed_segment_forces(n_elements, variables, upper_node, architecture, element_drag_fun, parameters):

    q_top, q_bottom, dq_top, dq_bottom = tether_element.get_upper_and_lower_pos_and_vel(variables, upper_node, architecture)
    diam = tether_element.get_element_diameter(variables, upper_node, architecture)
    segment_info_dict = {'q_upper': q_top,
                         'q_lower': q_bottom,
                         'dq_upper': dq_top,
                         'dq_lower': dq_bottom,
                         'diameter': diam}
    segment_info_columnized = tether_element.columnize_element_info(unpacked_as_dict=segment_info_dict)

    segment_drag_fun, _ = get_multielement_segment_drag_fun(n_elements, parameters, element_drag_fun)
    drag_distribution_fun, _ = get_multielement_drag_force_distribution_fun(n_elements, parameters, segment_drag_fun)

    columnized_node_forces = drag_distribution_fun(segment_info_columnized, parameters)

    node_forces_dict = unpack_node_force_column(columnized_node_forces)
    return node_forces_dict['lower'], node_forces_dict['upper']


def get_trivial_segment_drag_fun(atmos, wind, parameters):
    info_sym = cas.SX.sym('info_sym', tether_element.get_size_of_element_info_column())

    unpacked = tether_element.unpack_element_info_column(info_sym)
    q_upper = unpacked['q_upper']
    q_lower = unpacked['q_lower']
    dq_upper = unpacked['dq_upper']
    dq_lower = unpacked['dq_lower']
    diam = unpacked['diameter']

    length = vect_op.norm(q_upper - q_lower)
    q_average = 0.5 * (q_upper + q_lower)
    dq_average = 0.5 * (dq_upper + dq_lower)
    rho = atmos.get_density(q_average[2])

    u_a = wind.get_velocity(q_average[2]) - dq_average

    cd = parameters['theta0','tether','cd']
    info_to_add_to_applied_params_dict = {'params.tether.cd': cd}

    drag_force = cd * 0.5 * rho * vect_op.smooth_norm(u_a, 1e-6) * u_a * diam * length

    segment_drag_fun = cas.Function('segment_drag_fun', [info_sym, parameters], [drag_force])
    return segment_drag_fun, info_to_add_to_applied_params_dict


def get_half_half_drag_force_distribution_fun(parameters, segment_drag_fun):
    info_to_add_to_applied_params_dict = {}
    info_sym = cas.SX.sym('info_sym', tether_element.get_size_of_element_info_column())
    segment_drag = segment_drag_fun(info_sym, parameters)
    force_upper = segment_drag / 2.
    force_lower = segment_drag / 2.
    node_forces = columnize_node_forces(force_upper=force_upper, force_lower=force_lower)
    drag_distribution_fun = cas.Function('drag_distribution_fun', [info_sym, parameters], [node_forces])
    return drag_distribution_fun, info_to_add_to_applied_params_dict


def get_trivial_segment_forces(upper_node, architecture, variables, parameters, atmos=None, wind=None, segment_drag_fun=None, drag_distribution_fun=None):

    columnized_node_forces = cas.DM.zeros((6, 1))

    if drag_distribution_fun is not None:
        pass
    else:
        if segment_drag_fun is not None:
            pass
        elif (atmos is not None) and (wind is not None):
            segment_drag_fun, _ = get_trivial_segment_drag_fun(atmos, wind, parameters)
        else:
            message = 'not enough information available to get the trivial segment forces'
            print_op.log_and_raise_error(message)
        drag_distribution_fun, _ = get_half_half_drag_force_distribution_fun(parameters, segment_drag_fun)

    if upper_node in architecture.kite_nodes:
        q_upper, q_lower, dq_upper, dq_lower = tether_element.get_upper_and_lower_pos_and_vel(variables, upper_node,
                                                                                       architecture)
        diam = tether_element.get_element_diameter(variables, upper_node, architecture)

        columnized_info = tether_element.columnize_element_info(q_upper=q_upper, q_lower=q_lower, dq_upper=dq_upper, dq_lower=dq_lower, diameter=diam)
        columnized_node_forces = drag_distribution_fun(columnized_info, parameters)

    node_forces_dict = unpack_node_force_column(columnized_node_forces)
    return [node_forces_dict['lower'], node_forces_dict['upper']]


def get_kite_only_segment_drag_fun(atmos, parameters, reynolds_fun=None, cd_tether_fun=None):
    info_to_add_to_applied_params_dict = {}
    info_sym = cas.SX.sym('info_sym', tether_element.get_size_of_element_info_column(kite_only=True))

    unpacked = tether_element.unpack_element_info_column(info_sym, kite_only=True)
    q_upper = unpacked['q_upper']
    q_lower = unpacked['q_lower']
    diam = unpacked['diameter']
    ehat_1 = unpacked['ehat_1']
    ehat_3 = unpacked['ehat_3']
    alpha = unpacked['alpha']
    kite_dyn_pressure = unpacked['kite_dynamic_pressure']
    air_velocity = unpacked['air_velocity']

    d_hat = cas.cos(alpha) * ehat_1 + cas.sin(alpha) * ehat_3
    length = vect_op.norm(q_upper - q_lower)

    if reynolds_fun is not None:
        q_local = (q_upper + q_lower) / 2
        re_number = reynolds_fun(q_local, air_velocity, diam, parameters)
    else:
        re_number = tether_reynolds.get_reynolds_number(atmos, diam=diam, ua_local=air_velocity, q_upper=q_upper,
                                                    q_lower=q_lower)

    cd_tether = cd_tether_fun(re_number, parameters)

    d_mag = (1. / 4.) * cd_tether * kite_dyn_pressure * diam * length

    drag_force = d_mag * d_hat
    segment_drag_fun = cas.Function('segment_drag_fun', [info_sym, parameters], [drag_force])
    return segment_drag_fun, info_to_add_to_applied_params_dict

def get_kite_only_drag_force_distribution_fun(parameters, segment_drag_fun):
    info_to_add_to_applied_params_dict = {}
    info_sym = cas.SX.sym('info_sym', tether_element.get_size_of_element_info_column(kite_only=True))
    segment_drag = segment_drag_fun(info_sym, parameters)
    force_upper = segment_drag
    force_lower = cas.DM.zeros((3, 1))
    node_forces = columnize_node_forces(force_upper=force_upper, force_lower=force_lower)
    drag_distribution_fun = cas.Function('drag_distribution_fun', [info_sym, parameters], [node_forces])
    return drag_distribution_fun, info_to_add_to_applied_params_dict


def get_kite_only_segment_forces(upper_node, architecture, variables, parameters, outputs, atmos=None, segment_drag_fun=None, cd_tether_fun=None, reynolds_fun=None):

    force_lower = cas.DM.zeros((3, 1))
    force_upper = cas.DM.zeros((3, 1))

    if segment_drag_fun is not None:
        pass
    elif (atmos is not None) and (cd_tether_fun is not None):
        segment_drag_fun, _ = get_kite_only_segment_drag_fun(atmos, parameters, reynolds_fun=reynolds_fun, cd_tether_fun=cd_tether_fun)
    else:
        message = 'not enough information available to get the trivial segment forces'
        print_op.log_and_raise_error(message)


    if upper_node in architecture.kite_nodes:

        kite = upper_node

        ehat_1 = outputs['aerodynamics']['ehat_chord' + str(kite)]
        ehat_3 = outputs['aerodynamics']['ehat_span' + str(kite)]
        alpha = outputs['aerodynamics']['alpha' + str(kite)]
        kite_dyn_pressure = outputs['aerodynamics']['dyn_pressure' + str(kite)]
        air_velocity = outputs['aerodynamics']['air_velocity' + str(kite)]

        q_upper, q_lower, dq_upper, dq_lower = tether_element.get_upper_and_lower_pos_and_vel(variables, upper_node,
                                                                                       architecture)
        diam = tether_element.get_element_diameter(variables, upper_node, architecture)

        columnized_info = tether_element.columnize_element_info(q_upper=q_upper, q_lower=q_lower, dq_upper=dq_upper, dq_lower=dq_lower, diameter=diam, ehat_1=ehat_1, ehat_3=ehat_3, alpha=alpha, air_velocity=air_velocity, kite_dynamic_pressure=kite_dyn_pressure, kite_only=True)
        force_upper = segment_drag_fun(columnized_info, parameters)

    return force_lower, force_upper

def get_segment_reynolds_number(variables, atmos, wind, upper_node, architecture):
    diam = tether_element.get_element_diameter(variables, upper_node, architecture)
    q_upper, q_lower, dq_upper, dq_lower = tether_element.get_upper_and_lower_pos_and_vel(variables, upper_node, architecture)
    ua = tether_element.get_uapp(q_upper, q_lower, dq_upper, dq_lower, wind)
    re_number = tether_reynolds.get_reynolds_number(atmos, diam=diam, ua_local=ua, q_upper=q_upper, q_lower=q_lower)
    return re_number
