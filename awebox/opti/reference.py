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
makes a reference path based on the initialization path
_python-3.5 / casadi-3.4.5
- _author: rachel leuthold, jochem de schutter, thilo bronnenmeyer (alu-fr, 2017/18)
'''

import copy
import pdb
import casadi.tools as cas
import numpy as np

import awebox.tools.vector_operations as vect_op
import awebox.tools.print_operations as print_op
import awebox.tools.struct_operations as struct_op

from awebox.logger.logger import Logger as awelogger

def get_reference(nlp, model, V_init, options):

    # -------------------
    # generate tracking reference
    # -------------------

    nk = nlp.n_k
    if nlp.discretization == 'direct_collocation':
        direct_collocation = True
        d = nlp.d
    else:
        direct_collocation = False

    V_ref = copy.deepcopy(V_init)

    for k in range(nk):

        # first node
        V_ref['x', k, 'q10'] += get_stagger_distance(options, model, V_init['x', k, 'q10'], np.zeros(3), 1, 0)
        # other nodes
        for n in range(2, model.architecture.number_of_nodes):
            parent = model.architecture.parent_map[n]
            grandparent = model.architecture.parent_map[parent]
            V_ref['x', k, 'q' + str(n) + str(parent)] += get_stagger_distance(options, model,
                                                                        V_init['x', k, 'q' + str(n) + str(parent)],
                                                                        V_init['x', k, 'q' + str(parent) + str(grandparent)],
                                                                        n, parent)


        if direct_collocation:

            for j in range(d):

                # first node
                V_ref['coll_var', k, j, 'x', 'q10'] += get_stagger_distance(options, model,
                                                                            V_init['coll_var', k, j, 'x', 'q10'],
                                                                            np.zeros(3), 1, 0)
                # other nodes
                for n in range(2, model.architecture.number_of_nodes):
                    parent = model.architecture.parent_map[n]
                    grandparent = model.architecture.parent_map[parent]
                    V_ref['coll_var', k, j, 'x', 'q' + str(n) + str(parent)] += get_stagger_distance(options, model,
                                                                                V_init['coll_var', k, j, 'x', 'q' + str(n) + str(parent)],
                                                                                V_init['coll_var', k, j, 'x', 'q' + str(parent) + str(grandparent)],
                                                                                n, parent)

    return V_ref


def check_reference(options, V_ref, p_fix_num, nlp):

    orders_of_magnitude = options['initialization']['check_scaling']['orders_of_magnitude']
    zero_value_threshold = options['initialization']['check_scaling']['zero_value_threshold']

    check_reference_scaled_magnitudes(V_ref, orders_of_magnitude=orders_of_magnitude, zero_value_threshold=zero_value_threshold)
    check_reference_feasibility(options, V_ref, p_fix_num, nlp=nlp)

    return None


def check_reference_scaled_magnitudes(V_ref, orders_of_magnitude=1, zero_value_threshold=1.e-8):

    desirable = 1.
    minimum = desirable * 10.**(-1. * orders_of_magnitude)
    maximum = desirable * 10.**(+1. * orders_of_magnitude)

    larger_than_max = []
    smaller_than_min = []
    for idx in range(V_ref.shape[0]):
        local_val = float(V_ref.cat[idx])
        local_is_zero = np.abs(local_val) < zero_value_threshold
        local_larger_than_max = np.abs(local_val) > maximum
        local_smaller_than_min = (np.abs(local_val) < minimum) and not local_is_zero
        larger_than_max += [local_larger_than_max]
        smaller_than_min += [local_smaller_than_min]

    warning_message_start = 'some (scaled) reference values are more than '
    warning_message_start += str(orders_of_magnitude) + ' orders of magnitude '
    warning_message_end = ' than the goal value (' + str(desirable) + '). you may want '
    warning_message_end += 'to consider adjusting either the initialization or the scaling of:'

    if any(larger_than_max):
        message = warning_message_start + 'larger' + warning_message_end
        print_op.base_print(message, level='warning')

        dict_larger_than_max = {}
        for idx in range(len(larger_than_max)):
            if larger_than_max[idx]:
                dict_larger_than_max[V_ref.labels()[idx]] = V_ref.cat[idx]
        print_op.print_dict_as_table(dict_larger_than_max, level='warning')

    if any(smaller_than_min):
        message = warning_message_start + 'smaller' + warning_message_end
        print_op.base_print(message, level='warning')
        dict_smaller_than_min = {}

        for idx in range(len(smaller_than_min)):
            if smaller_than_min[idx]:
                dict_smaller_than_min[V_ref.labels()[idx]] = V_ref.cat[idx]
        print_op.print_dict_as_table(dict_smaller_than_min, level='warning')

    return None


def check_reference_feasibility(solver_options, V_ref, p_fix_num, nlp):
    path_constraint_theshold = solver_options['initialization']['check_feasibility']['path_constraint_threshold']
    raise_exception = solver_options['initialization']['check_feasibility']['raise_exception']

    for cstr_type in ['ineq']:

        name_list = nlp.ocp_cstr_list.get_name_list(cstr_type)
        cstr_fun = nlp.ocp_cstr_list.get_function(solver_options, nlp.V, nlp.P, cstr_type)
        cstr_vals = cstr_fun(V_ref, p_fix_num)

        is_violated = []
        for idx in range(cstr_vals.shape[0]):

            local_violation_is_reported = cstr_vals[idx] > path_constraint_theshold
            # if (cstr_type == 'eq' and 'dynamics' in name_list[idx]):
            #     local_violation_is_reported = False  # untrue, but
            # else:
            #     if cstr_type == 'ineq':
            #         local_violation_is_reported = cstr_vals[idx] > path_constraint_theshold
            #     elif cstr_type == 'eq':
            #         local_violation_is_reported = cstr_vals[idx]**2. > path_constraint_theshold**2.

            is_violated += [local_violation_is_reported]

        if any(is_violated):
            message = cstr_type + ' constraints violated at:'
            print_op.base_print(message, level='warning')

            dict_is_violated = {}
            for idx in range(len(is_violated)):
                if is_violated[idx]:
                    dict_is_violated[name_list[idx]] = cstr_vals[idx]
            print_op.print_dict_as_table(dict_is_violated, level='warning')

            if raise_exception:
                print_op.log_and_raise_error('initial guess violates path constraints')

    return None


def get_stagger_distance(options, model, q_init, q_parent, n, parent):

    ehat_l_init = vect_op.normalize(q_init - q_parent)

    if parent == 0:
        stagger_vector_si = options['tracking']['stagger_distance'] * ehat_l_init / 2.
    else:
        stagger_vector_si = options['tracking']['stagger_distance'] * ehat_l_init

    stagger_vector_scaled = struct_op.var_si_to_scaled('x', 'q' + str(n) + str(parent), stagger_vector_si, model.scaling)

    return stagger_vector_scaled