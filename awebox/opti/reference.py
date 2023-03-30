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

    if options['initialization']['use_reference_to_check_scaling']:
        check_order_of_magnitudes_of_reference_values(V_ref, model_scaling=model.scaling)

    return V_ref


def check_order_of_magnitudes_of_reference_values(V_ref, orders_of_magnitude_threshold=1., zero_threshold=1e-8, model_scaling=None):

    desirable = 1.
    minimum = desirable / 10.**orders_of_magnitude_threshold
    maximum = desirable * 10.**orders_of_magnitude_threshold

    unreasonably_scaled = {}

    for vdx in range(V_ref.cat.shape[0]):
        local_value = V_ref.cat[vdx]
        has_zero_value = local_value.is_zero() or (abs(local_value) < zero_threshold)
        within_desirable_range = (abs(local_value) > minimum) and (abs(local_value) < maximum)
        if not (has_zero_value or within_desirable_range):
            unreasonably_scaled[str(V_ref.getCanonicalIndex(vdx))] = local_value

    some_variables_unreasonably_scaled = len(unreasonably_scaled.keys()) > 0
    if some_variables_unreasonably_scaled:
        message = "some of the reference values used here, indicate that variable scaling could be improved for better performance"
        print_op.base_print(message, level='warning')
        print_op.print_dict_as_table(unreasonably_scaled, level='warning')

    return None


def get_stagger_distance(options, model, q_init, q_parent, n, parent):

    ehat_l_init = vect_op.normalize(q_init - q_parent)

    if parent == 0:
        stagger_vector_si = options['tracking']['stagger_distance'] * ehat_l_init / 2.
    else:
        stagger_vector_si = options['tracking']['stagger_distance'] * ehat_l_init

    stagger_vector_scaled = struct_op.var_si_to_scaled('x', 'q' + str(n) + str(parent), stagger_vector_si, model.scaling)

    return stagger_vector_scaled