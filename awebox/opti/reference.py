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

import numpy as np

import awebox.tools.vector_operations as vect_op

from awebox.logger.logger import Logger as awelogger

def get_reference(nlp, model, V_init, options):

    # -------------------
    # generate tracking reference
    # -------------------
    awelogger.logger.info('generate tracking reference...')
    nk = nlp.n_k
    if nlp.discretization == 'direct_collocation':
        direct_collocation = True
        d = nlp.d
    else:
        direct_collocation = False

    V_ref = copy.deepcopy(V_init)

    for k in range(nk):

        # first node
        V_ref['xd',k,'q10'] += get_stagger_distance(options, model, V_init['xd',k,'q10'], np.zeros(3), 1, 0)
        # other nodes
        for n in range(2, model.architecture.number_of_nodes):
            parent = model.architecture.parent_map[n]
            grandparent = model.architecture.parent_map[parent]
            V_ref['xd',k,'q' + str(n) + str(parent)] += get_stagger_distance(options, model,
                                                                        V_init['xd', k, 'q' + str(n) + str(parent)],
                                                                        V_init['xd', k, 'q' + str(parent) + str(grandparent)],
                                                                        n, parent)


        if direct_collocation:

            for j in range(d):

                # first node
                V_ref['coll_var',k,j, 'xd','q10'] += get_stagger_distance(options, model, V_init['coll_var',k,j, 'xd','q10'], np.zeros(3), 1, 0)
                # other nodes
                for n in range(2, model.architecture.number_of_nodes):
                    parent = model.architecture.parent_map[n]
                    grandparent = model.architecture.parent_map[parent]
                    V_ref['coll_var',k,j, 'xd','q' + str(n) + str(parent)] += get_stagger_distance(options, model,
                                                                                V_init['coll_var',k,j, 'xd','q' + str(n) + str(parent)],
                                                                                V_init['coll_var',k,j, 'xd','q' + str(parent) + str(grandparent)],
                                                                                n, parent)

    return V_ref

def get_stagger_distance(options, model, q_init, q_parent, n, parent):

    ehat_l_init = vect_op.normalize(q_init - q_parent)

    scale_stagger = model.scaling['xd']['q'+str(n)+str(parent)]
    stagger = options['tracking']['stagger_distance'] / scale_stagger

    if parent == 0:
        q_ref = stagger * ehat_l_init / 2.
    else:
        q_ref = stagger * ehat_l_init

    return q_ref
