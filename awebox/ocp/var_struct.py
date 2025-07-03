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
keep the variable construction separate,
so that non-sequential code can be used non-sequentially
python-3.5 / casadi-3.4.5
- authors: elena malz 2016
           rachel leuthold, jochem de schutter alu-fr 2017-20, jakob harzer 2024
'''

import casadi.tools as cas
import awebox.tools.struct_operations as struct_op
from awebox.mdl.model import Model
import awebox.ocp.collocation as collocation
import awebox.tools.print_operations as print_op


def setup_nlp_v(nlp_options: dict, model: Model, coll_instance: collocation.Collocation=None):
    """
    Generate the variable structure for the NLP.

    :param nlp_options: dictionary of the nlp options e.g. `options['nlp']`
    :param model: system model
    :param collocation:
    :return: a casadi struct containing all variables for the NLP
    """

    # extract necessary inputs
    variables_dict = model.variables_dict
    nk = nlp_options['n_k']
    multiple_shooting = (nlp_options['discretization'] == 'multiple_shooting')
    direct_collocation = (nlp_options['discretization'] == 'direct_collocation')

    # define interval struct entries for controls and states
    entry_tuple = (
        cas.entry('x', repeat = [nk+1], struct = variables_dict['x']),
        )

    # add additional variables according to provided options
    if multiple_shooting or nlp_options['collocation']['u_param'] == 'zoh':
        entry_tuple += (
            cas.entry('u',  repeat = [nk],   struct = variables_dict['u']),
        )

        # add state derivative variables at interval
        entry_tuple += (
            cas.entry('xdot', repeat = [nk], struct= variables_dict['xdot']),
        )

        # add algebraic variables at shooting nodes for constraint evaluation
        entry_tuple += (cas.entry('z', repeat = [nk],   struct= variables_dict['z']),)

    if direct_collocation:

        # add collocation node variables
        if coll_instance is None:
            message = 'a None instance of Collocation was passed to the NLP variable structure generator'
            print_op.log_and_raise_error(message)

        d = nlp_options['collocation']['d'] # interpolating polynomial order
        coll_var = coll_instance.get_collocation_variables_struct(variables_dict, nlp_options['collocation']['u_param'])
        entry_tuple += (cas.entry('coll_var', struct = coll_var, repeat= [nk,d]),)

    # add global entries
    theta = construct_theta(nlp_options, variables_dict)  # build the parameter struct (t_f, ...)

    # when the global variables are before the discretized variables, it leads to prettier kkt matrix spy plots
    entry_list = [
        cas.entry('theta', struct = theta),
        cas.entry('phi',   struct = model.parameters_dict['phi']),
        entry_tuple
    ]

    # add SAM variables if necessary
    if nlp_options['SAM']['use']:
        invariant_names_to_constrain = [key for key in model.outputs_dict['invariants'].keys() if key.startswith(tuple(['c', 'dc', 'orthonormality']))]
        entry_list += [
                       cas.entry('x_macro', struct=model.variables_dict['x'], repeat=[2]),
                       cas.entry('x_macro_coll', struct=model.variables_dict['x'], repeat=[nlp_options['SAM']['d']]),
                       cas.entry('v_macro_coll', struct=model.variables_dict['x'], repeat=[nlp_options['SAM']['d']]),
                       cas.entry('x_micro_minus', struct=model.variables_dict['x'], repeat=[nlp_options['SAM']['d']]),
                       cas.entry('x_micro_plus', struct=model.variables_dict['x'], repeat=[nlp_options['SAM']['d']]),
                       cas.entry('lam_SAM', shape=cas.vertcat(*model.outputs['invariants',invariant_names_to_constrain]).shape, repeat = [nlp_options['SAM']['d']+1]),
                       ]

    # generate structure
    return cas.struct_symMX(entry_list)


def construct_theta(nlp_options: dict, variables_dict: dict) -> cas.struct_symSX:
    """ Build the parameter struct for the NLP. The contents depend on the discretization method,
    especially the time-scaling parameter 't_f'.

    :param nlp_options: dictionary of the nlp options e.g. `options['nlp']`
    :param variables_dict: dictionary of the system variables e.g. `model.variables_dict`
    :return: a casadi struct containing the parameter variables for the NLP ('t_f', ....)
    """

    entry_list = []
    for name in list(variables_dict['theta'].keys()):
        if name == 't_f':
            # timescaling parameters? Their number depends on the discretization
            entry_list.append(cas.entry('t_f', shape=(get_number_of_tf(nlp_options), 1)))

        else:
            # other parameters? just copy
            entry_list.append(cas.entry(name, shape=variables_dict['theta'][name].shape))

    return cas.struct_symSX(entry_list)


def get_number_of_tf(nlp_options) -> int:
    """ Determine the number of time-scaling parameters in the NLP. This depends on the discretization method."""
    if nlp_options['SAM']['use']:
        n_tf = nlp_options['SAM']['d'] + 1  # d_SAM microintegrations regions in reel-out + reel-in regions
    elif nlp_options['phase_fix'] == 'single_reelout':
        n_tf = 2  # in this case, the reel-out and reel-in phases are treated separately
    else:
        n_tf = 1  # no seperation of reel-out and reel-in phases
    return n_tf
