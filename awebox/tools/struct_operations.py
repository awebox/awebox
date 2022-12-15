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
file to provide structure operations to the awebox,
_python-3.5 / casadi-3.4.5
- author: thilo bronnenmeyer, jochem de schutter, rachel leuthold, 2017-20
'''
import pdb

import casadi.tools as cas
import numpy as np

import operator

import copy
from functools import reduce
from awebox.logger.logger import Logger as awelogger
import awebox.tools.print_operations as print_op
import awebox.tools.vector_operations as vect_op
from itertools import chain

def subkeys(casadi_struct, key):

    if key in casadi_struct.keys():
        indices = np.array(casadi_struct.f[key])
        number_index = indices.shape[0]

        subkeys = set()
        for idx in range(number_index):
            canonical = casadi_struct.getCanonicalIndex(indices[idx])
            new_key = canonical[1]
            subkeys.add(new_key)

        subkey_list = sorted(subkeys)
    else:
        subkey_list = []

    return subkey_list

def count_shooting_nodes(nlp_options):
    return nlp_options['n_k']

def get_shooting_vars(nlp_options, V, P, Xdot, model):

    shooting_nodes = count_shooting_nodes(nlp_options)

    shooting_vars = []
    for kdx in range(shooting_nodes):
        var_at_time = get_variables_at_time(nlp_options, V, Xdot, model.variables, kdx)
        shooting_vars = cas.horzcat(shooting_vars, var_at_time)

    return shooting_vars

def get_shooting_params(nlp_options, V, P, model):

    shooting_nodes = count_shooting_nodes(nlp_options)

    parameters = model.parameters
    coll_params = cas.repmat(parameters(cas.vertcat(P['theta0'], V['phi'])), 1, (shooting_nodes))
    return coll_params



def get_coll_vars(nlp_options, V, P, Xdot, model):

    n_k = nlp_options['n_k']
    d = nlp_options['collocation']['d']

    # construct list of all collocation node variables and parameters
    coll_vars = []
    for kdx in range(n_k):
        for ddx in range(d):
            var_at_time = get_variables_at_time(nlp_options, V, Xdot, model.variables, kdx, ddx)
            coll_vars = cas.horzcat(coll_vars, var_at_time)

    return coll_vars


def get_coll_params(nlp_options, V, P, model):

    n_k = nlp_options['n_k']
    d = nlp_options['collocation']['d']
    N_coll = n_k * d # collocation points

    parameters = model.parameters
    coll_params = cas.repmat(parameters(cas.vertcat(P['theta0'], V['phi'])), 1, N_coll)
    return coll_params


def get_ms_vars(nlp_options, V, P, Xdot, model):

    n_k = nlp_options['n_k']

    # construct list of all multiple-shooting node variables and parameters
    ms_vars = []
    for kdx in range(n_k):
        var_at_time = get_variables_at_time(nlp_options, V, Xdot, model.variables, kdx)
        ms_vars = cas.horzcat(ms_vars, var_at_time)

    return ms_vars


def get_ms_params(nlp_options, V, P, Xdot, model):
    n_k = nlp_options['n_k']
    N_ms = n_k  # collocation points

    parameters = model.parameters

    ms_params = cas.repmat(parameters(cas.vertcat(P['theta0'], V['phi'])), 1, N_ms)

    return ms_params


def get_algebraics_at_time(nlp_options, V, model_variables, kdx, ddx=None):

    var_type = 'z'

    if (ddx is None):
        if var_type in list(V.keys()):
            return V[var_type, kdx]
        else:
            return V['coll_var', kdx, 0, var_type]
    else:
        return V['coll_var', kdx, ddx, var_type]


def get_states_at_time(nlp_options, V, model_variables, kdx, ddx=None):

    var_type = 'x'

    direct_collocation = (nlp_options['discretization'] == 'direct_collocation')
    at_control_node = (ddx is None)

    if at_control_node:
        return V[var_type, kdx]
    elif direct_collocation:
        return V['coll_var', kdx, ddx, var_type]


def get_controls_at_time(nlp_options, V, model_variables, kdx, ddx=None):

    var_type = 'u'

    multiple_shooting = (nlp_options['discretization'] == 'multiple_shooting')
    direct_collocation = (nlp_options['discretization'] == 'direct_collocation')

    piecewise_constant_controls = not (nlp_options['collocation']['u_param'] == 'poly')
    at_control_node = (ddx is None)
    before_last_node = kdx < nlp_options['n_k']

    if direct_collocation and piecewise_constant_controls and before_last_node:
        return V[var_type, kdx]

    elif direct_collocation and piecewise_constant_controls and (not before_last_node):
        return V[var_type, -1]

    elif direct_collocation and (not piecewise_constant_controls) and at_control_node:
        return V['coll_var', kdx, 0, var_type]

    elif direct_collocation and (not piecewise_constant_controls):
        return V['coll_var', kdx, ddx, var_type]

    elif multiple_shooting:
        return V[var_type, kdx]



def get_derivs_at_time(nlp_options, V, Xdot, model_variables, kdx, ddx=None):

    var_type = 'xdot'

    at_control_node = (ddx is None)
    lifted_derivs = ('xdot' in list(V.keys()))
    passed_Xdot_is_meaningful = (Xdot is not None) and not (Xdot == Xdot(0.))

    if at_control_node and lifted_derivs:
        return V[var_type, kdx]
    elif at_control_node and passed_Xdot_is_meaningful:
        return Xdot['x', kdx]
    elif lifted_derivs and ('coll' in V.keys()):
        return V['coll', kdx, ddx, 'xdot']
    elif passed_Xdot_is_meaningful:
        return Xdot['coll_x', kdx, ddx]
    else:
        attempted_reassamble = []
        for idx in range(model_variables.shape[0]):
            can_index = model_variables.getCanonicalIndex(idx)
            local_variable_has_a_derivative = (can_index[0] == 'x')
            if local_variable_has_a_derivative():

                var_name = can_index[1]
                dim = can_index[2]

                deriv_name = 'd' + var_name
                deriv_name_in_states = deriv_name in subkeys(model_variables, 'x')
                deriv_name_in_controls = deriv_name in subkeys(model_variables, 'u')

                if at_control_node and deriv_name_in_states:
                    local_val = V['x', kdx, deriv_name, dim]
                elif at_control_node and deriv_name_in_controls:
                    local_val = V['u', kdx, deriv_name, dim]
                elif deriv_name_in_states and ('coll' in V.keys()):
                    local_val = V['coll', kdx, ddx, 'x', deriv_name, dim]
                elif deriv_name_in_controls and ('coll' in V.keys()):
                    local_val = V['coll', kdx, ddx, 'u', deriv_name, dim]
                else:
                    local_val = cas.DM.zeros((1, 1))

                attempted_reassamble = cas.vertcat(attempted_reassamble, local_val)
        return attempted_reassamble


def get_variables_at_time(nlp_options, V, Xdot, model_variables, kdx, ddx=None):

    var_list = []
    # make list of variables at specific time
    for var_type in model_variables.keys():

        if var_type == 'z':
            local_var = get_algebraics_at_time(nlp_options, V, model_variables, kdx, ddx)

        elif var_type == 'x':
            local_var = get_states_at_time(nlp_options, V, model_variables, kdx, ddx)

        elif var_type == 'u':
            local_var = get_controls_at_time(nlp_options, V, model_variables, kdx, ddx)

        elif var_type == 'theta':
            local_var = get_V_theta(V, nlp_options, kdx)

        elif var_type == 'xdot':
            local_var = get_derivs_at_time(nlp_options, V, Xdot, model_variables, kdx, ddx)

        var_list.append(local_var)

    var_at_time = model_variables(cas.vertcat(*var_list))

    return var_at_time



def get_variables_at_final_time(nlp_options, V, Xdot, model):

    multiple_shooting = (nlp_options['discretization'] == 'multiple_shooting')

    scheme = nlp_options['collocation']['scheme']
    direct_collocation = (nlp_options['discretization'] == 'direct_collocation')
    radau_collocation = (direct_collocation and scheme == 'radau')
    other_collocation = (direct_collocation and (not scheme == 'radau'))

    terminal_constraint = nlp_options['mpc']['terminal_point_constr']

    if radau_collocation and not terminal_constraint:
        var_at_time = get_variables_at_time(nlp_options, V, Xdot, model.variables, -1, -1)
    elif direct_collocation or multiple_shooting:
        var_at_time = get_variables_at_time(nlp_options, V, Xdot, model.variables, -1)
    else:
        message = 'unfamiliar discretization option chosen: ' + nlp_options['discretization']
        print_op.log_and_raise_error(message)

    return var_at_time

def get_parameters_at_time(nlp_options, P, V, Xdot, model_variables, model_parameters, kdx=None, ddx=None):
    param_list = []

    parameters = model_parameters

    for var_type in list(parameters.keys()):
        if var_type == 'phi':
            param_list.append(V[var_type])
        if var_type == 'theta0':
            param_list.append(P[var_type])
        if var_type == 'lin':
            linearized_vars = get_variables_at_time(nlp_options, V(P['lin']), Xdot, model_variables, kdx, ddx)
            param_list.append(linearized_vars)

    param_at_time = parameters(cas.vertcat(*param_list))

    return param_at_time

def get_var_ref_at_time(nlp_options, P, V, Xdot, model, kdx, ddx=None):
    V_from_P = V(P['p', 'ref'])
    var_at_time = get_variables_at_time(nlp_options, V_from_P, Xdot, model.variables, kdx, ddx)
    return var_at_time

def get_var_ref_at_final_time(nlp_options, P, V, Xdot, model):
    V_from_P = V(P['p', 'ref'])
    var_at_time = get_variables_at_final_time(nlp_options, V_from_P, Xdot, model)
    return var_at_time

def get_V_theta(V, nlp_numerics_options, k):

    nk = nlp_numerics_options['n_k']
    k = list(range(nk+1))[k]

    if V['theta','t_f'].shape[0] == 1:
        theta = V['theta']
    else:
        theta = []
        tf_index = V.f['theta','t_f']
        theta_index = V.f['theta']
        for idx in theta_index:
            if idx == tf_index[0] and k < round(nk * nlp_numerics_options['phase_fix_reelout']):
                theta.append(V.cat[idx])
            elif idx == tf_index[1] and k >= round(nk * nlp_numerics_options['phase_fix_reelout']) :
                theta.append(V.cat[idx])
            elif idx not in tf_index:
                theta.append(V.cat[idx])
        theta = cas.vertcat(*theta)

    return theta


def calculate_tf(params, V, k):

    nk = params['n_k']

    if params['phase_fix'] == 'single_reelout':
        if k < round(nk * params['phase_fix_reelout']):
            tf = V['theta', 't_f', 0]
        else:
            tf = V['theta', 't_f', 1]
    else:
        tf = V['theta', 't_f']

    return tf

def calculate_kdx(params, V, t):

    n_k = params['n_k']

    if params['phase_fix'] == 'single_reelout':
        k_reelout = round(n_k * params['phase_fix_reelout'])
        t_reelout = k_reelout*V['theta','t_f',0]/n_k
        if t <= t_reelout:
            kdx = int(n_k * t / V['theta','t_f',0])
            tau = t / V['theta', 't_f',0]*n_k - kdx
        else:
            kdx = int(k_reelout + int(n_k * (t - t_reelout) / V['theta','t_f',1]))
            tau = (t - t_reelout)/ V['theta','t_f',1]*n_k - (kdx-k_reelout)
    else:
        t = t % V['theta', 't_f', 0].full()[0][0]
        kdx = int(n_k * t / V['theta','t_f'])
        tau = t / V['theta', 't_f']*n_k - kdx

    if kdx == n_k:
        kdx = n_k - 1
        tau = 1.0

    return kdx, tau

def variables_si_to_scaled(model_variables, variables_si, scaling):

    variables_scaled = copy.deepcopy(variables_si)

    for idx in range(model_variables.shape[0]):
        canonical = model_variables.getCanonicalIndex(idx)
        var_type = canonical[0]
        var_name = canonical[1]
        kdx = canonical[2]

        variables_scaled[var_type, var_name, kdx] = var_si_to_scaled(var_type, var_name, variables_scaled.cat[idx], scaling)

    return variables_scaled

def variables_scaled_to_si(model_variables, variables_scaled, scaling):

    stacked = []
    for idx in range(model_variables.shape[0]):
        canonical = model_variables.getCanonicalIndex(idx)
        var_type = canonical[0]
        var_name = canonical[1]
        kdx = canonical[2]

        new = var_scaled_to_si(var_type, var_name, variables_scaled.cat[idx], scaling)
        stacked = cas.vertcat(stacked, new)

    variables_si = model_variables(stacked)
    return variables_si


def var_si_to_scaled(var_type, var_name, var_si, scaling):

    scaling_defined_for_variable = (var_type in scaling.keys()) and (var_name in scaling[var_type].keys())
    if scaling_defined_for_variable:

        scale = scaling[var_type][var_name]

        if scale.shape == (1, 1):

            use_unit_scaling = (scale == cas.DM(1.)) or (scale == 1.)
            if use_unit_scaling:
                return var_si
            else:
                var_scaled = var_si / scale
                if type(var_si) == np.ndarray:
                    var_scaled = var_scaled.full()
                return var_scaled

        else:
            matrix_factor = cas.inv(cas.diag(scale))
            return cas.mtimes(matrix_factor, var_si)

    else:
        return var_si


def var_scaled_to_si(var_type, var_name, var_scaled, scaling):

    scaling_defined_for_variable = (var_type in scaling.keys()) and (var_name in scaling[var_type].keys())
    if scaling_defined_for_variable:

        scale = scaling[var_type][var_name]

        if scale.shape == (1, 1):

            use_unit_scaling = (scale == cas.DM(1.)) or (scale == 1.)
            if use_unit_scaling:
                return var_scaled
            else:
                return var_scaled * scale
        else:
            matrix_factor = cas.diag(scale)
            return cas.mtimes(matrix_factor, var_scaled)

    else:
        return var_scaled


def get_distinct_V_indices(V):

    distinct_indices = set([])

    number_V_entries = V.shape[0]
    for edx in range(number_V_entries):
        index = V.getCanonicalIndex(edx)

        distinct_indices.add(index[:-1])

    return distinct_indices

def si_to_scaled(V_ori, scaling):
    V = copy.deepcopy(V_ori)

    distinct_V_indices = get_distinct_V_indices(V)
    for index in distinct_V_indices:

        if len(index) == 2:
            var_type = index[0]
            var_name = index[1]
            var_si = V[var_type, var_name]
            V[var_type, var_name] = var_si_to_scaled(var_type, var_name, var_si, scaling)

        elif len(index) == 3:
            var_type = index[0]
            kdx = index[1]
            var_name = index[2]
            var_si = V[var_type, kdx, var_name]
            V[var_type, kdx, var_name] = var_si_to_scaled(var_type, var_name, var_si, scaling)

        elif (len(index) == 5) and (index[0] == 'coll_var'):
            kdx = index[1]
            ddx = index[2]
            var_type = index[3]
            var_name = index[4]
            var_si = V['coll_var', kdx, ddx, var_type, var_name]
            V['coll_var', kdx, ddx, var_type, var_name] = var_si_to_scaled(var_type, var_name, var_si, scaling)
        else:
            message = 'unexpected variable found at canonical index: ' + str(index) + ' while scaling variables from si'
            print_op.log_and_raise_error(message)

    return V


def scaled_to_si(V_ori, scaling):
    V = copy.deepcopy(V_ori)

    distinct_V_indices = get_distinct_V_indices(V)
    for index in distinct_V_indices:

        if len(index) == 2:
            var_type = index[0]
            var_name = index[1]
            var_scaled = V[var_type, var_name]
            V[var_type, var_name] = var_scaled_to_si(var_type, var_name, var_scaled, scaling)

        elif len(index) == 3:
            var_type = index[0]
            kdx = index[1]
            var_name = index[2]
            var_scaled = V[var_type, kdx, var_name]
            V[var_type, kdx, var_name] = var_scaled_to_si(var_type, var_name, var_scaled, scaling)

        elif (len(index) == 5) and (index[0] == 'coll_var'):
            kdx = index[1]
            ddx = index[2]
            var_type = index[3]
            var_name = index[4]
            var_scaled = V['coll_var', kdx, ddx, var_type, var_name]
            V['coll_var', kdx, ddx, var_type, var_name] = var_scaled_to_si(var_type, var_name, var_scaled, scaling)
        else:
            message = 'unexpected variable found at canonical index: ' + str(index) + ' while scaling variables to si'
            print_op.log_and_raise_error(message)

    return V


def coll_slice_to_vec(coll_slice):

    coll_list = []

    for i in range(len(coll_slice)):
        coll_list.append(cas.vertcat(*coll_slice[i]))

    coll_vec = cas.vertcat(*coll_list)

    return coll_vec

def coll_slice_to_norm(coll_slice):

    slice_norm = []
    for i in range(len(coll_slice)):
        slice_norm.append(np.linalg.norm(coll_slice[i]))

    return slice_norm

def interval_slice_to_vec(interval_slice):

    interval_vec = cas.vertcat(*interval_slice)

    return interval_vec

def get_variable_type(model, name):

    if type(model) == dict:
        variables_dict = model

    elif model.type == 'Model':
        variables_dict = model.variables_dict

    for variable_type in set(variables_dict.keys()) - set(['xdot']):
        if name in list(variables_dict[variable_type].keys()):
            return variable_type

    if name in list(variables_dict['xdot'].keys()):
        return 'xdot'

    message = 'variable ' + name + ' not found in variables dictionary'
    print_op.log_and_raise_error(message)

    return None



def convert_return_status_string_to_number(return_string):

    numeric_dictionary = get_return_status_dictionary()

    for num in list(numeric_dictionary.keys()):
        if numeric_dictionary[num] == return_string:
            return num

    return 17

def convert_return_status_number_to_string(return_number):

    numeric_dictionary = get_return_status_dictionary()

    if return_number in list(numeric_dictionary.keys()):
        return numeric_dictionary[return_number]

    return 'Other'

def get_return_status_dictionary():
    # ipopt return status described here:
    # https: // www.coin - or.org / Ipopt / documentation / node36.html

    ret_stats = {}
    ret_stats[1] = 'Solve_Succeeded'
    ret_stats[2] = 'Solved_To_Acceptable_Level'
    ret_stats[3] = 'Feasible_Point_Found'
    ret_stats[4] = 'Infeasible_Problem_Detected'
    ret_stats[5] = 'Search_Direction_Becomes_Too_Small'
    ret_stats[6] = 'Diverging_Iterates'
    ret_stats[7] = 'User_Requested_Stop'
    ret_stats[8] = 'Maximum_Iterations_Exceeded'
    ret_stats[9] = 'Maximum_CpuTime_Exceeded'
    ret_stats[10] = 'Restoration_Failed'
    ret_stats[11] = 'Error_In_Step_Computation'
    ret_stats[12] = 'Invalid_Option'
    ret_stats[13] = 'Not_Enough_Degrees_Of_Freedom'
    ret_stats[14] = 'Invalid_Problem_Definition'
    ret_stats[15] = 'NonIpopt_Exception_Thrown'
    ret_stats[16] = 'Insufficient_Memory'
    ret_stats[17] = 'IPOPT_DOA'

    return ret_stats

def get_V_index(canonical):

    var_is_coll_var = (canonical[0] == 'coll_var')

    length = len(canonical)

    if var_is_coll_var:
        # coll_var, kdx, ddx, type, name

        var_type = canonical[3]
        kdx = canonical[1]
        ddx = canonical[2]
        name = canonical[4]
        dim = None

    else:
        var_type = canonical[0]
        dim = None
        kdx = None
        ddx = None

        if length == 4:
            kdx = canonical[1]
            ddx = canonical[2]
            name = canonical[3]

        elif length == 3:
            kdx = canonical[1]
            name = canonical[2]

        elif length == 2:
            name = canonical[1]

        else:
            message = 'unexpected (distinct) canonical_index handing'
            print_op.log_and_raise_error(message)

    return [var_is_coll_var, var_type, kdx, ddx, name, dim]

def construct_Xdot_struct(nlp_options, variables_dict):
    ''' Construct a symbolic structure for the
        discretized state derivatives.

    @param nlp_options - discretization options
    @param model - awebox model
    @return Vdot - discretized state derivatives sym. struct.
    '''

    # extract information
    nk = nlp_options['n_k']
    x = variables_dict['x']
    z = variables_dict['z']

    # derivatives at interval nodes
    entry_tuple = (cas.entry('x', repeat=[nk], struct=x),)

    # add derivatives on collocation nodes
    if nlp_options['discretization'] == 'direct_collocation':
        d = nlp_options['collocation']['d']
        entry_tuple += (cas.entry('coll_x', repeat=[nk,d], struct=x),)
        entry_tuple += (cas.entry('coll_z', repeat=[nk,d], struct=z),)

    # make new symbolic structure
    Xdot = cas.struct_symMX([entry_tuple])

    return Xdot

##
#  @brief Method to recursively generate a casadi structure out of a nested dict.
#  @param v A (possibly nested) dictionary
#  @return subdict_struct Casadi struct_symSX with same structure as v.
def generate_nested_dict_struct(v):

    # empty entry list
    entry_list = []

    # iterate over all dict values
    for k1, v1 in v.items():

        if isinstance(v1, dict):

            # if value is a dict, recursively generate subdict struct
            substruct = generate_nested_dict_struct(v1)
            entry_list.append(cas.entry(k1,struct= substruct))

        else:
            if isinstance(v1,float) or isinstance(v1,int):
                shape = (1,1)
            else:
                shape = v1.shape
            # append value to entry list
            entry_list.append(cas.entry(k1, shape= shape))

    # make overall structure
    subdict_struct = cas.struct_symSX(entry_list)

    return subdict_struct

def get_from_dict(data_dict, mapList):
    return reduce(operator.getitem, mapList, data_dict)

def set_in_dict(data_dict, mapList, value):
    get_from_dict(data_dict, mapList[:-1])[mapList[-1]] = value
    return None

def generate_nested_dict_keys(d):

    keys_list = []

    for k, v in d.items():
        if isinstance(v,dict) and k != 'parent_map':
            subkeys = generate_nested_dict_keys(v)
            for i in range(len(subkeys)):
                keys_list += [[k]+subkeys[i]]
        else:
            keys_list += [[k]]

    return keys_list

def initialize_nested_dict(d,keys):

    if len(keys) == 1:
        d[keys[0]] = None
    else:
        if keys[0] in list(d.keys()):
            d[keys[0]] = initialize_nested_dict(d[keys[0]],keys[1:])
        else:
            d[keys[0]] = initialize_nested_dict({},keys[1:])

    return d

def setup_warmstart_data(nlp, warmstart_solution_dict):

    options_in_keys = 'options' in warmstart_solution_dict.keys()
    if options_in_keys:
        nlp_discretization = warmstart_solution_dict['options']['nlp']['discretization']

    if options_in_keys and not (nlp.discretization == nlp_discretization):

        if nlp.discretization == 'multiple_shooting':

            # number of shooting intervals
            n_k = nlp.n_k

            # initialize and extract
            V_init_proposed = nlp.V(0.0)
            V_coll = warmstart_solution_dict['V_opt']
            Xdot_coll = warmstart_solution_dict['Xdot_opt']

            lam_x_proposed = nlp.V_bounds['ub'](0.0)
            lam_x_coll = V_coll(warmstart_solution_dict['opt_arg']['lam_x0'])

            lam_g_proposed = nlp.g(0.0)
            lam_g_coll = warmstart_solution_dict['g_opt'](warmstart_solution_dict['opt_arg']['lam_g0'])
            g_coll = warmstart_solution_dict['g_opt']

            # initialize regular variables
            for var_type in set(['x', 'theta', 'phi', 'xi', 'z', 'xdot']):
                V_init_proposed[var_type] = V_coll[var_type]
                lam_x_proposed[var_type] = lam_x_coll[var_type]

            if 'u' in list(V_coll.keys()):
                V_init_proposed['u'] = V_coll['u']
                lam_x_proposed['u'] = lam_x_coll['u']
            else:
                for i in range(n_k):
                    # note: this does not give the actual mean, implement with quadrature weights instead
                    V_init_proposed['u', i] = np.mean(cas.horzcat(*V_coll['coll_var', i, :, 'u']))
                    lam_x_proposed['u', i] = np.mean(cas.horzcat(*lam_x_coll['coll_var', i, :, 'u']))

            # initialize path constraint multipliers
            for i in range(n_k):
                lam_g_proposed['path', i] = lam_g_coll['path', i]

            # initialize periodicity multipliers
            if 'periodic' in list(lam_g_coll.keys()) and 'periodic' in list(lam_g_proposed.keys()):
                lam_g_proposed['periodic'] = lam_g_coll['periodic']

            # initialize continuity multipliers
            lam_g_proposed['continuity'] = lam_g_coll['continuity']

        else:
            raise ValueError('Warmstart from multiple shooting to collocation not supported')

    else:

        V_init_proposed = warmstart_solution_dict['V_opt']
        if 'lam_x0' in warmstart_solution_dict['opt_arg'].keys():
            lam_x_proposed  = warmstart_solution_dict['opt_arg']['lam_x0']
            lam_g_proposed  = warmstart_solution_dict['opt_arg']['lam_g0']
        else:
            lam_x_proposed = np.zeros(nlp.V_bounds['ub'].shape)
            lam_g_proposed = np.zeros(nlp.g.shape)

    V_shape_matches = (V_init_proposed.cat.shape == nlp.V.cat.shape)
    if not V_shape_matches:
        raise ValueError('Variables of specified warmstart do not correspond to NLP requirements.')

    lam_x_shape_matches = (lam_x_proposed.shape == nlp.V_bounds['ub'].shape)
    if not lam_x_shape_matches:
        raise ValueError('Variable bound multipliers of specified warmstart do not correspond to NLP requirements.')

    lam_g_shape_matches = (lam_g_proposed.shape == nlp.g.shape)
    if not lam_g_shape_matches:
        raise ValueError('Constraint multipliers of specified warmstart do not correspond to NLP requirements.')

    return [V_init_proposed, lam_x_proposed, lam_g_proposed]

def dissolve_top_layer_of_struct(struct):

    dissolved_struct = {}
    for variable_type in list(struct.keys()):
        for variable in subkeys(struct, variable_type):
            dissolved_struct[variable] = struct[variable_type,variable]

    return dissolved_struct

def recursive_strip(d):
    if type(d) != dict:
        try:
            d = d(0)
        except:
            d = None
        return d
    else:
        for key in list(d.keys()):
            d[key] = recursive_strip(d[key])

def strip_of_contents(d):

    stripped_d = copy.deepcopy(d)
    recursive_strip(stripped_d)
    
    return stripped_d

def evaluate_cost_dict(cost_fun, V_plot, p_fix_num):

    cost = {}
    for name in list(cost_fun.keys()):
        if 'problem' not in name and 'objective' not in name:
            cost[name[:-4]] = cost_fun[name](V_plot, p_fix_num)

    return cost

def split_name_and_node_identifier(name):

    var_name = name
    kiteparent = ''

    while var_name[-1].isdigit():

        kiteparent = var_name[-1] + kiteparent
        var_name = var_name[:-1]

    return var_name, kiteparent


def split_kite_and_parent(kiteparent, architecture):

    for idx in range(1, len(kiteparent)):

        kite_try = int(kiteparent[:idx])
        parent_try = int(kiteparent[idx:])

        parent_of_kite_try = int(architecture.parent_map[kite_try])

        if parent_of_kite_try == parent_try:
            return kite_try, parent_try

    return None, None

def generate_variable_struct(variable_list):

    structs = {}
    for name in list(variable_list.keys()):
        structs[name] = cas.struct_symSX([cas.entry(variable_list[name][i][0], shape=variable_list[name][i][1])
                        for i in range(len(variable_list[name]))])

    variable_struct = cas.struct_symSX([cas.entry(name, struct=structs[name])
                        for name in list(variable_list.keys())])

    return variable_struct, structs

def find_output_idx(outputs, output_type, output_name, output_dim = 0):

    kk = 0
    can_index = outputs.getCanonicalIndex(kk)
    while not (can_index[0] == output_type and can_index[1] == output_name and can_index[2] == output_dim):
        kk += 1
        can_index = outputs.getCanonicalIndex(kk)

    return kk

def get_variable_from_model_or_reconstruction(variables, var_type, name):

    if var_type in variables.keys():
        sub_variables = variables[var_type]

    try:
        if isinstance(sub_variables, cas.MX):
            local_var = variables[var_type, name]
        elif isinstance(sub_variables, cas.structure3.SXStruct):
            local_var = variables[var_type][name]
        elif '[' + var_type + ',' + name + ',0]' in variables.labels():
            local_var = variables[var_type, name]
        elif isinstance(sub_variables, cas.DM):
            local_var = variables[var_type, name]
        else:
            local_var = variables[var_type, name]
    except:
        message = 'variable ' + name + ' is not in expected position (' + var_type + ') wrt variables.'
        print_op.log_and_raise_error(message)

    return local_var

def interpolate_solution(local_options, time_grids, variables_dict, V_opt, outputs_dict, outputs_opt, integral_output_names, integral_outputs_opt, Collocation=None, timegrid_label='ip', n_points=None):
    '''
    Postprocess tracking reference data from V-structure to (interpolated) data vectors
        with associated time grid
    :param nlp_options: the options of the nlp. notice that this should contain information about the nlp discretization, as well as the number of points desired for interpolation
    :param V_opt: the solution that we want to interpolate. remember, that for 'si' output, the numeric V input here, should also be in 'si' units. (for interpolating over the reference V, input optimization.V_ref here)
    :param time_grids: the time grids of the discretized problem. (for interpolating over the reference V, input optimization.time_grids['ref'] here)
    :return: a dictionary with entries corresponding to the interpolated variables
    '''

    nlp_discretization = local_options['discretization']
    collocation_scheme = local_options['collocation']['scheme']
    control_parametrization = local_options['collocation']['u_param']
    interpolation_type = local_options['interpolation']['type']
    n_k = local_options['n_k']
    collocation_d = local_options['collocation']['d']

    if n_points is None:
        n_points = local_options['interpolation']['n_points']

    if Collocation is not None:
        collocation_interpolator = Collocation.build_interpolator(local_options, V_opt)
        integral_collocation_interpolator = Collocation.build_interpolator(local_options, V_opt, integral_outputs=integral_outputs_opt)
    else:
        control_parametrization = 'zoh'
        collocation_interpolator = None
        integral_collocation_interpolator = None

    # add states and outputs to plotting dict
    interpolation = {'x': {}, 'u': {}, 'z': {}, 'theta': {}, 'time_grids': {}, 'outputs': {}, 'integral_outputs': {}}

    # time
    time_grid_interpolated = build_time_grid_for_interpolation(time_grids, n_points)
    time_grids['ip'] = time_grid_interpolated
    interpolation['time_grids'] = time_grids

    # x-values
    Vx_interpolated = interpolate_Vx(time_grids, variables_dict, V_opt, interpolation_type, nlp_discretization,
                                     collocation_scheme=collocation_scheme,
                                     collocation_interpolator=collocation_interpolator,
                                     timegrid_label=timegrid_label)
    interpolation['x'] = Vx_interpolated

    # z-values
    Vz_interpolated = interpolate_Vz(time_grids, variables_dict, V_opt, nlp_discretization,
                                     collocation_scheme=collocation_scheme,
                                     collocation_interpolator=collocation_interpolator,
                                     timegrid_label=timegrid_label)
    interpolation['z'] = Vz_interpolated

    # u-values
    Vu_interpolated = interpolate_Vu(time_grids, variables_dict, V_opt,
                                     control_parametrization=control_parametrization,
                                     collocation_interpolator=collocation_interpolator,
                                     timegrid_label=timegrid_label)
    interpolation['u'] = Vu_interpolated

    # theta values
    for name in list(subkeys(V_opt, 'theta')):
        interpolation['theta'][name] = V_opt['theta', name].full()[0][0]

    # output values
    interpolation['outputs'] = interpolate_outputs(time_grids, outputs_dict, outputs_opt, nlp_discretization,
                                                   n_k, collocation_d,
                                                   collocation_scheme=collocation_scheme,
                                                   timegrid_label=timegrid_label,
                                                   control_parametrization=control_parametrization)

    # integral-output values
    interpolation['integral_outputs'] = interpolate_integral_outputs(time_grids, integral_output_names,
                                                                     integral_outputs_opt, nlp_discretization,
                                                                     collocation_scheme=collocation_scheme,
                                                                     timegrid_label=timegrid_label,
                                                                     integral_collocation_interpolator=integral_collocation_interpolator)

    return interpolation

def build_time_grid_for_interpolation(time_grids, n_points):
    time_grid_interpolated = np.linspace(float(time_grids['x'][0]), float(time_grids['x'][-1]), n_points)
    return time_grid_interpolated

def interpolate_integral_outputs(time_grids, integral_output_names, integral_outputs_opt, nlp_discretization, collocation_scheme='radau', timegrid_label='ip', integral_collocation_interpolator=None):

    integral_outputs_interpolated = {}

    # integral-output values
    for name in integral_output_names:
        if name not in list(integral_outputs_interpolated.keys()):
            integral_outputs_interpolated[name] = []

        integral_output_dimension = integral_outputs_opt['int_out', 0, name].shape[0]

        for dim in range(integral_output_dimension):
            if (nlp_discretization == 'direct_collocation'):
                if (integral_collocation_interpolator is not None):
                    values_ip = integral_collocation_interpolator(time_grids[timegrid_label], name, dim, 'int_out')
                else:
                    message = 'awebox is not yet able to interpolate integral_outputs for direct collocation without the use of the integral_collocation_interpolator'
                    print_op.log_and_raise_error(message)
            else:
                output_values = cas.DM(integral_outputs_opt['int_out', :, name, dim])
                tgrid = time_grids['x']

                # make list of time grid and values
                tgrid = list(chain.from_iterable(tgrid.full().tolist()))
                output_values = output_values.full().squeeze()

                values_ip = vect_op.spline_interpolation(tgrid, output_values, time_grids[timegrid_label])

            integral_outputs_interpolated[name] += [values_ip]

    return integral_outputs_interpolated

def interpolate_outputs_by_index(time_grids, outputs_opt, odx, nlp_discretization, n_k, collocation_d, collocation_scheme='radau', control_parametrization='poly', timegrid_label='ip'):

    # merge values
    values, time_grid = merge_output_values(outputs_opt, odx, time_grids, nlp_discretization, n_k, collocation_d, collocation_scheme=collocation_scheme, control_parametrization=control_parametrization)

    # interpolate
    values_ip = vect_op.spline_interpolation(time_grid, values, time_grids[timegrid_label])
    return values_ip


def interpolate_outputs(time_grids, outputs_dict, outputs_opt, nlp_discretization, n_k, collocation_d, collocation_scheme='radau', timegrid_label='ip', control_parametrization='poly'):

    outputs_interpolated = {}

    expected_number_of_outputs = outputs_opt.shape[0]

    odx = 0
    for output_type in outputs_dict.keys():
        if output_type not in list(outputs_interpolated.keys()):
            outputs_interpolated[output_type] = {}

        for name in outputs_dict[output_type].keys():
            if name not in list(outputs_interpolated[output_type].keys()):
                outputs_interpolated[output_type][name] = []

            for dim in range(outputs_dict[output_type][name].shape[0]):
                # interpolate
                values_ip = interpolate_outputs_by_index(time_grids, outputs_opt, odx, nlp_discretization, n_k, collocation_d, collocation_scheme=collocation_scheme, timegrid_label=timegrid_label, control_parametrization=control_parametrization)

                # store
                outputs_interpolated[output_type][name] += [values_ip]
                odx += 1

    if not odx == expected_number_of_outputs:
        message = 'something went wrong when interpolating outputs. the number of outputs that were interpolated is ' + str(odx) + ' when it should have been ' + str(expected_number_of_outputs)
        print_op.log_and_raise_error(message)

    return outputs_interpolated


def interpolate_Vu(time_grids, variables_dict, V, control_parametrization='zoh', collocation_interpolator=None, timegrid_label='ip'):

    Vu_interpolated = {}

    var_type = 'u'
    variable_names = variables_dict[var_type].keys()
    for name in variable_names:
        Vu_interpolated[name] = []

        variable_dimension = variables_dict[var_type][name].shape[0]
        for dim in range(variable_dimension):
            if control_parametrization == 'zoh':
                control = V['u', :, name, dim]
                values_ip = sample_and_hold_controls(time_grids, control, timegrid_label=timegrid_label)

            elif (control_parametrization == 'poly') and (collocation_interpolator is not None):
                values_ip = collocation_interpolator(time_grids[timegrid_label], name, dim, 'u')

            Vu_interpolated[name] += [values_ip]

    return Vu_interpolated


def interpolate_Vz(time_grids, variables_dict, V, nlp_discretization, collocation_scheme='radau', collocation_interpolator=None, include_collocation=True, timegrid_label='ip'):

    Vz_interpolated = {}

    var_type = 'z'
    variable_names = variables_dict[var_type].keys()
    for name in variable_names:
        Vz_interpolated[name] = []

        variable_dimension = variables_dict[var_type][name].shape[0]
        for dim in range(variable_dimension):

            if (nlp_discretization == 'direct_collocation') and (collocation_interpolator is not None):
                values_ip = collocation_interpolator(time_grids[timegrid_label], name, dim, 'z')
            else:
                values, time_grid_data = merge_z_values(V, name, dim, time_grids, nlp_discretization,
                                                   collocation_scheme=collocation_scheme,
                                                   include_collocation=include_collocation)
                # interpolate
                values_ip = vect_op.spline_interpolation(time_grid_data, values, time_grids[timegrid_label])

            if hasattr(values_ip, 'full'):
                Vz_interpolated[name] += [values_ip.full()]
            else:
                Vz_interpolated[name] += [values_ip]

    return Vz_interpolated


def interpolate_Vx(time_grids, variables_dict, V, interpolation_type, nlp_discretization, collocation_scheme='radau', collocation_interpolator=None, include_collocation=True, timegrid_label='ip'):

    Vx_interpolated = {}

    var_type = 'x'
    variable_names = variables_dict[var_type].keys()
    for name in variable_names:
        Vx_interpolated[name] = []

        variable_dimension = variables_dict[var_type][name].shape[0]
        for dim in range(variable_dimension):

            # interpolate
            if (interpolation_type == 'spline') or (nlp_discretization == 'multiple_shooting'):
                values_data, time_grid_data = merge_x_values(V, name, dim, time_grids, nlp_discretization,
                                                             collocation_scheme=collocation_scheme,
                                                             include_collocation=include_collocation)
                values_ip = vect_op.spline_interpolation(time_grid_data, values_data, time_grids[timegrid_label])

            elif (interpolation_type == 'poly') and (nlp_discretization == 'direct_collocation') and (collocation_interpolator is not None):
                values_ip = collocation_interpolator(time_grids[timegrid_label], name, dim, 'x')

            else:
                message = 'interpolation not yet enabled for the combination of interpolation_type (' + interpolation_type + ') and nlp_discretization (' + nlp_discretization + ')'
                print_op.log_and_raise_error(message)

            if hasattr(values_ip, 'full'):
                Vx_interpolated[name] += [values_ip.full()]
            else:
                Vx_interpolated[name] += [values_ip]

    return Vx_interpolated


def merge_output_values(outputs_opt, odx, time_grids, nlp_discretization, n_k, collocation_d, collocation_scheme='radau', control_parametrization='poly'):

    using_collocation = (nlp_discretization == 'direct_collocation')

    output_values = outputs_opt[odx, :]

    # read in inputs
    if using_collocation:
        using_radau = (collocation_scheme == 'radau')
        if using_radau:
            tgrid = time_grids['coll']

            using_zoh_controls = (control_parametrization == 'zoh')
            if using_zoh_controls:
                outputs_reshaped = output_values.reshape((collocation_d+1, n_k)).T
                outputs_without_duplicate_gridpoints = outputs_reshaped[:, 1:]
                output_values = vect_op.columnize(outputs_without_duplicate_gridpoints)

        else:
            tgrid = time_grids['x_coll'][:-1]
    else:
        tgrid = time_grids['x'][:-1]

    # make list of time grid and values
    tgrid = list(chain.from_iterable(tgrid.full().tolist()))
    values = output_values.full().squeeze()

    assert (not vect_op.data_is_obviously_uninterpolatable(tgrid, values))

    return values, tgrid

def merge_z_values(V, name, dim, time_grids, nlp_discretization, collocation_scheme='radau', include_collocation=True):

    # read in inputs
    if nlp_discretization == 'direct_collocation':
        tgrid_coll = time_grids['coll']
        # total time points
        tgrid_z_coll = time_grids['x_coll'][:-1]

    # interval time points
    tgrid_z = time_grids['u']
    n_k = tgrid_z.shape[0]

    if nlp_discretization == 'multiple_shooting':
        # take interval values
        z_values = np.array(cas.vertcat(*V['z', :, name, dim]).full())
        tgrid = tgrid_z

    elif nlp_discretization == 'direct_collocation':
        if collocation_scheme == 'radau':
            if include_collocation:
                # add node values
                z_values = np.array(coll_slice_to_vec(V['coll_var', :, :, 'z', name, dim]))
                tgrid = tgrid_coll
            else:
                z_values = []
                tgrid = []
        else:
            z_values = []
            # merge interval and node values
            for ndx in range(n_k):
                # add interval values
                z_values = cas.vertcat(z_values, V['z', ndx, name, dim])
                if include_collocation:
                    # add node values
                    z_values = cas.vertcat(z_values, cas.vertcat(*V['coll_var', ndx, :, 'z', name, dim]))
            z_values = np.array(z_values)

            if include_collocation:
                tgrid = tgrid_z_coll
            else:
                tgrid = tgrid_z

    # make list of time grid and values
    tgrid = list(chain.from_iterable(tgrid.full().tolist()))
    values = list(chain.from_iterable(z_values))

    assert (not vect_op.data_is_obviously_uninterpolatable(tgrid, values))

    return values, tgrid


def merge_x_values(V, name, dim, time_grids, nlp_discretization, collocation_scheme='radau', include_collocation=True):

    # read in inputs
    if nlp_discretization == 'direct_collocation':
        tgrid_coll = time_grids['coll']
        # total time points
        tgrid_x_coll = time_grids['x_coll']

    # interval time points
    tgrid_x = time_grids['x']
    n_k = tgrid_x.shape[0] - 1

    if nlp_discretization == 'multiple_shooting':
        # take interval values
        x_values = np.array(cas.vertcat(*V['x', :, name, dim]).full())
        tgrid = tgrid_x

    elif nlp_discretization == 'direct_collocation':
        if collocation_scheme == 'radau':
            if include_collocation:
                # add node values
                x_values = np.array(coll_slice_to_vec(V['coll_var',:, :, 'x', name,dim]))
                tgrid = tgrid_coll
            else:
                x_values = []
                tgrid = []

        else:
            x_values = []
            # merge interval and node values
            for ndx in range(n_k + 1):
                # add interval values
                x_values = cas.vertcat(x_values, V['x', ndx, name, dim])
                if include_collocation and (ndx < n_k):
                    # add node values
                    x_values = cas.vertcat(x_values, cas.vertcat(*V['coll_var', ndx, :, 'x', name, dim]).full())

            x_values = np.array(x_values)
            if include_collocation:
                tgrid = tgrid_x_coll
            else:
                tgrid = tgrid_x

    # make list of time grid
    tgrid = list(chain.from_iterable(tgrid.full().tolist()))
    values = list(chain.from_iterable(x_values))

    assert (not vect_op.data_is_obviously_uninterpolatable(tgrid, values))

    return values, tgrid



def sample_and_hold_controls(time_grids, control, timegrid_label='ip'):

    tgrid_u = time_grids['u']
    tgrid_ip = time_grids[timegrid_label]
    values_ip = np.zeros(len(tgrid_ip),)
    for idx in range(len(tgrid_ip)):
        for jdx in range(tgrid_u.shape[0] - 1):
            if tgrid_u[jdx] < tgrid_ip[idx] and tgrid_ip[idx] < tgrid_u[jdx + 1]:
                values_ip[idx] = control[jdx]
                break
        if tgrid_u[-1] < tgrid_ip[idx]:
            values_ip[idx] = control[-1]

    tgrid = tgrid_ip
    values = values_ip

    if vect_op.data_is_obviously_uninterpolatable(tgrid, values):
        return None

    return values

def test():
    # todo
    awelogger.logger.warning('no tests currently defined for struct_operations')
    return None