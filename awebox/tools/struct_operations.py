#
#    This file is part of awebox.
#
#    awebox -- A modeling and optimization framework for multi-kite AWE systems.
#    Copyright (C) 2017-2020 Jochem De Schutter, Rachel Leuthold, Moritz Diehl,
#                            ALU Freiburg.
#    Copyright (C) 2018-2019 Thilo Bronnenmeyer, Kiteswarms Ltd.
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

import casadi.tools as cas
import numpy as np

import operator

import copy
from functools import reduce
from awebox.logger.logger import Logger as awelogger

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

    use_vortex_linearization = 'lin' in parameters.keys()
    if use_vortex_linearization:
        Xdot = construct_Xdot_struct(nlp_options, model.variables_dict)(0.)

        coll_params = []
        for kdx in range(n_k):
            for ddx in range(d):
                loc_params = get_parameters_at_time(nlp_options, P, V, Xdot, model.variables, model.parameters, kdx, ddx)
                coll_params = cas.horzcat(coll_params, loc_params)

    else:
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

    use_vortex_linearization = 'lin' in parameters.keys()
    if use_vortex_linearization:
        message = 'vortex induction model not yet supported for multiple shooting problems.'
        awelogger.logger.error(message)

    ms_params = cas.repmat(parameters(cas.vertcat(P['theta0'], V['phi'])), 1, N_ms)

    return ms_params


def get_variables_at_time(nlp_options, V, Xdot, model_variables, kdx, ddx=None):

    var_list = []

    # extract discretization type
    if nlp_options['discretization'] == 'direct_collocation':
        direct_collocation = True
        scheme = nlp_options['collocation']['scheme']
        u_param = nlp_options['collocation']['u_param']
    else:
        direct_collocation = False

    # extract variables
    variables = model_variables

    # make list of variables at specific time
    for var_type in list(variables.keys()):

        # algebraic variables
        if var_type in {'xl', 'xa'}:

            if direct_collocation and (scheme == 'radau'):
                # note that this shifting pattern is not strictly true,
                # but is requried to prevent licq errors for simple xl = 0 constraints
                # at nodes (d+1) and (d) from equivalence
                if ddx == None:
                    var_list.append(V['coll_var', kdx, 0, var_type])
                else:
                    var_list.append(V['coll_var', kdx, ddx, var_type])

            elif direct_collocation and (scheme != 'radau'):
                if ddx == None:
                    if var_type in list(V.keys()): # check if alg vars are lifted
                        var_list.append(V[var_type, kdx])
                    else: # not lifted
                        var_list.append(np.zeros(variables[var_type].shape)) # implicit function of other states
                else:
                    var_list.append(V['coll_var', kdx, ddx, var_type])
            else:
                if var_type in list(V.keys()): # check if lifted
                    var_list.append(V[var_type, kdx])
                else:
                    var_list.append(np.zeros(variables[var_type].shape)) # implicit function of other states

        # differential states
        elif var_type == 'xd':
            if ddx == None:
                var_list.append(V[var_type, kdx])
            else:
                var_list.append(V['coll_var', kdx, ddx, var_type])

        # controls
        elif var_type == 'u':

            if direct_collocation:
                if (u_param == 'poly'):
                    if ddx == None:
                        var_list.append(V['coll_var', kdx, 0, var_type])
                    else:
                        var_list.append(V['coll_var', kdx, ddx, var_type])
                else:
                    var_list.append(V[var_type, kdx])
            else:
                var_list.append(V[var_type, kdx])

        # parameters
        elif var_type == 'theta':
            var_list.append(get_V_theta(V, nlp_options, kdx))

        # state derivatives
        elif var_type == 'xddot':
            if ddx == None:
                if var_type in list(V.keys()): #  check if xddot is lifted
                    var_list.append(V[var_type, kdx])
                else: # not lifted
                    var_list.append(np.zeros(variables[var_type].shape)) # implicit function of other states

            else:
                var_list.append(Xdot['coll_xd', kdx, ddx])

        else:
            raise ValueError("iterating over non-supported model variable type")

    var_at_time = variables(cas.vertcat(*var_list))

    return var_at_time

def get_variables_at_final_time(nlp_options, V, Xdot, model):
    nk = nlp_options['n_k']

    var_list = []

    # extract variables
    variables = model.variables

    # make list of variables at specific time
    for var_type in list(variables.keys()):

        # algebraic variables
        if var_type in {'xa','xl','xddot','u'}:

            var_list.append(np.zeros(variables[var_type].shape))

        # differential states
        elif var_type == 'xd':

            var_list.append(V['xd', nk])

        # parameters
        elif var_type == 'theta':
            var_list.append(get_V_theta(V, nlp_options, nk))

        else:
            raise ValueError("iterating over non-supported model variable type")

    var_at_time = variables(cas.vertcat(*var_list))

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

    var_list = []

    # extract discretization type
    if nlp_options['discretization'] == 'direct_collocation':
        direct_collocation = True
        scheme = nlp_options['collocation']['scheme']
        u_param = nlp_options['collocation']['u_param']
    else:
        direct_collocation = False

    variables = model.variables
    for var_type in list(variables.keys()):

        # algebraic variables
        if var_type in {'xl', 'xa'}:

            if direct_collocation and (scheme == 'radau'):
                # note that this shifting pattern is not strictly true,
                # but is requried to prevent licq errors for simple xl = 0 constraints
                # at nodes (d+1) and (d) from equivalence
                if ddx == None:
                    var_list.append(np.zeros(variables[var_type].shape))
                else:
                    var_list.append(P['p', 'ref','coll_var', kdx, ddx, var_type])

            elif direct_collocation and (scheme != 'radau'):
                if ddx == None: # interval node
                    if var_type in list(V.keys()): # check if alg vars are lifted
                        var_list.append(P['p', 'ref',var_type, kdx])
                    else: # not lifted
                        var_list.append(np.zeros(variables[var_type].shape)) # will not be used
                else:
                    var_list.append(P['p', 'ref','coll_var', kdx, ddx, var_type])
            else: # multiple shooting
                if var_type in list(V.keys()): # check if lifted
                    var_list.append(P['p', 'ref',var_type, kdx])
                else: # not lifted
                    var_list.append(np.zeros(variables[var_type].shape)) # will not be used
        # differential states
        elif var_type == 'xd':
            if ddx == None:
                var_list.append(P['p', 'ref',var_type, kdx])
            else:
                var_list.append(P['p', 'ref','coll_var', kdx, ddx, var_type])

        # controls
        elif var_type == 'u':

            if direct_collocation:
                if (u_param == 'poly'):
                    if ddx == None:
                        var_list.append(np.zeros(variables[var_type].shape))
                    else:
                        var_list.append(P['p', 'ref','coll_var', kdx, ddx, var_type])
                else:
                    var_list.append(P['p', 'ref',var_type, kdx])
            else:
                var_list.append(P['p', 'ref',var_type, kdx])

        # parameters
        elif var_type == 'theta':
            var_list.append(get_P_theta(P, nlp_options, kdx))

        # state derivatives
        elif var_type == 'xddot':
            if ddx == None: # interval node
                if var_type in list(V.keys()): # check if xddot is lifted
                    var_list.append(V[var_type, kdx])
                else: # not lifted
                    var_list.append(np.zeros(variables[var_type].shape)) # will not be used
            else:
                var_list.append(Xdot['coll_xd', kdx, ddx])

        else:
            raise ValueError("iterating over non-supported model variable type")

    var_at_time = variables(cas.vertcat(*var_list))

    return var_at_time

def get_var_ref_at_final_time(nlp_options, P, Xdot, model):

    var_list = []
    nk = nlp_options['n_k']

    # extract variables
    variables = model.variables

    # make list of variables at specific time
    for var_type in list(variables.keys()):

        # algebraic variables
        if var_type in {'xa','xl','xddot','u'}:

            var_list.append(np.zeros(variables[var_type].shape))

        # differential states
        elif var_type == 'xd':

            var_list.append(P['p','ref','xd', nk])

        # parameters
        elif var_type == 'theta':
            var_list.append(get_P_theta(P, nlp_options, nk))

        else:
            raise ValueError("iterating over non-supported model variable type")

    var_at_time = variables(cas.vertcat(*var_list))

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

def get_P_theta(P, params, k):

    nk = params['n_k']
    k = list(range(nk+1))[k]

    if P['p','ref','theta','t_f'].shape[0] == 1:
        theta = P['p','ref','theta']
    else:
        tf_index = P.f['p','ref','theta','t_f']
        theta_index = P.f['p','ref','theta']
        theta = []
        for idx in theta_index:
            if idx == tf_index[0] and k < round(nk * params['phase_fix_reelout']):
                theta.append(P.cat[idx])
            elif idx == tf_index[1] and k >= round(nk * params['phase_fix_reelout']) :
                theta.append(P.cat[idx])
            elif idx not in tf_index:
                theta.append(P.cat[idx])
        theta = cas.vertcat(*theta)

    return theta

def calculate_tf(params, V, k):

    nk = params['n_k']

    if params['phase_fix'] == True:
        if k < round(nk * params['phase_fix_reelout']):
            tf = V['theta', 't_f', 0]
        else:
            tf = V['theta', 't_f', 1]
    else:
        tf = V['theta', 't_f']

    return tf

def calculate_kdx(params, V, t):

    n_k = params['n_k']

    if params['phase_fix'] == True:
        k_reelout = round(n_k * params['phase_fix_reelout'])
        t_reelout = k_reelout*V['theta','t_f',0]/n_k
        if t <= t_reelout:
            kdx = int(n_k * t / V['theta','t_f',0])
            tau = t / V['theta', 't_f',0]*n_k - kdx
        else:
            kdx = int(k_reelout + int(n_k * (t - t_reelout) / V['theta','t_f',1]))
            tau = (t - t_reelout)/ V['theta','t_f',1]*n_k - (kdx-k_reelout)
    else:
        kdx = int(n_k * t / V['theta','t_f'])
        tau = t / V['theta', 't_f']*n_k - kdx

    if kdx == n_k:
        kdx = n_k - 1
        tau = 1.0

    return kdx, tau

def si_to_scaled(model, V_ori):
    V = copy.deepcopy(V_ori)


    n_k = len(V_ori['xd']) - 1
    if 'coll_var' in list(V.keys()):
        d = len(V_ori['coll_var',0,:,'xd'])
        direct_collocation = True
    else:
        direct_collocation = False

    for variable_type in list(model.variables.keys()):

        for name in subkeys(model.variables, variable_type):

            if variable_type == 'theta':
                V[variable_type, name] = V[variable_type, name] / model.scaling[variable_type][name]

            elif variable_type == 'u':
                for kdx in range(n_k):
                    if variable_type in list(V.keys()):
                        V[variable_type, kdx, name] = V[variable_type, kdx, name] / model.scaling[variable_type][name]
                    else:
                        for ddx in range(d):
                            V['coll_var', kdx, ddx, variable_type, name] = V['coll_var', kdx, ddx, variable_type, name] / model.scaling[variable_type][name]

            elif variable_type in set(['xa', 'xl','xd','xddot']):
                if variable_type in list(V.keys()):
                    if variable_type == 'xd':
                        for kdx in range(n_k+1):
                            V[variable_type, kdx, name] = V[variable_type, kdx, name] / model.scaling[variable_type][name]
                    else:
                        for kdx in range(n_k):
                            V[variable_type, kdx, name] = V[variable_type, kdx, name] / model.scaling[variable_type][name]

                if (direct_collocation and variable_type != 'xddot'):
                    for kdx in range(n_k):
                        for ddx in range(d):
                            V['coll_var', kdx, ddx, variable_type, name] = V['coll_var', kdx, ddx, variable_type, name] / model.scaling[variable_type][name]

    return V


def scaled_to_si(variables, scaling, n_k, d, V_ori):
    V = copy.deepcopy(V_ori)

    if 'coll_var' in list(V.keys()):
        direct_collocation = True
    else:
        direct_collocation = False

    for variable_type in list(variables.keys()):

        for name in subkeys(variables, variable_type):

            if variable_type == 'theta':
                V[variable_type, name] = V[variable_type, name] * scaling[variable_type][name]

            elif variable_type == 'u':
                for kdx in range(n_k):
                    if variable_type in list(V.keys()):
                        V[variable_type, kdx, name] = V[variable_type, kdx, name] * scaling[variable_type][name]
                    else:
                        for ddx in range(d):
                            V['coll_var', kdx, ddx, variable_type, name] = V['coll_var', kdx, ddx, variable_type, name] *scaling[variable_type][name]

            elif variable_type in set(['xa', 'xl','xd','xddot']):
                if variable_type in list(V.keys()):
                    if variable_type == 'xd':
                        for kdx in range(n_k+1):
                            V[variable_type, kdx, name] = V[variable_type, kdx, name] * scaling[variable_type][name]
                    else:
                        for kdx in range(n_k):
                            V[variable_type, kdx, name] = V[variable_type, kdx, name] * scaling[variable_type][name]

                if (direct_collocation and variable_type != 'xddot'):
                    for kdx in range(n_k):
                        for ddx in range(d):
                            V['coll_var', kdx, ddx, variable_type, name] = V['coll_var', kdx, ddx, variable_type, name] * scaling[variable_type][name]

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

    for variable_type in set(variables_dict.keys()) - set(['xddot']):
        if name in list(variables_dict[variable_type].keys()):
            var_type = variable_type

    return var_type

def get_node_variable_name(name):

    var_name = name
    while var_name[-1].isdigit():
        var_name = var_name[:-1]

    return var_name

def get_scaling_name(scaling_options, variable_type, name):

    scaling_name = name
    while not scaling_name in list(scaling_options[variable_type].keys()) and len(scaling_name) > 0:
        scaling_name = scaling_name[1:]

    return scaling_name

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


    if canonical[0] == 'coll_var':
        coll_flag = True
        canonical = (canonical[3],) + canonical[1:3] + canonical[4:]

    elif canonical[0] == 'us':
        coll_flag = None
        var_type = None
        kdx = None
        ddx = None
        name = None
        dim = None

    else:
        coll_flag = False

    if coll_flag is not None:
        length = len(canonical)

        var_type = canonical[0]
        kdx = None
        ddx = None

        if length == 5:
            kdx = canonical[1]
            ddx = canonical[2]
            name = canonical[3]
            dim = canonical[4]

        elif length == 4:
            kdx = canonical[1]
            name = canonical[2]
            dim = canonical[3]

        elif length == 3:
            name = canonical[1]
            dim = canonical[2]


    return [coll_flag, var_type, kdx, ddx, name, dim]

def construct_Xdot_struct(nlp_options, variables_dict):
    ''' Construct a symbolic structure for the
        discretized state derivatives.

    @param nlp_options - discretization options
    @param model - awebox model
    @return Vdot - discretized state derivatives sym. struct.
    '''

    # extract information
    nk = nlp_options['n_k']
    xd = variables_dict['xd']

    # derivatives at interval nodes
    entry_tuple = (cas.entry('xd', repeat=[nk], struct=xd),)

    # add derivatives on collocation nodes
    if nlp_options['discretization'] == 'direct_collocation':
        d = nlp_options['collocation']['d']
        entry_tuple += (cas.entry('coll_xd', repeat=[nk,d], struct=xd),)

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

            lam_x_proposed  = nlp.V_bounds['ub'](0.0)
            lam_x_coll = V_coll(warmstart_solution_dict['opt_arg']['lam_x0'])

            lam_g_proposed  = nlp.g(0.0)
            lam_g_coll = warmstart_solution_dict['g_opt'](warmstart_solution_dict['opt_arg']['lam_g0'])
            g_coll = warmstart_solution_dict['g_opt']

            # initialize regular variables
            for var_type in set(['xd','theta','phi','xi']):
                V_init_proposed[var_type] = V_coll[var_type]
                lam_x_proposed[var_type]  = lam_x_coll[var_type]

            if 'u' in list(V_coll.keys()):
                V_init_proposed['u'] = V_coll['u']
                lam_x_proposed['u']  = lam_x_coll['u']
            else:
                for i in range(n_k):
                    # note: this does not give the actual mean, implement with quadrature weights instead
                    V_init_proposed['u',i] = np.mean(cas.horzcat(*V_coll['coll_var',i,:,'u']))
                    lam_x_proposed['u',i]  = np.mean(cas.horzcat(*lam_x_coll['coll_var',i,:,'u']))

            if 'xddot' in list(V_init_proposed.keys()):
                V_init_proposed['xddot'] = Xdot_coll['xd']

            # initialize slacks
            if 'us' in list(V_init_proposed.keys()):
                for i in range(n_k-1):
                    V_init_proposed['us',i+1] = g_coll['stage_constraints',i,-1,'path_constraints','inequality']

            # initialize algebraic variables on all but first interval node
            for var_type in set(['xa','xl']):
                if var_type in list(V_init_proposed.keys()):
                    for i in range(n_k-1):
                        V_init_proposed[var_type,i+1] = V_coll['coll_var',i,-1,var_type]
                        lam_x_proposed[var_type,i+1]  = lam_x_coll['coll_var',i,-1,var_type]

            # initialize path constraint multipliers on all but first interval node
            for i in range(n_k-1):
                lam_g_proposed['path_constraints',i+1] = lam_g_coll['stage_constraints',i,-1,'path_constraints']

            # initialize multipliers and alg_vars on first interval depending on periodicity
            if 'periodic' in list(lam_g_coll.keys()) and 'periodic' in list(lam_g_proposed.keys()):
                lam_g_proposed['periodic'] = lam_g_coll['periodic']
                lam_g_proposed['path_constraints',0] = lam_g_coll['stage_constraints',-1,-1,'path_constraints']
                for var_type in set(['xa','xl']):
                    if var_type in list(V_init_proposed.keys()):
                        V_init_proposed[var_type,0] = V_coll['coll_var',-1,-1,var_type]
                        lam_x_proposed[var_type,0]  = lam_x_coll['coll_var',-1,-1,var_type]
                if 'us' in list(V_init_proposed.keys()):
                    V_init_proposed['us',0] = g_coll['stage_constraints',-1,-1,'path_constraints']

            else:
                lam_g_proposed['path_constraints',0] = lam_g_coll['stage_constraints',0,0,'path_constraints']
                for var_type in set(['xa','xl']):
                    if var_type in list(V_init_proposed.keys()):
                        V_init_proposed[var_type,0] = V_coll['coll_var',0,0,var_type]
                        lam_x_proposed[var_type,0]  = lam_x_coll['coll_var',0,0,var_type]
                if 'us' in list(V_init_proposed.keys()):
                    V_init_proposed['us',0] = g_coll['stage_constraints',0,0,'path_constraints']

            # initialize continuity multipliers
            lam_g_proposed['continuity'] = lam_g_coll['continuity']

        else:
            raise ValueError('Warmstart from multiple shooting to collocation not supported')

    else:

        V_init_proposed = warmstart_solution_dict['V_opt']
        lam_x_proposed  = warmstart_solution_dict['opt_arg']['lam_x0']
        lam_g_proposed  = warmstart_solution_dict['opt_arg']['lam_g0']


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