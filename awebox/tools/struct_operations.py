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

import casadi.tools as cas
import numpy as np

import operator

import copy
from functools import reduce
from awebox.logger.logger import Logger as awelogger
import awebox.tools.print_operations as print_op
import awebox.tools.performance_operations as perf_op


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
    n_k = nlp_options['n_k']
    periodic = perf_op.determine_if_periodic(nlp_options)
    if periodic:
        shooting_nodes = n_k
    else:
        shooting_nodes = n_k + 1

    return shooting_nodes

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

    use_vortex_linearization = 'lin' in parameters.keys()
    if use_vortex_linearization:
        Xdot = construct_Xdot_struct(nlp_options, model.variables_dict)(0.)

        coll_params = []
        for kdx in range(shooting_nodes):
            loc_params = get_parameters_at_time(nlp_options, P, V, Xdot, model.variables, model.parameters, kdx)
            coll_params = cas.horzcat(coll_params, loc_params)

    else:
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

def get_algebraics_at_time(nlp_options, V, model_variables, var_type, kdx, ddx=None):

    if (ddx is None):
        return V[var_type, kdx]
    else:
        return V['coll_var', kdx, ddx, var_type]


def get_states_at_time(nlp_options, V, model_variables, kdx, ddx=None):

    var_type = 'xd'

    direct_collocation = (nlp_options['discretization'] == 'direct_collocation')
    at_control_node = (ddx is None)

    if at_control_node:
        return V[var_type, kdx]
    elif direct_collocation:
        return V['coll_var', kdx, ddx, var_type]
    else:
        return no_available_var_info(model_variables, var_type)


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

    else:
        return no_available_var_info(model_variables, var_type)


def get_derivs_at_time(nlp_options, V, Xdot, model_variables, kdx, ddx=None):

    var_type = 'xddot'

    at_control_node = (ddx is None)
    lifted_derivs = (var_type in list(V.keys()))

    if Xdot is not None:
        empty_Xdot = Xdot(0.)
        passed_Xdot_is_meaningful = not (Xdot == empty_Xdot)
    else:
        passed_Xdot_is_meaningful = False

    if at_control_node:
        if lifted_derivs:
            return V[var_type, kdx]
        elif passed_Xdot_is_meaningful:
            return Xdot['xd', kdx]
    elif passed_Xdot_is_meaningful:
        return Xdot['coll_xd', kdx, ddx]
    else:
        return no_available_var_info(model_variables, var_type)


def get_variables_at_time(nlp_options, V, Xdot, model_variables, kdx, ddx=None):

    var_list = []
    # make list of variables at specific time
    for var_type in model_variables.keys():

        if var_type in {'xl', 'xa'}:
            local_var = get_algebraics_at_time(nlp_options, V, model_variables, var_type, kdx, ddx)

        elif var_type == 'xd':
            local_var = get_states_at_time(nlp_options, V, model_variables, kdx, ddx)

        elif var_type == 'u':
            local_var = get_controls_at_time(nlp_options, V, model_variables, kdx, ddx)

        elif var_type == 'theta':
            local_var = get_V_theta(V, nlp_options, kdx)

        elif var_type == 'xddot':
            local_var = get_derivs_at_time(nlp_options, V, Xdot, model_variables, kdx, ddx)

        else:
            local_var = no_available_var_info(model_variables, var_type)

        var_list.append(local_var)

    var_at_time = model_variables(cas.vertcat(*var_list))

    return var_at_time



def get_variables_at_final_time(nlp_options, V, Xdot, model):

    multiple_shooting = (nlp_options['discretization'] == 'multiple_shooting')

    scheme = nlp_options['collocation']['scheme']
    direct_collocation = (nlp_options['discretization'] == 'direct_collocation')
    radau_collocation = (direct_collocation and scheme == 'radau')
    other_collocation = (direct_collocation and (not scheme == 'radau'))

    if radau_collocation:
        var_at_time = get_variables_at_time(nlp_options, V, Xdot, model.variables, -1, -1)
    elif other_collocation or multiple_shooting:
        var_at_time = get_variables_at_time(nlp_options, V, Xdot, model.variables, -1)
    else:
        message = 'unfamiliar discretization option chosen: ' + nlp_options['discretization']
        awelogger.logger.error(message)
        raise Exception(message)

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
        kdx = int(n_k * t / V['theta','t_f'])
        tau = t / V['theta', 't_f']*n_k - kdx

    if kdx == n_k:
        kdx = n_k - 1
        tau = 1.0

    return kdx, tau

def var_si_to_scaled(var_type, var_name, var_si, scaling):

    scaling_defined_for_variable = (var_type in scaling.keys()) and (var_name in scaling[var_type].keys())
    if scaling_defined_for_variable:

        scale = scaling[var_type][var_name]

        if scale.shape == (1, 1):

            use_unit_scaling = (scale == cas.DM(1.)) or (scale == 1.)
            if use_unit_scaling:
                return var_si
            else:
                return var_si / scale

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
            awelogger.logger.error(message)
            raise Exception(message)

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
            awelogger.logger.error(message)
            raise Exception(message)

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
            return variable_type

    if name in list(variables_dict['xddot'].keys()):
        return 'xddot'

    message = 'variable ' + name + ' not found in variables dictionary'
    awelogger.logger.error(message)
    raise Exception(message)

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
            awelogger.logger.error(message)
            raise Exception(message)

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
            for var_type in set(['xd','theta','phi','xi','xa','xl','xddot']):
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

            # initialize path constraint multipliers
            for i in range(n_k):
                lam_g_proposed['path',i] = lam_g_coll['path',i]

            # initialize periodicity multipliers
            if 'periodic' in list(lam_g_coll.keys()) and 'periodic' in list(lam_g_proposed.keys()):
                lam_g_proposed['periodic'] = lam_g_coll['periodic']

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

def get_variable_from_model_or_reconstruction(variables, var_type, name):

    if var_type in variables.keys():
        sub_variables = variables[var_type]

    try:
        if isinstance(sub_variables, cas.DM):
            local_var = variables[var_type, name]
        elif isinstance(sub_variables, cas.MX):
            local_var = variables[var_type, name]
        elif isinstance(sub_variables, cas.structure3.SXStruct):
            local_var = variables[var_type][name]
        else:
            local_var = variables[var_type, name]
    except:

        message = 'variable ' + name + ' is not in expected position (' + var_type + ') wrt variables.'
        awelogger.logger.error(message)
        raise Exception(message)

    return local_var